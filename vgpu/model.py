"""TinyGPT, written as a chain of pure functions.

The model is deliberately *not* an nn.Module with .parameters().  Instead it
is a list of `LayerSpec`s, each of which is

    (name, [(param_name, shape), ...], fn(x, P) -> y)

where `P` is a plain dict of tensors.  That indirection is the whole point:
it lets ZeRO-3 materialise a layer's weights (by all-gathering them) one
microsecond before the layer runs and throw them away one microsecond after,
which is impossible if the weights are permanently bolted onto a module.

Architecture: a pre-LayerNorm decoder-only transformer (GPT-2 shaped).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 512
    block_size: int = 32
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 128

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_head


@dataclass
class LayerSpec:
    name: str
    params: list          # [(pname, shape)]
    fn: object            # fn(x, P) -> y

    @property
    def numel(self) -> int:
        return sum(int(torch.tensor(s).prod()) for _, s in self.params)


# --------------------------------------------------------------------------
# layer functions
# --------------------------------------------------------------------------
def _embed_fn(cfg):
    def fn(idx, P):
        T = idx.shape[1]
        return P["wte"][idx] + P["wpe"][:T]
    return fn


def _block_fn(cfg):
    nh, dh = cfg.n_head, cfg.d_head

    def fn(x, P):
        B, T, C = x.shape
        # --- attention (pre-LN) ---
        h = F.layer_norm(x, (C,), P["ln1_w"], P["ln1_b"])
        qkv = F.linear(h, P["attn_w"], P["attn_b"])            # (B,T,3C)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, nh, dh).transpose(1, 2)
        k = k.view(B, T, nh, dh).transpose(1, 2)
        v = v.view(B, T, nh, dh).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).contiguous().view(B, T, C)
        x = x + F.linear(a, P["proj_w"], P["proj_b"])
        # --- MLP (pre-LN) ---
        h = F.layer_norm(x, (C,), P["ln2_w"], P["ln2_b"])
        h = F.gelu(F.linear(h, P["fc_w"], P["fc_b"]), approximate="tanh")
        x = x + F.linear(h, P["fcp_w"], P["fcp_b"])
        return x
    return fn


def _head_fn(cfg):
    def fn(x, P):
        C = x.shape[-1]
        h = F.layer_norm(x, (C,), P["lnf_w"], P["lnf_b"])
        return F.linear(h, P["head_w"])
    return fn


def build_specs(cfg: GPTConfig) -> list[LayerSpec]:
    """The model as an ordered list of ZeRO parameter-groups."""
    C, V, L = cfg.d_model, cfg.vocab_size, cfg.block_size
    specs = [LayerSpec("embed", [("wte", (V, C)), ("wpe", (L, C))], _embed_fn(cfg))]
    for i in range(cfg.n_layer):
        specs.append(LayerSpec(
            f"block{i}",
            [("ln1_w", (C,)), ("ln1_b", (C,)),
             ("attn_w", (3 * C, C)), ("attn_b", (3 * C,)),
             ("proj_w", (C, C)), ("proj_b", (C,)),
             ("ln2_w", (C,)), ("ln2_b", (C,)),
             ("fc_w", (4 * C, C)), ("fc_b", (4 * C,)),
             ("fcp_w", (C, 4 * C)), ("fcp_b", (C,))],
            _block_fn(cfg)))
    specs.append(LayerSpec(
        "head",
        [("lnf_w", (C,)), ("lnf_b", (C,)), ("head_w", (V, C))],
        _head_fn(cfg)))
    return specs


# --------------------------------------------------------------------------
# deterministic initialisation
# --------------------------------------------------------------------------
def init_group(spec: LayerSpec, cfg: GPTConfig, seed: int) -> torch.Tensor:
    """Flat fp32 init vector for one parameter-group -- identical on every rank."""
    g = torch.Generator().manual_seed(seed)
    parts = []
    for pname, shape in spec.params:
        n = 1
        for s in shape:
            n *= s
        if pname.endswith("_b"):
            t = torch.zeros(n)
        elif pname.endswith("ln1_w") or pname.endswith("ln2_w") or pname == "lnf_w":
            t = torch.ones(n)
        else:
            std = 0.02 / math.sqrt(2 * cfg.n_layer) if pname in ("proj_w", "fcp_w") else 0.02
            t = torch.randn(n, generator=g) * std
        parts.append(t)
    return torch.cat(parts)


def unflatten(flat: torch.Tensor, spec: LayerSpec) -> dict:
    """Views into a flat buffer -- no copy, so autograd flows back to `flat`."""
    P, off = {}, 0
    for pname, shape in spec.params:
        n = 1
        for s in shape:
            n *= s
        P[pname] = flat[off:off + n].view(*shape)
        off += n
    return P


def param_count(cfg: GPTConfig) -> int:
    return sum(s.numel for s in build_specs(cfg))


def make_batch(cfg: GPTConfig, batch: int, seed: int):
    """A small *learnable* synthetic task: predict token+1 (mod vocab).

    Random targets would make the loss sit flat at ln(vocab) forever, which
    would turn the "every stage traces the same curve" plot into a test of
    nothing.  A rule the model can actually fit means the curves visibly
    descend, so agreeing on them is a real statement.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, cfg.vocab_size, (batch, cfg.block_size), generator=g)
    y = (x + 1) % cfg.vocab_size
    return x, y
