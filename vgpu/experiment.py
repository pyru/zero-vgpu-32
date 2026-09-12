"""Drivers: run one stage across the whole virtual cluster and collect stats."""
from __future__ import annotations

import time

import torch

from .fabric import Fabric
from .memory import GB, MB, fmt
from .model import GPTConfig, make_batch, param_count
from .zero import ZeroEngine


def run_stage(stage, cfg, world_size=32, steps=10, global_batch=32, lr=3e-3,
              seed=1234, interconnect="nvlink4", keep_gathered=False,
              data_seed=7, checkpoint=False):
    """Train `steps` steps of `stage` on `world_size` virtual GPUs.

    The global batch is split evenly across ranks and the loss is normalised
    by the *global* token count, so summing gradients with an all-reduce /
    reduce-scatter reproduces single-device training exactly.
    """
    world = 1 if stage == "baseline" else world_size
    fabric = Fabric(world, interconnect=interconnect)
    assert global_batch % world == 0
    local_bs = global_batch // world
    ntokens_global = global_batch * cfg.block_size

    X, Y = make_batch(cfg, global_batch * steps, seed=data_seed)

    losses_holder = [None] * world
    t_start = time.perf_counter()

    def worker(rank, fab):
        torch.manual_seed(seed)
        eng = ZeroEngine(rank, fab, cfg, stage, lr=lr, seed=seed,
                         keep_gathered=keep_gathered, checkpoint=checkpoint)
        losses = []
        for s in range(steps):
            base = s * global_batch
            lo = base + rank * local_bs
            idx = X[lo:lo + local_bs]
            tgt = Y[lo:lo + local_bs]
            rep = eng.train_step(idx, tgt, ntokens_global, s)
            # the reported loss is only this rank's share -> all-reduce it so
            # every rank prints the true global loss
            lt = torch.tensor([rep.loss])
            # charge=False: this is a reporting convenience, not part of the
            # training algorithm, so it must not pollute the comm ledger
            lt = (fab.all_reduce(rank, lt, charge=False)
                  if fab.world_size > 1 else lt)
            losses.append(float(lt))
        losses_holder[rank] = losses
        return eng

    engines = fabric.run(worker)
    wall = time.perf_counter() - t_start

    g0 = fabric.gpu(0)
    eng0 = engines[0]
    psi = param_count(cfg)
    out = dict(
        stage=stage, world_size=world, steps=steps, wall_s=wall,
        params=psi, param_bytes=psi * 4,
        losses=losses_holder[0],
        peak_total=max(g.mem.peak_total for g in fabric.gpus),
        peak_by_cat={c: max(g.mem.peak_cat[c] for g in fabric.gpus)
                     for c in g0.mem.peak_cat},
        # "model state" in the paper's sense = params + grads + optimizer
        peak_model_state=(g0.mem.peak_cat["params"] + g0.mem.peak_cat["grads"]
                          + g0.mem.peak_cat["optimizer"]),
        resident_params=eng0.mem.cur["params"],
        resident_optimizer=eng0.mem.cur["optimizer"],
        comm_bytes_per_rank_per_step=g0.comm.total_bytes / max(steps, 1),
        comm_calls_per_step=g0.comm.total_calls / max(steps, 1),
        comm_seconds_modelled=g0.comm.total_seconds,
        comm_by_op={op: b / max(steps, 1) for op, b in g0.comm.wire_bytes.items()},
        step_wall_seconds=sum(g.compute_seconds for g in fabric.gpus) / world,
        fabric=fabric, engines=engines,
        timeline=fabric.gpu(0).mem.timeline,
    )
    out["comm_ratio_psi"] = out["comm_bytes_per_rank_per_step"] / (psi * 4)
    return out


def compare(results, ref_key="baseline"):
    """Numerical agreement between every stage and the single-device run."""
    ref = results[ref_key]["losses"]
    rows = []
    for k, r in results.items():
        d = max(abs(a - b) for a, b in zip(ref, r["losses"]))
        rows.append(dict(stage=k, final_loss=r["losses"][-1], max_abs_dev=d))
    return rows


# --------------------------------------------------------------------------
# analytic model -- lets us extrapolate to model sizes that will not fit here
# --------------------------------------------------------------------------
def analytic_memory(psi, N, precision="mixed"):
    """Bytes of *model state* per GPU for each stage.

    mixed  : fp16 params (2) + fp16 grads (2) + Adam fp32 master/m/v (12) = 16*psi
    fp32   : fp32 params (4) + fp32 grads (4) + Adam m/v (8)              = 16*psi
    """
    if precision == "mixed":
        p, g, o = 2.0, 2.0, 12.0
    else:
        p, g, o = 4.0, 4.0, 8.0
    return {
        "DDP (no ZeRO)": (p + g + o) * psi,
        "ZeRO-1 (Pos)":  (p + g) * psi + o * psi / N,
        "ZeRO-2 (Pos+g)": p * psi + (g + o) * psi / N,
        "ZeRO-3 (Pos+g+p)": (p + g + o) * psi / N,
    }


def analytic_comm(psi_bytes, N):
    """Wire bytes per rank per step (ring algorithms)."""
    f = (N - 1) / N if N > 1 else 0.0
    return {
        "DDP (no ZeRO)": 2 * f * psi_bytes,
        "ZeRO-1 (Pos)": 2 * f * psi_bytes,
        "ZeRO-2 (Pos+g)": 2 * f * psi_bytes,
        "ZeRO-3 (Pos+g+p)": 3 * f * psi_bytes,
    }


def final_params(result):
    """Reassemble the full fp32 weight vector a run ended with.

    For ZeRO-3 the weights only exist as shards, so we concatenate them back
    together across ranks; for the other stages every rank holds a full copy.
    """
    engines = result["engines"]
    out = []
    for gi, g0 in enumerate(engines[0].groups):
        if engines[0].stage == "zero3":
            full = torch.cat([e.groups[gi]["p"].data for e in engines])
        else:
            full = engines[0].groups[gi]["p"].data.clone()
        out.append(full[:g0["spec"].numel])       # drop the divisibility padding
    return torch.cat(out)


def weight_agreement(results, ref_key="baseline"):
    """Max |w - w_single_gpu| over every weight, for each stage."""
    ref = final_params(results[ref_key])
    rows = []
    for k, r in results.items():
        w = final_params(r)
        assert w.numel() == ref.numel()
        rows.append(dict(stage=k,
                         max_abs_weight_diff=float((w - ref).abs().max()),
                         rel=float((w - ref).abs().max() / ref.abs().max())))
    return rows
