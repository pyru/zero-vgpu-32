"""ZeRO stages 1, 2 and 3 implemented on top of the virtual GPU fabric.

The three stages differ in exactly one thing: *which of the three model-state
tensors is allowed to exist in full on every rank.*

    stage      params      grads              optimizer (Adam m,v)
    -------    --------    ---------------    --------------------
    DDP        replicated  replicated         replicated
    ZeRO-1     replicated  replicated         SHARDED
    ZeRO-2     replicated  SHARDED            SHARDED
    ZeRO-3     SHARDED     SHARDED            SHARDED

Everything else -- the maths, the loss, the learning rate -- is identical,
which is why all four produce matching loss curves.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .fabric import Fabric
from .memory import ActivationMeter, nbytes
from .model import GPTConfig, build_specs, init_group, unflatten

STAGES = ("baseline", "ddp", "zero1", "zero2", "zero3")


# ==========================================================================
# ZeRO-3: gather a layer's weights just-in-time, then throw them away
# ==========================================================================
class ZeroLayerFn(torch.autograd.Function):
    """One transformer layer whose weights live sharded across the fabric.

    forward :  all_gather(shard) -> weights -> compute -> FREE weights
    backward:  all_gather(shard) -> weights -> recompute -> local backward
               -> reduce_scatter(grad) -> keep only *my* 1/N of the gradient

    That is ZeRO-3 in a nutshell: the full weight tensor exists for the
    duration of one layer's matmuls and at no other time, and the full
    gradient tensor is destroyed by the reduce-scatter before the next
    layer's backward even begins.
    """

    @staticmethod
    def forward(ctx, x, shard, engine, gi):
        g = engine.groups[gi]
        spec, mem = g["spec"], engine.mem
        engine.layer_evals += 1
        with torch.no_grad():
            full = engine.fabric.all_gather(engine.rank, shard)
            mem.alloc("comm", nbytes(full))
            mem.mark("fwd/" + spec.name + "/gathered")
            P = unflatten(full[:spec.numel], spec)
            out = spec.fn(x, P)
            if engine.keep_gathered:
                g["_kept"] = full            # trade memory for comm volume
            else:
                mem.free("comm", nbytes(full))
                del P, full                  # <-- the weights evaporate here
            mem.mark("fwd/" + spec.name + "/released")
        ctx.save_for_backward(x, shard)
        ctx.engine, ctx.gi = engine, gi
        return out

    @staticmethod
    def backward(ctx, gout):
        x, shard = ctx.saved_tensors
        engine, gi = ctx.engine, ctx.gi
        g = engine.groups[gi]
        spec, mem = g["spec"], engine.mem

        engine.layer_evals += 1          # ZeRO-3 recomputes the layer here
        with torch.enable_grad():
            if engine.keep_gathered:
                full = g.pop("_kept").detach().requires_grad_(True)
            else:
                # ZeRO-3 pays a *second* all-gather here.  This is exactly why
                # its comm volume is 3*psi instead of DDP's 2*psi.
                full = engine.fabric.all_gather(engine.rank, shard.detach())
                full = full.requires_grad_(True)
                mem.alloc("comm", nbytes(full))
            wants_x = x.is_floating_point()
            xin = x.detach().requires_grad_(True) if wants_x else x
            P = unflatten(full[:spec.numel], spec)
            out = spec.fn(xin, P)
            inputs = [full] + ([xin] if wants_x else [])
            grads = torch.autograd.grad(out, inputs, gout)

        gfull = grads[0]
        gx = grads[1] if wants_x else None
        mem.alloc("grads", nbytes(gfull))
        mem.mark("bwd/" + spec.name + "/full_grad")
        gshard = engine.fabric.reduce_scatter(engine.rank, gfull)
        mem.free("grads", nbytes(gfull))
        mem.alloc("grads", nbytes(gshard))   # the 1/N slice we keep for good
        mem.free("comm", nbytes(full))
        del gfull, full, P, out
        mem.mark("bwd/" + spec.name + "/scattered")
        return gx, gshard, None, None


class CheckpointLayerFn(torch.autograd.Function):
    """Recompute-in-backward, but with *replicated* weights.

    This exists purely for fairness.  ZeRO-3 has to recompute -- it deleted the
    weights its backward would have needed -- which hands it activation
    checkpointing savings the other stages do not get.  Switching this on for
    DDP / ZeRO-1 / ZeRO-2 puts every stage on the same activation footing, so
    the difference that remains is only the model-state sharding that ZeRO is
    actually about.
    """

    @staticmethod
    def forward(ctx, x, p, engine, gi):
        spec = engine.groups[gi]["spec"]
        engine.layer_evals += 1
        with torch.no_grad():
            out = spec.fn(x, unflatten(p[:spec.numel], spec))
        ctx.save_for_backward(x, p)
        ctx.engine, ctx.gi = engine, gi
        return out

    @staticmethod
    def backward(ctx, gout):
        x, p = ctx.saved_tensors
        engine, gi = ctx.engine, ctx.gi
        spec = engine.groups[gi]["spec"]
        engine.layer_evals += 1          # checkpointed recompute
        with torch.enable_grad():
            pin = p.detach().requires_grad_(True)
            wants_x = x.is_floating_point()
            xin = x.detach().requires_grad_(True) if wants_x else x
            out = spec.fn(xin, unflatten(pin[:spec.numel], spec))
            inputs = [pin] + ([xin] if wants_x else [])
            grads = torch.autograd.grad(out, inputs, gout)
        return (grads[1] if wants_x else None), grads[0], None, None


# ==========================================================================
# the per-rank engine
# ==========================================================================
@dataclass
class StepReport:
    step: int
    loss: float
    step_wall_s: float          # includes barrier waits -- NOT pure compute
    comm_s: float               # modelled comm time for THIS step only
    mem_total: int


class ZeroEngine:
    def __init__(self, rank, fabric, cfg, stage, lr=3e-4, seed=1234,
                 keep_gathered=False, betas=(0.9, 0.95), eps=1e-8,
                 checkpoint=False):
        assert stage in STAGES, stage
        self.rank, self.fabric, self.cfg, self.stage = rank, fabric, cfg, stage
        self.world = fabric.world_size
        self.mem = fabric.gpu(rank).mem
        self.gpu = fabric.gpu(rank)
        self.lr, self.betas, self.eps = lr, betas, eps
        self.keep_gathered = keep_gathered
        # ZeRO-3 always recomputes (it deleted the weights); `checkpoint` makes
        # the other stages do it too, for a like-for-like activation comparison
        self.checkpoint = checkpoint
        self.shard_params = (stage == "zero3")
        self.shard_grads = (stage in ("zero2", "zero3"))
        self.shard_optim = (stage in ("zero1", "zero2", "zero3"))
        self.t = 0
        self._comm_prev = 0.0
        # --- compute counters -------------------------------------------
        # layer_evals : how many times a layer's fn() actually runs.  ZeRO-3
        #               runs every layer TWICE (forward + recompute in
        #               backward), which is real extra arithmetic.
        # optim_elems : how many parameter elements this rank's Adam touches.
        #               DDP updates all of them on every rank -- N-way
        #               redundant work that ZeRO-1 removes along with the
        #               memory.
        self.layer_evals = 0
        self.optim_elems = 0

        self.groups = []
        for gi, spec in enumerate(build_specs(cfg)):
            full = init_group(spec, cfg, seed=seed + 17 * gi)
            pad = (-full.numel()) % self.world
            if pad:
                full = torch.cat([full, torch.zeros(pad)])
            full_n = full.numel()
            shard_n = full_n // self.world
            lo = rank * shard_n

            if self.shard_params:
                p = torch.nn.Parameter(full[lo:lo + shard_n].clone())
            else:
                p = torch.nn.Parameter(full.clone())
            self.mem.alloc("params", nbytes(p))

            own_n = shard_n if self.shard_optim else full_n
            g = dict(spec=spec, p=p, gi=gi, full_n=full_n, shard_n=shard_n,
                     lo=lo, own_n=own_n, gshard=None, charged=False,
                     m=torch.zeros(own_n), v=torch.zeros(own_n))
            self.mem.alloc("optimizer", nbytes(g["m"]) + nbytes(g["v"]))
            self.groups.append(g)
            del full

        self.param_ptrs = set()
        for g in self.groups:
            self.param_ptrs.add(g["p"].untyped_storage().data_ptr())
        self.mem.mark("init")

        if self.stage == "zero2":
            for gi, g in enumerate(self.groups):
                g["p"].register_post_accumulate_grad_hook(self._zero2_hook(gi))
        elif self.stage in ("baseline", "ddp", "zero1"):
            # purely observational: lets us watch the full gradient buffers
            # appear one layer at a time during the backward pass
            for g in self.groups:
                g["p"].register_post_accumulate_grad_hook(self._grow_grad_hook(g))

    # -- properties ------------------------------------------------------
    @property
    def n_params(self):
        return sum(g["spec"].numel for g in self.groups)

    @property
    def padded_params(self):
        return sum(g["full_n"] for g in self.groups)

    def _zero2_hook(self, gi):
        def hook(p):
            """Fires the instant this layer's gradient is complete.

            ZeRO-2's trick: do not wait for the whole backward pass.  Reduce
            -scatter this layer's gradient right now, keep 1/N of it, and
            release the full buffer before the next layer's backward starts.
            """
            g = self.groups[gi]
            self.mem.alloc("grads", nbytes(p.grad))
            self.mem.mark("bwd/" + g["spec"].name + "/full_grad")
            gsh = self.fabric.reduce_scatter(self.rank, p.grad)
            self.mem.free("grads", nbytes(p.grad))
            if g["gshard"] is None:
                g["gshard"] = gsh
                self.mem.alloc("grads", nbytes(gsh))
            else:
                g["gshard"] += gsh
            p.grad = None
            self.mem.mark("bwd/" + g["spec"].name + "/scattered")
        return hook

    # -- forward ---------------------------------------------------------
    def forward(self, idx, targets, ntokens_global):
        x = idx
        for gi, g in enumerate(self.groups):
            spec = g["spec"]
            if self.shard_params:
                x = ZeroLayerFn.apply(x, g["p"], self, gi)
            elif self.checkpoint:
                x = CheckpointLayerFn.apply(x, g["p"], self, gi)
                self.mem.mark("fwd/" + spec.name)
            else:
                P = unflatten(g["p"][:spec.numel], spec)
                x = spec.fn(x, P)
                self.layer_evals += 1
                self.mem.mark("fwd/" + spec.name)
        logits = x
        loss = F.cross_entropy(
            logits.view(-1, self.cfg.vocab_size), targets.reshape(-1),
            reduction="sum") / ntokens_global
        return loss

    # -- gradient reduction ----------------------------------------------
    def reduce_gradients(self):
        if self.stage == "baseline":
            return
        if self.stage == "ddp":
            # every rank ends up holding the *full* summed gradient
            for g in self.groups:
                # the all-reduce output is a second full buffer that coexists
                # with the old gradient for a moment.  Real NCCL pays this too,
                # which is what the bucket-size knobs exist to bound.
                self.mem.alloc("comm", nbytes(g["p"].grad))
                g["p"].grad = self.fabric.all_reduce(self.rank, g["p"].grad)
                self.mem.mark("ar/" + g["spec"].name)
                self.mem.free("comm", nbytes(g["p"].grad))
            self.mem.mark("after_all_reduce")
        elif self.stage == "zero1":
            # full gradients were held for the entire backward pass; only now
            # do we scatter them.  Peak grad memory == the whole model.
            for g in self.groups:
                gsh = self.fabric.reduce_scatter(self.rank, g["p"].grad)
                self.mem.free("grads", nbytes(g["p"].grad))
                g["gshard"] = gsh
                self.mem.alloc("grads", nbytes(gsh))
                g["p"].grad = None
                self.mem.mark("rs/" + g["spec"].name)
            self.mem.mark("after_reduce_scatter")
        # zero2: already done inside the backward hooks
        # zero3: gradients arrived pre-sharded from ZeroLayerFn.backward

    # -- optimizer --------------------------------------------------------
    def step_optimizer(self):
        self.t += 1
        b1, b2 = self.betas
        bc1 = 1 - b1 ** self.t
        bc2 = 1 - b2 ** self.t
        for g in self.groups:
            if self.stage in ("baseline", "ddp", "zero3"):
                own, grad = g["p"].data, g["p"].grad
            else:                                   # zero1 / zero2
                own = g["p"].data[g["lo"]:g["lo"] + g["shard_n"]]
                grad = g["gshard"]
            self.optim_elems += own.numel()
            g["m"].mul_(b1).add_(grad, alpha=1 - b1)
            g["v"].mul_(b2).addcmul_(grad, grad, value=1 - b2)
            denom = (g["v"] / bc2).sqrt_().add_(self.eps)
            own.addcdiv_(g["m"] / bc1, denom, value=-self.lr)

    def refresh_params(self):
        """ZeRO-1/2 only: each rank updated 1/N of the weights -- re-assemble."""
        if self.stage in ("zero1", "zero2"):
            for g in self.groups:
                own = g["p"].data[g["lo"]:g["lo"] + g["shard_n"]].clone()
                self.mem.alloc("comm", nbytes(own) + g["full_n"] * 4)
                full = self.fabric.all_gather(self.rank, own)
                g["p"].data.copy_(full)
                self.mem.mark("ag/" + g["spec"].name)
                self.mem.free("comm", nbytes(own) + g["full_n"] * 4)
            self.mem.mark("after_param_all_gather")

    def zero_grad(self):
        for g in self.groups:
            g["p"].grad = None
            g["gshard"] = None
            g["charged"] = False
        self.mem.set("grads", 0)

    def _grow_grad_hook(self, g):
        """Observational only: lets the timeline show full gradient buffers
        appearing one layer at a time.  Charges each group at most once per
        step, so repeated firing cannot inflate the ledger."""
        def hook(p):
            if not g["charged"]:
                g["charged"] = True
                self.mem.alloc("grads", nbytes(p.grad))
            self.mem.mark("bwd/" + g["spec"].name + "/full_grad")
        return hook

    # -- one training step -------------------------------------------------
    def train_step(self, idx, targets, ntokens_global, step):
        self.mem.new_step(step)
        self.mem.set("activations", 0)
        t0 = time.perf_counter()

        with ActivationMeter(self.mem, exclude_ptrs=self.param_ptrs):
            loss = self.forward(idx, targets, ntokens_global)
        self.mem.mark("after_forward")

        loss.backward()
        if self.stage in ("baseline", "ddp", "zero1"):
            # these stages materialise a full gradient for every parameter
            got = 0
            for g in self.groups:
                if g["p"].grad is not None:
                    got += nbytes(g["p"].grad)
            self.mem.set("grads", got)
        self.mem.set("activations", 0)     # the graph is released by backward()
        self.mem.mark("after_backward")

        self.reduce_gradients()
        self.step_optimizer()
        self.refresh_params()
        self.mem.mark("after_step")

        # NOTE: this is step WALL time.  On a thread-backed fabric it includes
        # every barrier wait, so it is not a clean compute measurement.
        wall_s = time.perf_counter() - t0
        self.gpu.compute_seconds += wall_s
        comm_now = self.gpu.comm.total_seconds
        rp = StepReport(step, float(loss.detach()), wall_s,
                        comm_now - self._comm_prev, self.mem.total)
        self._comm_prev = comm_now
        self.zero_grad()
        return rp
