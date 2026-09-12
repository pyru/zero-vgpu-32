"""A virtual GPU cluster built out of CPU threads.

Each "vGPU" is one worker thread with its own memory ledger.  The Fabric
provides the four collectives every data-parallel / ZeRO implementation
actually needs:

    broadcast, all_reduce, reduce_scatter, all_gather

They are implemented for real (the numbers that come out are the numbers a
real NCCL call would produce), and every call is charged:

  * **bytes on the wire** using the standard ring-algorithm cost model, and
  * **modelled time**  = latency + wire_bytes / bandwidth

so we can compare communication volume between ZeRO stages the same way the
ZeRO paper does, without needing 32 real GPUs.
"""
from __future__ import annotations

import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import torch

from .memory import MemoryTracker, nbytes


# --------------------------------------------------------------------------
# interconnect cost model
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Interconnect:
    name: str
    bandwidth_GBps: float   # effective unidirectional bandwidth per rank
    latency_us: float       # per-collective fixed cost


PROFILES = {
    # intra-node (inside one 8-GPU box): the GPUs talk over NVLink/NVSwitch
    "nvlink450": Interconnect("NVLink / NVSwitch (intra-node)", 450.0, 2.0),
    "nvlink4":   Interconnect("NVLink-4 (intra-node, conservative)", 300.0, 2.0),
    # inter-node (box to box): an order of magnitude slower, and this is the
    # single most important number in multi-node training
    "internode50": Interconnect("Inter-node fabric (box to box)", 50.0, 5.0),
    "ib400":       Interconnect("InfiniBand NDR 400 Gb/s", 50.0, 5.0),
    "pcie4":       Interconnect("PCIe 4.0 x16", 25.0, 8.0),
    "eth100":      Interconnect("100 Gb Ethernet", 12.5, 40.0),
}


@dataclass
class CommStats:
    """Per-rank communication ledger."""
    calls: dict = field(default_factory=lambda: {})
    wire_bytes: dict = field(default_factory=lambda: {})
    seconds: dict = field(default_factory=lambda: {})

    def add(self, op: str, wire: float, secs: float) -> None:
        self.calls[op] = self.calls.get(op, 0) + 1
        self.wire_bytes[op] = self.wire_bytes.get(op, 0.0) + wire
        self.seconds[op] = self.seconds.get(op, 0.0) + secs

    @property
    def total_bytes(self) -> float:
        return sum(self.wire_bytes.values())

    @property
    def total_seconds(self) -> float:
        return sum(self.seconds.values())

    @property
    def total_calls(self) -> int:
        return sum(self.calls.values())


class VirtualGPU:
    """One simulated accelerator: an id, a memory ledger, a comm ledger."""

    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.name = f"vGPU:{rank:02d}"
        self.mem = MemoryTracker(rank)
        self.comm = CommStats()
        self.compute_seconds = 0.0

    def __repr__(self) -> str:
        return f"<{self.name} mem_peak={self.mem.peak_total/2**20:.1f}MB>"


class Fabric:
    """The interconnect + rank bookkeeping for `world_size` virtual GPUs."""

    def __init__(self, world_size: int = 32, interconnect: str = "nvlink450",
                 gpus_per_node: int | None = None,
                 inter_node: str = "internode50"):
        """A virtual cluster of `world_size` GPUs.

        `gpus_per_node` makes the cluster *hierarchical*, the way a real one is:
        8 GPUs sit in a box and talk over NVLink at ~450 GB/s, and boxes talk to
        each other over a fabric roughly 9x slower.  Leave it None (or >=
        world_size) for a flat single-node cluster.

        A ring collective is a pipeline, so it runs at the rate of its slowest
        hop: the moment a ring spans two boxes, the *whole* collective is paced
        by the inter-node link.  That is the single most important fact about
        multi-node training, and it is why NCCL actually uses hierarchical
        (intra-node then inter-node) algorithms rather than one flat ring.
        """
        self.world_size = world_size
        self.gpus_per_node = gpus_per_node or world_size
        self.n_nodes = math.ceil(world_size / self.gpus_per_node)
        self.intra_link = PROFILES[interconnect]
        self.inter_link = PROFILES[inter_node]
        # a flat ring over several boxes is paced by the slow hop
        self.link = self.intra_link if self.n_nodes == 1 else self.inter_link
        self.gpus = [VirtualGPU(r, world_size) for r in range(world_size)]
        self._barrier = threading.Barrier(world_size)
        self._slots: list = [None] * world_size
        self._result = None

    # -- helpers --------------------------------------------------------
    def gpu(self, rank: int) -> VirtualGPU:
        return self.gpus[rank]

    def node_of(self, rank: int) -> int:
        """Which physical box this rank lives in."""
        return rank // self.gpus_per_node

    def spans_nodes(self) -> bool:
        return self.n_nodes > 1

    def topology(self) -> str:
        if self.n_nodes == 1:
            return (f"{self.world_size} GPUs in 1 node, all on "
                    f"{self.intra_link.name} @ {self.intra_link.bandwidth_GBps} GB/s")
        return (f"{self.world_size} GPUs = {self.n_nodes} nodes x "
                f"{self.gpus_per_node} GPUs; intra-node "
                f"{self.intra_link.bandwidth_GBps} GB/s, inter-node "
                f"{self.inter_link.bandwidth_GBps} GB/s "
                f"({self.intra_link.bandwidth_GBps/self.inter_link.bandwidth_GBps:.0f}x slower)")

    def barrier(self) -> None:
        if self.world_size > 1:
            self._barrier.wait()

    def _charge(self, rank: int, op: str, wire_bytes: float, hops: int) -> None:
        """Charge one collective.

        A ring collective is not one hop: all_gather / reduce_scatter are N-1
        sequential steps and all_reduce is 2(N-1), so the fixed latency is paid
        that many times.  On a slow link with many ranks this term dominates,
        which is exactly why NCCL switches to tree algorithms when a message is
        latency-bound -- we model the ring only.
        """
        secs = (self.link.latency_us * 1e-6 * hops
                + wire_bytes / (self.link.bandwidth_GBps * 1e9))
        self.gpus[rank].comm.add(op, wire_bytes, secs)

    # -- collectives ----------------------------------------------------
    # Ring-algorithm wire cost per rank (this is the textbook model that
    # NCCL's own tuning docs use):
    #   all_gather / reduce_scatter :  (N-1)/N * S
    #   all_reduce                  : 2(N-1)/N * S     (= RS + AG)
    #   broadcast                   :  (N-1)/N * S
    # where S is the size of the *full* (unsharded) buffer.

    def all_reduce(self, rank: int, t: torch.Tensor, charge: bool = True) -> torch.Tensor:
        """Sum `t` across all ranks; every rank gets the full result."""
        N = self.world_size
        if N == 1:
            return t
        S = nbytes(t)
        self._slots[rank] = t
        self.barrier()
        if rank == 0:
            self._result = torch.stack(self._slots, 0).sum(0)
        self.barrier()
        out = self._result.clone()
        self.barrier()
        if charge:
            self._charge(rank, "all_reduce", 2.0 * (N - 1) / N * S, 2 * (N - 1))
        return out

    def reduce_scatter(self, rank: int, t: torch.Tensor) -> torch.Tensor:
        """Sum `t` across ranks, but rank r keeps only chunk r.

        This is the collective that makes ZeRO work: the output is 1/N the
        size of the input, so gradients never have to exist in full.
        """
        N = self.world_size
        if N == 1:
            return t
        assert t.numel() % N == 0, f"{t.numel()} not divisible by {N}"
        S = nbytes(t)
        self._slots[rank] = t
        self.barrier()
        if rank == 0:
            self._result = torch.stack(self._slots, 0).sum(0)
        self.barrier()
        chunk = self._result.numel() // N
        out = self._result[rank * chunk:(rank + 1) * chunk].clone()
        self.barrier()
        self._charge(rank, "reduce_scatter", (N - 1) / N * S, N - 1)
        return out

    def all_gather(self, rank: int, shard: torch.Tensor) -> torch.Tensor:
        """Concatenate every rank's shard; every rank gets the full buffer."""
        N = self.world_size
        if N == 1:
            return shard
        self._slots[rank] = shard
        self.barrier()
        out = torch.cat(self._slots, 0)
        self.barrier()
        self._charge(rank, "all_gather", (N - 1) / N * nbytes(out), N - 1)
        return out

    def broadcast(self, rank: int, t: torch.Tensor, src: int = 0) -> torch.Tensor:
        N = self.world_size
        if N == 1:
            return t
        if rank == src:
            self._result = t
        self.barrier()
        out = self._result.clone()
        self.barrier()
        self._charge(rank, "broadcast", (N - 1) / N * nbytes(t), N - 1)
        return out

    # -- launcher -------------------------------------------------------
    def run(self, worker, *args, **kwargs):
        """Launch `worker(rank, fabric, *args)` on every virtual GPU."""
        self._barrier = threading.Barrier(self.world_size)
        results = [None] * self.world_size

        errors = []

        def _entry(r):
            torch.set_num_threads(1)          # each vGPU is one compute unit
            try:
                results[r] = worker(r, self, *args, **kwargs)
            except BaseException as e:        # noqa: BLE001
                # Without this, one rank dying leaves the other 31 blocked in
                # barrier.wait() forever and the user sees a silent hang
                # instead of a traceback.
                errors.append((r, e))
                self._barrier.abort()
                raise

        if self.world_size == 1:
            _entry(0)
            return results
        with ThreadPoolExecutor(max_workers=self.world_size) as pool:
            futures = [pool.submit(_entry, r) for r in range(self.world_size)]
            for f in futures:
                try:
                    f.result()
                except Exception:
                    pass
        if errors:
            r, e = errors[0]
            raise RuntimeError(f"rank {r} failed: {e!r}") from e
        self._slots = [None] * self.world_size     # do not pin the last payload
        self._result = None
        return results

    # -- reporting ------------------------------------------------------
    def comm_totals(self) -> dict:
        ops = {}
        for g in self.gpus:
            for op, b in g.comm.wire_bytes.items():
                ops[op] = ops.get(op, 0.0) + b
        return ops

    def __repr__(self) -> str:
        return f"<Fabric {self.topology()}>"
