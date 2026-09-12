"""Per-vGPU memory accounting.

Everything here is *measured* from real torch tensors (element_size * numel);
the tracker only decides which bucket a tensor belongs to.  The four buckets
are exactly the four things the ZeRO paper accounts for:

    params      -- the model weights this rank physically holds
    grads       -- gradient buffers this rank physically holds
    optimizer   -- Adam's fp32 moments (m, v) for the slice this rank owns
    activations -- tensors the autograd graph keeps alive for the backward pass
    comm        -- transient buffers a collective needs while it is in flight
"""
from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field

import torch

CATEGORIES = ("params", "grads", "optimizer", "activations", "comm")

MB = 1024.0 ** 2
GB = 1024.0 ** 3


def nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


@dataclass
class MemorySample:
    step: int
    phase: str
    total: int
    per_cat: dict


class MemoryTracker:
    """A tiny allocator-shaped ledger for one virtual GPU."""

    def __init__(self, rank: int):
        self.rank = rank
        self.cur = {c: 0 for c in CATEGORIES}
        self.peak_cat = {c: 0 for c in CATEGORIES}
        self.peak_total = 0
        self.timeline: list[MemorySample] = []
        self._step = 0
        self._lock = threading.Lock()

    # -- ledger ---------------------------------------------------------
    def alloc(self, cat: str, n: int) -> None:
        with self._lock:
            self.cur[cat] += int(n)
            self.peak_cat[cat] = max(self.peak_cat[cat], self.cur[cat])
            self.peak_total = max(self.peak_total, self.total)

    def free(self, cat: str, n: int) -> None:
        with self._lock:
            self.cur[cat] = max(0, self.cur[cat] - int(n))

    def set(self, cat: str, n: int) -> None:
        with self._lock:
            self.cur[cat] = int(n)
            self.peak_cat[cat] = max(self.peak_cat[cat], self.cur[cat])
            self.peak_total = max(self.peak_total, self.total)

    @property
    def total(self) -> int:
        return sum(self.cur.values())

    # -- instrumentation ------------------------------------------------
    def mark(self, phase: str) -> None:
        self.timeline.append(
            MemorySample(self._step, phase, self.total, dict(self.cur))
        )

    def new_step(self, step: int) -> None:
        self._step = step

    def summary(self) -> dict:
        d = {f"peak_{c}": self.peak_cat[c] for c in CATEGORIES}
        d["peak_total"] = self.peak_total
        return d


class ActivationMeter:
    """Measures how many bytes of activations the graph pins for backward.

    Uses ``saved_tensors_hooks``: every tensor autograd stashes for the
    backward pass passes through ``pack``.  We de-duplicate by storage
    pointer and skip anything that is a parameter/grad we already counted,
    so what is left is genuinely "activation memory".
    """

    def __init__(self, tracker: MemoryTracker, exclude_ptrs: set | None = None):
        self.tracker = tracker
        self.exclude = exclude_ptrs if exclude_ptrs is not None else set()
        self.seen: set[int] = set()
        self.bytes = 0

    def __enter__(self):
        self.seen = set()
        self.bytes = 0
        self._ctx = torch.autograd.graph.saved_tensors_hooks(self._pack, self._unpack)
        self._ctx.__enter__()
        return self

    def __exit__(self, *exc):
        self._ctx.__exit__(*exc)
        return False

    def _pack(self, t: torch.Tensor):
        # charge the whole STORAGE, not the view: q,k,v from a .split() share
        # one storage, and charging each view's numel undercounts the real
        # allocation.  Charge incrementally so the memory timeline is right
        # during the forward pass, not only at the end of it.
        try:
            st = t.untyped_storage()
            ptr, n = st.data_ptr(), st.nbytes()
        except Exception:
            ptr, n = id(t), nbytes(t)
        if ptr not in self.seen and ptr not in self.exclude:
            self.seen.add(ptr)
            self.bytes += n
            self.tracker.alloc("activations", n)
        return t

    def _unpack(self, t):
        return t


def fmt(b: float) -> str:
    if b >= GB:
        return f"{b / GB:.2f} GB"
    if b >= MB:
        return f"{b / MB:.2f} MB"
    return f"{b / 1024:.1f} KB"
