"""Tables and figures for the ZeRO simulation."""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .memory import GB, MB, fmt

LABEL = {"baseline": "1 GPU (no DP)", "ddp": "DDP (no ZeRO)", "zero1": "ZeRO-1  $P_{os}$",
         "zero2": "ZeRO-2  $P_{os+g}$", "zero3": "ZeRO-3  $P_{os+g+p}$"}
CATCOL = {"params": "#4C72B0", "grads": "#DD8452", "optimizer": "#55A868",
          "activations": "#C44E52", "comm": "#8172B3"}
STAGE_ORDER = ["ddp", "zero1", "zero2", "zero3"]


def _style(ax, title="", xlabel="", ylabel=""):
    ax.set_title(title, fontsize=12, fontweight="bold", loc="left")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return ax


# --------------------------------------------------------------------------
def fig_memory_breakdown(results, stages=STAGE_ORDER, path=None):
    """Stacked bar: what actually occupies each vGPU, per stage."""
    cats = ["params", "grads", "optimizer", "activations", "comm"]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    bottom = np.zeros(len(stages))
    for c in cats:
        vals = np.array([results[s]["peak_by_cat"][c] / MB for s in stages])
        ax.bar([LABEL[s] for s in stages], vals, bottom=bottom, label=c,
               color=CATCOL[c], edgecolor="white", linewidth=0.7)
        bottom += vals
    for i, s in enumerate(stages):
        ax.text(i, bottom[i] + 0.15, f"{bottom[i]:.2f} MB", ha="center",
                fontsize=9, fontweight="bold")
    _style(ax, "Peak memory per virtual GPU  (N = %d)" % results[stages[0]]["world_size"],
           "", "MB per GPU")
    ax.legend(frameon=False, ncol=5, fontsize=9, loc="upper right")
    ax.set_ylim(0, bottom.max() * 1.22)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def fig_scaling(sweep, path=None):
    """Measured peak memory per GPU as the cluster grows 1 -> 32."""
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    Ns = sorted(sweep.keys())
    for s in STAGE_ORDER:
        ys = [sweep[n][s]["peak_model_state"] / MB for n in Ns]
        ax.plot(Ns, ys, marker="o", lw=2, label=LABEL[s])
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(Ns)
    ax.set_xticklabels([str(n) for n in Ns])
    _style(ax, "Model-state memory per GPU vs. cluster size (measured)",
           "number of virtual GPUs (N)", "MB per GPU  (params+grads+optimizer)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def fig_comm(results, stages=STAGE_ORDER, path=None):
    """Communication volume per rank per step, in units of psi."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    ratios = [results[s]["comm_ratio_psi"] for s in stages]
    bars = ax1.bar([LABEL[s] for s in stages], ratios,
                   color=["#8C8C8C", "#4C72B0", "#55A868", "#C44E52"],
                   edgecolor="white")
    for b, r in zip(bars, ratios):
        ax1.text(b.get_x() + b.get_width() / 2, r + 0.04, f"{r:.2f}$\\psi$",
                 ha="center", fontweight="bold", fontsize=10)
    N = results[stages[0]]["world_size"]
    ax1.axhline(2 * (N - 1) / N, ls="--", c="k", lw=1, alpha=0.6)
    ax1.axhline(3 * (N - 1) / N, ls=":", c="k", lw=1, alpha=0.6)
    ax1.text(3.45, 2 * (N - 1) / N + 0.05, "theory 2$\\psi$", fontsize=8, ha="right")
    ax1.text(3.45, 3 * (N - 1) / N + 0.05, "theory 3$\\psi$", fontsize=8, ha="right")
    _style(ax1, "Communication volume per GPU per step", "", "multiples of $\\psi$ (param bytes)")
    ax1.set_ylim(0, 3.5)
    ax1.tick_params(axis="x", labelrotation=12)

    ops = ["all_reduce", "reduce_scatter", "all_gather"]
    w, xs = 0.26, np.arange(len(stages))
    for i, op in enumerate(ops):
        vals = [results[s]["comm_by_op"].get(op, 0) / MB for s in stages]
        ax2.bar(xs + (i - 1) * w, vals, w, label=op,
                color=["#4C72B0", "#DD8452", "#55A868"][i], edgecolor="white")
    ax2.set_xticks(xs)
    ax2.set_xticklabels([LABEL[s] for s in stages], rotation=12)
    _style(ax2, "...split by collective", "", "MB per GPU per step")
    ax2.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def fig_timeline(results, path=None):
    """Memory *within* one training step.

    Left : total footprint of every stage, x-axis normalised to one step.
    Right: ZeRO-3 alone, on its own scale, so the gather/release sawtooth
           is actually visible -- each tooth is one layer being materialised
           and thrown away again.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6))

    for s in STAGE_ORDER:
        tl = [m for m in results[s]["timeline"] if m.step == 1] or results[s]["timeline"]
        ys = np.array([m.total / MB for m in tl])
        xs = np.linspace(0, 1, len(ys))
        ax1.step(xs, ys, where="post", lw=2, label=LABEL[s])
    _style(ax1, "Total memory through one step (all stages)",
           "progress through one training step", "MB per GPU")
    ax1.legend(frameon=False, fontsize=9)

    tl = [m for m in results["zero3"]["timeline"] if m.step == 1] or results["zero3"]["timeline"]
    xs = np.arange(len(tl))
    bottom = np.zeros(len(tl))
    for c in ["params", "grads", "optimizer", "activations", "comm"]:
        ys = np.array([m.per_cat[c] / MB for m in tl])
        ax2.fill_between(xs, bottom, bottom + ys, label=c, color=CATCOL[c],
                         alpha=0.92, step="post")
        bottom += ys
    nf = sum(1 for m in tl if m.phase.startswith("fwd"))
    ax2.axvline(nf, color="k", ls="--", lw=1, alpha=0.6)
    ax2.text(nf * 0.5, bottom.max() * 1.05, "forward", ha="center", fontsize=9,
             style="italic")
    ax2.text((nf + len(tl)) * 0.5, bottom.max() * 1.05, "backward + step",
             ha="center", fontsize=9, style="italic")
    _style(ax2, "ZeRO-3 only: the all-gather / release sawtooth",
           "instrumented events within one step", "MB per GPU")
    ax2.set_xticks([])
    ax2.set_ylim(0, bottom.max() * 1.38)
    ax2.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def fig_losses(results, path=None):
    """All five configurations must trace the same curve."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ref = results["baseline"]["losses"]
    for s in ["baseline"] + STAGE_ORDER:
        ax1.plot(results[s]["losses"], marker="o", ms=4, lw=1.6, alpha=0.85,
                 label=LABEL[s])
    _style(ax1, "Training loss -- all stages overlap exactly", "step", "loss")
    ax1.legend(frameon=False, fontsize=9)
    for s in STAGE_ORDER:
        d = [abs(a - b) for a, b in zip(ref, results[s]["losses"])]
        ax2.semilogy([max(x, 1e-16) for x in d], marker="o", ms=4, label=LABEL[s])
    ax2.axhline(1e-6, ls="--", c="k", lw=1, alpha=0.5)
    _style(ax2, "|loss - single-GPU loss|  (fp32 noise floor)", "step", "absolute deviation")
    ax2.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def fig_analytic(psi=7.5e9, Ns=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024), path=None):
    """Extrapolate to a 7.5B-parameter model -- the paper's headline table."""
    from .experiment import analytic_memory
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    keys = list(analytic_memory(psi, 2).keys())
    for k in keys:
        ys = [analytic_memory(psi, n)[k] / GB for n in Ns]
        ax.plot(Ns, ys, marker="o", lw=2, label=k)
    ax.axhline(80, ls="--", c="crimson", lw=1.4)
    ax.text(Ns[0], 88, "80 GB (one H100)", color="crimson", fontsize=9)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    ax.set_xticks(list(Ns))
    ax.set_xticklabels([str(n) for n in Ns])
    _style(ax, f"Analytic: model-state memory per GPU, psi = {psi/1e9:.1f}B params, mixed-precision Adam",
           "number of GPUs (N)", "GB per GPU")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def table_summary(results, stages=("baseline",) + tuple(STAGE_ORDER)):
    import pandas as pd
    rows = []
    for s in stages:
        r = results[s]
        pk = r["peak_by_cat"]
        rows.append({
            "stage": LABEL[s],
            "N": r["world_size"],
            "params (MB)": pk["params"] / MB,
            "grads (MB)": pk["grads"] / MB,
            "optim (MB)": pk["optimizer"] / MB,
            "activations (MB)": pk["activations"] / MB,
            "model state (MB)": r["peak_model_state"] / MB,
            "PEAK all (MB)": r["peak_total"] / MB,
            "comm/step (MB)": r["comm_bytes_per_rank_per_step"] / MB,
            "comm (x psi)": r["comm_ratio_psi"],
            "final loss": r["losses"][-1],
        })
    return pd.DataFrame(rows).set_index("stage").round(4)


def fig_topology(single, multi, path=None):
    """The cost of leaving the box.

    `single` / `multi` are dicts stage -> run_stage result, one with all 32
    vGPUs inside a single node, one spread over 4 nodes of 8.  Same bytes on
    the wire in both cases -- only the slowest hop in the ring changed.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    stages = STAGE_ORDER
    xs, w = np.arange(len(stages)), 0.36

    def ms(d, s):
        return d[s]["comm_seconds_modelled"] * 1e3 / d[s]["steps"]

    a = [ms(single, s) for s in stages]
    b = [ms(multi, s) for s in stages]
    ax1.bar(xs - w / 2, a, w, label="1 node x 32 GPUs (NVLink 450 GB/s)",
            color="#55A868", edgecolor="white")
    ax1.bar(xs + w / 2, b, w, label="4 nodes x 8 GPUs (inter-node 50 GB/s)",
            color="#C44E52", edgecolor="white")
    for i, (u, v) in enumerate(zip(a, b)):
        ax1.text(i + w / 2, v * 1.02, f"{v/u:.1f}x", ha="center", fontsize=9,
                 fontweight="bold")
    ax1.set_xticks(xs)
    ax1.set_xticklabels([LABEL[s] for s in stages], rotation=12)
    _style(ax1, "Same bytes, different wires: the cost of leaving the box",
           "", "modelled comm time per step (ms)")
    ax1.legend(frameon=False, fontsize=9)

    # why a ring, and not "gather everything to rank 0"?
    Ns = np.array([2, 4, 8, 16, 32, 64, 128, 256])
    ring = 2 * (Ns - 1) / Ns
    central = 2 * (Ns - 1)
    ax2.plot(Ns, ring, marker="o", lw=2, label=r"ring all-reduce  $2\frac{N-1}{N}\psi$")
    ax2.plot(Ns, central, marker="s", lw=2,
             label=r"gather to rank 0, broadcast back  $2(N-1)\psi$")
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log", base=2)
    ax2.set_xticks(Ns)
    ax2.set_xticklabels([str(n) for n in Ns])
    _style(ax2, "Why every GPU does the same redundant reduction",
           "number of GPUs (N)", r"traffic on the busiest link ($\times\psi$)")
    ax2.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def fig_compute(results, psi, path=None):
    """What ZeRO changes about *arithmetic*, not memory.

    Three per-rank quantities, each normalised to DDP:
      * model FLOPs      -- unchanged, except ZeRO-3 pays +1/3 for recompute
      * optimizer work   -- falls by N: DDP runs Adam over every parameter on
                            every rank, which is N-way redundant
      * communication    -- unchanged for stages 1-2, 1.5x for stage 3
    """
    from .experiment import analytic_flops
    stages = STAGE_ORDER
    ddp = results["ddp"]

    def flops(r):
        return analytic_flops(psi, r["tokens_per_rank"],
                              r["layer_evals_per_step"] > r["n_groups"])

    series = {
        "model FLOPs / rank": [flops(results[s]) / flops(ddp) for s in stages],
        "optimizer work / rank": [results[s]["optim_elems_per_step"]
                                  / ddp["optim_elems_per_step"] for s in stages],
        "communication / rank": [results[s]["comm_ratio_psi"]
                                 / ddp["comm_ratio_psi"] for s in stages],
    }
    colors = ["#4C72B0", "#55A868", "#C44E52"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    xs, w = np.arange(len(stages)), 0.26
    for i, (k, v) in enumerate(series.items()):
        bars = ax1.bar(xs + (i - 1) * w, v, w, label=k, color=colors[i],
                       edgecolor="white")
        for b, val in zip(bars, v):
            ax1.text(b.get_x() + b.get_width() / 2, val + 0.03,
                     f"{val:.2f}" if val >= 0.1 else f"{val:.3f}",
                     ha="center", fontsize=7.5)
    ax1.axhline(1.0, ls="--", c="k", lw=1, alpha=0.5)
    ax1.set_xticks(xs)
    ax1.set_xticklabels([LABEL[s] for s in stages], rotation=12)
    _style(ax1, "Per-GPU work, relative to DDP", "", "x DDP  (1.0 = same work)")
    ax1.legend(frameon=False, fontsize=9)
    ax1.set_ylim(0, 1.75)

    # the optimizer-redundancy point, on a log axis where it is visible
    opt = [results[s]["optim_elems_per_step"] for s in stages]
    ax2.bar([LABEL[s] for s in stages], opt, color="#55A868", edgecolor="white")
    ax2.axhline(psi, ls="--", c="crimson", lw=1.3)
    ax2.text(0.02, psi * 1.15, f"the whole model ({psi:,} params)",
             color="crimson", fontsize=9, transform=ax2.get_yaxis_transform())
    ax2.set_yscale("log")
    ax2.tick_params(axis="x", labelrotation=12)
    _style(ax2, "Adam element-updates per GPU per step", "",
           "parameter elements updated")
    n = results["zero1"]["world_size"]
    ax2.annotate(f"{n}x less\n(DDP does this work\n{n} times over)",
                 xy=(1, opt[1]), xytext=(1.6, opt[0] * 0.28),
                 arrowprops=dict(arrowstyle="->", lw=1.2), fontsize=9)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig
