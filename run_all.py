"""Run the whole experiment: 5 configurations on 32 vGPUs + a scaling sweep.

    python run_all.py            # full run, writes assets/*.png and results.json
    python run_all.py --quick    # fewer steps / skip the sweep
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")

from vgpu import report
from vgpu.experiment import (analytic_comm, analytic_memory, compare, run_stage,
                             weight_agreement)
from vgpu.memory import GB, MB, fmt
from vgpu.model import GPTConfig, param_count

STAGES = ["baseline", "ddp", "zero1", "zero2", "zero3"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, default=32)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-sweep", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.steps = 3

    os.makedirs("assets", exist_ok=True)
    cfg = GPTConfig()
    psi = param_count(cfg)
    print(f"TinyGPT: {psi:,} parameters  ({psi*4/MB:.2f} MB fp32)")
    print(f"virtual cluster: {args.world} vGPUs, global batch {args.batch}, "
          f"{args.steps} steps\n")

    results = {}
    for s in STAGES:
        r = run_stage(s, cfg, world_size=args.world, steps=args.steps,
                      global_batch=args.batch)
        results[s] = r
        print(f"  {report.LABEL[s]:22s} model_state={fmt(r['peak_model_state']):>9s}  "
              f"peak={fmt(r['peak_total']):>9s}  comm={r['comm_ratio_psi']:.3f} psi  "
              f"loss {r['losses'][0]:.4f}->{r['losses'][-1]:.4f}  ({r['wall_s']:.1f}s)")

    print("\n--- fairness control: every stage recomputes, so activations match ---")
    fair = {}
    for s in ["ddp", "zero1", "zero2", "zero3"]:
        fair[s] = run_stage(s, cfg, world_size=args.world,
                            steps=max(2, args.steps // 2),
                            global_batch=args.batch, checkpoint=True)
        print(f"  {report.LABEL[s]:22s} "
              f"activations={fmt(fair[s]['peak_by_cat']['activations']):>9s}"
              f"  model_state={fmt(fair[s]['peak_model_state']):>9s}"
              f"  peak={fmt(fair[s]['peak_total']):>9s}")


    print("\n--- correctness ---")
    for row in compare(results):
        print(f"  {row['stage']:9s} final={row['final_loss']:.6f}  "
              f"max|dloss|={row['max_abs_dev']:.3e}")
    for row in weight_agreement(results):
        print(f"  {row['stage']:9s} max|dW|={row['max_abs_weight_diff']:.3e}")

    print("\n--- figures ---")
    report.fig_losses(results, "assets/01_losses.png")
    report.fig_memory_breakdown(results, path="assets/02_memory_breakdown.png")
    report.fig_comm(results, path="assets/03_communication.png")
    report.fig_timeline(results, path="assets/04_timeline.png")
    report.fig_analytic(path="assets/06_analytic_7B.png")
    print("  wrote assets/01..04, 06")

    sweep = {}
    if not (args.no_sweep or args.quick):
        print("\n--- scaling sweep ---")
        for n in [1, 2, 4, 8, 16, 32]:
            sweep[n] = {}
            for s in ["ddp", "zero1", "zero2", "zero3"]:
                sweep[n][s] = run_stage(s, cfg, world_size=n, steps=2,
                                        global_batch=args.batch)
            ms = {s: sweep[n][s]["peak_model_state"] / MB for s in sweep[n]}
            print(f"  N={n:3d}  " + "  ".join(f"{k}={v:6.2f}MB" for k, v in ms.items()))
        report.fig_scaling(sweep, "assets/05_scaling.png")
        print("  wrote assets/05")

    # ---- dump a json summary (no tensors) ----
    dump = {}
    for s, r in results.items():
        dump[s] = {k: v for k, v in r.items()
                   if k not in ("fabric", "engines", "timeline")}
    dump["_meta"] = dict(params=psi, world=args.world, steps=args.steps,
                         batch=args.batch, torch=__import__("torch").__version__)
    dump["_sweep"] = {str(n): {s: dict(
        peak_model_state=v["peak_model_state"],
        peak_total=v["peak_total"], comm_ratio=v["comm_ratio_psi"])
        for s, v in d.items()} for n, d in sweep.items()}
    dump["_fair_checkpointed"] = {s: dict(
        activations=v["peak_by_cat"]["activations"],
        peak_model_state=v["peak_model_state"], peak_total=v["peak_total"])
        for s, v in fair.items()}
    dump["_analytic_7p5B_N32"] = {k: v / GB for k, v in
                                  analytic_memory(7.5e9, 32).items()}
    with open("results.json", "w") as f:
        json.dump(dump, f, indent=2)
    print("\nwrote results.json")


if __name__ == "__main__":
    main()
