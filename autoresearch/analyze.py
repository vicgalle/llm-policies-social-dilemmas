#!/usr/bin/env python3
"""
analyze.py — Analyze autoresearch experiment results for SSD policy synthesis.

Usage:
    python3 autoresearch/analyze.py [results_dir_or_file...]

Outputs convergence curves, strategy breakdown, and comparison across runs.
"""

import sys
import os
import csv
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_results(filepath):
    results = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                row["experiment"] = int(row["experiment"])
                row["efficiency"] = float(row["efficiency"])
                row["equality"] = float(row.get("equality", 0))
                row["sustainability"] = float(row.get("sustainability", 0))
                row["peace"] = float(row.get("peace", 0))
                row["reward_avg"] = float(row.get("reward_avg", 0))
                delta_str = row.get("delta_eff", "0").replace("+", "")
                row["delta_float"] = float(delta_str) if delta_str not in ("n/a", "") else 0.0
                row["run_time_s"] = float(row.get("run_time_s", 0))
            except (ValueError, KeyError):
                pass
            results.append(row)
    return results


def summarize(results, label=""):
    prefix = f"[{label}] " if label else ""

    total = len(results)
    kept = [r for r in results if r.get("status") == "keep"]
    discarded = [r for r in results if r.get("status") == "discard"]
    crashed = [r for r in results if r.get("status") == "crash"]

    non_baseline = [r for r in kept if r["experiment"] > 0]
    baseline = results[0] if results else None
    best = max(results, key=lambda r: r.get("efficiency", 0)) if results else None

    print(f"\n{'='*60}")
    print(f"{prefix}AUTORESEARCH RUN SUMMARY")
    print(f"{'='*60}")
    print(f"Total experiments:    {total}")
    print(f"  Kept:               {len(kept)}")
    print(f"  Discarded:          {len(discarded)}")
    print(f"  Crashed:            {len(crashed)}")
    print(f"  Success rate:       {len(kept)/max(total,1)*100:.1f}%")
    print()

    if baseline:
        print(f"Baseline efficiency:  {baseline.get('efficiency', '?')}")
    if best:
        print(f"Best efficiency:      {best.get('efficiency', '?')} (experiment {best.get('experiment', '?')})")
    if baseline and best:
        gain = best.get("efficiency", 0) - baseline.get("efficiency", 0)
        print(f"Total improvement:    {gain:+.4f}")
    print()

    # Strategy breakdown
    if non_baseline:
        print("--- Kept Experiments ---")
        for r in non_baseline:
            desc = r.get("description", "")
            eff = r.get("efficiency", 0)
            delta = r.get("delta_float", 0)
            print(f"  exp {r['experiment']:3d}: eff={eff:.4f} ({delta:+.4f}) | {desc}")

    # Timing
    run_times = [r["run_time_s"] for r in results if r.get("run_time_s", 0) > 0]
    if run_times:
        print(f"\n--- Timing ---")
        print(f"  Mean run:   {sum(run_times)/len(run_times):.0f}s")
        print(f"  Total:      {sum(run_times)/60:.1f} min")

    # Multi-metric view
    if non_baseline:
        print(f"\n--- Multi-Metric View (kept experiments) ---")
        print(f"  {'Exp':>4} {'Efficiency':>11} {'Equality':>9} {'Sustain':>9} {'Peace':>7} {'Reward':>8}")
        print(f"  {'-'*52}")
        for r in [baseline] + non_baseline:
            print(
                f"  {r['experiment']:4d} "
                f"{r.get('efficiency',0):11.4f} "
                f"{r.get('equality',0):9.4f} "
                f"{r.get('sustainability',0):9.1f} "
                f"{r.get('peace',0):7.2f} "
                f"{r.get('reward_avg',0):8.1f}"
            )


def plot_convergence(runs, output_path=None):
    if not HAS_MATPLOTLIB:
        print("matplotlib not installed — skipping plots.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics = [
        ("efficiency", "Efficiency (U)", axes[0, 0]),
        ("equality", "Equality (E)", axes[0, 1]),
        ("sustainability", "Sustainability (S)", axes[1, 0]),
        ("peace", "Peace (P)", axes[1, 1]),
    ]

    for label, results in runs.items():
        kept = [r for r in results if r.get("status") == "keep"]
        exps = [r["experiment"] for r in kept]

        for metric_key, metric_title, ax in metrics:
            vals = [r.get(metric_key, 0) for r in kept]
            ax.plot(exps, vals, marker="o", markersize=4, label=label)
            ax.set_xlabel("Outer Iteration")
            ax.set_ylabel(metric_title)
            ax.set_title(metric_title)
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = output_path or "autoresearch/convergence.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {save_path}")


def main():
    args = sys.argv[1:]

    tsv_files = {}
    if not args:
        default_path = "autoresearch/results.tsv"
        if os.path.exists(default_path):
            tsv_files["default"] = default_path
        else:
            print(f"No results found at {default_path}")
            sys.exit(1)
    else:
        for arg in args:
            p = Path(arg)
            if p.is_dir():
                tsv = p / "results.tsv"
                if tsv.exists():
                    tsv_files[p.name] = str(tsv)
            elif p.is_file():
                tsv_files[p.stem] = str(p)

    all_runs = {}
    for label, filepath in tsv_files.items():
        results = load_results(filepath)
        all_runs[label] = results
        summarize(results, label=label if len(tsv_files) > 1 else "")

    plot_convergence(all_runs)

    if len(all_runs) > 1:
        print(f"\n{'='*60}")
        print("COMPARISON ACROSS RUNS")
        print(f"{'='*60}")
        print(f"{'Run':<20} {'Exps':>6} {'Kept':>6} {'Best Eff':>10} {'Δ Eff':>10}")
        print("-" * 55)
        for label, results in all_runs.items():
            total = len(results)
            kept = len([r for r in results if r.get("status") == "keep" and r["experiment"] > 0])
            baseline_eff = results[0].get("efficiency", 0) if results else 0
            best_eff = max(r.get("efficiency", 0) for r in results) if results else 0
            delta = best_eff - baseline_eff
            print(f"{label:<20} {total:>6} {kept:>6} {best_eff:>10.4f} {delta:>+10.4f}")


if __name__ == "__main__":
    main()
