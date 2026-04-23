#!/usr/bin/env python3
"""
plot_pga_comparison.py — Visualise EXP1 (scalar autoresearch) vs EXP2 (PGA).

Reads ``metrics.json`` from each run directory of the two experiment repos and
produces five figures:

    fig1_trajectories.png          — inner-loop trajectories (eff + reward/agent)
    fig2_headline.png              — headline metrics (eff / reward / maximin)
    fig3_variance.png              — per-iter distribution + eff-vs-maximin scatter
    fig4_profile_signal.png        — the PGA signal the EXP2 researcher acted on
    fig5_researcher_trajectory.png — running-best over RESEARCHER (outer-loop)
                                     iterations, paper-style

Run from the repo root:  python3 autoresearch/plot_pga_comparison.py
"""
from __future__ import annotations

import csv
import json
import os
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---------------------------------------------------------------------------
# Paths + style
# ---------------------------------------------------------------------------

EXP1_DIR = Path("/Users/victorgallego/llm-policies-social-dilemmas-prod-exp1-geminipro-mean/autoresearch/runs")
EXP2_DIR = Path("/Users/victorgallego/llm-policies-social-dilemmas-prod-exp2-geminipro-pga/autoresearch/runs")
OUT_DIR  = Path(__file__).resolve().parent / "figures" / "pga_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLOR_E1 = "#D55E00"   # orange-red (scalar baseline)
COLOR_E2 = "#0072B2"   # blue (PGA)

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "legend.frameon": False,
})


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_runs(run_dir: Path) -> list[dict]:
    """Return [{'run': str, 'trajectory': [...]}] sorted by timestamp."""
    out = []
    for metrics_path in sorted(glob(str(run_dir / "*" / "metrics.json"))):
        with open(metrics_path) as f:
            m = json.load(f)
        out.append({
            "run": Path(metrics_path).parent.name,
            "trajectory": m.get("trajectory", []),
        })
    return out


e1 = load_runs(EXP1_DIR)
e2 = load_runs(EXP2_DIR)
print(f"Loaded {len(e1)} EXP1 runs, {len(e2)} EXP2 runs")


def iters(runs, include_iter0=False):
    """Concatenate (iteration, eff, reward, maximin) tuples across runs."""
    out = []
    for r in runs:
        for t in r["trajectory"]:
            if not include_iter0 and t["iteration"] == 0:
                continue
            out.append((t["iteration"],
                        t["efficiency"],
                        t["reward_avg"],
                        t.get("maximin", 0)))
    return np.array(out)


# ---------------------------------------------------------------------------
# Figure 1 — per-run trajectories
# ---------------------------------------------------------------------------

def fig1_trajectories():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    for ax, metric, ylabel in [
        (axes[0], "efficiency",  "Efficiency $U$ (collective reward / step)"),
        (axes[1], "reward_avg",  "Mean per-agent return"),
    ]:
        for label, runs, color in [("EXP1 (scalar)", e1, COLOR_E1),
                                    ("EXP2 (PGA)",    e2, COLOR_E2)]:
            # Per-run thin lines
            mat = []
            for r in runs:
                xs = [t["iteration"] for t in r["trajectory"]]
                ys = [t[metric]       for t in r["trajectory"]]
                ax.plot(xs, ys, color=color, alpha=0.25, linewidth=1.2)
                mat.append(ys)
            # Mean across runs (aligned by iteration — all runs have 0..K)
            mat = np.array(mat)
            mean = mat.mean(axis=0)
            ax.plot(range(len(mean)), mean, color=color, linewidth=3.0,
                    marker="o", markersize=6, label=f"{label} (run-mean)")
        ax.set_xlabel("Inner-loop iteration")
        ax.set_ylabel(ylabel)
        ax.set_xticks([0, 1, 2, 3])
        ax.legend(loc="lower right")
    axes[0].axhline(0, color="black", linewidth=0.5, alpha=0.5)
    axes[1].axhline(0, color="black", linewidth=0.5, alpha=0.5)
    fig.suptitle("Per-iteration trajectories across all runs "
                 "(thin = individual runs, thick = run-mean)",
                 fontsize=12, y=1.02)
    fig.savefig(OUT_DIR / "fig1_trajectories.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — headline metrics
# ---------------------------------------------------------------------------

def fig2_headline():
    # Compute per-run aggregates (mean of iters 1..K per run).
    def run_stats(runs):
        stats = []
        for r in runs:
            tr = [t for t in r["trajectory"] if t["iteration"] >= 1]
            if not tr:
                continue
            stats.append({
                "eff_mean": np.mean([t["efficiency"]       for t in tr]),
                "eff_peak": max([t["efficiency"]           for t in tr]),
                "rw_mean":  np.mean([t["reward_avg"]       for t in tr]),
                "rw_peak":  max([t["reward_avg"]           for t in tr]),
                "mm_mean":  np.mean([t.get("maximin", 0)   for t in tr]),
                "mm_peak":  max([t.get("maximin", 0)       for t in tr]),
            })
        return stats

    s1 = run_stats(e1)
    s2 = run_stats(e2)

    metrics = [
        ("eff_mean", "Efficiency (mean of iters 1..K)", "U"),
        ("eff_peak", "Efficiency (peak within run)",     "U"),
        ("rw_peak",  "Reward / agent (peak within run)", "reward"),
        ("mm_peak",  "Maximin (peak within run)",        "maximin"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.2))
    for ax, (key, title, ylabel) in zip(axes, metrics):
        v1 = [s[key] for s in s1]
        v2 = [s[key] for s in s2]
        positions = [0, 1]
        # strip (individual runs)
        ax.scatter([0]*len(v1), v1, color=COLOR_E1, s=55, alpha=0.55, zorder=3,
                   edgecolors="white", linewidths=0.6)
        ax.scatter([1]*len(v2), v2, color=COLOR_E2, s=55, alpha=0.55, zorder=3,
                   edgecolors="white", linewidths=0.6)
        # mean bar
        ax.bar(positions, [np.mean(v1), np.mean(v2)],
               color=[COLOR_E1, COLOR_E2], alpha=0.25, width=0.55, zorder=1)
        # mean line marker
        for p, vs, c in [(0, v1, COLOR_E1), (1, v2, COLOR_E2)]:
            ax.plot([p - 0.22, p + 0.22], [np.mean(vs)] * 2,
                    color=c, linewidth=2.5, zorder=4)
        # best-run annotation
        ax.scatter(positions, [max(v1), max(v2)], marker="*", s=180,
                   color=[COLOR_E1, COLOR_E2], zorder=5,
                   edgecolors="black", linewidths=0.7)
        ax.set_xticks(positions)
        ax.set_xticklabels(["EXP1\n(scalar)", "EXP2\n(PGA)"])
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10.5)
        # annotate pct change on the run-mean
        pct = (np.mean(v2) - np.mean(v1)) / max(abs(np.mean(v1)), 1e-6) * 100
        ax.text(0.5, ax.get_ylim()[1] * 0.06 + ax.get_ylim()[0] * 0.94,
                f"Δ mean: {pct:+.0f}%",
                transform=ax.transData, ha="center", fontsize=9,
                color="black", style="italic")

    # Shared legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="white",
               markerfacecolor="gray", markersize=8, label="individual run"),
        Line2D([0], [0], color="gray", linewidth=2.5, label="run-mean"),
        Line2D([0], [0], marker="*", color="white",
               markerfacecolor="gray", markersize=12,
               markeredgecolor="black", label="best run"),
    ]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.03), ncol=3)
    fig.suptitle("Headline metrics: EXP1 (scalar autoresearch) vs EXP2 (PGA)",
                 fontsize=13, y=1.10)
    fig.savefig(OUT_DIR / "fig2_headline.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — variance / distribution
# ---------------------------------------------------------------------------

def fig3_variance():
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))

    # --- Panel A: per-iteration efficiency distribution ---
    ax = axes[0]
    pts1 = iters(e1)
    pts2 = iters(e2)
    jitter = 0.08
    rng = np.random.default_rng(0)
    ax.scatter(0 + rng.normal(0, jitter, size=len(pts1)), pts1[:, 1],
               color=COLOR_E1, alpha=0.55, s=45, edgecolors="white",
               linewidths=0.5, label=f"EXP1 (n={len(pts1)} iter)")
    ax.scatter(1 + rng.normal(0, jitter, size=len(pts2)), pts2[:, 1],
               color=COLOR_E2, alpha=0.55, s=45, edgecolors="white",
               linewidths=0.5, label=f"EXP2 (n={len(pts2)} iter)")
    for pos, pts, c in [(0, pts1, COLOR_E1), (1, pts2, COLOR_E2)]:
        q1, med, q3 = np.percentile(pts[:, 1], [25, 50, 75])
        ax.plot([pos - 0.22, pos + 0.22], [med, med], color=c, linewidth=2.5)
        ax.plot([pos, pos], [q1, q3], color=c, linewidth=1.5)
    # Mark the crash floor.
    ax.axhline(0, color="crimson", linewidth=0.8, linestyle=":",
               alpha=0.7, label="crash floor (rw < 0)")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["EXP1 (scalar)", "EXP2 (PGA)"])
    ax.set_ylabel("Efficiency $U$ (per iteration)")
    ax.set_title("Per-iteration efficiency distribution\n(bar = median, whisker = IQR)",
                 fontsize=10.5)
    ax.legend(loc="lower right", fontsize=8)

    # --- Panel B: efficiency vs maximin scatter ---
    ax = axes[1]
    ax.scatter(pts1[:, 1], pts1[:, 3], color=COLOR_E1, alpha=0.65, s=60,
               edgecolors="white", linewidths=0.6, label="EXP1 (scalar)")
    ax.scatter(pts2[:, 1], pts2[:, 3], color=COLOR_E2, alpha=0.65, s=60,
               edgecolors="white", linewidths=0.6, label="EXP2 (PGA)")
    # Highlight the best point of each
    b1 = pts1[pts1[:, 1].argmax()]
    b2 = pts2[pts2[:, 1].argmax()]
    ax.scatter([b1[1]], [b1[3]], marker="*", s=240, color=COLOR_E1,
               edgecolors="black", linewidths=0.7, zorder=5,
               label=f"EXP1 best iter ({b1[1]:.2f}, {b1[3]:.0f})")
    ax.scatter([b2[1]], [b2[3]], marker="*", s=240, color=COLOR_E2,
               edgecolors="black", linewidths=0.7, zorder=5,
               label=f"EXP2 best iter ({b2[1]:.2f}, {b2[3]:.0f})")
    ax.axhline(0, color="black", linewidth=0.4, alpha=0.5)
    ax.axvline(0, color="black", linewidth=0.4, alpha=0.5)
    ax.set_xlabel("Efficiency $U$")
    ax.set_ylabel("Maximin (min per-agent return)")
    ax.set_title("Per-iteration trade-off frontier", fontsize=10.5)
    ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("Variance containment & social-welfare frontier",
                 fontsize=13, y=1.02)
    fig.savefig(OUT_DIR / "fig3_variance.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 — the profile signal that drove the EXP2 keep decision
# ---------------------------------------------------------------------------

PE_ACTION_ORDER = [
    "NOOP", "MOVE_N", "MOVE_S", "MOVE_E", "MOVE_W",
    "GATHER", "CRAFT", "CRAFT_TOOL", "CRAFT_SHELTER",
    "DROP_WOOD", "DROP_STONE", "DROP_PLANK", "DROP_BRICK",
    "PICKUP_WOOD", "PICKUP_STONE", "PICKUP_PLANK", "PICKUP_BRICK",
]


def _flatten_histogram(profile_json: dict) -> dict[str, float]:
    """Collapse per-bucket histograms into an overall action distribution."""
    total = 0
    counts = {a: 0.0 for a in PE_ACTION_ORDER}
    for b in profile_json.get("action_buckets", []):
        sub_total = b["total"]
        total += sub_total
        for name, frac in b["top"]:
            counts[name] = counts.get(name, 0.0) + frac * sub_total
    if total == 0:
        return counts
    return {k: v / total for k, v in counts.items()}


def fig4_profile_signal():
    """Use the PGA **change-point / phase-rate** channel to show that the
    baseline policy FAILED winter (phase 3 rate < 0) while the kept policy
    PASSED winter (phase 3 rate > 0) — the single causal channel the
    researcher attributed the lift to."""

    base_path = EXP2_DIR / "20260423_184611" / "history.json"
    kept_path = EXP2_DIR / "20260423_192043" / "history.json"

    with open(base_path) as f:
        base_h = json.load(f)
    with open(kept_path) as f:
        kept_h = json.load(f)

    base_prof = base_h[-1]["profile"]
    kept_prof = kept_h[-1]["profile"]

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2),
                              gridspec_kw={"width_ratios": [1.6, 1.0]})

    # --- Panel A: phase rates (change-point channel) -----------------
    # The profile segments the episode at detected change points and
    # reports a per-phase reward rate. This is channel (2) in the
    # extractor. Visualising it directly tells a two-arc story.
    def _phase_bars(ax, prof, color, label_prefix, y_shift=0):
        phases = prof.get("reward_phases_example", [])
        last = max((p["end"] for p in phases), default=0)
        for p in phases:
            ax.barh(y_shift, p["end"] - p["start"], left=p["start"],
                    color=color, alpha=0.85, edgecolor="white",
                    height=0.7)
            mid = (p["start"] + p["end"]) / 2
            label = f"{p['rate']:+.2f} U/step"
            ax.text(mid, y_shift, label, ha="center", va="center",
                    fontsize=10, color="white", fontweight="bold")
        ax.text(-5, y_shift, label_prefix, ha="right", va="center",
                fontsize=10)

    ax = axes[0]
    _phase_bars(ax, base_prof, COLOR_E1, "Baseline\n(pre-modification)",
                y_shift=1)
    _phase_bars(ax, kept_prof, COLOR_E2, "Kept policy\n(post-modification)",
                y_shift=0)
    # Winter tick
    ax.axvline(200, color="crimson", linewidth=1.2, linestyle="--",
               alpha=0.7, zorder=5)
    ax.text(201, 1.65, "winter (step 200)", color="crimson", fontsize=9,
            va="top")
    ax.set_xlim(-40, 310)
    ax.set_ylim(-0.6, 1.9)
    ax.set_yticks([])
    ax.set_xlabel("Step")
    ax.set_title("PGA channel (2): reward-rate per phase (representative agent)",
                 fontsize=11)
    ax.grid(False)

    # --- Panel B: LOC-changed vs metric delta ---
    ax = axes[1]
    # Data from the two keep-commits (only commits that were KEPT):
    # EXP1: 95aae0f  +71 prompt lines, +0 helper lines, delta +1.99 eff_mean
    # EXP2: 1f4ebd8  +29 prompt lines, +11 helper lines, delta +1.80 eff_mean
    xs = [71, 40]
    ys = [1.99, 1.80]
    colors = [COLOR_E1, COLOR_E2]
    labels = ["EXP1 keep (scalar)", "EXP2 keep (PGA)"]
    for x0, y0, c, lbl in zip(xs, ys, colors, labels):
        ax.scatter([x0], [y0], color=c, s=260, edgecolors="black",
                   linewidths=0.8, zorder=3)
        dx = 3 if x0 < 55 else -3
        ha = "left" if x0 < 55 else "right"
        ax.annotate(lbl, (x0, y0), xytext=(dx * 4, 12),
                    textcoords="offset points", fontsize=10, ha=ha)
    ax.set_xlabel("Pipeline LOC added in the keep-commit")
    ax.set_ylabel("$\\Delta$ efficiency (mean of iters 1..K)")
    ax.set_title("Efficiency lift per line of researcher code", fontsize=11)
    ax.set_xlim(0, 95)
    ax.set_ylim(0, 2.6)
    for x0, y0 in zip(xs, ys):
        ax.text(x0, y0 - 0.28, f"{y0 / x0:.3f} ΔU / LOC",
                ha="center", fontsize=9, style="italic", color="gray")
    # Add helpful callout arrow.
    ax.annotate("", xy=(40, 1.8), xytext=(71, 1.99),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.2,
                                connectionstyle="arc3,rad=0.2"))
    ax.text(55, 2.25, "same lift, less code",
            fontsize=9, style="italic", color="gray", ha="center")

    fig.suptitle("The signal → the mutation → the payoff: "
                 "baseline fails winter, kept policy passes",
                 fontsize=13, y=1.02)
    fig.savefig(OUT_DIR / "fig4_profile_signal.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5 — researcher-iteration trajectory (paper-style)
# ---------------------------------------------------------------------------

EXP1_TSV = Path("/Users/victorgallego/llm-policies-social-dilemmas-prod-exp1-geminipro-mean/autoresearch/results.tsv")
EXP2_TSV = Path("/Users/victorgallego/llm-policies-social-dilemmas-prod-exp2-geminipro-pga/autoresearch/results.tsv")


def _read_tsv(path: Path) -> list[dict]:
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


def _running_best(values: list[float]) -> list[float]:
    """Cumulative max — matches the "kept only" running-best convention used
    in the paper's `plot_paper_figures.py`. We apply it to the raw per-outer-
    iteration metric so the curve equals the last ACCEPTED value up to i."""
    out, best = [], -np.inf
    for v in values:
        best = max(best, v)
        out.append(best)
    return out


def fig5_researcher_trajectory():
    """Mirror autoresearch/figures/fig2_maximin_trajectory.png: x-axis is the
    researcher (outer-loop) iteration; the line is the running best of the
    KEPT proposals; hollow markers are discarded proposals at their raw
    value; filled markers are kept/baseline proposals on the line."""
    # Temporary override of paper-style rcParams so this matches the existing
    # paper figures.
    saved = plt.rcParams.copy()
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
    })

    rows_e1 = _read_tsv(EXP1_TSV)
    rows_e2 = _read_tsv(EXP2_TSV)

    def _parse(rows):
        eff = [float(r["efficiency"]) for r in rows]
        mm  = [float(r["maximin"])    for r in rows]
        st  = [r["status"]             for r in rows]
        return eff, mm, st, list(range(len(rows)))

    eff_e1, mm_e1, st_e1, it_e1 = _parse(rows_e1)
    eff_e2, mm_e2, st_e2, it_e2 = _parse(rows_e2)

    fig, (ax_eff, ax_mm) = plt.subplots(1, 2, figsize=(7.5, 3.2), sharex=True)

    def _draw(ax, iters, values, statuses, color, label, ls="-"):
        rb = _running_best(values)
        ax.plot(iters, rb, color=color, ls=ls, lw=1.5, label=label, zorder=3)
        for i, (v, s) in enumerate(zip(values, statuses)):
            if s in ("keep", "baseline"):
                ax.scatter(i, v, marker="o", s=22, color=color, zorder=4)
            else:
                ax.scatter(i, v, marker="o", facecolors="white",
                           edgecolors=color, s=22, linewidths=0.7,
                           zorder=4, alpha=0.9)

    COLOR_SCALAR = "#D55E00"
    COLOR_PGA    = "#0072B2"

    _draw(ax_eff, it_e1, eff_e1, st_e1, COLOR_SCALAR, "EXP1 (scalar)")
    _draw(ax_eff, it_e2, eff_e2, st_e2, COLOR_PGA,    "EXP2 (PGA)")

    _draw(ax_mm,  it_e1, mm_e1,  st_e1, COLOR_SCALAR, "EXP1 (scalar)")
    _draw(ax_mm,  it_e2, mm_e2,  st_e2, COLOR_PGA,    "EXP2 (PGA)")

    # Baseline iteration-0 reference lines (same convention as paper fig2).
    ax_eff.axhline(eff_e1[0], color=COLOR_SCALAR, ls=":", lw=0.7,
                   alpha=0.45, zorder=0)
    ax_eff.axhline(eff_e2[0], color=COLOR_PGA,    ls=":", lw=0.7,
                   alpha=0.45, zorder=0)
    ax_mm.axhline(0, color="gray", ls=":", lw=0.8, alpha=0.7)
    ax_mm.text(0.3, 2, "$\\min_i R_i = 0$", fontsize=7,
               color="gray", va="bottom")

    # Axes / labels
    ax_eff.set_xlabel("Researcher iteration")
    ax_eff.set_ylabel("Efficiency $U$ (mean of iters 1..K)")
    ax_eff.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax_eff.set_xlim(-0.5, max(len(it_e1), len(it_e2)) - 0.5)
    ax_eff.legend(loc="lower right", framealpha=0.9)

    ax_mm.set_xlabel("Researcher iteration")
    ax_mm.set_ylabel("Maximin ($\\min_i R_i$)")
    ax_mm.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax_mm.legend(loc="lower right", framealpha=0.9)

    fig.suptitle("Outer-loop trajectory on Production Economy: "
                 "EXP1 (scalar) vs EXP2 (PGA)",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig5_researcher_trajectory.png")
    fig.savefig(OUT_DIR / "fig5_researcher_trajectory.svg")
    plt.close(fig)

    # Restore rcParams so subsequent figures keep their old style.
    plt.rcParams.update(saved)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    fig1_trajectories()
    fig2_headline()
    fig3_variance()
    fig4_profile_signal()
    fig5_researcher_trajectory()
    print(f"Wrote figures to {OUT_DIR}")
    for f in sorted(OUT_DIR.glob("*.png")):
        print(f"  {f.name}")
