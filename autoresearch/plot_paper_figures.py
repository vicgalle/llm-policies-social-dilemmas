#!/usr/bin/env python3
"""Generate publication figures for the autoresearch workshop paper.

Reads results.tsv files from all 12 experiment directories and produces:
  - Figure 1: Efficiency trajectory (running best) across researcher iterations
  - Figure 2: Maximin trajectory for the 4 maximin-targeted runs
  - Figure 3: Final efficiency vs equality (bar chart) by condition
  - Figure 4: Researcher behavior summary (iterations & keep rate)

Usage:
    python autoresearch/plot_paper_figures.py
"""

import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Style ──────────────────────────────────────────────────────────────────
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
})

# ── Experiment metadata ────────────────────────────────────────────────────
BASE = Path(os.path.expanduser("~"))

EXPERIMENTS = {
    # Cleanup efficiency
    "exp1":  {"dir": "llm-policies-social-dilemmas-exp1",  "game": "Cleanup", "llm": "Gemini", "target": "eff", "label": "C-Gem-eff-1"},
    "exp2":  {"dir": "llm-policies-social-dilemmas-exp2",  "game": "Cleanup", "llm": "Gemini", "target": "eff", "label": "C-Gem-eff-2"},
    "exp3":  {"dir": "llm-policies-social-dilemmas-exp3-sonnet",  "game": "Cleanup", "llm": "Sonnet", "target": "eff", "label": "C-Son-eff-1"},
    "exp4":  {"dir": "llm-policies-social-dilemmas-exp4-sonnet",  "game": "Cleanup", "llm": "Sonnet", "target": "eff", "label": "C-Son-eff-2"},
    # Cleanup maximin
    "exp5":  {"dir": "llm-policies-social-dilemmas-exp5-rawls",  "game": "Cleanup", "llm": "Gemini", "target": "max", "label": "C-Gem-max-1"},
    "exp6":  {"dir": "llm-policies-social-dilemmas-exp6-maximin", "game": "Cleanup", "llm": "Gemini", "target": "max", "label": "C-Gem-max-2"},
    "exp7":  {"dir": "llm-policies-social-dilemmas-exp7-sonnet-maximin", "game": "Cleanup", "llm": "Sonnet", "target": "max", "label": "C-Son-max-1"},
    "exp8":  {"dir": "llm-policies-social-dilemmas-exp8-sonnet-maximin", "game": "Cleanup", "llm": "Sonnet", "target": "max", "label": "C-Son-max-2"},
    # Gathering efficiency
    "gath1": {"dir": "llm-policies-social-dilemmas-gather-exp1-sonnet", "game": "Gathering", "llm": "Sonnet", "target": "eff", "label": "G-Son-eff-1"},
    "gath2": {"dir": "llm-policies-social-dilemmas-gather-exp2-gemini", "game": "Gathering", "llm": "Gemini", "target": "eff", "label": "G-Gem-eff-1"},
    "gath3": {"dir": "llm-policies-social-dilemmas-gather-exp3-gemini", "game": "Gathering", "llm": "Gemini", "target": "eff", "label": "G-Gem-eff-2"},
    "gath4": {"dir": "llm-policies-social-dilemmas-gather-exp4-sonnet", "game": "Gathering", "llm": "Sonnet", "target": "eff", "label": "G-Son-eff-2"},
}

# Colors by condition group
COLORS = {
    ("Cleanup", "Gemini", "eff"): "#2171b5",  # blue
    ("Cleanup", "Sonnet", "eff"): "#6baed6",  # light blue
    ("Cleanup", "Gemini", "max"): "#cb181d",  # red
    ("Cleanup", "Sonnet", "max"): "#fb6a4a",  # light red
    ("Gathering", "Gemini", "eff"): "#238b45",  # green
    ("Gathering", "Sonnet", "eff"): "#74c476",  # light green
}

LINESTYLES = {1: "-", 2: "--"}  # solid for first run, dashed for second


def read_tsv(exp_key: str) -> list[dict]:
    """Read a results.tsv and return list of row dicts."""
    meta = EXPERIMENTS[exp_key]
    tsv_path = BASE / meta["dir"] / "autoresearch" / "results.tsv"
    rows = []
    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)
    return rows


def running_best(values: list[float], maximize: bool = True) -> list[float]:
    """Compute running best (cumulative max or min)."""
    out = []
    best = -np.inf if maximize else np.inf
    for v in values:
        best = max(best, v) if maximize else min(best, v)
        out.append(best)
    return out


# ── Load all data ──────────────────────────────────────────────────────────
data = {}
for key in EXPERIMENTS:
    rows = read_tsv(key)
    meta = EXPERIMENTS[key]
    iters = list(range(len(rows)))
    effs = [float(r["efficiency"]) for r in rows]
    eqs = [float(r["equality"]) for r in rows]
    maxs = [float(r["maximin"]) for r in rows] if "maximin" in rows[0] else None
    statuses = [r["status"] for r in rows]
    data[key] = {
        "iters": iters,
        "efficiency": effs,
        "equality": eqs,
        "maximin": maxs,
        "status": statuses,
        "running_best_eff": running_best(effs),
        "running_best_max": running_best(maxs) if maxs else None,
        **meta,
    }

OUT_DIR = Path(__file__).parent / "figures"
OUT_DIR.mkdir(exist_ok=True)


# ── Baseline statistics (mean ± s.d. from iteration-0 across all runs) ────
BASELINES = {
    # Cleanup baselines (4 runs per LLM: 2 eff + 2 max)
    ("Cleanup", "Gemini", "eff"):  {"mean": 1.93, "sd": 0.73},
    ("Cleanup", "Sonnet", "eff"):  {"mean": 0.86, "sd": 0.62},
    ("Gathering", "Gemini", "eff"): {"mean": 2.04, "sd": 0.54},
    ("Gathering", "Sonnet", "eff"): {"mean": 0.03, "sd": 0.00},
    # Maximin baselines (from re-evaluation of all Cleanup exp0 policies)
    ("Cleanup", "Gemini", "maximin"): {"mean": -159, "sd": 78},
    ("Cleanup", "Sonnet", "maximin"): {"mean": -151, "sd": 73},
}

# Maximin at best efficiency for Φ_U runs (re-evaluated since metric was added later)
MAXIMIN_AT_BEST_EFF = {
    "exp1": -238.6,  # Gemini, eff=3.247
    "exp2": -182.4,  # Gemini, eff=3.150
    "exp3": -145.6,  # Sonnet, eff=3.097
    "exp4": -247.0,  # Sonnet, eff=3.137
}


# ── Figure 1: Efficiency trajectory (Cleanup left, Gathering right) ────────
def plot_efficiency_trajectory():
    fig, (ax_c, ax_g) = plt.subplots(1, 2, figsize=(5.5, 3.0),
                                      gridspec_kw={"width_ratios": [2.5, 1.5]},
                                      sharey=True)

    # Gathering markers to distinguish LLMs beyond color
    MARKERS_GATH = {"Gemini": "s", "Sonnet": "D"}

    legend_entries = {}
    run_counter = {}

    for key, d in data.items():
        cond = (d["game"], d["llm"], d["target"])
        run_counter.setdefault(cond, 0)
        run_counter[cond] += 1
        run_num = run_counter[cond]

        color = COLORS[cond]
        ls = LINESTYLES.get(run_num, "-")
        ax = ax_c if d["game"] == "Cleanup" else ax_g

        # Legend label
        target_lbl = "$\\Phi_U$" if d["target"] == "eff" else "$\\Phi_{\\min}$"
        label_str = f"{d['llm']}, {target_lbl}"
        if cond not in legend_entries:
            legend_entries[cond] = label_str
            lbl = label_str
        else:
            lbl = None

        # Running best line
        ax.plot(d["iters"], d["running_best_eff"], color=color, ls=ls,
                lw=1.5, label=lbl, zorder=3)

        # Individual experiment dots (kept = filled, discarded = hollow)
        mk = MARKERS_GATH.get(d["llm"], "o") if d["game"] == "Gathering" else "o"
        for i, (eff, st) in enumerate(zip(d["efficiency"], d["status"])):
            if st in ("keep", "baseline"):
                ax.scatter(i, eff, marker=mk, s=16, color=color,
                           edgecolors=color, zorder=2)
            else:
                ax.scatter(i, eff, marker=mk, facecolors="white",
                           edgecolors=color, s=8, linewidths=0.4,
                           zorder=2, alpha=0.35)

    # Cleanup baseline lines (mean only, no bands)
    bl_gem = BASELINES[("Cleanup", "Gemini", "eff")]
    bl_son = BASELINES[("Cleanup", "Sonnet", "eff")]
    ax_c.axhline(bl_gem["mean"], color=COLORS[("Cleanup", "Gemini", "eff")],
                 ls=":", lw=0.8, alpha=0.5, zorder=0)
    ax_c.axhline(bl_son["mean"], color=COLORS[("Cleanup", "Sonnet", "eff")],
                 ls=":", lw=0.8, alpha=0.5, zorder=0)
    ax_c.text(17.3, bl_gem["mean"], "BL (Gem)", fontsize=5.5,
              color=COLORS[("Cleanup", "Gemini", "eff")], ha="left", va="center", alpha=0.7)
    ax_c.text(17.3, bl_son["mean"], "BL (Son)", fontsize=5.5,
              color=COLORS[("Cleanup", "Sonnet", "eff")], ha="left", va="center", alpha=0.7)

    # Gathering baseline lines
    bl_gem_g = BASELINES[("Gathering", "Gemini", "eff")]
    bl_son_g = BASELINES[("Gathering", "Sonnet", "eff")]
    ax_g.axhline(bl_gem_g["mean"], color=COLORS[("Gathering", "Gemini", "eff")],
                 ls=":", lw=0.8, alpha=0.5, zorder=0)
    ax_g.axhline(bl_son_g["mean"], color=COLORS[("Gathering", "Sonnet", "eff")],
                 ls=":", lw=0.8, alpha=0.5, zorder=0)

    ax_c.set_xlabel("Researcher iteration")
    ax_c.set_ylabel("Efficiency ($U$)")
    ax_c.set_title("Cleanup ($N{=}10$)", fontsize=9)
    ax_c.set_xlim(-0.5, 17.5)
    ax_c.set_ylim(-0.5, 3.6)
    ax_c.xaxis.set_major_locator(ticker.MultipleLocator(4))
    ax_c.legend(loc="center right", framealpha=0.9, fontsize=7)

    ax_g.set_xlabel("Researcher iteration")
    ax_g.set_title("Gathering ($N{=}4$)", fontsize=9)
    ax_g.set_xlim(-0.5, 5.5)
    ax_g.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax_g.legend(loc="lower right", framealpha=0.9, fontsize=7)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig1_efficiency_trajectory.png")
    fig.savefig(OUT_DIR / "fig1_efficiency_trajectory.svg")
    plt.close(fig)
    print(f"  Saved fig1_efficiency_trajectory")


# ── Figure 2: Maximin trajectory (4 maximin runs) ─────────────────────────
def plot_maximin_trajectory():
    fig, ax = plt.subplots(figsize=(5.5, 3.2))

    maximin_keys = [k for k, d in data.items() if d["target"] == "max"]
    run_counter = {}

    for key in maximin_keys:
        d = data[key]
        cond = (d["game"], d["llm"], d["target"])
        run_counter.setdefault(cond, 0)
        run_counter[cond] += 1
        run_num = run_counter[cond]

        color = COLORS[cond]
        ls = LINESTYLES.get(run_num, "-")

        label_str = f"{d['llm']} (run {run_num})"
        ax.plot(d["iters"], d["running_best_max"], color=color, ls=ls,
                lw=1.5, label=label_str, zorder=3)

        # Individual experiment dots
        for i, (val, st) in enumerate(zip(d["maximin"], d["status"])):
            if st in ("keep", "baseline"):
                ax.scatter(i, val, marker="o", s=18, color=color, zorder=2)
            else:
                ax.scatter(i, val, marker="o", facecolors="white",
                           edgecolors=color, s=18, linewidths=0.7, zorder=2)

    ax.axhline(0, color="gray", ls=":", lw=0.8, alpha=0.7)
    ax.text(0.5, 5, "$\\min_i R_i = 0$", fontsize=7, color="gray", va="bottom")

    # Baseline reference line (mean of starting maximin across all 4 runs)
    bl_mean = np.mean([-98.8, -83.8, -188.6, -59.0])
    ax.axhline(bl_mean, color="gray", ls=":", lw=0.7, alpha=0.4, zorder=0)
    ax.text(17.3, bl_mean, "BL", fontsize=5.5,
            color="gray", ha="left", va="center", alpha=0.6)

    ax.set_xlabel("Researcher iteration")
    ax.set_ylabel("Maximin ($\\min_i R_i$)")
    ax.set_xlim(-0.5, 17.5)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.legend(loc="lower right", framealpha=0.9)
    fig.savefig(OUT_DIR / "fig2_maximin_trajectory.png")
    fig.savefig(OUT_DIR / "fig2_maximin_trajectory.svg")
    plt.close(fig)
    print(f"  Saved fig2_maximin_trajectory")


# ── Figure 3: Three-panel bar chart (U, E, min_i R_i) ───────────────────
def plot_efficiency_equality_bars():
    """Three-panel bar chart: final efficiency, equality, and maximin per condition."""
    conditions = [
        ("Gem\n$\\Phi_U$",        "Cleanup",    "Gemini", "eff"),
        ("Son\n$\\Phi_U$",        "Cleanup",    "Sonnet", "eff"),
        ("Gem\n$\\Phi_{\\min}$",  "Cleanup",    "Gemini", "max"),
        ("Son\n$\\Phi_{\\min}$",  "Cleanup",    "Sonnet", "max"),
        ("Gem\n(Gath)",            "Gathering",  "Gemini", "eff"),
        ("Son\n(Gath)",            "Gathering",  "Sonnet", "eff"),
    ]

    eff_means, eff_errs = [], []
    eq_means, eq_errs = [], []
    max_means, max_errs = [], []

    # Map exp keys to their condition for maximin lookup
    eff_exp_keys = {"exp1": 0, "exp2": 0, "exp3": 1, "exp4": 1}  # index into conditions

    for idx, (label, game, llm, target) in enumerate(conditions):
        runs = [d for d in data.values()
                if d["game"] == game and d["llm"] == llm and d["target"] == target]
        # Best efficiency and corresponding equality for each run
        best_effs = [max(r["efficiency"]) for r in runs]
        best_eqs = []
        for r in runs:
            best_idx = int(np.argmax(r["efficiency"]))
            best_eqs.append(r["equality"][best_idx])

        eff_means.append(np.mean(best_effs))
        eff_errs.append(np.std(best_effs) if len(best_effs) > 1 else 0)
        eq_means.append(np.mean(best_eqs))
        eq_errs.append(np.std(best_eqs) if len(best_eqs) > 1 else 0)

        # Maximin at best config
        if game == "Cleanup" and target == "eff":
            # Use re-evaluated maximin values (not in TSV)
            if llm == "Gemini":
                maxs = [MAXIMIN_AT_BEST_EFF["exp1"], MAXIMIN_AT_BEST_EFF["exp2"]]
            else:
                maxs = [MAXIMIN_AT_BEST_EFF["exp3"], MAXIMIN_AT_BEST_EFF["exp4"]]
        elif game == "Cleanup" and target == "max":
            # Maximin at best maximin iteration
            maxs = []
            for r in runs:
                best_max_idx = int(np.argmax(r["maximin"]))
                maxs.append(r["maximin"][best_max_idx])
        else:
            # Gathering: maximin available in data at best efficiency
            maxs = []
            for r in runs:
                best_idx = int(np.argmax(r["efficiency"]))
                maxs.append(r["maximin"][best_idx])

        max_means.append(np.mean(maxs))
        max_errs.append(np.std(maxs) if len(maxs) > 1 else 0)

    x = np.arange(len(conditions))
    width = 0.6
    colors = [COLORS[(c[1], c[2], c[3])] for c in conditions]

    fig, (ax_u, ax_e, ax_m) = plt.subplots(1, 3, figsize=(7.0, 2.8),
                                             gridspec_kw={"width_ratios": [1, 1, 1.15]})

    # Panel 1: Efficiency (U)
    ax_u.bar(x, eff_means, width, yerr=eff_errs, color=colors,
             edgecolor="white", linewidth=0.5, capsize=3, zorder=3)
    ax_u.set_ylabel("Efficiency ($U$)")
    ax_u.set_ylim(0, 3.8)
    ax_u.set_title("(a) Efficiency", fontsize=9)
    ax_u.axvline(3.5, color="gray", ls="--", lw=0.5, alpha=0.4)

    # Panel 2: Equality (E)
    ax_e.bar(x, eq_means, width, yerr=eq_errs, color=colors,
             edgecolor="white", linewidth=0.5, capsize=3, zorder=3)
    ax_e.set_ylabel("Equality ($E$)")
    ax_e.set_ylim(0, 1.15)
    ax_e.set_title("(b) Equality", fontsize=9)
    ax_e.axvline(3.5, color="gray", ls="--", lw=0.5, alpha=0.4)

    # Panel 3: Maximin (min_i R_i)
    bar_colors_m = []
    for i, m in enumerate(max_means):
        bar_colors_m.append(colors[i])
    ax_m.bar(x, max_means, width, yerr=max_errs, color=bar_colors_m,
             edgecolor="white", linewidth=0.5, capsize=3, zorder=3)
    ax_m.set_ylabel("Maximin ($\\min_i R_i$)")
    ax_m.set_title("(c) Maximin", fontsize=9)
    ax_m.axhline(0, color="gray", ls=":", lw=0.7, alpha=0.5)
    ax_m.axvline(3.5, color="gray", ls="--", lw=0.5, alpha=0.4)

    # Shared x-axis labels and game annotations
    for ax in (ax_u, ax_e, ax_m):
        ax.set_xticks(x)
        ax.set_xticklabels([c[0] for c in conditions], fontsize=7)

    # Game labels
    ax_u.text(1.5, 3.65, "Cleanup", ha="center", fontsize=7, color="gray")
    ax_u.text(4.5, 3.65, "Gathering", ha="center", fontsize=7, color="gray")
    ax_e.text(1.5, 1.10, "Cleanup", ha="center", fontsize=7, color="gray")
    ax_e.text(4.5, 1.10, "Gathering", ha="center", fontsize=7, color="gray")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig3_efficiency_equality.png")
    fig.savefig(OUT_DIR / "fig3_efficiency_equality.svg")
    plt.close(fig)
    print(f"  Saved fig3_efficiency_equality")


# ── Figure 4: Researcher behavior (iterations & keep rate) ────────────────
def plot_researcher_behavior():
    """Compact summary of researcher behavior across all runs."""
    run_labels = []
    n_iters_list = []
    keep_rates = []
    bar_colors = []

    for key in ["exp1", "exp2", "exp3", "exp4",
                "exp5", "exp6", "exp7", "exp8",
                "gath1", "gath2", "gath3", "gath4"]:
        d = data[key]
        n = len(d["iters"]) - 1  # subtract baseline
        n_kept = sum(1 for s in d["status"] if s == "keep") - (1 if d["status"][0] in ("keep", "baseline") else 0)
        # Adjust: baseline is always iter 0, count only non-baseline keeps
        total_non_baseline = len(d["iters"]) - 1
        keeps_non_baseline = sum(1 for i, s in enumerate(d["status"]) if s == "keep" and i > 0)
        kr = keeps_non_baseline / total_non_baseline if total_non_baseline > 0 else 0

        run_labels.append(d["label"])
        n_iters_list.append(total_non_baseline)
        keep_rates.append(kr)
        bar_colors.append(COLORS[(d["game"], d["llm"], d["target"])])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.5), sharey=True)

    y = np.arange(len(run_labels))

    # Number of iterations
    ax1.barh(y, n_iters_list, color=bar_colors, edgecolor="white", linewidth=0.5)
    ax1.set_xlabel("Researcher iterations")
    ax1.set_yticks(y)
    ax1.set_yticklabels(run_labels)
    ax1.invert_yaxis()
    for i, v in enumerate(n_iters_list):
        ax1.text(v + 0.3, i, str(v), va="center", fontsize=7)

    # Keep rate
    ax2.barh(y, [k * 100 for k in keep_rates], color=bar_colors,
             edgecolor="white", linewidth=0.5, alpha=0.7)
    ax2.set_xlabel("Keep rate (%)")
    ax2.set_xlim(0, 110)
    ax2.invert_yaxis()
    for i, v in enumerate(keep_rates):
        ax2.text(v * 100 + 1.5, i, f"{v:.0%}", va="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig4_researcher_behavior.png")
    fig.savefig(OUT_DIR / "fig4_researcher_behavior.svg")
    plt.close(fig)
    print(f"  Saved fig4_researcher_behavior")


# ── Figure 5 (combined): Cleanup trajectories + bar summary ───────────────
def plot_combined_cleanup():
    """Three-panel figure: Cleanup eff trajectory | maximin trajectory | bar chart."""
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.8),
                              gridspec_kw={"width_ratios": [1.2, 1.2, 1]})
    ax_eff, ax_max, ax_bar = axes

    # --- Panel A: Cleanup efficiency trajectories (8 runs) ---
    legend_entries = {}
    run_counter = {}
    cleanup_keys = [k for k, d in data.items() if d["game"] == "Cleanup"]

    for key in cleanup_keys:
        d = data[key]
        cond = (d["game"], d["llm"], d["target"])
        run_counter.setdefault(cond, 0)
        run_counter[cond] += 1
        run_num = run_counter[cond]
        color = COLORS[cond]
        ls = LINESTYLES.get(run_num, "-")

        target_lbl = "$\\Phi_U$" if d["target"] == "eff" else "$\\Phi_{\\min}$"
        label_str = f"{d['llm']}, {target_lbl}"
        lbl = label_str if cond not in legend_entries else None
        if lbl:
            legend_entries[cond] = True

        ax_eff.plot(d["iters"], d["running_best_eff"], color=color, ls=ls,
                    lw=1.3, label=lbl, zorder=3)
        for i, (eff, st) in enumerate(zip(d["efficiency"], d["status"])):
            if st in ("keep", "baseline"):
                ax_eff.scatter(i, eff, marker="o", s=12, color=color, zorder=2)
            else:
                ax_eff.scatter(i, eff, marker="o", facecolors="white",
                               edgecolors=color, s=10, linewidths=0.5,
                               zorder=2, alpha=0.5)

    ax_eff.axhline(2.75, color="gray", ls=":", lw=0.7, alpha=0.5)
    ax_eff.axhline(1.37, color="gray", ls=":", lw=0.7, alpha=0.5)
    ax_eff.set_xlabel("Researcher iteration")
    ax_eff.set_ylabel("Efficiency ($U$)")
    ax_eff.set_title("(a) Efficiency", fontsize=9)
    ax_eff.set_xlim(-0.5, 17.5)
    ax_eff.set_ylim(-0.5, 3.6)
    ax_eff.xaxis.set_major_locator(ticker.MultipleLocator(4))
    ax_eff.legend(loc="lower right", framealpha=0.9, fontsize=6.5, ncol=1)

    # --- Panel B: Maximin trajectories (4 runs) ---
    maximin_keys = [k for k, d in data.items() if d["target"] == "max"]
    run_counter2 = {}
    for key in maximin_keys:
        d = data[key]
        cond = (d["game"], d["llm"], d["target"])
        run_counter2.setdefault(cond, 0)
        run_counter2[cond] += 1
        run_num = run_counter2[cond]
        color = COLORS[cond]
        ls = LINESTYLES.get(run_num, "-")

        label_str = f"{d['llm']} ({run_num})"
        ax_max.plot(d["iters"], d["running_best_max"], color=color, ls=ls,
                    lw=1.3, label=label_str, zorder=3)
        for i, (val, st) in enumerate(zip(d["maximin"], d["status"])):
            if st in ("keep", "baseline"):
                ax_max.scatter(i, val, marker="o", s=12, color=color, zorder=2)
            else:
                ax_max.scatter(i, val, marker="o", facecolors="white",
                               edgecolors=color, s=10, linewidths=0.5,
                               zorder=2, alpha=0.5)

    ax_max.axhline(0, color="gray", ls=":", lw=0.7, alpha=0.5)
    ax_max.set_xlabel("Researcher iteration")
    ax_max.set_ylabel("Maximin ($\\min_i R_i$)")
    ax_max.set_title("(b) Maximin", fontsize=9)
    ax_max.set_xlim(-0.5, 17.5)
    ax_max.xaxis.set_major_locator(ticker.MultipleLocator(4))
    ax_max.legend(loc="lower right", framealpha=0.9, fontsize=6.5)

    # --- Panel C: Final efficiency vs equality summary ---
    conditions = [
        ("Gem\n$\\Phi_U$", "Cleanup", "Gemini", "eff"),
        ("Son\n$\\Phi_U$", "Cleanup", "Sonnet", "eff"),
        ("Gem\n$\\Phi_{\\min}$", "Cleanup", "Gemini", "max"),
        ("Son\n$\\Phi_{\\min}$", "Cleanup", "Sonnet", "max"),
    ]

    eff_vals, eq_vals = [], []
    for _, game, llm, target in conditions:
        runs = [d for d in data.values()
                if d["game"] == game and d["llm"] == llm and d["target"] == target]
        eff_vals.append([max(r["efficiency"]) for r in runs])
        best_eqs = []
        for r in runs:
            idx = int(np.argmax(r["efficiency"]))
            best_eqs.append(r["equality"][idx])
        eq_vals.append(best_eqs)

    x = np.arange(len(conditions))
    w = 0.35

    ax_bar.bar(x - w / 2, [np.mean(e) for e in eff_vals], w,
               yerr=[np.std(e) if len(e) > 1 else 0 for e in eff_vals],
               color=[COLORS[(c[1], c[2], c[3])] for c in conditions],
               edgecolor="white", linewidth=0.5, capsize=2, label="$U$", zorder=3)
    ax_bar.bar(x + w / 2, [np.mean(e) for e in eq_vals], w,
               yerr=[np.std(e) if len(e) > 1 else 0 for e in eq_vals],
               color=[COLORS[(c[1], c[2], c[3])] for c in conditions],
               edgecolor="white", linewidth=0.5, alpha=0.4, capsize=2,
               hatch="//", label="$E$", zorder=3)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([c[0] for c in conditions], fontsize=7)
    ax_bar.set_ylabel("Metric value")
    ax_bar.set_title("(c) Best config", fontsize=9)
    ax_bar.set_ylim(0, 3.8)
    ax_bar.legend(loc="upper right", framealpha=0.9, fontsize=7)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_combined_cleanup.png")
    fig.savefig(OUT_DIR / "fig_combined_cleanup.svg")
    plt.close(fig)
    print(f"  Saved fig_combined_cleanup")


# ── Figure 6: Efficiency vs Equality scatter ────────────────────────────────
def plot_efficiency_equality_scatter():
    """Scatter plot: each run's best result as a point (U vs E), colored by condition."""
    fig, ax = plt.subplots(figsize=(4.0, 3.2))

    MARKERS = {
        ("Cleanup", "eff"):  "o",
        ("Cleanup", "max"):  "s",
        ("Gathering", "eff"): "D",
    }

    legend_handles = {}

    for key, d in data.items():
        cond = (d["game"], d["llm"], d["target"])
        color = COLORS[cond]
        mk = MARKERS[(d["game"], d["target"])]

        # Best efficiency and equality at that point
        best_idx = int(np.argmax(d["efficiency"]))
        best_eff = d["efficiency"][best_idx]
        best_eq = d["equality"][best_idx]

        # Legend: one entry per condition
        target_lbl = "$\\Phi_U$" if d["target"] == "eff" else "$\\Phi_{\\min}$"
        game_lbl = "C" if d["game"] == "Cleanup" else "G"
        label_str = f"{d['llm']}, {target_lbl} ({game_lbl})"
        lbl = label_str if cond not in legend_handles else None
        if lbl:
            legend_handles[cond] = True

        ax.scatter(best_eff, best_eq, marker=mk, s=60, color=color,
                   edgecolors="white", linewidth=0.5, label=lbl, zorder=3)

    # Reference lines
    ax.axhline(0.9, color="gray", ls=":", lw=0.6, alpha=0.4)
    ax.text(0.3, 0.92, "$E = 0.9$", fontsize=6, color="gray", alpha=0.6)

    ax.set_xlabel("Efficiency ($U$)")
    ax.set_ylabel("Equality ($E$)")
    ax.set_xlim(0, 3.6)
    ax.set_ylim(-0.1, 1.1)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=6.5)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_scatter_eff_eq.png")
    fig.savefig(OUT_DIR / "fig_scatter_eff_eq.svg")
    plt.close(fig)
    print(f"  Saved fig_scatter_eff_eq")


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating paper figures...")
    plot_efficiency_trajectory()
    plot_maximin_trajectory()
    plot_efficiency_equality_bars()
    plot_researcher_behavior()
    plot_combined_cleanup()
    plot_efficiency_equality_scatter()
    print(f"All figures saved to {OUT_DIR}/")
