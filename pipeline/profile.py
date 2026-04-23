"""
Profile-Guided Autoresearch (PGA) — execution profile extractor.

This module is FIXED INFRASTRUCTURE. The researcher agent observes its
outputs but MUST NOT edit this file (the scientific claim of the
extractor is that it is environment-agnostic and stable across runs).

A :class:`Profile` packages five complementary channels derived from an
inner-loop trajectory:

1. ``action_histogram_over_buckets`` — what actions dominate each third of
   the horizon, plus which actions are never used at all.
2. ``change_point_detection`` — per-agent reward-stream phase transitions
   (sliding-window-mean heuristic). Distinguishes stationary streams from
   ones with detectable regime changes.
3. ``inter_agent_divergence`` — mean pairwise total-variation distance
   between per-agent action distributions, plus the fraction of behavioural
   variance explained by ``agent_id`` alone (the "brittle static role"
   signature).
4. ``ast_branch_coverage`` — percentage of the synthesized policy's
   executable lines that actually fired, and a short list of dead /
   coldest branches with their source.
5. ``precondition_failure_rates`` — per-action "ineffective-action"
   fraction (action invoked but no agent-visible state change), plus
   inventory-full frame fraction (environment-generic cap-saturation).

All channels are computed from a minimal trajectory bundle produced by
:func:`gathering_policy.run_episode` when called with
``capture_trajectory=True`` (and ``trace_line_hits`` for (4)).  No
environment-specific code is required; the extractor peers at
``env.inventory`` / ``env.has_tool`` only opportunistically.

Serialization (``serialize_profile``) produces a compact, prompt-friendly
Markdown block that the feedback builder (pipeline/feedback.py) embeds in
the policy LLM's iteration prompt — and that the inner-loop orchestrator
writes to ``history[i]["profile"]`` so the *outer-loop researcher* sees it
too.
"""
from __future__ import annotations

import ast
import linecache
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Profile:
    """Structured execution profile P (see pga_improvement_plan.md §PGA)."""

    # --- Action histograms over time-buckets --------------------------
    # One entry per bucket. Each entry: {"top": [(name, frac), ...],
    # "total": int}. Dominant actions per phase of the episode.
    action_buckets: List[dict] = field(default_factory=list)
    # Actions never invoked by any agent across the whole episode.
    actions_never_used: List[str] = field(default_factory=list)

    # --- Reward change-point detection --------------------------------
    # Mean number of detected change points per agent (0 = stationary
    # stream; 1+ = at least one regime change). Change points are step indices.
    change_points_per_agent_mean: float = 0.0
    # Representative example: change points for agent 0 (or the median
    # agent if all are similar).
    change_points_example: List[int] = field(default_factory=list)
    # Reward "phase" summary: [(start, end, rate_per_step), ...]
    reward_phases_example: List[dict] = field(default_factory=list)

    # --- Inter-agent divergence --------------------------------------
    # Mean pairwise total-variation distance between agents' action
    # distributions. 0 = perfectly homogeneous; high = differentiated.
    action_tv_mean: float = 0.0
    # Fraction of behavioural variance (per-action chi-square) explained
    # by agent_id. High → brittle static assignment.
    role_id_rsquared: float = 0.0

    # --- Branch coverage ---------------------------------------------
    # Percentage (0–100) of the policy's executable statements that fired.
    branch_coverage_pct: float = 100.0
    policy_n_executable_lines: int = 0
    # Dead branches with source: lines that are executable but never hit.
    dead_branches: List[dict] = field(default_factory=list)   # [{line, source}]
    # The N coldest live branches (smallest non-zero hit counts) — a
    # researcher signal that a case in the policy is rarely triggered.
    coldest_branches: List[dict] = field(default_factory=list)

    # --- Precondition-failure & cap saturation -----------------------
    # Per-action-name, fraction of invocations that did not change any
    # agent-visible state (pos/inventory/has_tool/reward). High values
    # flag unmet preconditions (e.g. CRAFT without inputs).
    action_ineffective_rate: Dict[str, float] = field(default_factory=dict)
    # Fraction of agent-frames where inventory is at cap (PE-specific but
    # computed environment-agnostically when env has .inventory).
    inventory_full_frame_rate: float = 0.0

    # --- Raw meta ----------------------------------------------------
    n_episodes: int = 0
    horizon: int = 0
    n_agents: int = 0


# ---------------------------------------------------------------------------
# Channel 1: action histograms over time-buckets
# ---------------------------------------------------------------------------


def action_histogram_over_buckets(
    action_history: Dict[int, List[int]],
    horizon: int,
    action_names: List[str],
    n_buckets: int = 3,
    top_k: int = 4,
) -> Tuple[List[dict], List[str]]:
    """Split the horizon into ``n_buckets`` equal thirds and report the
    ``top_k`` most-frequent actions per bucket, plus actions never used
    across the whole episode.
    """
    num_actions = len(action_names)
    bucket_edges = [int(round(horizon * k / n_buckets)) for k in range(n_buckets + 1)]
    buckets: List[dict] = []
    total_counts = Counter()
    for b in range(n_buckets):
        lo, hi = bucket_edges[b], bucket_edges[b + 1]
        counts = Counter()
        for aid, acts in action_history.items():
            for a in acts[lo:hi]:
                counts[int(a)] += 1
                total_counts[int(a)] += 1
        total = sum(counts.values()) or 1
        top = counts.most_common(top_k)
        entry = {
            "bucket": b,
            "steps": [lo, hi],
            "total": total,
            "top": [
                (action_names[a] if 0 <= a < num_actions else str(a),
                 round(c / total, 3))
                for a, c in top
            ],
        }
        buckets.append(entry)

    never = [action_names[a] for a in range(num_actions)
             if total_counts.get(a, 0) == 0]
    return buckets, never


# ---------------------------------------------------------------------------
# Channel 2: change-point detection on per-agent reward timelines
# ---------------------------------------------------------------------------


def _simple_change_points(series: List[float], min_window: int = 20,
                          z_threshold: float = 2.5) -> List[int]:
    """Windowed-variance heuristic change-point detector.

    Splits into windows of ``min_window`` steps, compares each window's
    mean to the running mean; flags windows where the z-score exceeds
    ``z_threshold``. A cheap but surprisingly effective detector for
    this use-case — the exact choice matters less than having *a*
    detector (see pga_improvement_plan.md §Open Questions).
    """
    arr = np.asarray(series, dtype=float)
    n = len(arr)
    if n < 2 * min_window:
        return []
    changes: List[int] = []
    running_mean = arr[:min_window].mean()
    running_sd = arr[:min_window].std() + 1e-9
    i = min_window
    while i + min_window <= n:
        w = arr[i:i + min_window]
        z = (w.mean() - running_mean) / running_sd
        if abs(z) >= z_threshold:
            changes.append(int(i + min_window // 2))
            running_mean = w.mean()
            running_sd = w.std() + 1e-9
        else:
            # Smoothly update the reference.
            running_mean = 0.8 * running_mean + 0.2 * w.mean()
            running_sd = 0.8 * running_sd + 0.2 * (w.std() + 1e-9)
        i += min_window
    return changes


def change_point_detection(
    ep_rewards: Dict[int, List[float]],
    horizon: int,
) -> Tuple[float, List[int], List[dict]]:
    """Detect reward-stream phase transitions. Returns ``(mean_cp_count,
    representative_cp_list, phase_summary)``.
    """
    cp_lists: Dict[int, List[int]] = {}
    for aid, rews in ep_rewards.items():
        cp_lists[aid] = _simple_change_points(rews)
    mean_count = (float(np.mean([len(v) for v in cp_lists.values()]))
                  if cp_lists else 0.0)

    # Representative: agent with the median number of change points.
    if cp_lists:
        rep_aid = sorted(cp_lists, key=lambda k: len(cp_lists[k]))[len(cp_lists) // 2]
        rep_cps = cp_lists[rep_aid]
    else:
        rep_aid = 0
        rep_cps = []

    # Phase summary on representative agent: split at change points, compute
    # reward rate per phase.
    rep_rewards = ep_rewards.get(rep_aid, [])
    phases: List[dict] = []
    boundaries = [0] + rep_cps + [len(rep_rewards)]
    for s, e in zip(boundaries[:-1], boundaries[1:]):
        if e <= s:
            continue
        chunk = rep_rewards[s:e]
        phases.append({
            "start": int(s),
            "end": int(e),
            "rate": round(float(sum(chunk) / max(len(chunk), 1)), 4),
        })
    return mean_count, rep_cps, phases


# ---------------------------------------------------------------------------
# Channel 3: inter-agent divergence
# ---------------------------------------------------------------------------


def inter_agent_divergence(
    action_history: Dict[int, List[int]],
    num_actions: int,
) -> Tuple[float, float]:
    """Return (mean_pairwise_TV, role_id_rsquared).

    * mean_pairwise_TV: average total-variation distance between all
      pairs of per-agent action distributions. 0 = homogeneous.
    * role_id_rsquared: fraction of the total action-count variance
      explained by agent_id. Computed as the sum-of-squares of
      per-agent-deviation-from-mean divided by the total sum-of-squares
      over all (agent, action) cells. High → behaviour is mostly
      predicted by agent_id (brittle static assignment).
    """
    agents = list(action_history.keys())
    if len(agents) < 2:
        return 0.0, 0.0
    dists = []
    for aid in agents:
        cnt = np.zeros(num_actions, dtype=float)
        for a in action_history[aid]:
            if 0 <= int(a) < num_actions:
                cnt[int(a)] += 1.0
        s = cnt.sum()
        dists.append(cnt / s if s > 0 else cnt)
    dists = np.stack(dists, axis=0)  # (n_agents, n_actions)

    # Mean pairwise TV
    tvs = []
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            tvs.append(0.5 * float(np.abs(dists[i] - dists[j]).sum()))
    mean_tv = float(np.mean(tvs)) if tvs else 0.0

    # Role-id R^2: ratio of between-agent SS to total SS on action counts.
    counts = np.zeros((len(agents), num_actions), dtype=float)
    for idx, aid in enumerate(agents):
        for a in action_history[aid]:
            if 0 <= int(a) < num_actions:
                counts[idx, int(a)] += 1.0
    grand_mean = counts.mean()
    total_ss = float(((counts - grand_mean) ** 2).sum())
    agent_means = counts.mean(axis=1, keepdims=True)
    between_ss = float(((agent_means - grand_mean) ** 2).sum() * num_actions)
    if total_ss <= 1e-9:
        r2 = 0.0
    else:
        r2 = min(1.0, max(0.0, between_ss / total_ss))
    return mean_tv, r2


# ---------------------------------------------------------------------------
# Channel 4: AST-branch coverage
# ---------------------------------------------------------------------------


def _executable_lines_from_ast(code: str, offset: int = 0) -> List[int]:
    """Return the list of line numbers that are executable statements in
    ``code``. Skips ``pass``, ``Ellipsis`` expressions, module-level
    docstrings, and nested function/class def lines themselves (the def
    line is not really "executed" in the same sense).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    lines: set = set()

    # Lines that are "executable statements" for coverage purposes: we
    # include branches, returns, calls, assignments, etc. but NOT the
    # ``def``/``class`` header line itself (which runs only once, during
    # ``exec``, not during tracing of policy calls). Also skip ``pass``,
    # ``Ellipsis``, and docstring expressions.
    skip_headers = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)

    class Visitor(ast.NodeVisitor):
        def visit(self, node):
            if isinstance(node, ast.stmt):
                if isinstance(node, skip_headers):
                    pass  # visit body but don't add the def/class line itself
                elif isinstance(node, ast.Pass):
                    pass
                elif (isinstance(node, ast.Expr)
                      and isinstance(node.value, ast.Constant)
                      and isinstance(node.value.value, (str, type(Ellipsis)))):
                    # docstring / ellipsis
                    pass
                else:
                    lines.add(node.lineno)
            self.generic_visit(node)

    Visitor().visit(tree)
    return sorted(lines)


def ast_branch_coverage(
    code: str,
    filename: str,
    line_hits: Dict[tuple, int],
    n_dead_to_show: int = 5,
    n_coldest_to_show: int = 3,
) -> Tuple[float, int, List[dict], List[dict]]:
    """Compute branch-coverage metrics for a single policy.

    ``line_hits`` maps (co_filename, lineno) → count (accumulated by the
    tracer in :func:`gathering_policy.run_episode`).

    Returns ``(coverage_pct, n_exec_lines, dead_branches, coldest_branches)``.
    """
    exec_lines = _executable_lines_from_ast(code)
    if not exec_lines:
        return 100.0, 0, [], []

    per_line_hits: Dict[int, int] = defaultdict(int)
    for (fn, ln), cnt in line_hits.items():
        if fn == filename:
            per_line_hits[int(ln)] += int(cnt)

    live = [ln for ln in exec_lines if per_line_hits.get(ln, 0) > 0]
    dead = [ln for ln in exec_lines if per_line_hits.get(ln, 0) == 0]
    coverage_pct = round(100.0 * len(live) / len(exec_lines), 2)

    def _src(lineno: int) -> str:
        line = linecache.getline(filename, lineno)
        if not line:
            # Fallback: split code directly.
            try:
                line = code.splitlines()[lineno - 1]
            except IndexError:
                line = ""
        return line.rstrip()

    dead_entries = [{"line": ln, "source": _src(ln)}
                    for ln in dead[:n_dead_to_show]]

    live_sorted = sorted(live, key=lambda ln: per_line_hits.get(ln, 0))
    coldest_entries = [
        {"line": ln, "hits": int(per_line_hits.get(ln, 0)),
         "source": _src(ln)}
        for ln in live_sorted[:n_coldest_to_show]
    ]
    return coverage_pct, len(exec_lines), dead_entries, coldest_entries


# ---------------------------------------------------------------------------
# Channel 5: precondition-failure rates
# ---------------------------------------------------------------------------


# Action names per environment are exposed via GameConfig.extra_namespace;
# we translate them into a ``{int: name}`` lookup (see ACTION_NAMES_BY_GAME
# below for the defaults).

ACTION_NAMES_BY_GAME: Dict[str, List[str]] = {
    "cleanup": [
        "FORWARD", "BACKWARD", "STEP_LEFT", "STEP_RIGHT",
        "ROTATE_LEFT", "ROTATE_RIGHT", "BEAM", "STAND", "CLEAN",
    ],
    "gathering": [
        "FORWARD", "BACKWARD", "STEP_LEFT", "STEP_RIGHT",
        "ROTATE_LEFT", "ROTATE_RIGHT", "BEAM", "STAND",
    ],
    "coop_mining": [
        "FORWARD", "BACKWARD", "STEP_LEFT", "STEP_RIGHT",
        "ROTATE_LEFT", "ROTATE_RIGHT", "MINE", "STAND",
    ],
    "production_economy": [
        "NOOP", "MOVE_N", "MOVE_S", "MOVE_E", "MOVE_W",
        "GATHER", "CRAFT", "CRAFT_TOOL", "CRAFT_SHELTER",
        "DROP_WOOD", "DROP_STONE", "DROP_PLANK", "DROP_BRICK",
        "PICKUP_WOOD", "PICKUP_STONE", "PICKUP_PLANK", "PICKUP_BRICK",
    ],
}

# Actions that we do NOT score for ineffective-action rate because their
# "no state change" is expected / intentional (stand still).
_INERT_ACTIONS = {"NOOP", "STAND"}


def precondition_failure_rates(
    action_history: Dict[int, List[int]],
    state_changed: Dict[int, List[bool]],
    action_names: List[str],
    top_k: int = 8,
) -> Dict[str, float]:
    """For each action, return the fraction of invocations that did NOT
    change the acting agent's visible state. Reported only for actions
    with at least ``min_count`` invocations.
    """
    counts = Counter()
    no_effect = Counter()
    for aid, acts in action_history.items():
        ch = state_changed[aid]
        for t, a in enumerate(acts):
            a = int(a)
            if 0 <= a < len(action_names):
                name = action_names[a]
            else:
                name = str(a)
            counts[name] += 1
            if not ch[t]:
                no_effect[name] += 1
    rates: Dict[str, float] = {}
    for name, c in counts.items():
        if name in _INERT_ACTIONS:
            continue
        if c < 5:  # too few samples to be informative
            continue
        rates[name] = round(no_effect[name] / c, 3)
    # Keep only the top_k by count (or by rate if you prefer); we pick the
    # top-k MOST INEFFECTIVE actions (worst offenders first) so the channel
    # surfaces unmet preconditions.
    ranked = sorted(rates.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return dict(ranked)


# ---------------------------------------------------------------------------
# Top-level extractor
# ---------------------------------------------------------------------------


def extract_profile(
    trajectory: dict,
    code: str,
    filename: str | None,
    game: str,
    line_hits: Optional[Dict[tuple, int]] = None,
) -> Profile:
    """Compose all five channels into a single :class:`Profile`.

    Parameters
    ----------
    trajectory
        The dict returned by ``run_episode(..., capture_trajectory=True)``.
    code
        Source of the synthesized policy (for branch coverage).
    filename
        Virtual filename used during ``compile(code, filename, 'exec')``
        (see ``load_policy(..., tag=...)``).
    game
        Game name; used to pick action names.
    line_hits
        Optional dict populated by the tracer in
        :func:`gathering_policy.run_episode`.
    """
    action_names = ACTION_NAMES_BY_GAME.get(game, [])
    horizon = int(trajectory.get("horizon", 0))
    n_agents = int(trajectory.get("n_agents", 0))

    action_history = trajectory.get("action_history") or {}
    state_changed = trajectory.get("state_changed") or {}
    ep_rewards = trajectory.get("ep_rewards") or {}
    inv_full = trajectory.get("inventory_full_frames") or {}

    # --- Channel 1 ---
    buckets, never = action_histogram_over_buckets(
        action_history, horizon, action_names,
    )

    # --- Channel 2 ---
    cp_mean, cp_example, phase_example = change_point_detection(
        ep_rewards, horizon,
    )

    # --- Channel 3 ---
    tv_mean, role_r2 = inter_agent_divergence(
        action_history, num_actions=max(len(action_names), 1),
    )

    # --- Channel 4 ---
    if line_hits is not None and filename is not None:
        coverage_pct, n_exec, dead_entries, coldest_entries = ast_branch_coverage(
            code, filename, line_hits,
        )
    else:
        coverage_pct, n_exec, dead_entries, coldest_entries = 100.0, 0, [], []

    # --- Channel 5 ---
    ineffective = precondition_failure_rates(
        action_history, state_changed, action_names,
    )
    if inv_full and horizon > 0 and n_agents > 0:
        inv_full_rate = round(
            sum(inv_full.values()) / (horizon * n_agents), 3,
        )
    else:
        inv_full_rate = 0.0

    return Profile(
        action_buckets=buckets,
        actions_never_used=never,
        change_points_per_agent_mean=round(cp_mean, 2),
        change_points_example=cp_example,
        reward_phases_example=phase_example,
        action_tv_mean=round(tv_mean, 3),
        role_id_rsquared=round(role_r2, 3),
        branch_coverage_pct=coverage_pct,
        policy_n_executable_lines=n_exec,
        dead_branches=dead_entries,
        coldest_branches=coldest_entries,
        action_ineffective_rate=ineffective,
        inventory_full_frame_rate=inv_full_rate,
        n_episodes=1,
        horizon=horizon,
        n_agents=n_agents,
    )


# ---------------------------------------------------------------------------
# Serialization (for prompt injection and JSON logging)
# ---------------------------------------------------------------------------


def serialize_profile(p: Profile, include_source: bool = True) -> str:
    """Render a :class:`Profile` as a short Markdown block for prompts."""
    lines: List[str] = []
    lines.append("### Execution profile (PGA)")
    lines.append(
        f"- Horizon {p.horizon}, {p.n_agents} agents, {p.n_episodes} trace episode(s)."
    )

    # Channel 1
    lines.append("")
    lines.append("**Action histogram by phase** "
                 "(fraction of actions taken in each third of the episode):")
    for b in p.action_buckets:
        top = ", ".join(f"{n}={f}" for n, f in b["top"])
        lines.append(
            f"- steps {b['steps'][0]}-{b['steps'][1]}: {top}"
        )
    if p.actions_never_used:
        lines.append(
            f"- Actions **never invoked by any agent**: "
            f"{', '.join(p.actions_never_used)}"
        )
    else:
        lines.append("- Every action was invoked at least once.")

    # Channel 2
    lines.append("")
    lines.append("**Reward-stream change points** (windowed-mean z-score detector):")
    lines.append(
        f"- Mean change-points per agent: {p.change_points_per_agent_mean}. "
        f"Representative agent change-points: {p.change_points_example}."
    )
    if p.reward_phases_example:
        phases = ", ".join(
            f"[{ph['start']}..{ph['end']}] rate={ph['rate']}"
            for ph in p.reward_phases_example
        )
        lines.append(f"- Reward phases on representative agent: {phases}.")

    # Channel 3
    lines.append("")
    lines.append("**Inter-agent divergence** (self-play specialization check):")
    lines.append(
        f"- Mean pairwise TV(action) = {p.action_tv_mean}. "
        f"Variance explained by agent_id (role-R²) = {p.role_id_rsquared}."
    )
    if p.action_tv_mean < 0.05:
        lines.append(
            "- Interpretation: agents are behaving nearly identically; no role "
            "specialization is emerging."
        )
    elif p.role_id_rsquared > 0.5:
        lines.append(
            "- Interpretation: behaviour is largely predicted by `agent_id` — "
            "likely a brittle static assignment rather than state-conditional "
            "specialization."
        )
    else:
        lines.append(
            "- Interpretation: agents diverge but not by `agent_id`, consistent "
            "with state-conditional role differentiation."
        )

    # Channel 4
    lines.append("")
    lines.append(
        f"**Branch coverage**: {p.branch_coverage_pct}% of "
        f"{p.policy_n_executable_lines} executable lines fired."
    )
    if p.dead_branches:
        lines.append("- **Dead branches** (never executed — silent bugs or unreached cases):")
        for d in p.dead_branches:
            src = d.get("source", "")
            lines.append(f"  - line {d['line']}: `{src.strip()}`")
    if p.coldest_branches:
        lines.append("- **Coldest live branches** (lowest hit counts — rare cases):")
        for c in p.coldest_branches:
            src = c.get("source", "")
            lines.append(
                f"  - line {c['line']} (hits={c['hits']}): `{src.strip()}`"
            )

    # Channel 5
    lines.append("")
    lines.append("**Precondition-failure / ineffective-action rates** "
                 "(action invoked but no agent-visible state change):")
    if not p.action_ineffective_rate:
        lines.append("- All invoked non-inert actions changed agent state at least sometimes.")
    else:
        for name, rate in p.action_ineffective_rate.items():
            lines.append(f"- `{name}`: {int(rate * 100)}% of invocations were no-ops.")
    if p.inventory_full_frame_rate > 0:
        lines.append(
            f"- Inventory-at-cap frame fraction: {p.inventory_full_frame_rate}. "
            "High values indicate the cap is a binding constraint on throughput."
        )
    return "\n".join(lines)


def profile_to_jsonable(p: Profile) -> dict:
    """Return a JSON-serialisable dict for logging (measure.sh / analyze.py)."""
    return {
        "action_buckets": p.action_buckets,
        "actions_never_used": p.actions_never_used,
        "change_points_per_agent_mean": p.change_points_per_agent_mean,
        "change_points_example": p.change_points_example,
        "reward_phases_example": p.reward_phases_example,
        "action_tv_mean": p.action_tv_mean,
        "role_id_rsquared": p.role_id_rsquared,
        "branch_coverage_pct": p.branch_coverage_pct,
        "policy_n_executable_lines": p.policy_n_executable_lines,
        "dead_branches": p.dead_branches,
        "coldest_branches": p.coldest_branches,
        "action_ineffective_rate": p.action_ineffective_rate,
        "inventory_full_frame_rate": p.inventory_full_frame_rate,
        "n_episodes": p.n_episodes,
        "horizon": p.horizon,
        "n_agents": p.n_agents,
    }
