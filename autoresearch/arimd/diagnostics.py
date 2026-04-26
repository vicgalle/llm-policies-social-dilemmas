"""Stackelberg diagnostics — invasion barrier, exploitability, stability gap.

All three are stated in arimd_plan.md §2.5 and reused verbatim from
the minimax notes §3.1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .evaluator import evaluate_mixed, MixedEvalResult


# ---------------------------------------------------------------------------
# Invasion barrier
# ---------------------------------------------------------------------------


def invasion_barrier(
    env_factory: Callable,
    blue_fn: Callable,
    red_fn: Optional[Callable],
    seeds: Sequence[int],
    *,
    lambdas: Sequence[float] = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5),
    welfare: str = "blue",
    n_assignments: int = 1,
) -> Tuple[float, List[Dict[str, float]]]:
    """Return (λ*, table).

    λ* = largest λ in ``lambdas`` at which V_λ ≥ V_0.  table is a list of
    {"lambda": λ, "v_blue": ..., "v_red": ..., "v_pop": ...} for plotting.
    """
    table: List[Dict[str, float]] = []
    base_v: Optional[float] = None
    star = 0.0

    for lam in sorted(lambdas):
        res = evaluate_mixed(
            env_factory, blue_fn, red_fn if lam > 0 else None,
            lambda_=lam, seeds=list(seeds),
            n_assignments=n_assignments, welfare=welfare,
        )
        v = res.primary(welfare)
        row = {
            "lambda": float(lam),
            "v_blue": res.v_blue,
            "v_red": res.v_red,
            "v_pop": res.v_pop,
            "n_red_per_ep": res.n_red_per_ep,
        }
        table.append(row)
        if lam == 0.0 or base_v is None:
            base_v = v
        if base_v is not None and v >= base_v:
            star = float(lam)

    return star, table


# ---------------------------------------------------------------------------
# Exploitability — drop in V_λ from a *fresh* Red search round
# ---------------------------------------------------------------------------


@dataclass
class ExploitabilityResult:
    v_before: float
    v_after: float
    drop: float          # v_before - v_after, positive = unstable
    seeds: int


def exploitability(
    v_before: float, v_after: float, n_seeds: int,
) -> ExploitabilityResult:
    """Compute the exploitability gap given a *fresh* Red eval at the same env.

    Caller is responsible for running an extra Red search round and
    re-evaluating; this function is the bookkeeping wrapper.
    """
    return ExploitabilityResult(
        v_before=float(v_before),
        v_after=float(v_after),
        drop=float(v_before - v_after),
        seeds=int(n_seeds),
    )


# ---------------------------------------------------------------------------
# Stability gap across Stackelberg rounds
# ---------------------------------------------------------------------------


def stability_gap(history: List[Dict]) -> List[float]:
    """Return |V_t - V_{t-1}| across accepted rounds.

    Only consecutive accepted rounds are compared (skipping rejected
    proposals — those didn't advance the env state).
    """
    accepted = [h for h in history if h.get("accepted", True)]
    gaps: List[float] = []
    for i in range(1, len(accepted)):
        v_prev = accepted[i - 1].get("v_blue", 0.0)
        v_cur = accepted[i].get("v_blue", 0.0)
        gaps.append(abs(v_cur - v_prev))
    return gaps


# ---------------------------------------------------------------------------
# Degeneracy detection — flag suspicious Red collapse modes
# ---------------------------------------------------------------------------


def detect_degenerate_red(
    res: MixedEvalResult,
    *,
    min_v_red_for_action: float = 0.0,
    max_raid_attempts_per_ep: float = 0.5,
) -> Optional[str]:
    """Return a string describing a degeneracy mode, or None if Red looks fine.

    Heuristics (intentionally cheap; flag for human review, don't auto-correct):
    - Red sits at zero or negative reward → likely a NOOP-like collapse.
    - Red attempts no raids in raid-capable envs → flag if this seems off.
    """
    if res.v_red <= min_v_red_for_action and res.v_blue > 0:
        return "Red looks idle: v_red ≤ 0 while Blue is positive — possible NOOP collapse."
    if res.raid_attempts < max_raid_attempts_per_ep and res.v_red < res.v_blue:
        # Only notable for envs with a raid mechanic — caller may filter.
        return ("Red made very few raid attempts and earns less than Blue — "
                "may be missing the dominant exploit.")
    return None
