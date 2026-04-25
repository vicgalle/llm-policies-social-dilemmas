"""
Population manager: multi-objective Pareto frontier over harnesses.

Each harness has a metrics.json (written by tools.py:eval) of the form::

    {
      "id": "h0042",
      "n_seeds": 5,
      "primary": {"efficiency": 9.7, "maximin": 230, "equality": 0.81, ...},
      "cost":    {"synthesis_token_cost": 12000, "inference_token_cost": 0},
      "raw_per_seed": [...]
    }

For Pareto purposes we project to a tuple of *higher-is-better* values:

    (efficiency, maximin, equality, -synthesis_cost, -inference_cost)

The proposer picks the objectives axis-by-axis via the ``objectives`` argument.
``equality`` is omitted by default in single-objective runs (the plan's E2
experiment opts in by passing ``--objectives efficiency,maximin,equality``).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


META_DIR = Path(__file__).resolve().parent
HARNESS_DIR = META_DIR / "harnesses"
PARETO_PATH = META_DIR / "pareto.json"


# ---------------------------------------------------------------------------
# Score extraction
# ---------------------------------------------------------------------------


def _score_tuple(metrics: Dict, objectives: List[str]) -> Tuple[float, ...]:
    """Project a metrics dict to a higher-is-better tuple in objective order.

    Recognised keys (case-sensitive):

      * ``efficiency``, ``maximin``, ``equality``, ``sustainability``, ``peace``
            → read from ``metrics["primary"]``.
      * ``neg_synthesis_token_cost`` / ``neg_inference_token_cost``
            → read from ``metrics["cost"]`` and negated (cost↓ → score↑).
    """
    primary = metrics.get("primary", {}) or {}
    cost = metrics.get("cost", {}) or {}
    out: List[float] = []
    for obj in objectives:
        if obj.startswith("neg_"):
            base = obj[4:]
            out.append(-float(cost.get(base, 0.0)))
        else:
            out.append(float(primary.get(obj, 0.0)))
    return tuple(out)


# ---------------------------------------------------------------------------
# Pareto domination
# ---------------------------------------------------------------------------


def dominates(a: Tuple[float, ...], b: Tuple[float, ...]) -> bool:
    """True iff ``a`` Pareto-dominates ``b``: all coords ≥, at least one >."""
    if len(a) != len(b):
        raise ValueError("score tuples of different arity")
    ge = all(ai >= bi for ai, bi in zip(a, b))
    gt = any(ai > bi for ai, bi in zip(a, b))
    return ge and gt


# ---------------------------------------------------------------------------
# Frontier maintenance
# ---------------------------------------------------------------------------


def load_all_metrics() -> Dict[str, Dict]:
    """Load metrics.json for every harness that has one. Keyed by id."""
    out: Dict[str, Dict] = {}
    for hd in sorted(HARNESS_DIR.glob("h*")):
        mp = hd / "metrics.json"
        if mp.exists():
            try:
                m = json.loads(mp.read_text())
                hid = m.get("id", hd.name)
                out[hid] = m
            except Exception:
                continue
    return out


def pareto_frontier(
    objectives: Optional[List[str]] = None,
    metrics_map: Optional[Dict[str, Dict]] = None,
) -> List[str]:
    """Return harness IDs on the current Pareto frontier in score order.

    Default objectives match pareto.json on disk; pass ``objectives`` to
    override (e.g., the E1/E2 experiment switches).
    """
    if metrics_map is None:
        metrics_map = load_all_metrics()
    if objectives is None:
        objectives = json.loads(PARETO_PATH.read_text()).get(
            "objectives",
            ["efficiency", "maximin", "equality",
             "neg_synthesis_token_cost", "neg_inference_token_cost"],
        )
    scored = {hid: _score_tuple(m, objectives) for hid, m in metrics_map.items()}
    frontier = []
    for hid, s in scored.items():
        dominated = False
        for hid2, s2 in scored.items():
            if hid == hid2:
                continue
            if dominates(s2, s):
                dominated = True
                break
        if not dominated:
            frontier.append(hid)
    # Sort by primary objective desc for stable display.
    frontier.sort(key=lambda h: -scored[h][0])
    return frontier


def update_frontier(
    new_hid: str,
    metrics: Dict,
    objectives: Optional[List[str]] = None,
) -> Dict:
    """Refresh pareto.json after a new harness's metrics are committed.

    Returns ``{dominates, dominated_by, on_frontier, frontier}`` so the
    proposer (and the outer loop) can act on the result without rereading.
    """
    metrics_map = load_all_metrics()
    metrics_map[new_hid] = metrics
    if objectives is None:
        objectives = json.loads(PARETO_PATH.read_text()).get(
            "objectives",
            ["efficiency", "maximin", "equality",
             "neg_synthesis_token_cost", "neg_inference_token_cost"],
        )
    new_score = _score_tuple(metrics, objectives)
    dominated_by: List[str] = []
    dominates_list: List[str] = []
    for hid, m in metrics_map.items():
        if hid == new_hid:
            continue
        s = _score_tuple(m, objectives)
        if dominates(s, new_score):
            dominated_by.append(hid)
        elif dominates(new_score, s):
            dominates_list.append(hid)

    frontier = pareto_frontier(objectives=objectives, metrics_map=metrics_map)
    on_frontier = new_hid in frontier

    PARETO_PATH.write_text(json.dumps({
        "objectives": objectives,
        "frontier": frontier,
        "scores": {hid: list(_score_tuple(m, objectives))
                   for hid, m in metrics_map.items()},
        "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2))

    return {
        "dominates": dominates_list,
        "dominated_by": dominated_by,
        "on_frontier": on_frontier,
        "frontier": frontier,
        "score": list(new_score),
    }


def cull_dominated(keep_n_per_objective: int = 5) -> List[str]:
    """Move strictly dominated harnesses (not on any single-objective top-N)
    to ``archive/`` to keep the listing tractable.

    A harness is kept if it is on the Pareto frontier OR ranked in the top
    ``keep_n_per_objective`` for any single objective.
    """
    metrics_map = load_all_metrics()
    objectives = json.loads(PARETO_PATH.read_text()).get(
        "objectives",
        ["efficiency", "maximin", "equality",
         "neg_synthesis_token_cost", "neg_inference_token_cost"],
    )
    frontier = set(pareto_frontier(objectives=objectives,
                                   metrics_map=metrics_map))

    keep = set(frontier)
    for axis_idx, _obj in enumerate(objectives):
        ranked = sorted(metrics_map.items(),
                        key=lambda kv: -_score_tuple(kv[1], objectives)[axis_idx])
        for hid, _m in ranked[:keep_n_per_objective]:
            keep.add(hid)

    moved: List[str] = []
    archive_dir = META_DIR / "archive"
    archive_dir.mkdir(exist_ok=True)
    for hid in metrics_map:
        if hid in keep:
            continue
        src = HARNESS_DIR / hid
        dst = archive_dir / hid
        if src.exists() and not dst.exists():
            src.rename(dst)
            moved.append(hid)
    return moved
