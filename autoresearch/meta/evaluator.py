"""
Adaptive evaluator — staged seed allocation for harness eval.

Stage 1 (n=2 seeds): cheap exploratory check. If the candidate is
clearly dominated by every existing frontier point, stop.

Stage 2 (n=5 seeds): standard self-play eval block. Computes
metrics + cost. Most candidates terminate here.

Stage 3 (n=20 seeds, with traces): only invoked when the candidate is
on or near the current Pareto frontier. Records per-seed JSONL traces
so the proposer (and later analyses) can read a high-fidelity diagnostic
log of the *committed* harness.

Returns a metrics dict in the format population.update_frontier expects.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from pipeline.harness import (
    EpisodeResult,
    Harness,
    action_names_for,
    load_harness,
    make_env,
    run_meta_episode,
)
from pipeline.trace import TraceRecorder, aggregate_seed_summaries
from autoresearch.meta import population

META_DIR = Path(__file__).resolve().parent
HARNESS_DIR = META_DIR / "harnesses"


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def _aggregate_episode_metrics(results: List[EpisodeResult]) -> Dict[str, float]:
    keys = list(results[0].metrics.keys())
    out = {}
    for k in keys:
        vals = [r.metrics[k] for r in results]
        out[k] = float(np.mean(vals))
        out[f"{k}_std"] = float(np.std(vals))
    out["agent_total_mean"] = float(np.mean([
        np.mean(list(r.total_rewards.values())) for r in results
    ]))
    return out


def _is_clearly_dominated(
    candidate_score: Dict[str, float],
    frontier_metrics: List[Dict],
    objectives: List[str],
    margin: float = 0.05,
) -> bool:
    """Stage-1 early-stop heuristic.

    A candidate is *clearly* dominated when every frontier point beats it
    on the primary objective by ≥ margin · |frontier_value|. We use a
    proportional margin so the heuristic adapts to the metric's scale.
    Variance-aware (std-based) gates would be more principled but the
    plan calls for a simple staged allocation; this is the simple version.
    """
    if not frontier_metrics:
        return False
    primary = objectives[0]
    cand_val = candidate_score.get(primary, 0.0)
    margin_threshold = cand_val + margin * max(abs(cand_val), 1.0)
    for fm in frontier_metrics:
        fp = fm.get("primary", {}).get(primary, 0.0)
        if fp <= margin_threshold:
            return False  # at least one frontier point isn't safely above us
    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def adaptive_eval(
    hid: str,
    *,
    objectives: Optional[List[str]] = None,
    stage1_seeds: int = 2,
    stage2_seeds: int = 5,
    stage3_seeds: int = 20,
    record_traces: bool = True,
    force_stage: Optional[int] = None,
    log_fn=print,
) -> Dict:
    """Evaluate a harness with staged seed allocation.

    Parameters
    ----------
    hid : str
        Harness identifier (must exist under autoresearch/meta/harnesses/).
    objectives : list[str] | None
        Pareto axes; first one is the primary used for early-stop.
    force_stage : int | None
        If 1/2/3, skip the early-stop logic and run that many seeds
        unconditionally. Useful for re-eval or audit.
    record_traces : bool
        If True, stage 3 records per-seed JSONL + aggregated summary.

    Returns the persisted metrics dict (also written to
    ``harnesses/<hid>/metrics.json``).
    """
    harness_dir = HARNESS_DIR / hid
    if not harness_dir.exists():
        raise FileNotFoundError(harness_dir)
    harness = load_harness(harness_dir)
    loaded = harness.load()
    env_factory = make_env(harness.manifest)
    game = harness.manifest.get("env", "production_economy")
    action_names = action_names_for(game)

    if objectives is None:
        objectives = json.loads((META_DIR / "pareto.json").read_text()).get(
            "objectives",
            ["efficiency", "maximin", "equality",
             "neg_synthesis_token_cost", "neg_inference_token_cost"],
        )

    log_fn(f"[adaptive_eval] {hid} mode={harness.mode} game={game} "
           f"obj={objectives[0]}")

    # ----- Stage 1: 2 seeds (skipped if force_stage>=2 or if no frontier yet)
    metrics_map = population.load_all_metrics()
    frontier_ids = population.pareto_frontier(
        objectives=objectives, metrics_map=metrics_map)
    frontier_metrics = [metrics_map[h] for h in frontier_ids if h != hid]

    seed_results: List[EpisodeResult] = []
    seed_summaries: List[Dict] = []
    n_done = 0
    target_total = 0
    stage_reached = 0

    def _run_seeds(start: int, stop: int, *, with_traces: bool):
        nonlocal n_done
        for s in range(start, stop):
            recorder = None
            if with_traces:
                trace_dir = harness_dir / "traces"
                trace_dir.mkdir(parents=True, exist_ok=True)
                recorder = TraceRecorder(
                    trace_dir / f"seed_{s:02d}.jsonl",
                    action_names=action_names,
                    env_meta={"game": game, "harness": hid, "seed": s,
                              "mode": harness.mode},
                )
            res = run_meta_episode(env_factory, loaded.fn, seed=s,
                                   recorder=recorder)
            seed_results.append(res)
            if recorder is not None:
                seed_summaries.append(recorder.finalize())
            n_done += 1
            log_fn(f"  seed {s}: efficiency="
                   f"{res.metrics.get('efficiency', 0):.3f}  "
                   f"maximin={res.metrics.get('maximin', 0):.2f}")

    primary = objectives[0]

    # Stage 1 (always runs).
    _run_seeds(0, stage1_seeds, with_traces=False)
    stage_reached = 1

    if force_stage is not None:
        # Forced cap: run cumulatively up to the requested stage's seed
        # count, but never beyond. No adaptive early-stop.
        if force_stage >= 2:
            _run_seeds(stage1_seeds, stage2_seeds, with_traces=False)
            stage_reached = 2
        if force_stage >= 3:
            _run_seeds(stage2_seeds, stage3_seeds,
                       with_traces=record_traces)
            stage_reached = 3
    else:
        # Adaptive path. Stage-1 stop if clearly dominated.
        primary_val = float(np.mean([r.metrics.get(primary, 0.0)
                                     for r in seed_results]))
        candidate_score = {primary: primary_val}
        if _is_clearly_dominated(candidate_score, frontier_metrics, objectives):
            log_fn(f"  stage 1 stop: clearly dominated by frontier "
                   f"({primary}={primary_val:.3f} vs frontier).")
        else:
            _run_seeds(stage1_seeds, stage2_seeds, with_traces=False)
            stage_reached = 2
            primary_val = float(np.mean([r.metrics.get(primary, 0.0)
                                         for r in seed_results]))
            on_or_near_frontier = (not frontier_ids) or any(
                primary_val >= fm.get("primary", {}).get(primary, 0.0) - 0.5
                for fm in frontier_metrics
            )
            if on_or_near_frontier:
                _run_seeds(stage2_seeds, stage3_seeds,
                           with_traces=record_traces)
                stage_reached = 3

    # ----- Aggregate -----
    primary_metrics = _aggregate_episode_metrics(seed_results)
    cost = {
        "synthesis_token_cost": float(loaded.synthesis_token_cost),
        "inference_token_cost": float(loaded.inference_token_cost),
    }

    metrics_record = {
        "id": hid,
        "mode": harness.mode,
        "env": game,
        "n_seeds": n_done,
        "stage_reached": stage_reached,
        "primary": primary_metrics,
        "cost": cost,
        "raw_per_seed": [
            {"seed": r.seed,
             "metrics": {k: float(v) for k, v in r.metrics.items()},
             "agent_totals": {int(k): float(v)
                              for k, v in r.total_rewards.items()},
             "wall_time_s": r.wall_time_s,
             "error": r.error}
            for r in seed_results
        ],
        "wall_time_s_total": round(sum(r.wall_time_s for r in seed_results), 2),
        "evaluated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (harness_dir / "metrics.json").write_text(
        json.dumps(metrics_record, indent=2))

    if seed_summaries:
        (harness_dir / "traces" / "summary.json").write_text(
            json.dumps(aggregate_seed_summaries(seed_summaries), indent=2))

    update = population.update_frontier(hid, metrics_record,
                                        objectives=objectives)
    metrics_record["pareto_update"] = update
    return metrics_record
