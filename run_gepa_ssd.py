#!/usr/bin/env python
"""
GEPA baseline for Sequential Social Dilemma experiments.
========================================================

Runs GEPA (Genetic-Pareto prompt Optimization) over the SSD verifier
environments. GEPA iteratively refines the system prompt so that the
LLM generates better policy code for multi-agent self-play.

This script produces the GEPA baseline numbers for the paper.

Model: Gemini 3.1 Pro (same model for generation and reflection).
Settings parallel llm_self_play.py: 10 agents, large maps, 3 iterations,
5 evaluation seeds per policy.

Usage::

    # Both games (gathering + cleanup)
    python run_gepa_ssd.py

    # Single game
    python run_gepa_ssd.py --game gathering

    # Override model or iteration count
    python run_gepa_ssd.py --model gemini-2.5-pro-preview --iterations 5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# ── Ensure SSD directory and local verifiers repo are importable ──────────────
_SSD_DIR = os.path.dirname(os.path.abspath(__file__))
_VERIFIERS_DIR = os.path.join(_SSD_DIR, "verifiers")
# Local verifiers repo takes precedence over any pip-installed version
if _VERIFIERS_DIR not in sys.path:
    sys.path.insert(0, _VERIFIERS_DIR)
if _SSD_DIR not in sys.path:
    sys.path.insert(0, _SSD_DIR)

import numpy as np

from gepa.api import optimize

import verifiers as vf
from verifiers.clients import resolve_client
from verifiers.gepa.adapter import (
    VerifiersGEPAAdapter,
    make_reflection_lm,
    _inject_system_prompt,
)
from verifiers.gepa.display import GEPADisplay
from verifiers.gepa.gepa_utils import save_gepa_results
from verifiers.types import ClientConfig

from ssd_verifier_env import load_environment, _evaluate_policy_code

# ── Defaults (aligned with llm_self_play.py) ─────────────────────────────────

MODEL = "gemini-3.1-pro-preview"
N_AGENTS = 10
N_EVAL_SEEDS = 5
MAX_ITERATIONS = 3       # comparable to llm_self_play --iterations 3

# Google Gemini via OpenAI-compatible endpoint
GEMINI_API_BASE_URL = (
    "https://generativelanguage.googleapis.com/v1beta/openai/"
)
GEMINI_API_KEY_VAR = "GEMINI_API_KEY"


def log(msg: str = ""):
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def _save_results_fallback(run_dir: Path, result, config: dict) -> None:
    """Minimal result saving when save_gepa_results fails due to
    gepa library version mismatches."""
    from datetime import datetime

    best_candidate = getattr(result, "best_candidate", {})
    best_idx = getattr(result, "best_idx", 0)
    val_scores = getattr(result, "val_aggregate_scores", [])
    candidates = getattr(result, "candidates", [])

    # Best prompt
    best_prompt = best_candidate.get("system_prompt", "")
    (run_dir / "best_prompt.txt").write_text(best_prompt)

    # Metadata
    best_score = (
        float(val_scores[best_idx])
        if val_scores and best_idx < len(val_scores)
        else None
    )
    metadata = {
        "num_candidates": len(candidates),
        "best_idx": best_idx,
        "best_score": best_score,
        "total_metric_calls": getattr(result, "total_metric_calls", None),
        "completed_at": datetime.now().isoformat(),
        "config": config,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


# ── Per-configuration runner ──────────────────────────────────────────────────

def run_gepa_for_config(
    game: str,
    model: str,
    n_agents: int,
    n_eval_seeds: int,
    max_iterations: int,
    output_dir: Path,
    api_base_url: str,
    api_key_var: str,
) -> dict:
    """Run GEPA for one game configuration and return a result dict."""

    mode = "code-reward-all"

    log(f"\n{'=' * 60}")
    log(f"  GEPA: {game} / {mode}")
    log(f"  Model: {model}  |  Iterations: {max_iterations}")
    log(f"{'=' * 60}")

    # ── Load environment ──────────────────────────────────────────────────
    env = load_environment(
        game=game, mode=mode, n_agents=n_agents, n_eval_seeds=n_eval_seeds,
    )

    # GEPA budget: 1 initial valset eval + 1 call per reflection iteration,
    # matching llm_self_play's 1 LLM call per iteration.
    max_metric_calls = 1 + max_iterations
    minibatch_size = 1   # single task per env
    num_train = 1
    num_val = 1

    # ── Client configs ────────────────────────────────────────────────────
    client_config = ClientConfig(
        api_key_var=api_key_var,
        api_base_url=api_base_url,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    env_id = f"ssd-{game}-{mode}"

    # ── Display ───────────────────────────────────────────────────────────
    display = GEPADisplay(
        env_id=env_id,
        model=model,
        reflection_model=model,
        max_metric_calls=max_metric_calls,
        num_train=num_train,
        num_val=num_val,
        log_file=output_dir / "gepa.log",
        screen=False,
    )

    with display:
        # Datasets
        trainset = env.get_dataset(n=num_train, seed=0).to_list()
        valset = env.get_eval_dataset(n=num_val, seed=0).to_list()

        valset_ids = [
            item.get("example_id", i) for i, item in enumerate(valset)
        ]
        display.set_valset_info(len(valset), valset_ids)
        display.num_train = len(trainset)
        display.num_val = len(valset)

        # Verifiers client (OpenAI-compatible, talks to Gemini)
        client = resolve_client(client_config)

        # GEPA adapter
        adapter = VerifiersGEPAAdapter(
            env=env,
            client=client,
            model=model,
            sampling_args={},
            max_concurrent=1,          # env eval is CPU-bound
            state_columns=[],
            display=display,
        )

        # Reflection LM (same model)
        reflection_lm = make_reflection_lm(
            client_config=client_config,
            model=model,
        )

        # Seed candidate = environment's existing system prompt
        seed_candidate = {"system_prompt": env.system_prompt or ""}

        # ── Run GEPA ─────────────────────────────────────────────────────
        log(f"  Starting GEPA (budget={max_metric_calls}, "
            f"minibatch={minibatch_size})...")

        t0 = time.time()
        result = optimize(
            seed_candidate=seed_candidate,
            trainset=trainset,
            valset=valset,
            adapter=adapter,
            reflection_lm=reflection_lm,
            max_metric_calls=max_metric_calls,
            reflection_minibatch_size=minibatch_size,
            run_dir=str(output_dir),
            seed=0,
            skip_perfect_score=False,  # SSD rewards are unbounded
            logger=display,
        )
        elapsed = time.time() - t0

        # ── Save ──────────────────────────────────────────────────────────
        run_config = {
            "game": game,
            "mode": mode,
            "model": model,
            "n_agents": n_agents,
            "n_eval_seeds": n_eval_seeds,
            "max_iterations": max_iterations,
            "max_metric_calls": max_metric_calls,
        }
        try:
            save_gepa_results(output_dir, result, config=run_config)
        except (AttributeError, TypeError) as e:
            # gepa lib version may store subscores as lists not dicts;
            # fall back to manual save of the essentials.
            log(f"  Warning: save_gepa_results hit {e}, saving manually")
            _save_results_fallback(output_dir, result, run_config)

        best_prompt = result.best_candidate.get("system_prompt", "")
        display.set_result(best_prompt=best_prompt, save_path=str(output_dir))

        # ── Final evaluation: re-run best candidate to capture metrics ────
        # The GEPA loop only tracks the scalar reward; we need the social
        # metrics (efficiency, equality, sustainability, peace) for the paper.
        log("  Running final evaluation for social metrics...")
        final_inputs = valset[:]
        final_inputs = _inject_system_prompt(final_inputs, best_prompt)
        import asyncio
        final_out = asyncio.get_event_loop().run_until_complete(
            env.generate(
                inputs=final_inputs,
                client=client,
                model=model,
                sampling_args={},
                max_concurrent=1,
            )
        )
        final_metrics = {}
        if final_out["outputs"]:
            final_metrics = final_out["outputs"][0].get("metrics", {})
            best_score = final_out["outputs"][0].get("reward")

    # ── Extract scores from saved metadata (fallback) ─────────────────────
    if best_score is None:
        metadata_path = output_dir / "metadata.json"
        if metadata_path.exists():
            meta = json.loads(metadata_path.read_text())
            best_score = meta.get("best_score")

    num_candidates = len(getattr(result, "candidates", []))

    # Save social metrics alongside GEPA artifacts
    social_path = output_dir / "social_metrics.json"
    social_path.write_text(json.dumps({
        "reward": best_score,
        "metrics": final_metrics,
    }, indent=2))

    log(f"  Done in {elapsed:.0f}s  |  candidates={num_candidates}  "
        f"|  best_score={best_score}")
    if final_metrics:
        for k, v in final_metrics.items():
            log(f"    {k:25s}: {v:.4f}")
    log(f"  Results saved to {output_dir}")

    return {
        "game": game,
        "model": model,
        "best_score": best_score,
        "num_candidates": num_candidates,
        "metrics": final_metrics,
        "elapsed_s": round(elapsed, 1),
        "output_dir": str(output_dir),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GEPA baseline for SSD experiments (paper)",
    )
    parser.add_argument(
        "--game", choices=["gathering", "cleanup"], default=None,
        help="Game to run (default: both)",
    )
    parser.add_argument(
        "--model", default=MODEL,
        help=f"Gemini model name (default: {MODEL})",
    )
    parser.add_argument(
        "--iterations", type=int, default=MAX_ITERATIONS,
        help=f"Max GEPA reflection iterations (default: {MAX_ITERATIONS})",
    )
    parser.add_argument(
        "--n-agents", type=int, default=N_AGENTS,
        help=f"Number of agents (default: {N_AGENTS})",
    )
    parser.add_argument(
        "--n-eval-seeds", type=int, default=N_EVAL_SEEDS,
        help=f"Evaluation seeds per policy (default: {N_EVAL_SEEDS})",
    )
    parser.add_argument(
        "--api-base-url", default=GEMINI_API_BASE_URL,
        help="OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--api-key-var", default=GEMINI_API_KEY_VAR,
        help="Environment variable holding the API key "
             f"(default: {GEMINI_API_KEY_VAR})",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override base output directory",
    )
    args = parser.parse_args()

    # Validate API key is available
    if not os.environ.get(args.api_key_var):
        log(f"ERROR: {args.api_key_var} environment variable not set.")
        log(f"Set it with:  export {args.api_key_var}=your_key_here")
        sys.exit(1)

    # ── Build list of game configurations ───────────────────────────────────
    games = [args.game] if args.game else ["gathering", "cleanup"]

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_dir = Path(args.output_dir or "gepa_results") / f"run_{timestamp}"

    # ── Run each configuration ────────────────────────────────────────────
    all_results = []
    for game in games:
        out = base_dir / f"{game}_code-reward-all"
        result = run_gepa_for_config(
            game=game,
            model=args.model,
            n_agents=args.n_agents,
            n_eval_seeds=args.n_eval_seeds,
            max_iterations=args.iterations,
            output_dir=out,
            api_base_url=args.api_base_url,
            api_key_var=args.api_key_var,
        )
        all_results.append(result)

    # ── Summary ───────────────────────────────────────────────────────────
    log(f"\n{'=' * 70}")
    log("  GEPA Baseline — Results Summary")
    log(f"{'=' * 70}")
    for r in all_results:
        score_str = f"{r['best_score']:.4f}" if r["best_score"] is not None else "n/a"
        log(f"\n  {r['game']} / code-reward-all")
        log(f"    reward:  {score_str}  ({r['num_candidates']} candidates, "
            f"{r['elapsed_s']:.0f}s)")
        m = r.get("metrics", {})
        if m:
            log(f"    U={m.get('ssd_reward', m.get('efficiency_metric', 0)):.4f}  "
                f"E={m.get('equality_metric', 0):.4f}  "
                f"S={m.get('sustainability_metric', 0):.1f}  "
                f"P={m.get('peace_metric', 0):.1f}")
    log(f"\n{'=' * 70}")

    # Save combined results
    combined_path = base_dir / "all_results.json"
    base_dir.mkdir(parents=True, exist_ok=True)
    combined_path.write_text(json.dumps(all_results, indent=2))
    log(f"  All results: {combined_path}\n")


if __name__ == "__main__":
    main()
