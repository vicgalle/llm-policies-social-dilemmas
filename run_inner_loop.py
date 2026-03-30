#!/usr/bin/env python3
"""
run_inner_loop.py — Fixed orchestrator for the autoresearch inner loop.

Composes the modifiable pipeline (pipeline/) with the frozen infrastructure
(llm_self_play.py, environments, evaluation) to run K iterations of LLM
policy synthesis and report social metrics.

This file is FROZEN — the researcher agent must NOT modify it.
The researcher modifies files in pipeline/ only.

Usage:
    uv run run_inner_loop.py --game cleanup --model gemini-3.1-pro-preview --map large --n-agents 10
    uv run run_inner_loop.py --game cleanup --model gemini-3.1-pro-preview --map large --n-agents 10 --output-dir autoresearch/runs/exp1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# --- Frozen infrastructure imports ---
# These come from the original codebase and MUST NOT be modified.

from llm_self_play import (
    _call_llm,
    extract_policy_code,
    validate_code_safety,
    load_policy,
    smoke_test_policy,
    evaluate_matchup,
    save_results,
    PolicyRecord,
    GameConfig,
    CLEANUP_CONFIG,
    GATHERING_CONFIG,
    get_opponents,
    log,
)
from gathering_env import make_gathering, make_gathering_large
from cleanup_env import make_cleanup

# --- Modifiable pipeline imports ---
# These are the files the researcher modifies.

from pipeline import prompts, feedback, helpers, config


async def run_inner_loop(
    game: str,
    model: str,
    env_factory: callable,
    game_config: GameConfig,
    output_dir: Path | None = None,
) -> dict:
    """Run the inner policy synthesis loop using pipeline configuration.

    Returns a dict with final metrics and trajectory.
    """
    n_iterations = config.N_ITERATIONS
    n_eval_seeds = config.N_EVAL_SEEDS
    max_retries = config.MAX_RETRIES

    system_prompt = prompts.get_system_prompt(game)
    extra_helpers = helpers.get_extra_helpers()

    # Merge game-specific namespace with pipeline helpers
    extra_ns = dict(game_config.extra_namespace)  # copy
    extra_ns.update(extra_helpers)

    max_action = game_config.max_action
    seeds = range(n_eval_seeds)

    policies = []
    history = []
    trajectory = []

    log(f"\n{'='*60}")
    log(f"  INNER LOOP: {game}, model={model}, K={n_iterations}, seeds={n_eval_seeds}")
    log(f"{'='*60}")

    t_start = time.time()

    for i in range(0, n_iterations + 1):
        log(f"\n{'='*60}")
        log(f"  ITERATION {i}/{n_iterations}")
        log(f"{'='*60}")

        opponent = policies[-1] if policies else None

        # Build prompt using pipeline feedback construction
        user_prompt = feedback.build_iteration_prompt(
            iteration=i,
            n_total=n_iterations,
            opponent=opponent,
            history=history,
            env_factory=env_factory,
            game_config=game_config,
        )

        # Generate policy with retries
        code = None
        fn = None
        reasoning = ""
        prompt_with_errors = user_prompt

        for attempt in range(max_retries):
            log(f"  Generating policy (attempt {attempt + 1}/{max_retries})...")
            t0 = time.time()

            try:
                full_text, reasoning = await _call_llm(
                    system_prompt, prompt_with_errors, model
                )
            except Exception as e:
                log(f"  LLM call failed: {e}")
                continue

            gen_time = time.time() - t0
            log(f"  Generated in {gen_time:.1f}s")

            # Extract code
            code = extract_policy_code(full_text)
            if code is None:
                error_msg = "Could not find policy function in response."
                log(f"  {error_msg}")
                prompt_with_errors += (
                    f"\n\nERROR from previous attempt: {error_msg}\n"
                    "You MUST provide exactly one ```python\\n...\\n``` block containing:\n"
                    "def policy(env, agent_id) -> int:\n"
                    '    """..."""\n'
                    "    ...\n"
                    f"    return <int 0-{max_action}>\n"
                )
                continue

            # Validate safety
            violations = validate_code_safety(code)
            if violations:
                error_msg = f"Safety violations: {'; '.join(violations)}"
                log(f"  {error_msg}")
                prompt_with_errors += (
                    f"\n\nERROR from previous attempt: {error_msg}\n"
                    "Please fix the violations and try again."
                )
                continue

            # Load policy
            try:
                fn = load_policy(code, extra_namespace=extra_ns)
            except Exception as e:
                error_msg = f"Failed to load policy: {e}"
                log(f"  {error_msg}")
                prompt_with_errors += (
                    f"\n\nERROR from previous attempt: {error_msg}\n"
                    "Please fix the code."
                )
                continue

            # Smoke test
            passed, smoke_err = smoke_test_policy(
                fn, env_factory=env_factory, max_action=max_action
            )
            if not passed:
                error_msg = f"Smoke test failed: {smoke_err}"
                log(f"  {error_msg}")
                prompt_with_errors += (
                    f"\n\nERROR from previous attempt: {error_msg}\n"
                    "Fix the bug above. Common mistakes:\n"
                    "- Unpacking a None result: always check `if result is None` before `dr, dc = result`\n"
                    "- Returning a tuple instead of int: `return direction_to_action(dr, dc, orient)` not `return (dr, dc)`\n"
                    "- Not casting numpy scalars: use `int(env.agent_orient[agent_id])`\n"
                    f"Make sure policy returns an integer 0-{max_action}."
                )
                fn = None
                continue

            break  # Success

        if fn is None:
            log(f"  WARNING: Failed to generate valid policy after {max_retries} retries. Skipping iteration.")
            continue

        # Evaluate self-play
        log(f"\n  Evaluating self-play...")
        t0 = time.time()
        try:
            results = evaluate_matchup(fn, fn, env_factory, seeds)
        except Exception as e:
            log(f"  Evaluation failed: {e}")
            results = {
                "reward_avg": 0.0,
                "metrics": {
                    "efficiency": 0.0,
                    "equality": 1.0,
                    "sustainability": 0.0,
                    "peace": 0.0,
                },
            }
        eval_time = time.time() - t0

        reward_avg = results["reward_avg"]
        metrics = results["metrics"]

        new_name = f"P{i}_pipeline"
        new_policy = PolicyRecord(new_name, code, fn, i, reasoning)
        policies.append(new_policy)

        history_entry = {
            "iteration": i,
            "new_policy": new_name,
            "opponent": new_name,
            "reward_avg": reward_avg,
            "metrics": metrics,
            "generation_time": gen_time,
            "eval_time": eval_time,
        }
        history.append(history_entry)

        trajectory.append({
            "iteration": i,
            "reward_avg": round(reward_avg, 2),
            "efficiency": round(metrics["efficiency"], 4),
            "equality": round(metrics.get("equality", 0), 4),
            "sustainability": round(metrics.get("sustainability", 0), 1),
            "peace": round(metrics.get("peace", 0), 2),
        })

        env_tmp = env_factory()
        log(f"  Self-play results ({eval_time:.1f}s):")
        log(f"    Average reward per agent ({env_tmp.n_agents} agents): {reward_avg:.1f}")
        for k, v in metrics.items():
            log(f"    {k:20s}: {v:.3f}")

    wall_time = time.time() - t_start

    # Save results if output_dir specified
    if output_dir:
        run_params = {
            "mode": "pipeline",
            "model": model,
            "game": game,
            "n_iterations": n_iterations,
            "eval_seeds": n_eval_seeds,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "wall_time_s": round(wall_time, 1),
        }
        save_results(policies, history, None, output_dir, run_params=run_params)

    # Final metrics (from the last successful iteration)
    if history:
        final = history[-1]
        result = {
            "efficiency": final["metrics"]["efficiency"],
            "equality": final["metrics"].get("equality", 0),
            "sustainability": final["metrics"].get("sustainability", 0),
            "peace": final["metrics"].get("peace", 0),
            "reward_avg": final["reward_avg"],
            "n_iterations": len(history),
            "wall_time_s": round(wall_time, 1),
            "trajectory": trajectory,
        }
    else:
        result = {
            "efficiency": 0.0,
            "equality": 0.0,
            "sustainability": 0.0,
            "peace": 0.0,
            "reward_avg": 0.0,
            "n_iterations": 0,
            "wall_time_s": round(wall_time, 1),
            "trajectory": [],
        }

    # Print metrics in parseable format
    print("\n=== INNER LOOP COMPLETE ===")
    print(json.dumps(result, indent=2))

    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inner policy synthesis loop with pipeline configuration"
    )
    parser.add_argument(
        "--game",
        choices=["gathering", "cleanup"],
        default="cleanup",
    )
    parser.add_argument(
        "--model",
        default="gemini-3.1-pro-preview",
    )
    parser.add_argument(
        "--map",
        choices=["small", "default", "large"],
        default="large",
    )
    parser.add_argument(
        "--n-agents",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save results (default: autoresearch/runs/<timestamp>)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Game config
    if args.game == "cleanup":
        game_config = CLEANUP_CONFIG
    else:
        game_config = GATHERING_CONFIG

    # Default agents
    if args.n_agents is None:
        if args.game == "cleanup":
            args.n_agents = 5 if args.map != "large" else 10
        elif args.map == "large":
            args.n_agents = 4
        else:
            args.n_agents = 2

    # Environment factory
    def env_factory():
        if args.game == "cleanup":
            return make_cleanup(n_agents=args.n_agents, small=(args.map == "small"))
        elif args.map == "large":
            return make_gathering_large(n_agents=args.n_agents)
        else:
            return make_gathering(n_agents=args.n_agents, small=(args.map == "small"))

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path("autoresearch") / "runs" / timestamp

    asyncio.run(
        run_inner_loop(
            game=args.game,
            model=args.model,
            env_factory=env_factory,
            game_config=game_config,
            output_dir=output_dir,
        )
    )
