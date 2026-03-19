"""
Verifier environment wrapper for Sequential Social Dilemma (SSD) environments.

Wraps the Gathering (large map) and Cleanup environments as verifiers-compatible
SingleTurnEnv environments for use with GEPA and other optimizers.

The LLM generates a Python policy function ``def policy(env, agent_id) -> int``.
The wrapper extracts, validates, and executes the policy in self-play (all N
agents run the same code). The reward is the average per-agent reward,
matching the code-reward mode from ``llm_self_play.py``.

Usage::

    from ssd_verifier_env import load_environment

    # Gathering (large map, 10 agents, code-reward)
    env = load_environment(game="gathering", mode="code-reward")

    # Cleanup (standard map, 10 agents, code-reward-all with social metrics)
    env = load_environment(game="cleanup", mode="code-reward-all")

Requires:
    - verifiers framework (``import verifiers as vf``)
    - SSD modules: gathering_env, cleanup_env, gathering_policy
    - llm_self_play (for prompts and code validation utilities)
"""

from __future__ import annotations

import textwrap
import traceback

import numpy as np
from datasets import Dataset

import verifiers as vf

# SSD environment and policy modules
from gathering_env import make_gathering_large
from cleanup_env import make_cleanup
from gathering_policy import run_episode

# Prompts, configs, and utilities from the self-play framework
from llm_self_play import (
    GATHERING_CONFIG,
    CLEANUP_CONFIG,
    extract_policy_code,
    validate_code_safety,
    load_policy,
    smoke_test_policy,
)


# ── Default Configuration ────────────────────────────────────────────────────

N_AGENTS = 10
N_EVAL_SEEDS = 5


# ── Environment Factories ────────────────────────────────────────────────────

def _make_env_factory(game: str, n_agents: int):
    """Return a zero-argument callable that creates a fresh env instance."""
    if game == "gathering":
        return lambda: make_gathering_large(n_agents=n_agents)
    elif game == "cleanup":
        return lambda: make_cleanup(n_agents=n_agents, small=False)
    else:
        raise ValueError(f"Unknown game: {game!r}. Use 'gathering' or 'cleanup'.")


# ── Policy Evaluation ────────────────────────────────────────────────────────

def _evaluate_policy_code(
    response_text: str,
    env_factory,
    game_config,
    n_eval_seeds: int,
) -> tuple[float, dict, str]:
    """Extract, validate, load, and evaluate a policy from LLM text.

    Returns
    -------
    avg_reward : float
        Average per-agent reward across all seeds (0.0 on failure).
    metrics : dict
        Social metrics (efficiency, equality, sustainability, peace).
        Empty dict on failure.
    error : str
        Empty on success; describes the failure otherwise.
    """
    # 1. Extract policy code from ```python``` block
    policy_code = extract_policy_code(response_text)
    if policy_code is None:
        return 0.0, {}, "Could not extract policy function from response."

    # 2. AST safety check
    violations = validate_code_safety(policy_code)
    if violations:
        return 0.0, {}, f"Safety violations: {'; '.join(violations)}"

    # 3. Load policy into sandboxed namespace
    try:
        fn = load_policy(policy_code, extra_namespace=game_config.extra_namespace)
    except Exception as e:
        return 0.0, {}, f"Failed to load policy: {e}"

    # 4. Apply wrapper (e.g., scent-only proxy) if configured
    if game_config.policy_wrapper:
        fn = game_config.policy_wrapper(fn)

    # 5. Smoke test (short self-play to catch crashes early)
    passed, smoke_err = smoke_test_policy(
        fn,
        env_factory=env_factory,
        max_action=game_config.max_action,
        n_steps=50,
    )
    if not passed:
        return 0.0, {}, f"Smoke test failed: {smoke_err}"

    # 6. Full evaluation: self-play across seeds
    try:
        all_rewards = []
        all_metrics = []
        for seed in range(n_eval_seeds):
            env = env_factory()
            agent_fns = {i: fn for i in range(env.n_agents)}
            result = run_episode(env, agent_fns, seed=seed, verbose=False)
            per_agent = [result["total_rewards"][i] for i in range(env.n_agents)]
            all_rewards.extend(per_agent)
            all_metrics.append(result["metrics"])

        avg_reward = float(np.mean(all_rewards))
        avg_metrics = {}
        for key in all_metrics[0]:
            avg_metrics[key] = float(np.mean([m[key] for m in all_metrics]))

        # Normalize to [0, 1] range (episode length = 1000 steps)
        return avg_reward / 1000.0, avg_metrics, ""

    except Exception as e:
        tb = traceback.format_exc()
        return 0.0, {}, f"Evaluation crashed: {e}\n{tb}"


# ── User Prompt Construction ─────────────────────────────────────────────────

def _build_user_prompt(game: str, mode: str, env_factory, game_config) -> str:
    """Build the initial user prompt for policy generation.

    Mirrors the iteration-0 prompt from ``build_iteration_prompt_reward``
    and ``build_iteration_prompt_reward_all`` in llm_self_play.py.
    """
    env = env_factory()
    env_desc = (
        f"{env.n_agents} agents on a {env.width}x{env.height} map "
        f"with ~{env.n_apples} apple spawns"
    )
    max_action = game_config.max_action
    env_hint = game_config.env_hint

    parts = [
        "## Write a policy\n",
        "All agents will run the same code. "
        "Your task is to write a policy that maximizes per-agent reward.\n",
    ]

    # Note: metric definitions are intentionally NOT included in the user
    # prompt.  GEPA only feeds back the scalar reward to its reflection LM,
    # so embedding metric definitions here would be information leakage
    # (the policy LLM would know what to optimise without GEPA discovering
    # it).  The mode flag still controls whether zero-weight metric
    # functions are registered in the rubric for logging purposes.

    parts.append(textwrap.dedent(f"""\
    ## Instructions

    Write a policy that maximizes per-agent reward. All agents will run your
    exact same code simultaneously. There are {env_desc}.
    {env_hint}

    Write your `policy(env, agent_id) -> int` function (returns 0-{max_action}).
    """))

    return "\n".join(parts)


# ── Entry Point ──────────────────────────────────────────────────────────────

def load_environment(
    game: str = "gathering",
    mode: str = "code-reward",
    n_agents: int = N_AGENTS,
    n_eval_seeds: int = N_EVAL_SEEDS,
) -> vf.Environment:
    """Load an SSD verifier environment.

    Parameters
    ----------
    game : str
        ``"gathering"`` (large 38x16 map) or ``"cleanup"`` (standard 18x25 map).
    mode : str
        ``"code-reward"`` — reward-only feedback.
        ``"code-reward-all"`` — reward + social metric definitions in prompt;
        social metrics are tracked as rubric metrics (weight 0).
    n_agents : int
        Number of agents (default 10).
    n_eval_seeds : int
        Number of random seeds for policy evaluation (default 5).

    Returns
    -------
    vf.SingleTurnEnv
        Verifiers-compatible environment.  Each rollout: the LLM generates a
        ``policy(env, agent_id) -> int`` function; the wrapper executes it in
        self-play and returns the average per-agent reward.
    """
    if game not in ("gathering", "cleanup"):
        raise ValueError(f"game must be 'gathering' or 'cleanup', got {game!r}")
    if mode not in ("code-reward", "code-reward-all"):
        raise ValueError(
            f"mode must be 'code-reward' or 'code-reward-all', got {mode!r}"
        )

    # ── Game configuration ────────────────────────────────────────────────
    if game == "gathering":
        game_config = GATHERING_CONFIG
    else:
        game_config = CLEANUP_CONFIG

    system_prompt = game_config.system_prompt_reward
    env_factory = _make_env_factory(game, n_agents)

    # ── User prompt ───────────────────────────────────────────────────────
    user_prompt = _build_user_prompt(game, mode, env_factory, game_config)

    # ── Dataset (single task: "write a policy for this game") ─────────────
    dataset = Dataset.from_dict({
        "question": [user_prompt],
        "answer": [""],
        "info": [{}],
        "task": [f"ssd-{game}"],
    })

    # ── Parser (default: returns full assistant text including code blocks) ─
    parser = vf.Parser()

    # ── Reward and metric functions ───────────────────────────────────────
    def ssd_reward(completion, state, **kwargs) -> float:
        """Execute the generated policy in self-play and return avg reward."""
        text = parser.parse_answer(completion) or ""
        reward, metrics, error = _evaluate_policy_code(
            text, env_factory, game_config, n_eval_seeds,
        )
        state["ssd_metrics"] = metrics
        state["ssd_error"] = error
        return reward

    funcs = [ssd_reward]
    weights = [1.0]

    # In code-reward-all mode, also track social metrics (weight=0: logged,
    # not added to the reward signal).
    if mode == "code-reward-all":

        def efficiency_metric(state, **kwargs) -> float:
            return state.get("ssd_metrics", {}).get("efficiency", 0.0)

        def equality_metric(state, **kwargs) -> float:
            return state.get("ssd_metrics", {}).get("equality", 0.0)

        def sustainability_metric(state, **kwargs) -> float:
            return state.get("ssd_metrics", {}).get("sustainability", 0.0)

        def peace_metric(state, **kwargs) -> float:
            return state.get("ssd_metrics", {}).get("peace", 0.0)

        funcs += [
            efficiency_metric,
            equality_metric,
            sustainability_metric,
            peace_metric,
        ]
        weights += [0.0, 0.0, 0.0, 0.0]

    rubric = vf.Rubric(funcs=funcs, weights=weights)

    # ── Build verifier environment ────────────────────────────────────────
    vf_env = vf.SingleTurnEnv(
        dataset=dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
    )
    return vf_env
