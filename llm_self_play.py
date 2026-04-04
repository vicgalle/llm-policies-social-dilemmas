"""
Iterative LLM Self-Play for Sequential Social Dilemmas
=======================================================

Uses Claude (Agent SDK) or Gemini (Google GenAI SDK) to iteratively generate policies:
  1. Generate an initial policy from environment description
  2. Ask Claude to generate an improved policy based on reward feedback
  3. Evaluate the new policy in self-play
  4. Repeat, building a payoff matrix and tracking convergence

Usage:
    python llm_self_play.py --iterations 5
    python llm_self_play.py --iterations 1 --eval-seeds 1   # smoke test
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import re
import signal
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path

import os
# Allow launching Claude Code SDK from within a Claude Code session
os.environ.pop("CLAUDECODE", None)

import numpy as np


def log(msg: str = ""):
    """Print to stderr so output isn't captured by SDK subprocess transport."""
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()

from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ThinkingBlock,
)
from claude_agent_sdk._errors import MessageParseError

from gathering_env import (
    GatheringEnv,
    Action,
    Orientation,
    _ROTATIONS,
    NUM_ACTIONS,
    make_gathering,
    make_gathering_large,
)
from gathering_policy import (
    run_episode,
    greedy_action,
    exploitative_action,
    bfs_nearest_apple,
    bfs_to_target_set,
    bfs_toward,
    direction_to_action,
    _beam_targets_for_orient,
    _rotation_distance,
)
from cleanup_env import (
    CleanupEnv,
    CleanupAction,
    NUM_CLEANUP_ACTIONS,
    make_cleanup,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N_ITERATIONS = 5
N_EVAL_SEEDS = 5
MAX_RETRIES = 3
MODEL = "claude-sonnet-4-6"
THINKING_BUDGET = 16000
EVAL_TIMEOUT = 600  # seconds per matchup

# ---------------------------------------------------------------------------
# LLM call abstraction (supports Claude and Gemini)
# ---------------------------------------------------------------------------

def _is_gemini_model(model: str) -> bool:
    """Check if the model name refers to a Gemini model."""
    return model.startswith("gemini")


async def _call_llm(system_prompt: str, user_prompt: str, model: str) -> tuple[str, str]:
    """Call an LLM and return (full_text, reasoning).

    Dispatches to Claude Agent SDK or Google GenAI based on model name.
    """
    if _is_gemini_model(model):
        return await _call_gemini(system_prompt, user_prompt, model)
    else:
        return await _call_claude(system_prompt, user_prompt, model)


async def _call_claude(system_prompt: str, user_prompt: str, model: str) -> tuple[str, str]:
    """Call Claude via the Agent SDK."""
    try:
        options = ClaudeAgentOptions(
            system_prompt=system_prompt,
            max_turns=1,
            model=model,
            effort="high",
        )
    except TypeError:
        options = ClaudeAgentOptions(
            system_prompt=system_prompt,
            max_turns=1,
            model=model,
            effort="high",
        )

    text_parts = []
    thinking_parts = []

    try:
        async for message in query(prompt=user_prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        text_parts.append(block.text)
                    elif isinstance(block, ThinkingBlock):
                        thinking_parts.append(block.thinking)
            elif isinstance(message, ResultMessage):
                if message.total_cost_usd:
                    log(f"  API cost: ${message.total_cost_usd:.4f}")
    except MessageParseError as e:
        log(f"  Warning: SDK parse error (skipped): {e}")
        if not text_parts:
            raise

    full_text = "\n".join(text_parts)
    reasoning = "\n".join(thinking_parts) if thinking_parts else ""
    return full_text, reasoning


async def _call_gemini(system_prompt: str, user_prompt: str, model: str) -> tuple[str, str]:
    """Call Gemini via the Google GenAI SDK.

    Requires the ``google-genai`` package and a GEMINI_API_KEY (or
    GOOGLE_API_KEY) environment variable.
    """
    from google import genai
    from google.genai import types

    client = genai.Client()

    response = await client.aio.models.generate_content(
        model=model,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            thinking_config=types.ThinkingConfig(thinking_budget=16384),
        ),
    )

    full_text = ""
    reasoning = ""
    if response.candidates and response.candidates[0].content:
        for part in response.candidates[0].content.parts:
            if hasattr(part, "thought") and part.thought:
                reasoning += part.text
            else:
                full_text += part.text

    return full_text, reasoning


# ---------------------------------------------------------------------------
# Dynamic environment description
# ---------------------------------------------------------------------------

def _env_description(env_factory: callable) -> str:
    """Generate a short description of the map from an env factory."""
    env = env_factory()
    return (f"{env.n_agents} agents on a {env.width}x{env.height} map "
            f"with ~{env.n_apples} apple spawns")


# ---------------------------------------------------------------------------
# System prompt for policy generation
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""\
You are an expert game-theoretic AI designing policies for a multi-agent
Sequential Social Dilemma (the Gathering game).

## Environment Summary

- 2D gridworld. Agents collect apples (+1 reward each). Apples respawn after
  25 steps. Agents can fire a "tagging beam" that temporarily removes rivals
  for 25 steps (requires 2 hits to tag in Gathering).
- Episode length: 1000 steps.
- 8 actions: FORWARD(0), BACKWARD(1), STEP_LEFT(2), STEP_RIGHT(3),
  ROTATE_LEFT(4), ROTATE_RIGHT(5), BEAM(6), STAND(7)
- Agents move in 4 cardinal directions WITHOUT needing to rotate first
  (strafe movement). Rotation only matters for the beam direction.

## Environment API (available in your policy's namespace)

```python
# env attributes you can read:
env.agent_pos        # np.array shape (n_agents, 2) — [row, col] per agent
env.agent_orient     # np.array shape (n_agents,) — 0=N, 1=E, 2=S, 3=W
env.agent_timeout    # np.array shape (n_agents,) — >0 means agent is removed
env.agent_beam_hits  # np.array shape (n_agents,) — hits accumulated toward tag
env.apple_alive      # np.array shape (n_apples,) bool — which apples exist
env._apple_pos       # np.array shape (n_apples, 2) — [row, col] per apple spawn
env.walls            # np.array shape (H, W) bool — wall map
env.height, env.width                  # map dimensions
env.n_agents, env.n_apples             # counts
env.beam_length, env.beam_width        # beam parameters (20, 1)
env.hits_to_tag, env.timeout_steps     # 2 hits to tag, 25 step timeout
```

## Helper functions available in your namespace

```python
from gathering_env import Action, Orientation, _ROTATIONS, NUM_ACTIONS

# BFS to nearest alive apple. Returns (dr, dc) of first step, or None.
bfs_nearest_apple(env, agent_id) -> Optional[Tuple[int,int]]

# BFS to nearest position in a set. Returns (dr, dc) or None.
bfs_to_target_set(env, agent_id, target_set) -> Optional[Tuple[int,int]]

# BFS toward a specific (row, col). Returns (dr, dc) or None.
bfs_toward(env, agent_id, target_r, target_c) -> Optional[Tuple[int,int]]

# Convert world-frame (dr, dc) step to an Action given agent orientation.
direction_to_action(dr, dc, orientation) -> int

# Build the opponents list for _beam_targets_for_orient (convenience helper).
# Returns list of (opp_id, row, col, manhattan_dist, beam_hits) for alive opponents.
get_opponents(env, agent_id) -> list

# Check which opponents would be hit by a beam from (ar, ac) facing orient_val.
# IMPORTANT: opponents must be a list of 5-tuples: [(id, row, col, manhattan_dist, beam_hits), ...]
# Use get_opponents(env, agent_id) to build this list easily.
# Returns list of (agent_id, fwd_dist, beam_hits).
_beam_targets_for_orient(env, ar, ac, orient_val, opponents) -> list

# Min rotations between two orientations (0-2).
_rotation_distance(cur, target) -> int

# Reference policies:
greedy_action(env, agent_id) -> int      # BFS to nearest apple, never beams
exploitative_action(env, agent_id) -> int # beams opportunistically + collects

# Also available: np (numpy), deque (from collections)
```

## Your task

Write a Python function called `policy` with this exact signature:

```python
def policy(env, agent_id) -> int:
    \"\"\"Return an action (int 0-7) for the given agent.\"\"\"
    ...
```

The function must:
1. Return an integer 0-7 (an Action value)
2. Be deterministic given the environment state
3. Only use the env attributes and helper functions listed above
4. Not import any modules (numpy and deque are pre-loaded)
5. Not use eval(), exec(), open(), or __import__

## Working Example (seed BFS policy)

This is a minimal working policy. Use it as a template for the structure and
return type. Note how bfs_nearest_apple returns Optional[Tuple[int,int]] —
you MUST handle the None case.

```python
def policy(env, agent_id) -> int:
    \"\"\"BFS greedy: go to nearest apple, never beam.\"\"\"
    if int(env.agent_timeout[agent_id]) > 0:
        return 7  # STAND while removed
    result = bfs_nearest_apple(env, agent_id)
    if result is None:
        return 7  # No reachable apple — stand
    dr, dc = result
    return direction_to_action(dr, dc, int(env.agent_orient[agent_id]))
```

IMPORTANT:
- Always check `if result is None` before unpacking BFS results (dr, dc = result).
- Always cast env arrays to int when comparing: `int(env.agent_timeout[agent_id])`.
- Always return a plain int (0-7), never a tuple or None.
- Put your code in a single ```python ... ``` block.
- Before the code block, explain your game-theoretic reasoning for WHY this
  counter-strategy should exploit the opponent's weaknesses.
""")

# Neutral, reward-only feedback variant of the system prompt.
SYSTEM_PROMPT_REWARD = SYSTEM_PROMPT.replace(
    "Before the code block, explain your game-theoretic reasoning for WHY this\n"
    "  counter-strategy should exploit the opponent's weaknesses.",
    "Before the code block, explain your reasoning for the policy design.",
)


# ---------------------------------------------------------------------------
# Cleanup game prompts
# ---------------------------------------------------------------------------

CLEANUP_SYSTEM_PROMPT = textwrap.dedent("""\
You are an expert game-theoretic AI designing policies for a multi-agent
Sequential Social Dilemma (the Cleanup game).

## Environment Summary

- 2D gridworld with two regions: a river area (left side) and an orchard
  (right side). A stream separates the two regions.
- Agents collect apples in the orchard (+1 reward each).
- Waste (pollution) accumulates in the river over time.
- Episode length: 1000 steps.
- 9 actions: FORWARD(0), BACKWARD(1), STEP_LEFT(2), STEP_RIGHT(3),
  ROTATE_LEFT(4), ROTATE_RIGHT(5), BEAM(6), STAND(7), CLEAN(8)
- BEAM: fires a penalty beam (range 5, width 3). Costs -1 reward to fire.
  Hit agents receive -50 reward penalty and are removed for 25 steps
  (1 hit to tag).
- CLEAN: fires a cleaning beam (range 5, width 3). Costs -1 reward to fire.
  Removes waste cells in the beam's path, restoring clean river.
- Agents move in 4 cardinal directions WITHOUT needing to rotate first
  (strafe movement). Rotation only matters for the beam/clean direction.

## Environment API (available in your policy's namespace)

```python
# env attributes you can read:
env.agent_pos        # np.array shape (n_agents, 2) — [row, col] per agent
env.agent_orient     # np.array shape (n_agents,) — 0=N, 1=E, 2=S, 3=W
env.agent_timeout    # np.array shape (n_agents,) — >0 means agent is removed
env.agent_beam_hits  # np.array shape (n_agents,) — hits accumulated toward tag
env.apple_alive      # np.array shape (n_apples,) bool — which apples exist
env._apple_pos       # np.array shape (n_apples, 2) — [row, col] per apple spawn
env.walls            # np.array shape (H, W) bool — wall map
env.waste            # np.array shape (H, W) bool — True where waste exists
env.river_cells_set  # set of (row, col) — all river cell positions
env.stream_cells_set # set of (row, col) — stream cell positions
env.height, env.width                  # map dimensions
env.n_agents, env.n_apples             # counts
env.beam_length, env.beam_width        # beam/clean parameters (5, 3)
env.hits_to_tag, env.timeout_steps     # 1 hit to tag, 25 step timeout
```

## Helper functions available in your namespace

```python
from cleanup_env import CleanupAction, NUM_CLEANUP_ACTIONS
from gathering_env import Orientation, _ROTATIONS

# BFS to nearest alive apple. Returns (dr, dc) of first step, or None.
bfs_nearest_apple(env, agent_id) -> Optional[Tuple[int,int]]

# BFS to nearest position in a set. Returns (dr, dc) or None.
bfs_to_target_set(env, agent_id, target_set) -> Optional[Tuple[int,int]]

# BFS toward a specific (row, col). Returns (dr, dc) or None.
bfs_toward(env, agent_id, target_r, target_c) -> Optional[Tuple[int,int]]

# Convert world-frame (dr, dc) step to an Action given agent orientation.
direction_to_action(dr, dc, orientation) -> int

# Build the opponents list for _beam_targets_for_orient.
get_opponents(env, agent_id) -> list

# Check which opponents would be hit by beam from (ar, ac) facing orient_val.
_beam_targets_for_orient(env, ar, ac, orient_val, opponents) -> list

# Min rotations between two orientations (0-2).
_rotation_distance(cur, target) -> int

# Reference policy (BFS to nearest apple, never beams or cleans):
greedy_action(env, agent_id) -> int

# Also available: np (numpy), deque (from collections)
```

## Your task

Write a Python function called `policy` with this exact signature:

```python
def policy(env, agent_id) -> int:
    \"\"\"Return an action (int 0-8) for the given agent.\"\"\"
    ...
```

The function must:
1. Return an integer 0-8 (a CleanupAction value)
2. Be deterministic given the environment state
3. Only use the env attributes and helper functions listed above
4. Not import any modules (numpy and deque are pre-loaded)
5. Not use eval(), exec(), open(), or __import__

## Working Example (seed BFS policy)

This is a minimal working policy. Use it as a template for the structure and
return type. Note how bfs_nearest_apple returns Optional[Tuple[int,int]] —
you MUST handle the None case.

```python
def policy(env, agent_id) -> int:
    \"\"\"BFS greedy: go to nearest apple, never beam or clean.\"\"\"
    if int(env.agent_timeout[agent_id]) > 0:
        return 7  # STAND while removed
    result = bfs_nearest_apple(env, agent_id)
    if result is None:
        return 7  # No reachable apple — stand
    dr, dc = result
    return direction_to_action(dr, dc, int(env.agent_orient[agent_id]))
```

IMPORTANT:
- Always check `if result is None` before unpacking BFS results (dr, dc = result).
- Always cast env arrays to int when comparing: `int(env.agent_timeout[agent_id])`.
- Always return a plain int (0-8), never a tuple or None.
- Put your code in a single ```python ... ``` block.
- Before the code block, explain your game-theoretic reasoning for WHY this
  counter-strategy should exploit the opponent's weaknesses.
""")

CLEANUP_SYSTEM_PROMPT_REWARD = CLEANUP_SYSTEM_PROMPT.replace(
    "Before the code block, explain your game-theoretic reasoning for WHY this\n"
    "  counter-strategy should exploit the opponent's weaknesses.",
    "Before the code block, explain your reasoning for the policy design.",
)


# ---------------------------------------------------------------------------
# Game configuration
# ---------------------------------------------------------------------------

@dataclass
class GameConfig:
    """Bundle of game-specific configuration for the self-play framework."""
    name: str                       # "gathering" or "cleanup"
    system_prompt_reward: str       # system prompt (neutral, reward-only feedback)
    max_action: int                 # 7 for gathering, 8 for cleanup
    extra_namespace: dict           # additional items for load_policy namespace
    env_hint: str                   # game-specific reminder for iteration prompts

# Game config instances
GATHERING_CONFIG = GameConfig(
    name="gathering",
    system_prompt_reward=SYSTEM_PROMPT_REWARD,
    max_action=7,
    extra_namespace={},
    env_hint="Apples respawn every 25 steps. It takes 2 beam hits to tag out an agent.",
)

CLEANUP_CONFIG = GameConfig(
    name="cleanup",
    system_prompt_reward=CLEANUP_SYSTEM_PROMPT_REWARD,
    max_action=8,
    extra_namespace={
        "CleanupAction": CleanupAction,
        "NUM_CLEANUP_ACTIONS": NUM_CLEANUP_ACTIONS,
    },
    env_hint=(
        "Waste accumulates in the river over time. "
        "BEAM costs -1 to fire (-50 to target, 1 hit tags out for 25 steps). "
        "CLEAN costs -1 to fire (removes waste in beam path)."
    ),
)

from coop_mining_env import Action as MiningAction, NUM_ACTIONS as NUM_MINING_ACTIONS

COOP_MINING_CONFIG = GameConfig(
    name="coop_mining",
    system_prompt_reward=SYSTEM_PROMPT_REWARD,
    max_action=7,
    extra_namespace={
        "MiningAction": MiningAction,
        "NUM_MINING_ACTIONS": NUM_MINING_ACTIONS,
        "IRON": 0,
        "GOLD": 1,
    },
    env_hint=(
        "Two ore types: Iron (mine alone, +1) and Gold (needs 2 miners within 3 steps, +8 each). "
        "Action 6 = MINE (beam range 3, width 1). Gold flashes when first mined — "
        "a second miner must hit it within 3 steps. No tagging in this game."
    ),
)


# ---------------------------------------------------------------------------
# PolicyRecord
# ---------------------------------------------------------------------------

@dataclass
class PolicyRecord:
    name: str
    code: str
    fn: callable
    iteration: int
    reasoning: str = ""


# ---------------------------------------------------------------------------
# Code extraction & validation
# ---------------------------------------------------------------------------

def extract_policy_code(text: str) -> str | None:
    """Extract the ```python``` block containing policy from LLM output."""
    # Try ```python fenced blocks first
    pattern = r"```python\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    for block in matches:
        if "def policy" in block:
            return block.strip()
    # Fallback: try without language tag
    pattern = r"```\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    for block in matches:
        if "def policy" in block:
            return block.strip()
    # Last resort: extract raw def block from unformatted text
    match = re.search(
        r"(def policy\(env,\s*agent_id\).*?)(?=\ndef [a-zA-Z]|\Z)",
        text, re.DOTALL,
    )
    if match:
        code = match.group(1).strip()
        if len(code.splitlines()) >= 3:  # at least signature + body
            return code
    return None


_FORBIDDEN_IMPORTS = {"os", "sys", "subprocess", "socket", "shutil", "pathlib"}
_FORBIDDEN_CALLS = {"__import__", "eval", "exec", "open", "compile", "globals"}


def get_opponents(env, agent_id) -> list:
    """Build the opponents list expected by _beam_targets_for_orient.

    Returns list of (opp_id, row, col, manhattan_dist, beam_hits) for each
    alive opponent.
    """
    ar, ac = int(env.agent_pos[agent_id][0]), int(env.agent_pos[agent_id][1])
    opponents = []
    for j in range(env.n_agents):
        if j == agent_id or int(env.agent_timeout[j]) > 0:
            continue
        jr, jc = int(env.agent_pos[j][0]), int(env.agent_pos[j][1])
        manhattan = abs(jr - ar) + abs(jc - ac)
        opponents.append((j, jr, jc, manhattan, int(env.agent_beam_hits[j])))
    return opponents


def validate_code_safety(code: str) -> list[str]:
    """AST-based safety check. Returns list of violation messages (empty = safe)."""
    violations = []
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"Syntax error: {e}"]

    for node in ast.walk(tree):
        # Check imports
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            else:
                names = [node.module.split(".")[0]] if node.module else []
            for name in names:
                if name in _FORBIDDEN_IMPORTS:
                    violations.append(f"Forbidden import: {name}")

        # Check dangerous calls
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _FORBIDDEN_CALLS:
                violations.append(f"Forbidden call: {func.id}()")
            if isinstance(func, ast.Attribute) and func.attr in _FORBIDDEN_CALLS:
                violations.append(f"Forbidden call: .{func.attr}()")

    return violations


def load_policy(code: str, extra_namespace: dict | None = None) -> callable:
    """Execute policy code in a sandboxed namespace and return the function."""
    namespace = {
        # Environment types
        "GatheringEnv": GatheringEnv,
        "Action": Action,
        "Orientation": Orientation,
        "_ROTATIONS": _ROTATIONS,
        "NUM_ACTIONS": NUM_ACTIONS,
        # Helper functions
        "bfs_nearest_apple": bfs_nearest_apple,
        "bfs_to_target_set": bfs_to_target_set,
        "bfs_toward": bfs_toward,
        "direction_to_action": direction_to_action,
        "_beam_targets_for_orient": _beam_targets_for_orient,
        "_rotation_distance": _rotation_distance,
        "get_opponents": get_opponents,
        "greedy_action": greedy_action,
        "exploitative_action": exploitative_action,
        # Utilities
        "np": np,
        "deque": __import__("collections").deque,
        # Builtins (restricted)
        "__builtins__": {
            k: v for k, v in __builtins__.items()
            if k not in {"__import__", "eval", "exec", "open", "compile"}
        } if isinstance(__builtins__, dict) else {
            k: getattr(__builtins__, k)
            for k in dir(__builtins__)
            if k not in {"__import__", "eval", "exec", "open", "compile"}
        },
    }

    if extra_namespace:
        namespace.update(extra_namespace)

    exec(code, namespace)

    if "policy" not in namespace:
        raise ValueError("Code does not define 'policy' function")

    return namespace["policy"]


def smoke_test_policy(fn: callable, env_factory: callable = None, max_action: int = 7,
                      n_steps: int = 50) -> tuple[bool, str]:
    """Run a short episode to verify the policy doesn't crash.

    Runs n_steps of self-play (all agents use fn) to exercise more code paths
    than a single-call test.  Returns (passed, error_detail).
    """
    import traceback
    env = env_factory() if env_factory is not None else make_gathering(n_agents=2, small=True)
    env.reset(seed=0)
    try:
        for step in range(n_steps):
            actions = {}
            for aid in range(env.n_agents):
                result = fn(env, aid)
                if not isinstance(result, (int, np.integer)) or not (0 <= int(result) <= max_action):
                    msg = (
                        f"Step {step}, agent {aid}: policy returned {result!r} "
                        f"(type {type(result).__name__}), "
                        f"but must return an int in 0-{max_action}."
                    )
                    log(f"  Smoke test failed: {msg}")
                    return False, msg
                actions[aid] = int(result)
            env.step(actions)
        return True, ""
    except Exception as e:
        tb = traceback.format_exc()
        log(f"  Smoke test failed at step {step}: {e}")
        return False, f"Runtime error at step {step}: {e}\nTraceback:\n{tb}"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

class EvalTimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise EvalTimeoutError("Evaluation timed out")


def evaluate_matchup(
    fn_a: callable,
    fn_b: callable,
    env_factory: callable,
    seeds: range,
    timeout: int = EVAL_TIMEOUT,
) -> dict:
    """Evaluate two policies against each other.

    Runs both orderings (A as agent 0, B as agent 0) across all seeds.
    Returns mean rewards and social metrics.
    """
    results_a_as_0 = []
    results_b_as_0 = []

    # Set timeout
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)

    try:
        for seed in seeds:
            env = env_factory()
            n = env.n_agents
            # A as agent 0, B as agent 1 (remaining agents get fn_a by default)
            fns_a0 = {0: fn_a, 1: fn_b}
            for k in range(2, n):
                fns_a0[k] = fn_a  # fill remaining slots with fn_a
            r = run_episode(env, fns_a0, seed=seed, verbose=False)
            results_a_as_0.append(r)

            # B as agent 0, A as agent 1
            env = env_factory()
            fns_b0 = {0: fn_b, 1: fn_a}
            for k in range(2, n):
                fns_b0[k] = fn_b
            r = run_episode(env, fns_b0, seed=seed, verbose=False)
            results_b_as_0.append(r)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    # Aggregate: average reward across all agents and both orderings
    all_rewards = []
    for r in results_a_as_0 + results_b_as_0:
        all_rewards.extend(r["total_rewards"][i] for i in range(len(r["total_rewards"])))
    reward_avg = float(np.mean(all_rewards))

    # Average metrics
    all_metrics = [r["metrics"] for r in results_a_as_0 + results_b_as_0]
    avg_metrics = {}
    for key in all_metrics[0]:
        avg_metrics[key] = float(np.mean([m[key] for m in all_metrics]))

    return {
        "reward_avg": reward_avg,
        "metrics": avg_metrics,
    }


def build_payoff_matrix(
    policies: list[PolicyRecord],
    env_factory: callable,
    seeds: range,
) -> np.ndarray:
    """Build N x N payoff matrix. M[i,j] = mean reward of policy i vs policy j."""
    n = len(policies)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                # Self-play — all agents use the same policy
                env = env_factory()
                fns = {k: policies[i].fn for k in range(env.n_agents)}
                r = run_episode(env, fns, seed=0, verbose=False)
                matrix[i, j] = np.mean([r["total_rewards"][k] for k in range(env.n_agents)])
            else:
                result = evaluate_matchup(
                    policies[i].fn, policies[j].fn, env_factory, seeds,
                )
                matrix[i, j] = result["reward_avg"]

    return matrix


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_iteration_prompt_reward(
    iteration: int,
    n_total: int,
    opponent: PolicyRecord | None,
    history: list[dict],
    env_factory: callable = None,
    game_config: GameConfig = None,
) -> str:
    """Build the user prompt for code-reward mode.

    Neutral framing: states that all agents share the same code (true),
    shows previous policy and per-agent reward only. No raw metrics,
    no adversarial language.

    When opponent is None (iteration 0), prompts from scratch with no
    prior policy — only environment description and template.
    """
    parts = []

    if opponent is None:
        # Iteration 0: no prior policy
        parts.append(f"## Iteration {iteration}/{n_total}: Write the initial policy\n")
        parts.append("No prior policy exists yet. All agents will run the same code. "
                     "Your task is to write a first policy that maximizes per-agent reward.\n")
    else:
        parts.append(f"## Iteration {iteration}/{n_total}: Write an improved policy\n")
        parts.append("The following policy is currently used by all agents. "
                     "All agents run the same code. "
                     "Your task is to write an improved version that maximizes per-agent reward.\n")
        parts.append(f"### Current policy: **{opponent.name}**\n")
        parts.append(f"```python\n{opponent.code}\n```\n")

    if history:
        parts.append("## Results from previous iterations\n")
        for h in history:
            parts.append(
                f"- Iteration {h['iteration']}: "
                f"Avg agent reward={h['reward_avg']:.1f}"
            )
        parts.append("")

    env_desc = _env_description(env_factory) if env_factory else "2 agents on a small map with ~12 apple spawns"
    env_hint = game_config.env_hint if game_config else GATHERING_CONFIG.env_hint
    max_action = game_config.max_action if game_config else 7
    parts.append(textwrap.dedent(f"""\
    ## Instructions

    Write a policy that maximizes per-agent reward. All agents will run your
    exact same code simultaneously. There are {env_desc}.
    {env_hint}

    Write your `policy(env, agent_id) -> int` function (returns 0-{max_action}).
    """))

    return "\n".join(parts)


# Social metric definitions shown to the LLM in code-reward-all mode.
METRIC_DEFINITIONS = (
    "- **Efficiency**: collective apple collection rate across all agents "
    "(higher = more apples collected per step).\n"
    "- **Equality**: fairness of reward distribution between agents "
    "(1.0 = perfectly equal, negative = highly unequal).\n"
    "- **Sustainability**: long-term apple availability — measures whether "
    "resources are preserved over the episode (higher = apples remain available "
    "later in the episode).\n"
    "- **Peace**: absence of aggressive beaming — counts agents not involved in "
    "attack beam conflicts (higher = less aggression). Using the CLEAN beam to "
    "remove waste does NOT reduce peace."
)


def build_iteration_prompt_reward_all(
    iteration: int,
    n_total: int,
    opponent: PolicyRecord | None,
    history: list[dict],
    env_factory: callable = None,
    game_config: GameConfig = None,
) -> str:
    """Build the user prompt for code-reward-all mode.

    Same as code-reward but also shows social metrics with definitions.
    """
    parts = []

    if opponent is None:
        parts.append(f"## Iteration {iteration}/{n_total}: Write the initial policy\n")
        parts.append("No prior policy exists yet. All agents will run the same code. "
                     "Your task is to write a first policy that maximizes per-agent reward.\n")
    else:
        parts.append(f"## Iteration {iteration}/{n_total}: Write an improved policy\n")
        parts.append("The following policy is currently used by all agents. "
                     "All agents run the same code. "
                     "Your task is to write an improved version that maximizes per-agent reward.\n")
        parts.append(f"### Current policy: **{opponent.name}**\n")
        parts.append(f"```python\n{opponent.code}\n```\n")

    if history:
        parts.append("## Results from previous iterations\n")
        parts.append("### Social Metrics (definitions)\n")
        parts.append(METRIC_DEFINITIONS)
        parts.append("")
        for h in history:
            m = h['metrics']
            parts.append(
                f"- Iteration {h['iteration']}: "
                f"Avg agent reward={h['reward_avg']:.1f} | "
                f"efficiency={m['efficiency']:.3f}, "
                f"equality={m.get('equality', 0):.3f}, "
                f"sustainability={m.get('sustainability', 0):.1f}, "
                f"peace={m.get('peace', 0):.1f}"
            )
        parts.append("")

    env_desc = _env_description(env_factory) if env_factory else "2 agents on a small map with ~12 apple spawns"
    env_hint = game_config.env_hint if game_config else GATHERING_CONFIG.env_hint
    max_action = game_config.max_action if game_config else 7
    parts.append(textwrap.dedent(f"""\
    ## Instructions

    Write a policy that maximizes per-agent reward. All agents will run your
    exact same code simultaneously. There are {env_desc}.
    {env_hint}

    Write your `policy(env, agent_id) -> int` function (returns 0-{max_action}).
    """))

    return "\n".join(parts)



async def generate_reward_policy(
    iteration: int,
    n_total: int,
    opponent: PolicyRecord | None,
    history: list[dict],
    model: str,
    thinking_budget: int,
    env_factory: callable = None,
    game_config: GameConfig = None,
) -> tuple[str, callable, str]:
    """Generate a policy with code-reward framing (reward-only feedback).

    Uses neutral system prompt and build_iteration_prompt_reward (no raw metrics).
    """
    gc = game_config
    _sys_prompt = gc.system_prompt_reward if gc else SYSTEM_PROMPT_REWARD
    _max_action = gc.max_action if gc else 7
    _extra_ns = gc.extra_namespace if gc else None

    user_prompt = build_iteration_prompt_reward(
        iteration, n_total, opponent, history,
        env_factory=env_factory, game_config=gc,
    )

    for attempt in range(MAX_RETRIES):
        log(f"  Generating policy (attempt {attempt + 1}/{MAX_RETRIES})...")

        full_text, reasoning = await _call_llm(_sys_prompt, user_prompt, model)

        code = extract_policy_code(full_text)
        if code is None:
            error_msg = "Could not find policy function in response."
            log(f"  {error_msg}")
            user_prompt += (
                f"\n\nERROR from previous attempt: {error_msg}\n"
                "You MUST provide exactly one ```python\\n...\\n``` block containing:\n"
                "def policy(env, agent_id) -> int:\n"
                '    """..."""\n'
                "    ...\n"
                f"    return <int 0-{_max_action}>\n"
            )
            continue

        violations = validate_code_safety(code)
        if violations:
            error_msg = f"Safety violations: {'; '.join(violations)}"
            log(f"  {error_msg}")
            user_prompt += f"\n\nERROR from previous attempt: {error_msg}\nPlease fix the violations and try again."
            continue

        try:
            fn = load_policy(code, extra_namespace=_extra_ns)
        except Exception as e:
            error_msg = f"Failed to load policy: {e}"
            log(f"  {error_msg}")
            user_prompt += f"\n\nERROR from previous attempt: {error_msg}\nPlease fix the code."
            continue

        passed, smoke_err = smoke_test_policy(fn, env_factory=env_factory, max_action=_max_action)
        if not passed:
            error_msg = f"Smoke test failed: {smoke_err}"
            log(f"  {error_msg}")
            user_prompt += (
                f"\n\nERROR from previous attempt: {error_msg}\n"
                "Fix the bug above. Common mistakes:\n"
                "- Unpacking a None result: always check `if result is None` before `dr, dc = result`\n"
                "- Returning a tuple instead of int: `return direction_to_action(dr, dc, orient)` not `return (dr, dc)`\n"
                "- Not casting numpy scalars: use `int(env.agent_orient[agent_id])`\n"
                f"Make sure policy returns an integer 0-{_max_action}."
            )
            continue

        return code, fn, reasoning

    raise RuntimeError(f"Failed to generate valid policy after {MAX_RETRIES} attempts")


async def generate_reward_all_policy(
    iteration: int,
    n_total: int,
    opponent: PolicyRecord | None,
    history: list[dict],
    model: str,
    thinking_budget: int,
    env_factory: callable = None,
    game_config: GameConfig = None,
) -> tuple[str, callable, str]:
    """Generate a policy with code-reward-all framing (reward + social metrics).

    Same logic as generate_reward_policy but uses build_iteration_prompt_reward_all.
    """
    gc = game_config
    _sys_prompt = gc.system_prompt_reward if gc else SYSTEM_PROMPT_REWARD
    _max_action = gc.max_action if gc else 7
    _extra_ns = gc.extra_namespace if gc else None

    user_prompt = build_iteration_prompt_reward_all(
        iteration, n_total, opponent, history,
        env_factory=env_factory, game_config=gc,
    )

    for attempt in range(MAX_RETRIES):
        log(f"  Generating policy (attempt {attempt + 1}/{MAX_RETRIES})...")

        full_text, reasoning = await _call_llm(_sys_prompt, user_prompt, model)

        code = extract_policy_code(full_text)
        if code is None:
            error_msg = "Could not find policy function in response."
            log(f"  {error_msg}")
            user_prompt += (
                f"\n\nERROR from previous attempt: {error_msg}\n"
                "You MUST provide exactly one ```python\\n...\\n``` block containing:\n"
                "def policy(env, agent_id) -> int:\n"
                '    """..."""\n'
                "    ...\n"
                f"    return <int 0-{_max_action}>\n"
            )
            continue

        violations = validate_code_safety(code)
        if violations:
            error_msg = f"Safety violations: {'; '.join(violations)}"
            log(f"  {error_msg}")
            user_prompt += f"\n\nERROR from previous attempt: {error_msg}\nPlease fix the violations and try again."
            continue

        try:
            fn = load_policy(code, extra_namespace=_extra_ns)
        except Exception as e:
            error_msg = f"Failed to load policy: {e}"
            log(f"  {error_msg}")
            user_prompt += f"\n\nERROR from previous attempt: {error_msg}\nPlease fix the code."
            continue

        passed, smoke_err = smoke_test_policy(fn, env_factory=env_factory, max_action=_max_action)
        if not passed:
            error_msg = f"Smoke test failed: {smoke_err}"
            log(f"  {error_msg}")
            user_prompt += (
                f"\n\nERROR from previous attempt: {error_msg}\n"
                "Fix the bug above. Common mistakes:\n"
                "- Unpacking a None result: always check `if result is None` before `dr, dc = result`\n"
                "- Returning a tuple instead of int: `return direction_to_action(dr, dc, orient)` not `return (dr, dc)`\n"
                "- Not casting numpy scalars: use `int(env.agent_orient[agent_id])`\n"
                f"Make sure policy returns an integer 0-{_max_action}."
            )
            continue

        return code, fn, reasoning

    raise RuntimeError(f"Failed to generate valid policy after {MAX_RETRIES} attempts")


# ---------------------------------------------------------------------------
# Analysis & output
# ---------------------------------------------------------------------------

def analyze_and_print(matrix: np.ndarray, policies: list[PolicyRecord]):
    """Print payoff matrix and convergence analysis."""
    n = len(policies)
    names = [p.name for p in policies]

    # Header
    log("\n" + "=" * 70)
    log("  PAYOFF MATRIX (row = player, col = opponent, value = row's reward)")
    log("=" * 70)

    # Column headers
    max_name = max(len(name) for name in names)
    header = " " * (max_name + 2)
    for name in names:
        header += f"{name:>12s}"
    log(header)

    for i in range(n):
        row = f"{names[i]:<{max_name + 2}}"
        for j in range(n):
            row += f"{matrix[i, j]:12.1f}"
        log(row)

    # Exploitability: for each policy, max reward any opponent gets against it
    log("\n" + "-" * 70)
    log("  EXPLOITABILITY (max opponent reward against each policy)")
    log("-" * 70)

    for j in range(n):
        col = matrix[:, j]
        # Exclude self-play
        opponent_rewards = [matrix[i, j] for i in range(n) if i != j]
        if opponent_rewards:
            max_exploit = max(opponent_rewards)
            exploiter = names[np.argmax([matrix[i, j] if i != j else -1e9 for i in range(n)])]
            log(f"  {names[j]:20s}: {max_exploit:.1f} (by {exploiter})")

    # Check for cycles (rock-paper-scissors dynamics)
    log("\n" + "-" * 70)
    log("  DOMINANCE CHECK")
    log("-" * 70)

    for i in range(n):
        dominated_by = []
        for j in range(n):
            if i != j and matrix[j, i] > matrix[i, i] and matrix[j, j] > matrix[i, j]:
                dominated_by.append(names[j])
        if dominated_by:
            log(f"  {names[i]:20s} dominated by: {', '.join(dominated_by)}")
        else:
            log(f"  {names[i]:20s} not dominated")

    # Trajectory: how does the latest policy compare to all previous?
    if n >= 3:
        log("\n" + "-" * 70)
        log("  REWARD TRAJECTORY (each policy vs its predecessor)")
        log("-" * 70)
        for i in range(1, n):
            log(
                f"  {names[i]:20s} vs {names[i-1]:20s}: "
                f"new={matrix[i, i-1]:.1f}, old={matrix[i-1, i]:.1f}, "
                f"delta={matrix[i, i-1] - matrix[i-1, i]:+.1f}"
            )


def save_results(
    policies: list[PolicyRecord],
    history: list[dict],
    matrix: np.ndarray,
    output_dir: Path,
    run_params: dict | None = None,
):
    """Save all results to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save run parameters
    if run_params:
        (output_dir / "params.json").write_text(
            json.dumps(run_params, indent=2)
        )

    # Save each policy
    for p in policies:
        policy_file = output_dir / f"policy_{p.name}.py"
        header = f'"""\nPolicy: {p.name} (iteration {p.iteration})\n'
        if p.reasoning:
            # Truncate reasoning to first 500 chars for the file header
            short_reason = p.reasoning[:500] + ("..." if len(p.reasoning) > 500 else "")
            header += f"\nReasoning:\n{short_reason}\n"
        header += '"""\n\n'
        policy_file.write_text(header + p.code + "\n")

    # Save payoff matrix
    if matrix is not None:
        matrix_data = {
            "policies": [p.name for p in policies],
            "matrix": matrix.tolist(),
        }
        (output_dir / "payoff_matrix.json").write_text(
            json.dumps(matrix_data, indent=2)
        )

    # Save history
    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2)
    )

    # Save summary
    summary_lines = ["Self-Play Results Summary", "=" * 40, ""]
    for i, p in enumerate(policies):
        summary_lines.append(f"[{p.name}] iteration={p.iteration}")
    summary_lines.append("")
    if matrix is not None:
        summary_lines.append("Payoff Matrix:")
        names = [p.name for p in policies]
        n = len(policies)
        max_name = max(len(name) for name in names)
        header = " " * (max_name + 2)
        for name in names:
            header += f"{name:>12s}"
        summary_lines.append(header)
        for i in range(n):
            row = f"{names[i]:<{max_name + 2}}"
            for j in range(n):
                row += f"{matrix[i, j]:12.1f}"
            summary_lines.append(row)
        summary_lines.append("")
    for h in history:
        summary_lines.append(
            f"{h['new_policy']} vs {h['opponent']}: "
            f"avg_reward={h['reward_avg']:.1f}"
        )
    (output_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n")

    log(f"\nResults saved to {output_dir}/")


# ---------------------------------------------------------------------------
# Main self-play loop
# ---------------------------------------------------------------------------

async def run_self_play(
    n_iterations: int,
    env_factory: callable,
    model: str,
    thinking_budget: int,
    eval_seeds: int,
    output_dir: Path,
    *,
    generator_fn=None,
    policy_prefix: str = "reward",
    mode_label: str = "reward-only",
    game_config: GameConfig = None,
    compute_payoff_matrix: bool = False,
):
    """Run the iterative self-play loop.

    generator_fn: async function to generate policies (default: generate_reward_policy)
    policy_prefix: label for generated policies, e.g. "reward" or "rall"
    mode_label: saved in run params, e.g. "reward-only" or "reward+social"
    """
    if generator_fn is None:
        generator_fn = generate_reward_policy
    policies = []
    history = []

    log(f"\nStarting self-play: {n_iterations} iterations")
    log(f"  Model: {model}")
    log(f"  Thinking budget: {thinking_budget}")
    log(f"  Eval seeds: {eval_seeds}")
    log()

    seeds = range(eval_seeds)

    for i in range(0, n_iterations + 1):
        log(f"\n{'='*60}")
        log(f"  ITERATION {i}/{n_iterations}")
        log(f"{'='*60}")

        opponent = policies[-1] if policies else None
        if opponent:
            log(f"  Previous policy: {opponent.name}")
        else:
            log(f"  No prior policy — generating from scratch")

        # Generate policy
        t0 = time.time()
        code, fn, reasoning = await generator_fn(
            i, n_iterations, opponent, history, model, thinking_budget,
            env_factory=env_factory, game_config=game_config,
        )
        gen_time = time.time() - t0
        log(f"  Generated in {gen_time:.1f}s")

        new_name = f"P{i}_{policy_prefix}"
        new_policy = PolicyRecord(new_name, code, fn, i, reasoning)

        # Print first few lines of the generated code
        code_preview = "\n".join(code.split("\n")[:8])
        log(f"  Code preview:\n    {code_preview.replace(chr(10), chr(10) + '    ')}")

        # Evaluate: self-play (all agents use the new policy)
        # Test whether the generated policy works well when everyone runs the same code.
        log(f"\n  Evaluating {new_name} vs itself (self-play)...")
        t0 = time.time()
        try:
            results = evaluate_matchup(fn, fn, env_factory, seeds)
        except EvalTimeoutError:
            log("  WARNING: Evaluation timed out! Using fallback scores.")
            results = {"reward_avg": 0.0, "metrics": {
                "efficiency": 0.0, "equality": 1.0, "sustainability": 0.0, "peace": 2.0,
            }}
        eval_time = time.time() - t0

        reward_avg = results["reward_avg"]

        history_entry = {
            "iteration": i,
            "new_policy": new_name,
            "opponent": new_name,
            "reward_avg": reward_avg,
            "metrics": results["metrics"],
            "generation_time": gen_time,
            "eval_time": eval_time,
        }
        history.append(history_entry)
        policies.append(new_policy)

        env_tmp = env_factory()
        n_ag = env_tmp.n_agents
        log(f"  Self-play results ({eval_time:.1f}s):")
        log(f"    Average reward per agent ({n_ag} agents): {reward_avg:.1f}")
        for k, v in results["metrics"].items():
            log(f"    {k:20s}: {v:.3f}")

    # Full payoff matrix (optional, expensive)
    matrix = None
    if compute_payoff_matrix:
        log(f"\n{'='*60}")
        log("  Building full payoff matrix...")
        log(f"{'='*60}")
        matrix = build_payoff_matrix(policies, env_factory, seeds)
        analyze_and_print(matrix, policies)

    run_params = {
        "mode": mode_label,
        "model": model,
        "thinking_budget": thinking_budget,
        "n_iterations": n_iterations,
        "eval_seeds": eval_seeds,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    save_results(policies, history, matrix, output_dir, run_params=run_params)

    return policies, history, matrix


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

async def main(args):
    # Select game config
    if args.game == "cleanup":
        game_config = CLEANUP_CONFIG
    else:
        game_config = GATHERING_CONFIG

    # Environment factory
    def env_factory():
        if args.game == "cleanup":
            return make_cleanup(n_agents=args.n_agents, small=(args.map == "small"))
        elif args.map == "large":
            return make_gathering_large(n_agents=args.n_agents)
        else:
            return make_gathering(n_agents=args.n_agents, small=(args.map == "small"))

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path("self_play_results") / f"{args.game}_{args.mode}_{timestamp}"

    if args.mode == "reward-only":
        await run_self_play(
            n_iterations=args.iterations,
            env_factory=env_factory,
            model=args.model,
            thinking_budget=args.thinking_budget,
            eval_seeds=args.eval_seeds,
            output_dir=output_dir,
            generator_fn=generate_reward_policy,
            policy_prefix="reward",
            mode_label="reward-only",
            game_config=game_config,
            compute_payoff_matrix=args.payoff_matrix,
        )
    elif args.mode == "reward+social":
        await run_self_play(
            n_iterations=args.iterations,
            env_factory=env_factory,
            model=args.model,
            thinking_budget=args.thinking_budget,
            eval_seeds=args.eval_seeds,
            output_dir=output_dir,
            generator_fn=generate_reward_all_policy,
            policy_prefix="rall",
            mode_label="reward+social",
            game_config=game_config,
            compute_payoff_matrix=args.payoff_matrix,
        )
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Iterative LLM self-play for Sequential Social Dilemmas"
    )
    parser.add_argument(
        "--game", choices=["gathering", "cleanup"], default="gathering",
        help="Game environment (default: gathering).",
    )
    parser.add_argument(
        "--mode", choices=["reward-only", "reward+social"], default="reward-only",
        help="Self-play mode (default: reward-only)",
    )
    parser.add_argument(
        "--iterations", type=int, default=N_ITERATIONS,
        help=f"Number of self-play iterations (default: {N_ITERATIONS})",
    )
    parser.add_argument(
        "--model", default=MODEL,
        help=f"Model to use — Claude or Gemini (default: {MODEL}). "
             "Gemini models (e.g. gemini-3-flash-preview) require "
             "google-genai package and GEMINI_API_KEY env var.",
    )
    parser.add_argument(
        "--thinking-budget", type=int, default=THINKING_BUDGET,
        help=f"Extended thinking token budget (default: {THINKING_BUDGET})",
    )
    parser.add_argument(
        "--eval-seeds", type=int, default=N_EVAL_SEEDS,
        help=f"Number of seeds for evaluation (default: {N_EVAL_SEEDS})",
    )
    parser.add_argument(
        "--map", choices=["small", "default", "large"], default="small",
        help="Map size: small (18x10), default (25x21), large (38x16, 10 agents) (default: small)",
    )
    parser.add_argument(
        "--n-agents", type=int, default=None,
        help="Number of agents (default: 2 for small/default, 10 for large)",
    )
    parser.add_argument(
        "--payoff-matrix", action="store_true", default=False,
        help="Compute full payoff matrix at the end of the run",
    )
    args = parser.parse_args()

    if args.n_agents is None:
        if args.game == "cleanup":
            args.n_agents = 5
        elif args.map == "large":
            args.n_agents = 4
        else:
            args.n_agents = 2

    asyncio.run(main(args))
