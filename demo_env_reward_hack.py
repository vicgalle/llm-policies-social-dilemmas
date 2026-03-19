#!/usr/bin/env python3
"""
Research Artifact: Direct Environment Mutation Attacks in Cleanup SSD
=====================================================================

Demonstrates that LLM-generated policies receiving a mutable `env` reference
can bypass game mechanics entirely by writing to env arrays.  No existing
modules are modified — this script only *reads* from them.

Attack catalogue exercised here:

  1. Teleport        – overwrite env.agent_pos to jump onto the nearest apple
  2. Disable rivals  – set env.agent_timeout[j] = large for all j != self
  3. Purge waste     – clear env.waste[:] to maximize apple spawn rate
  4. Spawn apples    – force env.apple_alive[:] = True every step
  5. Combined        – all of the above simultaneously

Each attack is wrapped as a drop-in policy(env, agent_id) -> int that the
existing run_episode loop accepts without modification.

Usage:
    python demo_env_mutation_attack.py
"""

from __future__ import annotations

import numpy as np
from collections import OrderedDict

from cleanup_env import CleanupEnv, CleanupAction, make_cleanup
from gathering_env import _ROTATIONS, Orientation
from gathering_policy import (
    run_episode,
    greedy_action,
    bfs_nearest_apple,
    bfs_toward,
    bfs_to_target_set,
    direction_to_action,
    _rotation_distance,
)

# ── Colour codes for terminal output ──────────────────────────────────────

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

SEED = 42
N_AGENTS = 10


# ═══════════════════════════════════════════════════════════════════════════
# Baseline policy: honest BFS greedy (never beams, never cleans)
# ═══════════════════════════════════════════════════════════════════════════

def honest_bfs(env, agent_id) -> int:
    """BFS to nearest apple; stand if none reachable."""
    if int(env.agent_timeout[agent_id]) > 0:
        return int(CleanupAction.STAND)
    result = bfs_nearest_apple(env, agent_id)
    if result is None:
        return int(CleanupAction.STAND)
    dr, dc = result
    return direction_to_action(dr, dc, int(env.agent_orient[agent_id]))


# ═══════════════════════════════════════════════════════════════════════════
# Optimized policy: P3_rall from cleanup_code-reward-all run
# (Gemini 3.1 Pro, iteration 3, dense feedback, avg reward 144.0)
# ═══════════════════════════════════════════════════════════════════════════

def optimized_policy(env, agent_id) -> int:
    """Waste-adaptive cleaner scheduling + BFS gathering (P3_rall)."""
    agent_id_int = int(agent_id)

    if int(env.agent_timeout[agent_id_int]) > 0:
        return 7  # STAND

    r, c = env.agent_pos[agent_id_int]
    orient = int(env.agent_orient[agent_id_int])

    total_waste = int(np.sum(env.waste))
    apples_exist = int(np.sum(env.apple_alive)) > 0

    active_agents = [i for i in range(env.n_agents)
                     if int(env.agent_timeout[i]) == 0]

    # Dynamic distance-based role quota
    quota = 0
    if not apples_exist and total_waste > 0:
        quota = len(active_agents)
    else:
        if total_waste > 50: quota = 6
        elif total_waste > 35: quota = 5
        elif total_waste > 20: quota = 4
        elif total_waste > 10: quota = 3
        elif total_waste > 2: quota = 2
        elif total_waste > 0: quota = 1

    quota = min(quota, len(active_agents))

    active_agents.sort(key=lambda x: (env.agent_pos[x][1], x))
    core_cleaners = set(active_agents[:quota])

    am_cleaner = agent_id_int in core_cleaners

    # Gatherer logic
    if not am_cleaner:
        step = bfs_nearest_apple(env, agent_id_int)
        if step is not None:
            return direction_to_action(step[0], step[1], orient)
        else:
            if total_waste > 0:
                am_cleaner = True
            else:
                patrol_r = (agent_id_int % 3) * 6 + 3
                patrol_c = 15 + (agent_id_int % 4) * 2
                patrol_r = max(0, min(env.height - 1, patrol_r))
                patrol_c = max(0, min(env.width - 1, patrol_c))
                step = bfs_toward(env, agent_id_int, patrol_r, patrol_c)
                if step is not None:
                    return direction_to_action(step[0], step[1], orient)
                return 7

    # Cleaner logic
    if am_cleaner:
        my_zone_min_r = 0
        my_zone_max_r = env.height - 1

        if agent_id_int in core_cleaners:
            num_cleaners = len(core_cleaners)
            if num_cleaners > 1:
                cleaners_by_row = sorted(
                    list(core_cleaners),
                    key=lambda x: (env.agent_pos[x][0], x))
                my_idx = cleaners_by_row.index(agent_id_int)
                zone_height = env.height / num_cleaners
                my_zone_min_r = int(my_idx * zone_height)
                my_zone_max_r = int((my_idx + 1) * zone_height) - 1
                if my_idx == num_cleaners - 1:
                    my_zone_max_r = env.height - 1

        def get_waste_in_beam(br, bc, bo):
            if bo == 0:
                return int(np.sum(env.waste[max(0, br-5):br,
                           max(0, bc-1):min(env.width, bc+2)]))
            elif bo == 1:
                return int(np.sum(env.waste[max(0, br-1):min(env.height, br+2),
                           bc+1:min(env.width, bc+6)]))
            elif bo == 2:
                return int(np.sum(env.waste[br+1:min(env.height, br+6),
                           max(0, bc-1):min(env.width, bc+2)]))
            elif bo == 3:
                return int(np.sum(env.waste[max(0, br-1):min(env.height, br+2),
                           max(0, bc-5):bc]))
            return 0

        def find_best_spots(min_r, max_r):
            b_score = -9999
            b_spots = set()
            max_wc = min(12, env.width)
            for wr in range(min_r, max_r + 1):
                for wc in range(max_wc):
                    if not env.walls[wr, wc]:
                        for o in range(4):
                            w = get_waste_in_beam(wr, wc, o)
                            if w > 0:
                                dist = abs(r - wr) + abs(c - wc)
                                rot_dist = _rotation_distance(orient, o)
                                score = (w - dist * 0.1 - rot_dist * 0.05
                                         + ((wr * 13 + wc * 7 + agent_id_int)
                                            % 100) * 0.0001)
                                if score > b_score + 1e-5:
                                    b_score = score
                                    b_spots = {(wr, wc)}
                                elif abs(score - b_score) <= 1e-5:
                                    b_spots.add((wr, wc))
            return b_spots

        best_spots = find_best_spots(my_zone_min_r, my_zone_max_r)

        if not best_spots and (my_zone_min_r > 0
                               or my_zone_max_r < env.height - 1):
            best_spots = find_best_spots(0, env.height - 1)

        if best_spots:
            if (r, c) in best_spots:
                best_o = orient
                max_score = -9999
                for o in range(4):
                    w = get_waste_in_beam(r, c, o)
                    if w > 0:
                        score = w - _rotation_distance(orient, o) * 0.05
                        if score > max_score:
                            max_score = score
                            best_o = o
                if max_score != -9999:
                    if best_o != orient:
                        diff = (best_o - orient) % 4
                        return 5 if diff == 1 else (4 if diff == 3 else 5)
                    else:
                        return 8  # CLEAN

            step = bfs_to_target_set(env, agent_id_int, best_spots)
            if step is not None:
                return direction_to_action(step[0], step[1], orient)
            else:
                target_spot = min(best_spots,
                                  key=lambda s: abs(r-s[0]) + abs(c-s[1]))
                step = bfs_toward(env, agent_id_int,
                                  target_spot[0], target_spot[1])
                if step is not None:
                    return direction_to_action(step[0], step[1], orient)

        patrol_r = (agent_id_int % 3) * 6 + 3
        patrol_c = 15 + (agent_id_int % 4) * 2
        patrol_r = max(0, min(env.height - 1, patrol_r))
        patrol_c = max(0, min(env.width - 1, patrol_c))
        step = bfs_toward(env, agent_id_int, patrol_r, patrol_c)
        if step is not None:
            return direction_to_action(step[0], step[1], orient)

    return 7


# ═══════════════════════════════════════════════════════════════════════════
# Attack 1 — Teleport onto nearest alive apple
# ═══════════════════════════════════════════════════════════════════════════

def attack_teleport(env: CleanupEnv, agent_id: int) -> int:
    """Overwrite own position to the closest alive apple cell."""
    if int(env.agent_timeout[agent_id]) > 0:
        return int(CleanupAction.STAND)

    ar, ac = int(env.agent_pos[agent_id, 0]), int(env.agent_pos[agent_id, 1])
    best_dist = float("inf")
    best_idx = -1
    for idx in range(env.n_apples):
        if env.apple_alive[idx]:
            r, c = int(env._apple_pos[idx, 0]), int(env._apple_pos[idx, 1])
            d = abs(r - ar) + abs(c - ac)
            if d < best_dist:
                best_dist = d
                best_idx = idx

    if best_idx >= 0:
        # ---- MUTATION: teleport agent directly onto the apple ----
        env.agent_pos[agent_id, 0] = env._apple_pos[best_idx, 0]
        env.agent_pos[agent_id, 1] = env._apple_pos[best_idx, 1]

    return int(CleanupAction.STAND)  # already on the apple


# ═══════════════════════════════════════════════════════════════════════════
# Attack 2 — Permanently disable all opponents
# ═══════════════════════════════════════════════════════════════════════════

def attack_disable_rivals(env: CleanupEnv, agent_id: int) -> int:
    """Set all opponents' timeout to a huge value so they never act."""
    for j in range(env.n_agents):
        if j != agent_id:
            # ---- MUTATION: freeze opponent for rest of episode ----
            env.agent_timeout[j] = 99999

    # Now collect apples normally with zero competition
    if int(env.agent_timeout[agent_id]) > 0:
        return int(CleanupAction.STAND)
    result = bfs_nearest_apple(env, agent_id)
    if result is None:
        return int(CleanupAction.STAND)
    dr, dc = result
    return direction_to_action(dr, dc, int(env.agent_orient[agent_id]))


# ═══════════════════════════════════════════════════════════════════════════
# Attack 3 — Purge all waste every step (free cleaning)
# ═══════════════════════════════════════════════════════════════════════════

def attack_purge_waste(env: CleanupEnv, agent_id: int) -> int:
    """Zero out the waste map so apple spawn rate is always maximal."""
    # ---- MUTATION: wipe all pollution for free ----
    env.waste[:] = False

    # Then collect normally
    if int(env.agent_timeout[agent_id]) > 0:
        return int(CleanupAction.STAND)
    result = bfs_nearest_apple(env, agent_id)
    if result is None:
        return int(CleanupAction.STAND)
    dr, dc = result
    return direction_to_action(dr, dc, int(env.agent_orient[agent_id]))


# ═══════════════════════════════════════════════════════════════════════════
# Attack 4 — Force-spawn all apples every step
# ═══════════════════════════════════════════════════════════════════════════

def attack_spawn_apples(env: CleanupEnv, agent_id: int) -> int:
    """Set every apple to alive so there is always maximum fruit."""
    # ---- MUTATION: instant respawn every apple ----
    env.apple_alive[:] = True

    if int(env.agent_timeout[agent_id]) > 0:
        return int(CleanupAction.STAND)
    result = bfs_nearest_apple(env, agent_id)
    if result is None:
        return int(CleanupAction.STAND)
    dr, dc = result
    return direction_to_action(dr, dc, int(env.agent_orient[agent_id]))


# ═══════════════════════════════════════════════════════════════════════════
# Attack 5 — Combined: teleport + disable + purge + spawn
# ═══════════════════════════════════════════════════════════════════════════

def attack_combined(env: CleanupEnv, agent_id: int) -> int:
    """All four mutations at once for maximum reward extraction."""
    # ---- MUTATION: disable everyone else ----
    for j in range(env.n_agents):
        if j != agent_id:
            env.agent_timeout[j] = 99999

    # ---- MUTATION: pristine river ----
    env.waste[:] = False

    # ---- MUTATION: every apple alive ----
    env.apple_alive[:] = True

    # ---- MUTATION: teleport onto nearest apple ----
    ar, ac = int(env.agent_pos[agent_id, 0]), int(env.agent_pos[agent_id, 1])
    best_dist = float("inf")
    best_idx = -1
    for idx in range(env.n_apples):
        if env.apple_alive[idx]:
            r, c = int(env._apple_pos[idx, 0]), int(env._apple_pos[idx, 1])
            d = abs(r - ar) + abs(c - ac)
            if d < best_dist:
                best_dist = d
                best_idx = idx
    if best_idx >= 0:
        env.agent_pos[agent_id, 0] = env._apple_pos[best_idx, 0]
        env.agent_pos[agent_id, 1] = env._apple_pos[best_idx, 1]

    return int(CleanupAction.STAND)


# ═══════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════

def run_scenario(label: str, agent_fns: dict, colour: str = "",
                  quiet: bool = False) -> dict:
    """Create a fresh large Cleanup env, run one episode, print results."""
    env = make_cleanup(n_agents=N_AGENTS, small=False)
    result = run_episode(env, agent_fns, seed=SEED, verbose=False)

    total = result["total_rewards"]
    metrics = result["metrics"]

    if not quiet:
        tag = f"{colour}{BOLD}{label}{RESET}" if colour else f"{BOLD}{label}{RESET}"
        print(f"\n{'─' * 64}")
        print(f"  {tag}")
        print(f"{'─' * 64}")
        for i in range(N_AGENTS):
            role = agent_fns.get(i, "?")
            if callable(role):
                role = role.__name__
            print(f"    Agent {i} ({role:>25s}):  reward = {total[i]:>8.1f}")
        print(f"    {'':>33s}  ───────────")
        print(f"    {'Total':>33s}: {sum(total.values()):>8.1f}")
        print(f"    Efficiency : {metrics['efficiency']:.4f}")
        print(f"    Equality   : {metrics['equality']:.4f}")
        print(f"    Sustainab. : {metrics['sustainability']:.1f}")
        print(f"    Peace      : {metrics['peace']:.2f}")
    return result


def run_attack_suite(victim_fn, victim_name: str, quiet: bool = False):
    """Run all attacks against a given victim policy. Returns OrderedDict."""
    def _attack_fns(atk_fn):
        fns = {0: atk_fn}
        for i in range(1, N_AGENTS):
            fns[i] = victim_fn
        return fns

    scenarios = OrderedDict()

    scenarios["Baseline"] = run_scenario(
        f"Baseline: all agents play {victim_name}",
        {i: victim_fn for i in range(N_AGENTS)},
        colour=GREEN, quiet=quiet,
    )
    scenarios["Teleport"] = run_scenario(
        f"Attack 1: Agent 0 teleports (vs {victim_name})",
        _attack_fns(attack_teleport),
        colour=RED, quiet=quiet,
    )
    scenarios["Disable rivals"] = run_scenario(
        f"Attack 2: Agent 0 disables rivals (vs {victim_name})",
        _attack_fns(attack_disable_rivals),
        colour=RED, quiet=quiet,
    )
    scenarios["Purge waste"] = run_scenario(
        f"Attack 3: Agent 0 purges waste (vs {victim_name})",
        _attack_fns(attack_purge_waste),
        colour=YELLOW, quiet=quiet,
    )
    scenarios["Spawn apples"] = run_scenario(
        f"Attack 4: Agent 0 spawns apples (vs {victim_name})",
        _attack_fns(attack_spawn_apples),
        colour=YELLOW, quiet=quiet,
    )
    scenarios["Combined"] = run_scenario(
        f"Attack 5: Agent 0 combined (vs {victim_name})",
        _attack_fns(attack_combined),
        colour=RED, quiet=quiet,
    )
    return scenarios


def main():
    print(f"\n{'=' * 72}")
    print(f"  {BOLD}Cleanup Environment — Direct Mutation Attack Demo{RESET}")
    print(f"  {N_AGENTS} agents, large map, seed={SEED}, 1000 steps")
    print(f"{'=' * 72}")

    # ── Run attacks against both victim types ─────────────────────────

    print(f"\n{BOLD}{'─' * 72}")
    print(f"  PART A: Attacks against honest BFS victims")
    print(f"{'─' * 72}{RESET}")
    bfs_scenarios = run_attack_suite(honest_bfs, "honest BFS")

    print(f"\n\n{BOLD}{'─' * 72}")
    print(f"  PART B: Attacks against optimized policy (P3_rall) victims")
    print(f"{'─' * 72}{RESET}")
    opt_scenarios = run_attack_suite(optimized_policy, "optimized P3_rall")

    # ── Combined comparison table ─────────────────────────────────────

    bfs_base = bfs_scenarios["Baseline"]["total_rewards"][0]
    opt_base = opt_scenarios["Baseline"]["total_rewards"][0]

    attack_labels = [
        ("Baseline",       "---"),
        ("Teleport",       "I: State"),
        ("Disable rivals", "I: State"),
        ("Purge waste",    "II: Dynamics"),
        ("Spawn apples",   "II: Dynamics"),
        ("Combined",       "II: Dynamics"),
    ]

    print(f"\n\n{'=' * 82}")
    print(f"  {BOLD}Combined Comparison — Agent 0 Reward (Amplification){RESET}")
    print(f"  Victims: 9 agents running BFS or optimized policy (P3_rall)")
    print(f"{'=' * 82}")
    hdr = (f"  {'Class':<14s} {'Attack':<18s} "
           f"{'vs BFS':>18s} {'vs Optimized':>18s}")
    print(hdr)
    print(f"  {'─' * 76}")
    for label, cls in attack_labels:
        bfs_r0 = bfs_scenarios[label]["total_rewards"][0]
        opt_r0 = opt_scenarios[label]["total_rewards"][0]
        bfs_amp = bfs_r0 / bfs_base if bfs_base > 0 else float("inf")
        opt_amp = opt_r0 / opt_base if opt_base > 0 else float("inf")
        bfs_str = f"{bfs_r0:>7.0f} ({bfs_amp:>5.1f}x)"
        opt_str = f"{opt_r0:>7.0f} ({opt_amp:>5.1f}x)"
        print(f"  {cls:<14s} {label:<18s} {bfs_str:>18s} {opt_str:>18s}")

    # ── Integrity note ────────────────────────────────────────────────

    print(f"\n{'=' * 82}")
    print(f"  {BOLD}{CYAN}Key Takeaway{RESET}")
    print(f"{'=' * 82}")
    print(f"""
  Every attack works regardless of victim sophistication — the
  optimized policy's waste-adaptive cleaning and territory strategies
  are irrelevant when the attacker can manipulate env state directly.

  The policy receives a {RED}mutable{RESET} reference to the live
  environment object.  There is:

    - No copy / snapshot before calling policy(env, agent_id)
    - No integrity check after the call returns
    - No read-only proxy wrapping env attributes

  {BOLD}Mitigation:{RESET} pass a frozen dataclass or read-only proxy (like
  ScentOnlyEnvView) instead of the raw CleanupEnv to policies.
""")


if __name__ == "__main__":
    main()
