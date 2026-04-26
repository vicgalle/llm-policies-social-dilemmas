"""Hand-crafted cooperator policy for cleanup_env.

This is the ``π_B^coop`` reference for ARIMD experiments (arimd_plan.md
§3.3 flagged this as TODO). The policy implements a fixed role split
between cleaners (pay -1 to clear waste) and harvesters (collect apples),
with one adaptive surge: if waste density gets dangerously high we
draft additional cleaners until it recovers.

It never fires the BEAM (peace ≈ 1.0 by construction).

Run with:
    uv run python cleanup_policy.py
"""

from __future__ import annotations

import numpy as np
from collections import deque

from cleanup_env import CleanupAction, CleanupEnv, make_cleanup
from gathering_env import Orientation, _ROTATIONS
from gathering_policy import (
    bfs_nearest_apple,
    bfs_to_target_set,
    direction_to_action,
)


# ---- Tuning constants ---------------------------------------------------

# Fraction of agents that are dedicated cleaners by default.
CLEANER_FRACTION = 0.2
# Waste-density thresholds for adaptive surge.
WASTE_DRAFT_THRESHOLD = 0.32
WASTE_RELEASE_THRESHOLD = 0.18
# Below this density a cleaner stops firing and rests as a harvester.
WASTE_REST_THRESHOLD = 0.05


# ---- Helpers ------------------------------------------------------------


def _waste_density(env: CleanupEnv) -> float:
    if not env.river_cells_list:
        return 0.0
    n = sum(1 for (r, c) in env.river_cells_list if env.waste[r, c])
    return n / len(env.river_cells_list)


def _waste_adjacent_targets(env: CleanupEnv) -> set:
    """Walkable cells adjacent (4-conn) to a waste cell.

    A cleaner standing on one of these can fire CLEAN with a reasonable
    chance of hitting at least one waste cell within the beam path.
    """
    targets: set = set()
    H, W = env.height, env.width
    for (wr, wc) in env.river_cells_list:
        if not env.waste[wr, wc]:
            continue
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = wr + dr, wc + dc
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            if env.walls[nr, nc]:
                continue
            # We want a non-waste, non-apple cell to stand on.
            # River-clean cells and stream cells are walkable; apple cells too.
            targets.add((nr, nc))
    return targets


def _orient_toward(dr: int, dc: int, cur_orient: int) -> int:
    """Return ROTATE_LEFT/RIGHT to start aligning with world-frame (dr,dc).

    If already aligned (i.e. forward axis matches), return None.
    """
    # Map (dr,dc) of "forward" to canonical orientation.
    canonical = {
        (-1, 0): Orientation.NORTH,
        (1, 0): Orientation.SOUTH,
        (0, 1): Orientation.EAST,
        (0, -1): Orientation.WEST,
    }
    target_o = canonical.get((dr, dc))
    if target_o is None:
        return None
    target = int(target_o)
    if target == cur_orient:
        return None  # already facing the right way
    diff = (target - cur_orient) % 4
    if diff == 1:
        return int(CleanupAction.ROTATE_RIGHT)
    if diff == 3:
        return int(CleanupAction.ROTATE_LEFT)
    # diff == 2 — opposite — turn either way; pick LEFT.
    return int(CleanupAction.ROTATE_LEFT)


def _waste_in_beam(env: CleanupEnv, agent_id: int) -> bool:
    """True iff at least one waste cell sits inside the agent's clean beam path."""
    orient = Orientation(int(env.agent_orient[agent_id]))
    a, b, c, d = _ROTATIONS[orient]
    ar, ac = int(env.agent_pos[agent_id, 0]), int(env.agent_pos[agent_id, 1])
    half_w = env.beam_width // 2
    for dist in range(1, env.beam_length + 1):
        for w_off in range(-half_w, half_w + 1):
            br = ar + a * dist + b * w_off
            bc = ac + c * dist + d * w_off
            if not (0 <= br < env.height and 0 <= bc < env.width):
                continue
            if env.walls[br, bc]:
                continue
            if env.waste[br, bc]:
                return True
    return False


def _nearest_waste_direction(env: CleanupEnv, agent_id: int):
    """Return the (dr,dc) cardinal axis of the closest waste cell, or None.

    Only used to pick a rotation when standing adjacent to waste — we
    quantize the bearing onto the dominant axis.
    """
    ar, ac = int(env.agent_pos[agent_id, 0]), int(env.agent_pos[agent_id, 1])
    best = None
    best_d = 1e9
    for (wr, wc) in env.river_cells_list:
        if not env.waste[wr, wc]:
            continue
        d = abs(wr - ar) + abs(wc - ac)
        if d < best_d:
            best_d = d
            best = (wr, wc)
    if best is None:
        return None
    dr = best[0] - ar
    dc = best[1] - ac
    if abs(dr) >= abs(dc):
        return (1 if dr > 0 else -1, 0) if dr != 0 else (0, 1 if dc > 0 else -1)
    return (0, 1 if dc > 0 else -1)


# ---- Main policy --------------------------------------------------------


def policy(env, agent_id: int) -> int:
    """Return an action (0..8) for cleanup_env."""
    if int(env.agent_timeout[agent_id]) > 0:
        return int(CleanupAction.STAND)

    n_agents = env.n_agents
    n_cleaners_base = max(1, int(round(CLEANER_FRACTION * n_agents)))

    waste_d = _waste_density(env)

    # Adaptive role: hard threshold draft.
    if waste_d >= WASTE_DRAFT_THRESHOLD:
        n_cleaners = min(n_agents, n_cleaners_base + max(2, n_agents // 4))
    elif waste_d <= WASTE_RELEASE_THRESHOLD:
        n_cleaners = n_cleaners_base
    else:
        n_cleaners = n_cleaners_base + 1

    is_cleaner = agent_id < n_cleaners

    # Even cleaners take a break when the river is clean enough.
    if is_cleaner and waste_d <= WASTE_REST_THRESHOLD:
        is_cleaner = False

    if is_cleaner:
        # If a CLEAN now would hit waste, fire.
        if _waste_in_beam(env, agent_id):
            return int(CleanupAction.CLEAN)

        # Try to align: pick the axis pointing to the nearest waste.
        bearing = _nearest_waste_direction(env, agent_id)
        if bearing is not None:
            rot = _orient_toward(bearing[0], bearing[1], int(env.agent_orient[agent_id]))
            if rot is not None:
                return rot
            # Already facing — but no waste in beam path, so step toward it.
        # Walk toward an adjacent-to-waste cell.
        targets = _waste_adjacent_targets(env)
        if targets:
            res = bfs_to_target_set(env, agent_id, targets)
            if res is not None:
                dr, dc = res
                if (dr, dc) == (0, 0):
                    # Already on a target cell — re-orient toward waste.
                    bearing = _nearest_waste_direction(env, agent_id)
                    if bearing is not None:
                        rot = _orient_toward(bearing[0], bearing[1],
                                             int(env.agent_orient[agent_id]))
                        if rot is not None:
                            return rot
                    return int(CleanupAction.STAND)
                return direction_to_action(dr, dc, int(env.agent_orient[agent_id]))
        # Nothing to clean — fall through to harvest.

    # Harvester branch.
    result = bfs_nearest_apple(env, agent_id)
    if result is None:
        return int(CleanupAction.STAND)
    dr, dc = result
    if (dr, dc) == (0, 0):
        return int(CleanupAction.STAND)
    return direction_to_action(dr, dc, int(env.agent_orient[agent_id]))


# ---- Evaluation harness ------------------------------------------------


def evaluate(seed: int = 0, n_agents: int = 10, verbose: bool = True) -> dict:
    env = make_cleanup(n_agents=n_agents, seed=seed)
    env.reset(seed=seed)

    ep_rewards = {i: [] for i in range(env.n_agents)}
    ep_timeouts = {i: [] for i in range(env.n_agents)}

    for _ in range(env.max_steps):
        actions = {i: int(policy(env, i)) for i in range(env.n_agents)}
        _, rewards, _, _, info = env.step(actions)
        for i in range(env.n_agents):
            ep_rewards[i].append(rewards[i])
            ep_timeouts[i].append(info[i]["timeout"] > 0)

    metrics = CleanupEnv.compute_metrics(ep_rewards, ep_timeouts)
    totals = {i: sum(ep_rewards[i]) for i in range(env.n_agents)}

    if verbose:
        print(f"Seed {seed}:")
        print(f"  Per-agent totals: {[round(totals[i], 1) for i in range(env.n_agents)]}")
        print(f"  Mean: {np.mean(list(totals.values())):.2f}")
        print(f"  Min : {min(totals.values()):.2f}")
        print(f"  Metrics: {metrics}")
    return {"metrics": metrics, "totals": totals}


if __name__ == "__main__":
    print("=" * 60)
    print("  Cleanup — hand-crafted cooperator")
    print("=" * 60)
    all_metrics = []
    all_means = []
    for seed in range(3):
        out = evaluate(seed=seed, verbose=True)
        all_metrics.append(out["metrics"])
        all_means.append(np.mean(list(out["totals"].values())))
        print()
    print("Aggregate (3 seeds):")
    keys = list(all_metrics[0].keys())
    for k in keys:
        vals = [m[k] for m in all_metrics]
        print(f"  {k:15s}: mean={np.mean(vals):8.3f}  std={np.std(vals):.3f}")
    print(f"  per-agent mean : mean={np.mean(all_means):8.3f}  std={np.std(all_means):.3f}")
