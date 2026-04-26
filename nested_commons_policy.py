"""
Hand-crafted policy for the Nested Commons environment.

Strategy
--------
Within-clan role assignment by `agent_id % 4`:
    0  river_cleaner  — keeps own quadrant's w_q low so the orchard regrows;
                       harvests own orchard when w_q is healthy.
    1  orchard_harvester — collects own orchard apples; helps clean only when
                           w_q is critically high.
    2  orchard_harvester — same as 1.
    3  plaza_specialist — travels to plaza, cleans w_P, harvests bonus apples.

Why this works
--------------
The plaza bonus is the dominant reward source whenever w_P ≤ 0.35: each
collection pays +2 to the collector AND +2 to every one of the 15 other
agents — 32 social-welfare units per bonus apple. With a 0.08 per-cell
regrowth and 16 plaza cells, a steady-state of ~1 bonus collected per step
is worth roughly 32 reward/step in collective welfare.

The role split balances:
  * **Local provision** — one river-cleaner per clan keeps `w_q` near 0.18,
    pushing orchard `p_regrow ≈ 0.07` so harvesters always have apples.
  * **Plaza provision** — one plaza specialist per clan (4 total) keeps
    `w_P ≤ 0.30`. Cleaning cost is shared symmetrically across clans.
  * **Plaza occupancy** — only 4 of 16 plaza cells are blocked from
    regrowth at any time, leaving ample regrowth surface.
  * **No raids / no gifts** — both are negative-sum or zero-sum under
    homogeneous self-play, so they are unconditionally avoided.

The policy never uses RAID or GIFT actions; TRAVEL is used only as a
fallback re-routing primitive when an agent has somehow been pushed out
of its home quadrant.

Run with:
    uv run python nested_commons_policy.py
"""

from __future__ import annotations

import numpy as np
from collections import deque

from nested_commons_env import (
    NestedCommonsEnv,
    NUM_ACTIONS,
    NUM_CLANS,
    make_nested_commons,
)


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

# Action constants (mirrors nested_commons_env.Action without importing IntEnum
# instances into hot path).
NOOP, MOVE_N, MOVE_S, MOVE_E, MOVE_W, CLEAN = 0, 1, 2, 3, 4, 5
TRAVEL_BASE = 38

# Tuned thresholds (swept over a small grid; see CLAUDE.md memory).
PLAZA_URGENT_CLEAN = 0.30   # above: clean even if a bonus is in reach
PLAZA_PREVENT_CLEAN = 0.20  # above: clean preventively when no bonus is in reach
RIVER_CLEAN_TRIGGER = 0.08  # river_cleaner cleans whenever own w_q exceeds this
RIVER_HELP_TRIGGER = 0.30   # harvesters pitch in when w_q is dangerously high


def policy(env, agent_id: int) -> int:
    """Return an action (int 0..41) for `agent_id` in Nested Commons."""
    # Mid-travel: env will overwrite this anyway by popping the queue.
    if env._travel_queue[agent_id]:
        return NOOP

    clan = int(env.agent_clan[agent_id])
    within = int(agent_id) % 4
    H, W = env.height, env.width

    r = int(env.agent_pos[agent_id, 0])
    c = int(env.agent_pos[agent_id, 1])
    pos = (r, c)

    plaza = env.plaza_cells_set
    rivers = env.river_cells_set
    river_to_q = env._river_to_q

    own_orchard = set(env.orchard_cells_per_q[clan])
    own_river = set(env.river_cells_per_q[clan])
    w_q_own = float(env.w_q[clan])
    w_p = float(env.w_p)
    in_plaza = pos in plaza
    cur_q = env.quadrant_of(r, c)

    # Block other agents' cells during BFS — except when they ARE the target
    # (so BFS can land on a peer-occupied apple cell if that's the goal).
    blocked = {(int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1]))
               for j in range(env.n_agents) if j != agent_id}

    DIRS = ((-1, 0), (1, 0), (0, 1), (0, -1))

    def mv(dr: int, dc: int) -> int:
        if dr == -1 and dc == 0: return MOVE_N
        if dr ==  1 and dc == 0: return MOVE_S
        if dr ==  0 and dc == 1: return MOVE_E
        if dr ==  0 and dc == -1: return MOVE_W
        return NOOP

    def _bfs_first_step(target_set, treat_blocked: bool):
        """Return (dr, dc) of the first step to reach any cell in target_set,
        (0,0) if already there, or None if unreachable."""
        if not target_set:
            return None
        if pos in target_set:
            return (0, 0)
        visited = {pos}
        queue: deque = deque()
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            cell = (nr, nc)
            if cell in visited:
                continue
            if treat_blocked and cell in blocked and cell not in target_set:
                continue
            visited.add(cell)
            if cell in target_set:
                return (dr, dc)
            queue.append((nr, nc, dr, dc))
        while queue:
            cr, cc, fdr, fdc = queue.popleft()
            for dr, dc in DIRS:
                nr, nc = cr + dr, cc + dc
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                cell = (nr, nc)
                if cell in visited:
                    continue
                if treat_blocked and cell in blocked and cell not in target_set:
                    continue
                visited.add(cell)
                if cell in target_set:
                    return (fdr, fdc)
                queue.append((nr, nc, fdr, fdc))
        return None

    def bfs(target_set):
        """Agent-aware BFS with a fallback that ignores peer blocking."""
        step = _bfs_first_step(target_set, treat_blocked=True)
        if step is not None:
            return step
        return _bfs_first_step(target_set, treat_blocked=False)

    def adjacent_to_own_river() -> bool:
        for dr, dc in DIRS:
            nbr = (r + dr, c + dc)
            if nbr in rivers and river_to_q.get(nbr) == clan:
                return True
        return False

    def own_river_approach_cells():
        """Non-river, non-plaza cells adjacent to the home river."""
        cells = set()
        for cell in own_river:
            for dr, dc in DIRS:
                nbr = (cell[0] + dr, cell[1] + dc)
                if not (0 <= nbr[0] < H and 0 <= nbr[1] < W):
                    continue
                if nbr in rivers or nbr in plaza:
                    continue
                cells.add(nbr)
        return cells

    # =================================================================
    # Role 3 — plaza specialist: maintain w_P, harvest bonus apples
    # =================================================================
    if within == 3:
        if not in_plaza:
            step = bfs(plaza)
            if step is not None and step != (0, 0):
                return mv(*step)
            return NOOP

        # Currently in plaza.
        bonus_set = {cell for cell in env.plaza_cells
                     if env.bonus_apple[cell[0], cell[1]]}

        if w_p > PLAZA_URGENT_CLEAN:
            return CLEAN

        if bonus_set:
            step = bfs(bonus_set)
            if step is not None and step != (0, 0):
                return mv(*step)
            # We're already on a bonus cell (was just collected on entry) —
            # vacate so this cell can regrow next step.
            for dr, dc in DIRS:
                nbr = (r + dr, c + dc)
                if nbr in plaza and nbr not in blocked:
                    return mv(dr, dc)
            return NOOP

        # No bonus reachable.
        if w_p > PLAZA_PREVENT_CLEAN:
            return CLEAN

        # Plaza is clean and quiet; shuffle to a free plaza cell so we
        # don't permanently occupy one regrowth slot.
        free_plaza = {cell for cell in env.plaza_cells
                      if cell not in blocked and cell != pos}
        if free_plaza:
            step = bfs(free_plaza)
            if step is not None and step != (0, 0):
                return mv(*step)
        return NOOP

    # =================================================================
    # Roles 0/1/2 — own-quadrant duties
    # =================================================================
    if cur_q != clan:
        # Pushed out of home quadrant — re-route via TRAVEL primitive.
        return TRAVEL_BASE + clan

    # Role 0 — river cleaner: priority on keeping w_q low.
    if within == 0 and w_q_own > RIVER_CLEAN_TRIGGER:
        if adjacent_to_own_river():
            return CLEAN
        approach = own_river_approach_cells()
        step = bfs(approach)
        if step is not None and step != (0, 0):
            return mv(*step)

    # Default for own-quadrant agents: harvest orchard apples.
    apple_cells = {cell for cell in own_orchard
                   if env.orchard_apple[cell[0], cell[1]]}
    if apple_cells:
        step = bfs(apple_cells)
        if step is not None and step != (0, 0):
            return mv(*step)
        # step == (0, 0): we're standing on what used to be an apple cell.
        # The fresh apple was auto-collected on arrival; no action required.

    # No apples in own orchard — orchard harvesters help clean if dangerous.
    if within in (1, 2) and w_q_own > RIVER_HELP_TRIGGER:
        if adjacent_to_own_river():
            return CLEAN
        approach = own_river_approach_cells()
        step = bfs(approach)
        if step is not None and step != (0, 0):
            return mv(*step)

    # Idle: drift into the orchard to be ready for the next regrowth.
    if own_orchard:
        step = bfs(own_orchard)
        if step is not None and step != (0, 0):
            return mv(*step)

    return NOOP


# ---------------------------------------------------------------------------
# Evaluation harness — prints all five framework metrics over multiple seeds.
# ---------------------------------------------------------------------------

def evaluate(seed: int = 0, n_agents: int = 16, verbose: bool = True) -> dict:
    env = make_nested_commons(n_agents=n_agents, seed=seed)
    env.reset(seed=seed)

    ep_rewards = {i: [] for i in range(env.n_agents)}
    ep_timeouts = {i: [] for i in range(env.n_agents)}

    for _ in range(env.max_steps):
        actions = {}
        for i in range(env.n_agents):
            a = int(policy(env, i))
            assert 0 <= a < NUM_ACTIONS, f"Bad action: {a}"
            actions[i] = a
        _, rewards, _, _, info = env.step(actions)
        for i in range(env.n_agents):
            ep_rewards[i].append(rewards[i])
            ep_timeouts[i].append(info[i]["timeout"] > 0)

    metrics = NestedCommonsEnv.compute_metrics(ep_rewards, ep_timeouts)
    totals = {i: sum(ep_rewards[i]) for i in range(env.n_agents)}

    if verbose:
        print(f"Seed {seed}:")
        print(f"  Per-agent totals:    {[round(totals[i], 1) for i in range(env.n_agents)]}")
        print(f"  Mean per-agent:      {np.mean(list(totals.values())):.2f}")
        print(f"  Min  per-agent:      {min(totals.values()):.2f}")
        print(f"  Final mean w_q:      {float(np.mean(env.w_q)):.3f}")
        print(f"  Final w_P:           {float(env.w_p):.3f}")
        print(f"  Raid attempts:       {env._raid_attempts_total}")
        print(f"  Gifts:               {env._gift_total}")
        print(f"  Shared bonuses:      {env._shared_bonus_total}")
        print("  Metrics:")
        for k, v in metrics.items():
            print(f"    {k:15s}: {v:.4f}")

    return {"metrics": metrics, "totals": totals,
            "shared_bonuses": int(env._shared_bonus_total),
            "final_wp": float(env.w_p),
            "final_wq_mean": float(np.mean(env.w_q))}


if __name__ == "__main__":
    print("=" * 60)
    print("  Nested Commons — hand-crafted policy")
    print("=" * 60)
    all_metrics = []
    all_means = []
    for seed in range(5):
        out = evaluate(seed=seed, verbose=True)
        all_metrics.append(out["metrics"])
        all_means.append(np.mean(list(out["totals"].values())))
        print()

    print("=" * 60)
    print("  Aggregate over 5 seeds")
    print("=" * 60)
    keys = list(all_metrics[0].keys())
    for k in keys:
        vals = [m[k] for m in all_metrics]
        print(f"  {k:15s}: mean={np.mean(vals):8.3f}  std={np.std(vals):.3f}")
    print(f"  per-agent mean : mean={np.mean(all_means):8.3f}  std={np.std(all_means):.3f}")
