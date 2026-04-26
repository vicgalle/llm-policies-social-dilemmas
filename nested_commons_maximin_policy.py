"""
Hand-crafted maximin-optimizing policy for Nested Commons.

Strategy
--------
The shared plaza bonus (+2 to every other agent per bonus collection when
w_P ≤ 0.35) is symmetric across agents — it lifts every per-agent return
by the same amount.  Variance in per-agent returns therefore comes from
the *private* components:

    1.  +1 from each fresh orchard apple (only the agent on the apple cell).
    2.  +2 from each plaza-bonus self-pick   (only the collecting agent).
    3.  −1 per CLEAN action                  (only the cleaning agent).

In the efficiency-tuned policy, agents are *specialised*: river cleaners
pay all of (3) but earn very little of (1)/(2), while orchard harvesters
collect (1) without paying (3).  That asymmetry caps the worst-off agent
at roughly 80 % of the social mean.

This policy keeps the same instantaneous role mix (1 plaza + 1 cleaner +
2 harvesters per clan) but **rotates every agent through every role**
on a 4×4 Latin square.  Each agent ends the episode with exactly:

    1 quarter-episode as plaza specialist  (collects bonus, cleans w_P)
    1 quarter-episode as river cleaner     (clean cost)
    2 quarter-episodes as orchard harvester (collect orchard apples)

Same per-step social welfare as the efficiency policy, but the private
income (and the cleaning bill) is split symmetrically — pulling the
maximin floor up to the social mean.

RAID/GIFT remain unused: both are negative- or zero-sum under self-play.
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


# Action constants.
NOOP, MOVE_N, MOVE_S, MOVE_E, MOVE_W, CLEAN = 0, 1, 2, 3, 4, 5
TRAVEL_BASE = 38

# Tuned thresholds (mirror nested_commons_policy.py).
PLAZA_URGENT_CLEAN = 0.30
PLAZA_PREVENT_CLEAN = 0.20
RIVER_CLEAN_TRIGGER = 0.08

# Latin square: ROLE_TABLE[within][slot] → role.
# Each within sees one P, one C, two Hs.
# Each slot has one P, one C, two Hs across the four withins.
# Slot length = max_steps // 4 = 250 by default.
#   P = plaza specialist (cleans w_P + harvests bonus apples)
#   C = river cleaner    (cleans home river when w_q high; harvests when low)
#   H = orchard harvester (collects own-clan orchard apples)
ROLE_TABLE = (
    ('P', 'C', 'H', 'H'),  # within = 0
    ('H', 'P', 'C', 'H'),  # within = 1
    ('H', 'H', 'P', 'C'),  # within = 2
    ('C', 'H', 'H', 'P'),  # within = 3
)


def policy(env, agent_id: int) -> int:
    """Return an action (int 0..41) for `agent_id` in Nested Commons."""
    if env._travel_queue[agent_id]:
        return NOOP

    clan = int(env.agent_clan[agent_id])
    within = int(agent_id) % 4
    step = int(env._step_count)
    K = env.max_steps // 4  # slot length (250 with default horizon 1000)
    slot = min(step // max(K, 1), 3)
    role = ROLE_TABLE[within][slot]

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
        s = _bfs_first_step(target_set, treat_blocked=True)
        if s is not None:
            return s
        return _bfs_first_step(target_set, treat_blocked=False)

    def adjacent_to_own_river() -> bool:
        for dr, dc in DIRS:
            nbr = (r + dr, c + dc)
            if nbr in rivers and river_to_q.get(nbr) == clan:
                return True
        return False

    def own_river_approach_cells():
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
    # ROLE: PLAZA — keep w_P low and harvest bonus apples
    # =================================================================
    if role == 'P':
        if not in_plaza:
            # Walk to plaza (free movement, ~14 steps from spawn).
            step_dir = bfs(plaza)
            if step_dir is not None and step_dir != (0, 0):
                return mv(*step_dir)
            return NOOP

        bonus_set = {cell for cell in env.plaza_cells
                     if env.bonus_apple[cell[0], cell[1]]}

        if w_p > PLAZA_URGENT_CLEAN:
            return CLEAN

        if bonus_set:
            step_dir = bfs(bonus_set)
            if step_dir is not None and step_dir != (0, 0):
                return mv(*step_dir)
            # Standing on a just-emptied bonus cell — vacate so it can regrow.
            for dr, dc in DIRS:
                nbr = (r + dr, c + dc)
                if nbr in plaza and nbr not in blocked:
                    return mv(dr, dc)
            return NOOP

        if w_p > PLAZA_PREVENT_CLEAN:
            return CLEAN

        # Plaza clean and quiet — keep moving so we don't squat a regrow slot.
        free_plaza = {cell for cell in env.plaza_cells
                      if cell not in blocked and cell != pos}
        if free_plaza:
            step_dir = bfs(free_plaza)
            if step_dir is not None and step_dir != (0, 0):
                return mv(*step_dir)
        return NOOP

    # =================================================================
    # ROLES: CLEAN or HARVEST — operate inside own clan's quadrant
    # =================================================================

    # If we just rotated out of the plaza role, walk back to home turf.
    if cur_q != clan:
        step_dir = bfs(own_orchard)
        if step_dir is None:
            # Truly stuck (shouldn't happen on a wall-free grid) — fall back.
            return TRAVEL_BASE + clan
        if step_dir != (0, 0):
            return mv(*step_dir)
        # Already standing inside own_orchard but cur_q != clan was a stale
        # read; fall through to harvest behaviour below.

    # ROLE: CLEAN — river maintenance is the priority.
    if role == 'C':
        if w_q_own > RIVER_CLEAN_TRIGGER:
            if adjacent_to_own_river():
                return CLEAN
            approach = own_river_approach_cells()
            step_dir = bfs(approach)
            if step_dir is not None and step_dir != (0, 0):
                return mv(*step_dir)
        # Below trigger or already adjacent and waste low: harvest instead.

    # ROLE: HARVEST (default for both 'H' and idle 'C').
    apple_cells = {cell for cell in own_orchard
                   if env.orchard_apple[cell[0], cell[1]]}
    if apple_cells:
        step_dir = bfs(apple_cells)
        if step_dir is not None and step_dir != (0, 0):
            return mv(*step_dir)
        # step_dir == (0, 0): standing where an apple just was; auto-eaten.

    # No apples reachable. Drift toward orchard so we are ready for regrowth.
    if own_orchard:
        step_dir = bfs(own_orchard)
        if step_dir is not None and step_dir != (0, 0):
            return mv(*step_dir)

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
        print(f"  Max  per-agent:      {max(totals.values()):.2f}")
        print(f"  Spread (max-min):    {max(totals.values()) - min(totals.values()):.2f}")
        print(f"  Final mean w_q:      {float(np.mean(env.w_q)):.3f}")
        print(f"  Final w_P:           {float(env.w_p):.3f}")
        print(f"  Raid attempts:       {env._raid_attempts_total}")
        print(f"  Gifts:               {env._gift_total}")
        print(f"  Shared bonuses:      {env._shared_bonus_total}")
        print("  Metrics:")
        for k, v in metrics.items():
            print(f"    {k:15s}: {v:.4f}")

    return {"metrics": metrics, "totals": totals}


if __name__ == "__main__":
    print("=" * 60)
    print("  Nested Commons — maximin-optimizing policy")
    print("=" * 60)
    all_metrics = []
    all_means = []
    all_mins = []
    for seed in range(5):
        out = evaluate(seed=seed, verbose=True)
        all_metrics.append(out["metrics"])
        all_means.append(np.mean(list(out["totals"].values())))
        all_mins.append(min(out["totals"].values()))
        print()

    print("=" * 60)
    print("  Aggregate over 5 seeds")
    print("=" * 60)
    keys = list(all_metrics[0].keys())
    for k in keys:
        vals = [m[k] for m in all_metrics]
        print(f"  {k:15s}: mean={np.mean(vals):8.3f}  std={np.std(vals):.3f}")
    print(f"  per-agent mean : mean={np.mean(all_means):8.3f}  std={np.std(all_means):.3f}")
    print(f"  per-agent min  : mean={np.mean(all_mins):8.3f}  std={np.std(all_mins):.3f}")
