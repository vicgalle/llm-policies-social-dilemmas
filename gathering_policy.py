"""
Programmatic policies for the Gathering environment.

Policies
--------
1. **Greedy BFS** – shortest path to nearest apple every step.
2. **Exploitative** – opportunistically beams opponents to remove them
   for 25 steps, actively chases half-tagged targets, and collects
   apples when no beaming opportunity exists.  Designed to maximise
   reward for a single agent competing against BFS opponents.
"""

from __future__ import annotations

import numpy as np
from collections import deque
from typing import Optional, Tuple

from gathering_env import (
    GatheringEnv,
    make_gathering,
    Action,
    Orientation,
    _ROTATIONS,
    NUM_ACTIONS,
)


# ── BFS pathfinding ──────────────────────────────────────────────────────────

def bfs_nearest_apple(
    env: GatheringEnv, agent_id: int
) -> Optional[Tuple[int, int]]:
    """BFS from the agent's position to the nearest alive apple.

    Returns
    -------
    (dr, dc) : the world-frame displacement of the *first* step on the
                shortest path, or ``(0, 0)`` if we're already on an apple,
                or ``None`` if no apple is reachable.
    """
    start = (int(env.agent_pos[agent_id, 0]),
             int(env.agent_pos[agent_id, 1]))

    # Alive apple positions
    apple_set: set = set()
    for idx in range(env.n_apples):
        if env.apple_alive[idx]:
            apple_set.add((int(env._apple_pos[idx, 0]),
                           int(env._apple_pos[idx, 1])))

    if not apple_set:
        return None

    # Already standing on an apple
    if start in apple_set:
        return (0, 0)

    # Standard BFS – each entry remembers the first step taken from `start`
    visited = {start}
    queue: deque = deque()
    DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for dr, dc in DIRS:
        nr, nc = start[0] + dr, start[1] + dc
        if (0 <= nr < env.height and 0 <= nc < env.width
                and not env.walls[nr, nc]):
            pos = (nr, nc)
            if pos not in visited:
                visited.add(pos)
                if pos in apple_set:
                    return (dr, dc)
                queue.append((nr, nc, dr, dc))

    while queue:
        r, c, first_dr, first_dc = queue.popleft()
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if (0 <= nr < env.height and 0 <= nc < env.width
                    and not env.walls[nr, nc]):
                pos = (nr, nc)
                if pos not in visited:
                    visited.add(pos)
                    if pos in apple_set:
                        return (first_dr, first_dc)
                    queue.append((nr, nc, first_dr, first_dc))

    return None


# ── Action selection ──────────────────────────────────────────────────────────

def direction_to_action(dr: int, dc: int, orientation: int) -> int:
    """Convert a desired world-frame step (dr, dc) into an Action
    given the agent's current orientation.

    The agent can move in all 4 cardinal directions without rotating
    (forward, backward, strafe-left, strafe-right), so we never
    need to rotate to reach an adjacent cell.
    """
    if dr == 0 and dc == 0:
        return Action.STAND

    orient = Orientation(orientation)
    a, b, c, d = _ROTATIONS[orient]
    # forward  → (a, c)
    # backward → (-a, -c)
    # left     → (-b, -d)
    # right    → (b, d)

    if (dr, dc) == (a, c):
        return Action.FORWARD
    if (dr, dc) == (-a, -c):
        return Action.BACKWARD
    if (dr, dc) == (-b, -d):
        return Action.STEP_LEFT
    if (dr, dc) == (b, d):
        return Action.STEP_RIGHT

    # Fallback (should never happen for cardinal BFS)
    return Action.FORWARD


def bfs_to_target_set(
    env: GatheringEnv, agent_id: int, target_set: set,
) -> Optional[Tuple[int, int]]:
    """BFS first step toward the nearest position in *target_set*."""
    start = (int(env.agent_pos[agent_id, 0]),
             int(env.agent_pos[agent_id, 1]))
    if not target_set:
        return None
    if start in target_set:
        return (0, 0)

    visited = {start}
    queue: deque = deque()
    DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for dr, dc in DIRS:
        nr, nc = start[0] + dr, start[1] + dc
        if (0 <= nr < env.height and 0 <= nc < env.width
                and not env.walls[nr, nc]):
            pos = (nr, nc)
            if pos not in visited:
                visited.add(pos)
                if pos in target_set:
                    return (dr, dc)
                queue.append((nr, nc, dr, dc))

    while queue:
        r, c, first_dr, first_dc = queue.popleft()
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if (0 <= nr < env.height and 0 <= nc < env.width
                    and not env.walls[nr, nc]):
                pos = (nr, nc)
                if pos not in visited:
                    visited.add(pos)
                    if pos in target_set:
                        return (first_dr, first_dc)
                    queue.append((nr, nc, first_dr, first_dc))
    return None


def greedy_action(env: GatheringEnv, agent_id: int) -> int:
    """Return the BFS-greedy action for *agent_id*."""
    if env.agent_timeout[agent_id] > 0:
        return Action.STAND

    result = bfs_nearest_apple(env, agent_id)
    if result is None:
        return Action.STAND

    dr, dc = result
    return direction_to_action(dr, dc, int(env.agent_orient[agent_id]))


# ── BFS to arbitrary target ──────────────────────────────────────────────────

def bfs_toward(
    env: GatheringEnv, agent_id: int, target_r: int, target_c: int
) -> Optional[Tuple[int, int]]:
    """BFS first step toward an arbitrary (row, col) target.

    Returns (dr, dc) of the first step, (0,0) if already there, None if
    unreachable.
    """
    start = (int(env.agent_pos[agent_id, 0]),
             int(env.agent_pos[agent_id, 1]))
    goal = (target_r, target_c)

    if start == goal:
        return (0, 0)

    visited = {start}
    queue: deque = deque()
    DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for dr, dc in DIRS:
        nr, nc = start[0] + dr, start[1] + dc
        if (0 <= nr < env.height and 0 <= nc < env.width
                and not env.walls[nr, nc]):
            pos = (nr, nc)
            if pos not in visited:
                visited.add(pos)
                if pos == goal:
                    return (dr, dc)
                queue.append((nr, nc, dr, dc))

    while queue:
        r, c, first_dr, first_dc = queue.popleft()
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if (0 <= nr < env.height and 0 <= nc < env.width
                    and not env.walls[nr, nc]):
                pos = (nr, nc)
                if pos not in visited:
                    visited.add(pos)
                    if pos == goal:
                        return (first_dr, first_dc)
                    queue.append((nr, nc, first_dr, first_dc))
    return None


# ── Exploitative policy ──────────────────────────────────────────────────────

def _beam_targets_for_orient(
    env: GatheringEnv,
    ar: int, ac: int,
    orient_val: int,
    opponents: list,
) -> list:
    """Return opponents hittable by a beam from (ar, ac) facing *orient_val*.

    Each entry is ``(agent_id, fwd_dist, beam_hits_so_far)``.
    Walls along the beam path are checked.
    """
    o = Orientation(orient_val)
    a, b, c, d = _ROTATIONS[o]
    # det = a*d - b*c  (always -1 for all four orientations)
    det = a * d - b * c
    half_w = env.beam_width // 2
    hits = []

    for j, jr, jc, _dist, bhits in opponents:
        dr, dc = jr - ar, jc - ac
        fwd_num = d * dr - b * dc
        right_num = a * dc - c * dr
        # Exact integer check (det divides evenly for grid positions)
        if fwd_num % det != 0 or right_num % det != 0:
            continue
        fwd = fwd_num // det
        right = right_num // det
        if not (1 <= fwd <= env.beam_length and abs(right) <= half_w):
            continue
        # Wall check along the beam column at this lateral offset
        blocked = False
        for f in range(1, fwd):
            wr = ar + a * f + b * right
            wc = ac + c * f + d * right
            if not (0 <= wr < env.height and 0 <= wc < env.width):
                blocked = True
                break
            if env.walls[wr, wc]:
                blocked = True
                break
        if not blocked:
            hits.append((j, fwd, bhits))
    return hits


def _rotation_distance(cur: int, target: int) -> int:
    """Minimum number of 90° turns from *cur* to *target* orientation."""
    diff = (target - cur) % 4
    return min(diff, 4 - diff)


def exploitative_action(env: GatheringEnv, agent_id: int) -> int:
    """Pick the action that maximises reward by combining beaming + apple BFS.

    Decision priority:
      1. BEAM  – if an opponent is already in our beam path, fire immediately.
      2. ROTATE – if one rotation puts an opponent in our beam, rotate first.
                  Prefer half-tagged targets (1 hit → 1 more tags them out).
      3. CHASE  – if an opponent already has 1 beam hit and is within chase
                  range, navigate toward a cell where we can fire at them.
      4. COLLECT – BFS to nearest apple (same as greedy policy).
    """
    if env.agent_timeout[agent_id] > 0:
        return Action.STAND

    ar = int(env.agent_pos[agent_id, 0])
    ac = int(env.agent_pos[agent_id, 1])
    cur_orient = int(env.agent_orient[agent_id])

    # Gather active opponent info: (id, row, col, manhattan_dist, beam_hits)
    opponents = []
    for j in range(env.n_agents):
        if j == agent_id or env.agent_timeout[j] > 0:
            continue
        jr = int(env.agent_pos[j, 0])
        jc = int(env.agent_pos[j, 1])
        opponents.append((j, jr, jc,
                          abs(jr - ar) + abs(jc - ac),
                          int(env.agent_beam_hits[j])))

    if not opponents:
        # Everyone tagged → just collect apples
        return greedy_action(env, agent_id)

    # ── 1. Can we fire RIGHT NOW? ────────────────────────────────────────
    cur_targets = _beam_targets_for_orient(env, ar, ac, cur_orient, opponents)
    if cur_targets:
        return Action.BEAM

    # ── 2. One rotation away? ────────────────────────────────────────────
    left_orient = (cur_orient - 1) % 4
    right_orient = (cur_orient + 1) % 4
    left_targets = _beam_targets_for_orient(env, ar, ac, left_orient, opponents)
    right_targets = _beam_targets_for_orient(env, ar, ac, right_orient, opponents)

    def _best_score(targets):
        """Higher = more desirable.  Prioritise half-tagged, then proximity."""
        if not targets:
            return -999
        return max(bhits * 1000 + (env.beam_length - fwd)
                   for _, fwd, bhits in targets)

    ls, rs = _best_score(left_targets), _best_score(right_targets)
    # Rotate if at least one target is half-tagged, or if cheap shot available
    if ls >= 0 or rs >= 0:
        return Action.ROTATE_LEFT if ls >= rs else Action.ROTATE_RIGHT

    # ── 3. Chase half-tagged opponents ───────────────────────────────────
    CHASE_RANGE = 12
    half_tagged = [(j, jr, jc, dist)
                   for j, jr, jc, dist, bhits in opponents if bhits >= 1]
    if half_tagged:
        # Pick closest
        half_tagged.sort(key=lambda x: x[3])
        tj, tr, tc, tdist = half_tagged[0]
        if tdist <= CHASE_RANGE:
            # Navigate toward a cell on the same row or column as the target
            # (so we can beam them once we arrive + rotate).
            # Try to reach a cell 1 step away on same row or col.
            best_step = None
            best_bfs_dist = float("inf")
            # Candidate firing positions: same row or same col as target,
            # within beam range, that are walkable.
            candidates = []
            for orient_val in range(4):
                o = Orientation(orient_val)
                a, b, c, d = _ROTATIONS[o]
                # We'd stand at (tr - a*fwd, tc - c*fwd) to beam them
                for fwd in range(1, min(env.beam_length + 1, 8)):
                    fr = tr - a * fwd
                    fc = tc - c * fwd
                    if (0 <= fr < env.height and 0 <= fc < env.width
                            and not env.walls[fr, fc]):
                        candidates.append((fr, fc))

            # BFS from our position to the nearest candidate
            start = (ar, ac)
            if candidates:
                cand_set = set(candidates)
                visited = {start}
                q: deque = deque()
                DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                # Check if already at a candidate
                if start in cand_set:
                    # We're in firing position but didn't match orient above.
                    # Just rotate toward the target.
                    dr, dc = tr - ar, tc - ac
                    # Pick orientation that faces (dr, dc)
                    for ov in range(4):
                        oa, ob, oc, od = _ROTATIONS[Orientation(ov)]
                        # Forward is (oa, oc); check if it points toward target
                        if dr != 0 and oa != 0 and (dr > 0) == (oa > 0) and dc == 0:
                            return _rotate_toward(cur_orient, ov)
                        if dc != 0 and oc != 0 and (dc > 0) == (oc > 0) and dr == 0:
                            return _rotate_toward(cur_orient, ov)
                else:
                    for ddr, ddc in DIRS:
                        nr, nc = start[0] + ddr, start[1] + ddc
                        pos = (nr, nc)
                        if (0 <= nr < env.height and 0 <= nc < env.width
                                and not env.walls[nr, nc]
                                and pos not in visited):
                            visited.add(pos)
                            if pos in cand_set:
                                best_step = (ddr, ddc)
                                best_bfs_dist = 1
                                break
                            q.append((nr, nc, ddr, ddc))

                    if best_step is None:
                        while q:
                            r, c, fdr, fdc = q.popleft()
                            for ddr, ddc in DIRS:
                                nr, nc = r + ddr, c + ddc
                                pos = (nr, nc)
                                if (0 <= nr < env.height and 0 <= nc < env.width
                                        and not env.walls[nr, nc]
                                        and pos not in visited):
                                    visited.add(pos)
                                    if pos in cand_set:
                                        best_step = (fdr, fdc)
                                        break
                                    q.append((nr, nc, fdr, fdc))
                            if best_step is not None:
                                break

                    if best_step is not None:
                        return direction_to_action(
                            best_step[0], best_step[1], cur_orient)

    # ── 4. Default: collect apples ───────────────────────────────────────
    return greedy_action(env, agent_id)


def _rotate_toward(cur_orient: int, target_orient: int) -> int:
    """Return a ROTATE action that moves from *cur* toward *target* orient."""
    if cur_orient == target_orient:
        return Action.STAND
    diff = (target_orient - cur_orient) % 4
    if diff == 1:
        return Action.ROTATE_RIGHT
    elif diff == 3:
        return Action.ROTATE_LEFT
    else:
        return Action.ROTATE_RIGHT  # 180°, pick either direction


# ── Cooperative policy ────────────────────────────────────────────────────────

def _compute_coop_zones(env: GatheringEnv) -> dict:
    """Partition apple indices among *cooperative* agents only.

    Uses ``env._coop_agent_ids`` (set by run_episode) to decide how many
    zones to create.  Falls back to all agents if the attribute is missing.

    Returns {agent_id: set_of_apple_indices}.  Cached on the env object
    so it's computed once per episode.
    """
    if hasattr(env, "_coop_zones"):
        return env._coop_zones

    coop_ids = getattr(env, "_coop_agent_ids", set(range(env.n_agents)))
    coop_list = sorted(coop_ids)
    n = len(coop_list)

    # ── 1. Sort apples spatially and split into n groups ─────────────
    sorted_idx = sorted(
        range(env.n_apples),
        key=lambda i: (int(env._apple_pos[i, 0]), int(env._apple_pos[i, 1])),
    )
    groups: list[list[int]] = []
    per = env.n_apples // n
    rem = env.n_apples % n
    pos = 0
    for g in range(n):
        count = per + (1 if g < rem else 0)
        groups.append(sorted_idx[pos : pos + count])
        pos += count

    # ── 2. Build position-sets for each group ────────────────────────
    group_positions = []
    for g in groups:
        group_positions.append(
            {(int(env._apple_pos[i, 0]), int(env._apple_pos[i, 1])) for i in g}
        )

    # ── 3. Greedy nearest-zone matching ──────────────────────────────
    # Compute BFS distance from each agent to each group (dist to closest
    # apple in the group).
    def _bfs_dist(start, targets):
        """BFS distance from *start* to nearest position in *targets*."""
        if start in targets:
            return 0
        visited = {start}
        queue = deque([(start, 0)])
        while queue:
            (r, c), d = queue.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                pos = (nr, nc)
                if (0 <= nr < env.height and 0 <= nc < env.width
                        and not env.walls[nr, nc] and pos not in visited):
                    if pos in targets:
                        return d + 1
                    visited.add(pos)
                    queue.append((pos, d + 1))
        return 9999

    dist = np.zeros((n, n))
    for ai, agent_id in enumerate(coop_list):
        start = (int(env.agent_pos[agent_id, 0]),
                 int(env.agent_pos[agent_id, 1]))
        for g in range(n):
            dist[ai, g] = _bfs_dist(start, group_positions[g])

    used_ai: set = set()
    used_groups: set = set()
    zones: dict[int, set] = {}

    for _ in range(n):
        best_d, best_ai, best_g = 1e9, 0, 0
        for ai in range(n):
            if ai in used_ai:
                continue
            for g in range(n):
                if g in used_groups:
                    continue
                if dist[ai, g] < best_d:
                    best_d, best_ai, best_g = dist[ai, g], ai, g
        zones[coop_list[best_ai]] = set(groups[best_g])
        used_ai.add(best_ai)
        used_groups.add(best_g)

    env._coop_zones = zones
    return zones


def cooperative_action(env: GatheringEnv, agent_id: int) -> int:
    """Cooperative zone-patrol policy.

    Each agent is assigned a spatial zone of ~n_apples/n_agents apple
    spawn points and always BFS-navigates to the nearest alive apple in
    its zone.  When the zone is empty (all apples consumed, awaiting
    respawn), the agent waits at the centroid of its zone so it's
    close when the next apple pops.

    Never fires the beam (maximises peace).
    """
    if env.agent_timeout[agent_id] > 0:
        return Action.STAND

    zones = _compute_coop_zones(env)
    my_indices = zones[agent_id]

    # Alive apples in my zone
    my_targets: set = set()
    for idx in my_indices:
        if env.apple_alive[idx]:
            my_targets.add(
                (int(env._apple_pos[idx, 0]), int(env._apple_pos[idx, 1]))
            )

    # ── Primary: go to nearest alive apple in my zone ─────────────────
    if my_targets:
        result = bfs_to_target_set(env, agent_id, my_targets)
        if result is not None:
            dr, dc = result
            return direction_to_action(dr, dc, int(env.agent_orient[agent_id]))

    # ── Zone empty: wait near the zone centroid ───────────────────────
    # (stay close so we grab the next respawn immediately)
    zone_rows = [int(env._apple_pos[i, 0]) for i in my_indices]
    zone_cols = [int(env._apple_pos[i, 1]) for i in my_indices]
    cr = int(round(np.mean(zone_rows)))
    cc = int(round(np.mean(zone_cols)))
    # Clamp to valid non-wall cell
    if env.walls[cr, cc]:
        # Nudge to nearest open cell
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                nr, nc = cr + dr, cc + dc
                if (0 <= nr < env.height and 0 <= nc < env.width
                        and not env.walls[nr, nc]):
                    cr, cc = nr, nc
                    break
            else:
                continue
            break

    ar = int(env.agent_pos[agent_id, 0])
    ac = int(env.agent_pos[agent_id, 1])
    if (ar, ac) != (cr, cc):
        result = bfs_toward(env, agent_id, cr, cc)
        if result is not None:
            dr, dc = result
            return direction_to_action(dr, dc, int(env.agent_orient[agent_id]))

    return Action.STAND


# ── Evaluation ────────────────────────────────────────────────────────────────

def run_episode(
    env: GatheringEnv,
    agent_fns: dict = None,
    seed: int = 42,
    verbose: bool = True,
    label: str = "",
) -> dict:
    """Run one full episode with per-agent action functions.

    Parameters
    ----------
    agent_fns : dict {agent_id: callable(env, agent_id) -> int}
        Action function for each agent.  Missing agents default to random.
        Special string shortcuts accepted as values:
        ``"bfs"``  → greedy_action,
        ``"exploit"`` → exploitative_action,
        ``"random"`` → uniform random.
    """
    if agent_fns is None:
        agent_fns = {}

    # Resolve string shortcuts
    fn_map = {}
    tag_map = {}
    for i in range(env.n_agents):
        spec = agent_fns.get(i, "random")
        if spec == "bfs":
            fn_map[i] = greedy_action
            tag_map[i] = "BFS"
        elif spec == "exploit":
            fn_map[i] = exploitative_action
            tag_map[i] = "EXPLOIT"
        elif spec == "coop":
            fn_map[i] = cooperative_action
            tag_map[i] = "COOP"
        elif callable(spec):
            fn_map[i] = spec
            tag_map[i] = spec.__name__
        else:
            fn_map[i] = None  # random
            tag_map[i] = "random"

    obs = env.reset(seed=seed)

    # Tell cooperative policy which agents are cooperating
    env._coop_agent_ids = {i for i in range(env.n_agents)
                           if tag_map[i] == "COOP"}
    # Clear cached zones from any previous episode on this env
    if hasattr(env, "_coop_zones"):
        del env._coop_zones

    ep_rewards = {i: [] for i in range(env.n_agents)}
    ep_timeouts = {i: [] for i in range(env.n_agents)}

    for step in range(env.max_steps):
        actions = {}
        for i in range(env.n_agents):
            if fn_map[i] is not None:
                actions[i] = fn_map[i](env, i)
            else:
                actions[i] = int(env.rng.integers(NUM_ACTIONS))

        obs, rewards, terminated, truncated, info = env.step(actions)

        for i in range(env.n_agents):
            ep_rewards[i].append(rewards[i])
            ep_timeouts[i].append(info[i]["timeout"] > 0)

    # ── Results ───────────────────────────────────────────────────────────
    total = {i: sum(ep_rewards[i]) for i in range(env.n_agents)}
    metrics = GatheringEnv.compute_metrics(ep_rewards, ep_timeouts)

    if verbose:
        if not label:
            tags = set(tag_map.values())
            label = " + ".join(sorted(tags))

        print("=" * 60)
        print(f"  {label}")
        print("=" * 60)
        print(f"  Map size        : {env.height} x {env.width}")
        print(f"  Episode length  : {env.max_steps}")
        print(f"  Apple spawns    : {env.n_apples}")
        print(f"  Respawn timer   : {env.apple_respawn_time} steps")
        print()
        print("  Per-agent total rewards:")
        for i in range(env.n_agents):
            print(f"    Agent {i}: {total[i]:6.0f}  {tag_map[i]}")
        print()
        print("  Social outcome metrics:")
        for k, v in metrics.items():
            print(f"    {k:20s}: {v:.4f}")
        print("=" * 60)

    return {
        "total_rewards": total,
        "metrics": metrics,
        "ep_rewards": ep_rewards,
        "ep_timeouts": ep_timeouts,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    N_AGENTS = 4
    SEED = 42

    # ── Scenario 1: 1 BFS vs 3 random ────────────────────────────────────
    env1 = make_gathering(n_agents=N_AGENTS, small=False, seed=0)
    r1 = run_episode(
        env1,
        agent_fns={0: "bfs", 1: "random", 2: "random", 3: "random"},
        seed=SEED,
        label="1 BFS vs 3 Random",
    )

    # ── Scenario 2: ALL 4 BFS ────────────────────────────────────────────
    print("\n")
    env2 = make_gathering(n_agents=N_AGENTS, small=False, seed=0)
    r2 = run_episode(
        env2,
        agent_fns={i: "bfs" for i in range(N_AGENTS)},
        seed=SEED,
        label="ALL 4 BFS",
    )

    # ── Scenario 3: 1 EXPLOIT vs 3 BFS ──────────────────────────────────
    print("\n")
    env3 = make_gathering(n_agents=N_AGENTS, small=False, seed=0)
    r3 = run_episode(
        env3,
        agent_fns={0: "exploit", 1: "bfs", 2: "bfs", 3: "bfs"},
        seed=SEED,
        label="1 EXPLOIT vs 3 BFS",
    )

    # ── Scenario 4: ALL 4 EXPLOIT ────────────────────────────────────────
    print("\n")
    env4 = make_gathering(n_agents=N_AGENTS, small=False, seed=0)
    r4 = run_episode(
        env4,
        agent_fns={i: "exploit" for i in range(N_AGENTS)},
        seed=SEED,
        label="ALL 4 EXPLOIT",
    )

    # ── Scenario 5: ALL 4 COOPERATIVE ────────────────────────────────────
    print("\n")
    env5 = make_gathering(n_agents=N_AGENTS, small=False, seed=0)
    r5 = run_episode(
        env5,
        agent_fns={i: "coop" for i in range(N_AGENTS)},
        seed=SEED,
        label="ALL 4 COOPERATIVE",
    )

    # ── Scenario 6: 2 COOP vs 2 EXPLOIT ──────────────────────────────────
    print("\n")
    env6 = make_gathering(n_agents=N_AGENTS, small=False, seed=0)
    r6 = run_episode(
        env6,
        agent_fns={0: "coop", 1: "coop", 2: "exploit", 3: "exploit"},
        seed=SEED,
        label="2 COOP vs 2 EXPLOIT",
    )

    # ── Scenario 7: ALL 4 random (baseline) ──────────────────────────────
    print("\n")
    env7 = make_gathering(n_agents=N_AGENTS, small=False, seed=0)
    r7 = run_episode(
        env7,
        agent_fns={},  # all random
        seed=SEED,
        label="ALL 4 Random (baseline)",
    )
