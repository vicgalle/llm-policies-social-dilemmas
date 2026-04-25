"""
Hand-crafted policy for the Production Economy environment.

Strategy
--------
Phase 0 — tool-acquire (cohort agents 0–6, step < 120):
    * Self-source 2p+1b cross-role: gather wood + stone, craft both at the
      workshops, forge tool. Agent 0 takes a pool shortcut: if the anchor
      seeded ≥2 bricks at a forge, walk over and pickup instead of doing the
      stone trip. Saves ~10 steps for the lowest-id wood agent.

Phase A — shelter-contribute (pre-winter, shelter < 6):
    * Agent 7 (anchor) skips Phase 0 and goes straight to shelter so the
      rush starts immediately — without an anchor, no one feeds the brick
      pool while the cohort is busy tool-acquiring.
    * Even-id agents specialise wood→plank, odd-id stone→brick.
    * Bricks pool at *both* forges (cap 3 each); traffic divides.
    * Wood agents bring 3p to whichever forge has ≥3 bricks pooled and
      trigger CRAFT_SHELTER (3p inv + 3b drops).

Phase B — tool cycle (post-shelter / post-winter):
    Three coordinated mechanisms drive near-continuous tool ownership:

    1. *Cap-aware pool drops.* Tooled agents drop intermediates only when
       (a) the per-type pool slot is below 3 and (b) the cell still has
       drop room. Without these guards the cell's 5-item cap silently
       fails drops and tooled agents jam on full forge cells, blocking
       tool-less peers.

    2. *Pool-budget production gate.* Production pauses once the pool
       already covers tool-less demand: target_p = 2*n_toolless + 2,
       target_b = n_toolless + 1 (clamped at 6p+4b). Agents idle off-forge
       once pool is saturated, freeing forge cells for re-toolers.

    3. *Pre-spoil personal kit.* When tool age ≥ 60 (20 steps before
       spoil), the tooled agent stops dropping into the pool and instead
       accumulates 2p+1b in inv via cross-role gathering. Once kit ready
       they park adjacent to a forge so the post-spoil step lands on the
       forge for an immediate CRAFT_TOOL — 2-step re-craft, no pool wait.

Tool-less agents prefer pool pickup over self-source: walk to whichever
forge has the components they need. Agent-aware BFS treats peers as walls
to avoid swap-dance deadlocks.

Run with:
    uv run python production_economy_policy.py
"""

from __future__ import annotations

import numpy as np
from collections import deque

from production_economy_env import (
    ProductionEconomyEnv,
    NUM_ACTIONS,
    make_production_economy,
)
from gathering_policy import bfs_to_target_set


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

def policy(env, agent_id: int) -> int:
    """Return an action (0-16) for the given agent in Production Economy."""
    # --- action constants ---
    NOOP = 0
    MOVE_N, MOVE_S, MOVE_E, MOVE_W = 1, 2, 3, 4
    GATHER = 5
    CRAFT = 6
    CRAFT_TOOL = 7
    CRAFT_SHELTER = 8
    DROP_WOOD = 9
    DROP_STONE = 10
    DROP_PLANK = 11
    DROP_BRICK = 12
    PICKUP_PLANK = 15
    PICKUP_BRICK = 16
    W_I, S_I, P_I, B_I = 0, 1, 2, 3

    # --- state ---
    r = int(env.agent_pos[agent_id, 0])
    c = int(env.agent_pos[agent_id, 1])
    pos = (r, c)
    inv = env.inventory[agent_id]
    iw = int(inv[W_I]); is_ = int(inv[S_I]); ip = int(inv[P_I]); ib = int(inv[B_I])
    inv_total = iw + is_ + ip + ib
    inv_cap = int(env.inventory_capacity)
    inv_free = inv_cap - inv_total
    has_tool = bool(env.has_tool[agent_id])
    step = int(getattr(env, "_step_count", 0))
    shelter = int(env.shelter_count)
    winter_step = int(env.winter_step)
    threshold = int(env.winter_threshold)

    sawmill = env.sawmill_cells_set
    masonry = env.masonry_cells_set
    forge = env.forge_cells_set
    forge_list = env.forge_cells_list
    forest = env.forest_cells_set
    quarry = env.quarry_cells_set

    # Forge role assignment: forge1 = shelter (brick pool), forge2 = tool (mixed pool).
    # forge_list comes out of _parse_map row-major: [(6,7), (7,7)].
    sorted_forges = sorted(forge_list)
    tool_forge = sorted_forges[1] if len(sorted_forges) > 1 else sorted_forges[0]
    tool_forge_set = {tool_forge}
    shelter_forge = sorted_forges[0]
    shelter_forge_set = {shelter_forge}

    # Currently stocked resource cells.
    stocked_f, stocked_q = set(), set()
    for idx in range(env.n_resources):
        if env.resource_stocked[idx]:
            rr = int(env.resource_pos[idx, 0])
            cc = int(env.resource_pos[idx, 1])
            if int(env.resource_type[idx]) == 0:
                stocked_f.add((rr, cc))
            else:
                stocked_q.add((rr, cc))

    # Other agents' positions — used to make BFS agent-aware.
    occupied = {(int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1]))
                for j in range(env.n_agents) if j != agent_id}

    # Per-forge pools.
    forge_p_pool = {fp: int(env.dropped_items[fp[0], fp[1], P_I]) for fp in forge_list}
    forge_b_pool = {fp: int(env.dropped_items[fp[0], fp[1], B_I]) for fp in forge_list}

    # --- helpers ---
    def mv(dr: int, dc: int) -> int:
        if dr == -1 and dc == 0: return MOVE_N
        if dr == 1 and dc == 0: return MOVE_S
        if dr == 0 and dc == 1: return MOVE_E
        if dr == 0 and dc == -1: return MOVE_W
        return NOOP

    def go_avoiding(target_set):
        if not target_set or pos in target_set:
            return None
        visited = {pos}
        queue = deque()
        DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in DIRS:
            nr, nc = pos[0] + dr, pos[1] + dc
            if not (0 <= nr < env.height and 0 <= nc < env.width):
                continue
            if env.walls[nr, nc]:
                continue
            cell = (nr, nc)
            if cell in occupied and cell not in target_set:
                continue
            if cell in visited:
                continue
            visited.add(cell)
            if cell in target_set:
                return mv(dr, dc)
            queue.append((nr, nc, dr, dc))
        while queue:
            r0, c0, fdr, fdc = queue.popleft()
            for dr, dc in DIRS:
                nr, nc = r0 + dr, c0 + dc
                if not (0 <= nr < env.height and 0 <= nc < env.width):
                    continue
                if env.walls[nr, nc]:
                    continue
                cell = (nr, nc)
                if cell in occupied and cell not in target_set:
                    continue
                if cell in visited:
                    continue
                visited.add(cell)
                if cell in target_set:
                    return mv(fdr, fdc)
                queue.append((nr, nc, fdr, fdc))
        return None

    def go(target_set):
        a = go_avoiding(target_set)
        if a is not None:
            return a
        if not target_set or pos in target_set:
            return None
        res = bfs_to_target_set(env, agent_id, target_set)
        if res is None:
            return None
        return mv(res[0], res[1])

    def near(target):
        if pos in target:
            return True
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            if (r + dr, c + dc) in target:
                return True
        return False

    def vacate_forge():
        """Step off the current forge cell to a free non-forge neighbor."""
        for dr, dc in ((-1, 0), (0, 1), (0, -1), (1, 0)):
            nr, nc = r + dr, c + dc
            if (0 <= nr < env.height and 0 <= nc < env.width
                    and not env.walls[nr, nc]
                    and (nr, nc) not in forge
                    and (nr, nc) not in occupied):
                return mv(dr, dc)
        return NOOP

    pre_winter = step < winter_step
    shelter_done = shelter >= threshold
    on_forge = pos in forge

    pile_p = int(env.dropped_items[r, c, P_I]) if on_forge else 0
    pile_b = int(env.dropped_items[r, c, B_I]) if on_forge else 0

    is_wood = (agent_id % 2 == 0)

    # ====================================================
    # PHASE PRIORITY:
    #   1. Shelter rush (pre-winter, shelter < 6) — non-negotiable; the
    #      ±50 winter swing dominates everything else.
    #   2. Tool-acquire (no tool, after shelter or post-winter) —
    #      assemble 2p+1b from pool first, fall back to self-source.
    #   3. Tool-maintain (have tool, after shelter or post-winter) —
    #      keep producing intermediates and dump at the tool forge so
    #      the next re-tool is a 4-action shop visit.
    # The "have tool while doing shelter" case is fine: we just stay in
    # phase 1 and the tool keeps paying out in the background.
    # ====================================================

    in_shelter_phase = pre_winter and not shelter_done

    # Step deadline for tool-first opening: any agent that doesn't have a
    # tool by this step gives up on tools until shelter is done. Keeps
    # shelter rush from starving while still allowing 30-35 steps for the
    # full self-source cycle (gather 2w+1s, craft 2p+1b, forge).
    TOOL_FIRST_DEADLINE = 120

    # =================================================================
    # PHASE 0: TOOL-FIRST (early steps, every agent self-sources its tool)
    # Self-source 2p + 1b cross-role (gather both wood and stone).
    # Pool-assisted shortcuts apply once a peer has dropped intermediates.
    # =================================================================
    def _tool_acquire():
        if on_forge:
            if ip >= 2 and ib >= 1:
                return CRAFT_TOOL
            need_p = max(0, 2 - ip)
            need_b = max(0, 1 - ib)
            if need_p > 0 and pile_p > 0 and inv_free > 0 and iw == 0:
                return PICKUP_PLANK
            if need_b > 0 and pile_b > 0 and inv_free > 0 and is_ == 0:
                return PICKUP_BRICK
            if inv_free == 0:
                # Drop excess raws first — they're not directly usable for
                # the tool and we want a free slot for cross-role gathering
                # or pool pickup.
                if iw > 0 and ip >= 2: return DROP_WOOD
                if is_ > 0 and ib >= 1: return DROP_STONE
                if ip > 2: return DROP_PLANK
                if ib > 1: return DROP_BRICK
                # Drop the type we already have enough of.
                if ip >= 2 and ib < 1: return DROP_PLANK
                if ib >= 1 and ip < 2: return DROP_BRICK
            # Otherwise leave the cell.
            return vacate_forge()

        if ip >= 2 and ib >= 1:
            a = go(forge)
            if a is not None: return a

        # Push raw through workshop.
        if iw >= 1 and ip < 2 and near(sawmill): return CRAFT
        if is_ >= 1 and ib < 1 and near(masonry): return CRAFT
        if iw >= 1 and ip < 2:
            a = go(sawmill)
            if a is not None: return a
        if is_ >= 1 and ib < 1:
            a = go(masonry)
            if a is not None: return a

        # Pool shortcut for the lowest-id wood agent only: if I have 2 planks
        # and the anchor has already seeded a brick at a forge, pick it up
        # rather than self-source a stone trip. Limited to agent 0 to avoid
        # the whole cohort piling on the shelter brick reserve.
        if (agent_id == 0 and ip >= 2 and ib < 1 and is_ == 0
                and inv_free > 0):
            brick_pool_forges = {fp for fp in forge_list if forge_b_pool[fp] >= 2}
            if brick_pool_forges:
                a = go(brick_pool_forges)
                if a is not None: return a

        # Gather missing raws.
        need_w_to_gather = max(0, max(0, 2 - ip) - iw)
        need_s_to_gather = max(0, max(0, 1 - ib) - is_)
        if pos in stocked_f and need_w_to_gather > 0 and inv_free > 0 and iw < inv_cap:
            return GATHER
        if pos in stocked_q and need_s_to_gather > 0 and inv_free > 0 and is_ < inv_cap:
            return GATHER

        # Decide which raw to fetch first. Wood needs 2 (heavier), so prefer it.
        if need_w_to_gather > 0 and inv_free > 0:
            a = go(stocked_f)
            if a is not None: return a
            a = go(forest)
            if a is not None: return a
        if need_s_to_gather > 0 and inv_free > 0:
            a = go(stocked_q)
            if a is not None: return a
            a = go(quarry)
            if a is not None: return a

        # Last resort
        a = go(forge)
        if a is not None: return a
        return NOOP

    # Only half the population takes the tool-first opening; the other half
    # goes straight to shelter so the rush isn't starved while the tool-first
    # cohort is gathering both raws cross-role.
    is_tool_first_cohort = (agent_id < 7)
    if not has_tool and step < TOOL_FIRST_DEADLINE and is_tool_first_cohort:
        return _tool_acquire()

    # Pre-spoil mode: shared with Phase B. When a tooled agent's tool is
    # ageing toward spoil, switch to building a personal 2p+1b kit so the
    # next CRAFT_TOOL is one action after spoil.
    tool_age_n = int(env.tool_age[agent_id])
    pre_spoil = has_tool and tool_age_n >= 60

    # =================================================================
    # PHASE: SHELTER-CONTRIBUTE (pre-winter, shelter < 6)
    # =================================================================
    if in_shelter_phase:
        # Drop targets: forges with brick room.
        drop_forges = {fp for fp in forge_list if forge_b_pool[fp] < 3}
        # Craft targets: forges where wood agent's 3p + pool 3b makes recipe.
        craft_forges = {fp for fp in forge_list if forge_b_pool[fp] >= 3}

        if on_forge:
            # 1. Craft shelter if recipe complete from inv + this cell's pool.
            if (ip + pile_p) >= 3 and (ib + pile_b) >= 3:
                return CRAFT_SHELTER
            # 2. Opportunistic tool craft — if we happen to be at a forge
            #    with 2p+1b in inv, take the tool. Free +160 over 79 steps.
            if not has_tool and ip >= 2 and ib >= 1:
                return CRAFT_TOOL
            # 3. Stone agents drop bricks (cap 3 per forge; cap-5 cell limit
            #    leaves room for the wood agent's 3 planks in inventory).
            if not is_wood and ib > 0 and pile_b < 3:
                return DROP_BRICK
            # 4. Otherwise vacate the cell.
            return vacate_forge()

        if is_wood and near(sawmill) and iw >= 1:
            return CRAFT
        if not is_wood and near(masonry) and is_ >= 1:
            return CRAFT
        if near(sawmill) and iw >= 1: return CRAFT
        if near(masonry) and is_ >= 1: return CRAFT

        if is_wood and pos in stocked_f and inv_free > 0 and iw < inv_cap:
            return GATHER
        if not is_wood and pos in stocked_q and inv_free > 0 and is_ < inv_cap:
            return GATHER

        primary_raw = iw if is_wood else is_
        primary_workshop = sawmill if is_wood else masonry
        primary_stocked = stocked_f if is_wood else stocked_q
        primary_all = forest if is_wood else quarry
        other_stocked = stocked_q if is_wood else stocked_f

        # Opportunistic: if we already have the tool recipe (2p+1b) and no
        # tool, head to any forge to craft it on the way to shelter work.
        if not has_tool and ip >= 2 and ib >= 1:
            a = go(forge)
            if a is not None: return a

        # Wood agents converge on a craft-ready forge ONLY when carrying 3p.
        if is_wood and ip >= 3 and craft_forges:
            a = go(craft_forges)
            if a is not None: return a

        # Stone agents head to a forge with brick-pool room.
        if not is_wood and ib >= 1 and drop_forges:
            a = go(drop_forges)
            if a is not None: return a

        if primary_raw >= 1:
            a = go(primary_workshop)
            if a is not None: return a

        if inv_free > 0:
            a = go(primary_stocked)
            if a is not None: return a
            a = go(other_stocked)
            if a is not None: return a
            a = go(primary_all)
            if a is not None: return a

        return NOOP

    # =================================================================
    # PHASE: TOOL CYCLE (post-shelter or post-winter)
    #
    # Hybrid: role-based production refills a forge-pool that tool-less peers
    # consume. Both forges are equally usable. Cap-aware: tooled agents pause
    # production once the pool covers the pending tool-less demand. Tool-less
    # agents keep self-sourcing as a fallback when the pool is empty.
    # =================================================================
    cell_drop_cap = int(env.cell_drop_capacity)
    forge_total = {fp: int(env.dropped_items[fp[0], fp[1]].sum()) for fp in forge_list}
    total_p_pooled = sum(forge_p_pool.values())
    total_b_pooled = sum(forge_b_pool.values())

    # How much pool we'd like to keep stocked. Each tool-less peer consumes
    # 2p+1b. Add a buffer of one tool-recipe so the next spoil round has
    # supplies waiting. Cell cap (5) bounds total pool to 6p+4b.
    n_toolless = sum(1 for j in range(env.n_agents) if not env.has_tool[j])
    target_p = min(2 * max(n_toolless, 1) + 2, 6)
    target_b = min(1 * max(n_toolless, 1) + 1, 4)
    pool_needs_p = total_p_pooled < target_p
    pool_needs_b = total_b_pooled < target_b

    if on_forge:
        if not has_tool and ip >= 2 and ib >= 1:
            return CRAFT_TOOL
        if not has_tool:
            if ip < 2 and pile_p > 0 and inv_free > 0:
                return PICKUP_PLANK
            if ib < 1 and pile_b > 0 and inv_free > 0:
                return PICKUP_BRICK
        cell_total = forge_total[pos]
        cell_has_room = cell_total < cell_drop_cap
        if has_tool and pre_spoil:
            # Protect the personal kit: only drop true excess.
            if ip > 2 and pile_p < 3 and cell_has_room: return DROP_PLANK
            if ib > 1 and pile_b < 3 and cell_has_room: return DROP_BRICK
            return vacate_forge()
        if has_tool:
            if ip > 0 and pile_p < 3 and cell_has_room: return DROP_PLANK
            if ib > 0 and pile_b < 3 and cell_has_room: return DROP_BRICK
        else:
            if ip > 2 and cell_has_room: return DROP_PLANK
            if ib > 1 and cell_has_room: return DROP_BRICK

    if near(sawmill) and iw >= 1: return CRAFT
    if near(masonry) and is_ >= 1: return CRAFT

    if not has_tool:
        need_p = max(0, 2 - ip)
        need_b = max(0, 1 - ib)
        if need_p > 0 and pos in stocked_f and inv_free > 0 and iw < inv_cap:
            return GATHER
        if need_b > 0 and pos in stocked_q and inv_free > 0 and is_ < inv_cap:
            return GATHER
    else:
        if is_wood and pos in stocked_f and inv_free > 0 and iw < inv_cap:
            return GATHER
        if not is_wood and pos in stocked_q and inv_free > 0 and is_ < inv_cap:
            return GATHER

    if not has_tool:
        need_p = max(0, 2 - ip)
        need_b = max(0, 1 - ib)
        if need_p == 0 and need_b == 0:
            a = go(forge)
            if a is not None: return a
            return NOOP
        if iw > 0 and need_p > 0:
            a = go(sawmill)
            if a is not None: return a
        if is_ > 0 and need_b > 0:
            a = go(masonry)
            if a is not None: return a
        # Prefer pickup from a forge if it has what we need.
        plank_pickup_forges = {fp for fp in forge_list if forge_p_pool[fp] > 0}
        brick_pickup_forges = {fp for fp in forge_list if forge_b_pool[fp] > 0}
        if need_p > 0 and inv_free > 0 and plank_pickup_forges:
            a = go(plank_pickup_forges)
            if a is not None: return a
        if need_b > 0 and inv_free > 0 and brick_pickup_forges:
            a = go(brick_pickup_forges)
            if a is not None: return a
        if need_p >= need_b and need_p > 0 and inv_free > 0:
            a = go(stocked_f)
            if a is not None: return a
            a = go(forest)
            if a is not None: return a
        if need_b > 0 and inv_free > 0:
            a = go(stocked_q)
            if a is not None: return a
            a = go(quarry)
            if a is not None: return a
        if need_p > 0 and inv_free > 0:
            a = go(stocked_f)
            if a is not None: return a
            a = go(forest)
            if a is not None: return a
        a = go(forge)
        if a is not None: return a
        return NOOP

    # When tool age is approaching spoil, hold intermediates in inv for a
    # personal re-craft. Park adjacent to a forge so the post-spoil step
    # lands directly on the forge cell.
    if pre_spoil and ip >= 2 and ib >= 1:
        # Personal kit complete — pre-position adjacent to a forge so the
        # post-spoil step lands on the forge cell.
        forge_park_cells = set()
        for fp in forge_list:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = fp[0] + dr, fp[1] + dc
                if (0 <= nr < env.height and 0 <= nc < env.width
                        and not env.walls[nr, nc]
                        and (nr, nc) not in forge):
                    forge_park_cells.add((nr, nc))
        if pos in forge_park_cells:
            return NOOP
        a = go(forge_park_cells)
        if a is not None: return a

    # has_tool: role-based gathering, drop into the cap-aware pool. Pause
    # production when the pool already covers tool-less demand.
    # Pre-spoil mode: build a personal kit (2p+1b in inv) for instant re-craft.
    if is_wood:
        if pre_spoil and ip < 2:
            # Build personal kit. Need 2 planks first.
            if iw >= 1 and near(sawmill): return CRAFT
            if pos in stocked_f and inv_free > 0 and iw < inv_cap: return GATHER
            if iw >= 1:
                a = go(sawmill)
                if a is not None: return a
            if inv_free > 0:
                a = go(stocked_f)
                if a is not None: return a
                a = go(forest)
                if a is not None: return a
            return NOOP
        if pre_spoil and ip >= 2 and ib < 1:
            # Need 1 brick. Try cross-role.
            if is_ >= 1 and near(masonry): return CRAFT
            if pos in stocked_q and inv_free > 0 and is_ < inv_cap: return GATHER
            # Pool pickup if available.
            brick_pickup = {fp for fp in forge_list if forge_b_pool[fp] > 0}
            if inv_free > 0 and brick_pickup:
                a = go(brick_pickup)
                if a is not None: return a
            if is_ >= 1:
                a = go(masonry)
                if a is not None: return a
            if inv_free > 0:
                a = go(stocked_q)
                if a is not None: return a
                a = go(quarry)
                if a is not None: return a
            return NOOP

        if ip >= 1 and pool_needs_p:
            drop_targets = {fp for fp in forge_list
                            if forge_p_pool[fp] < 3 and forge_total[fp] < cell_drop_cap}
            if drop_targets:
                a = go(drop_targets)
                if a is not None: return a
        if iw >= 1 and pool_needs_p:
            a = go(sawmill)
            if a is not None: return a
        if inv_free > 0 and pool_needs_p:
            a = go(stocked_f)
            if a is not None: return a
            a = go(forest)
            if a is not None: return a
        if pos in forge:
            return vacate_forge()
        return NOOP
    else:
        if pre_spoil and ib < 1:
            if is_ >= 1 and near(masonry): return CRAFT
            if pos in stocked_q and inv_free > 0 and is_ < inv_cap: return GATHER
            if is_ >= 1:
                a = go(masonry)
                if a is not None: return a
            if inv_free > 0:
                a = go(stocked_q)
                if a is not None: return a
                a = go(quarry)
                if a is not None: return a
            return NOOP
        if pre_spoil and ib >= 1 and ip < 2:
            if iw >= 1 and near(sawmill): return CRAFT
            if pos in stocked_f and inv_free > 0 and iw < inv_cap: return GATHER
            plank_pickup = {fp for fp in forge_list if forge_p_pool[fp] > 0}
            if inv_free > 0 and plank_pickup:
                a = go(plank_pickup)
                if a is not None: return a
            if iw >= 1:
                a = go(sawmill)
                if a is not None: return a
            if inv_free > 0:
                a = go(stocked_f)
                if a is not None: return a
                a = go(forest)
                if a is not None: return a
            return NOOP

        if ib >= 1 and pool_needs_b:
            drop_targets = {fp for fp in forge_list
                            if forge_b_pool[fp] < 3 and forge_total[fp] < cell_drop_cap}
            if drop_targets:
                a = go(drop_targets)
                if a is not None: return a
        if is_ >= 1 and pool_needs_b:
            a = go(masonry)
            if a is not None: return a
        if inv_free > 0 and pool_needs_b:
            a = go(stocked_q)
            if a is not None: return a
            a = go(quarry)
            if a is not None: return a
        if pos in forge:
            return vacate_forge()
        return NOOP


# ---------------------------------------------------------------------------
# Evaluation harness — prints all five framework metrics.
# ---------------------------------------------------------------------------

def evaluate(seed: int = 0, n_agents: int = 8, verbose: bool = True) -> dict:
    env = make_production_economy(n_agents=n_agents, seed=seed)
    env.reset(seed=seed)

    ep_rewards = {i: [] for i in range(env.n_agents)}
    ep_timeouts = {i: [] for i in range(env.n_agents)}

    for _ in range(env.max_steps):
        actions = {}
        for i in range(env.n_agents):
            a = int(policy(env, i))
            assert 0 <= a < NUM_ACTIONS, f"Bad action: {a}"
            actions[i] = a
        _, rewards, _, _, _ = env.step(actions)
        for i in range(env.n_agents):
            ep_rewards[i].append(rewards[i])
            ep_timeouts[i].append(False)

    metrics = ProductionEconomyEnv.compute_metrics(ep_rewards, ep_timeouts)
    totals = {i: sum(ep_rewards[i]) for i in range(env.n_agents)}

    if verbose:
        print(f"Seed {seed}:")
        print(f"  Per-agent totals:  {[round(totals[i], 1) for i in range(env.n_agents)]}")
        print(f"  Mean per-agent:    {np.mean(list(totals.values())):.2f}")
        print(f"  Min  per-agent:    {min(totals.values()):.2f}")
        print(f"  Final shelter:     {env.shelter_count}")
        print(f"  Tools equipped@end {sum(int(env.has_tool[i]) for i in range(env.n_agents))}")
        print("  Metrics:")
        for k, v in metrics.items():
            print(f"    {k:15s}: {v:.4f}")

    return {"metrics": metrics, "totals": totals,
            "shelter": env.shelter_count}


if __name__ == "__main__":
    print("=" * 60)
    print("  Production Economy — hand-crafted policy")
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
