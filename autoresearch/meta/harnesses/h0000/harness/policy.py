"""
h0000 — weak greedy seed for production_economy.

Deliberately incompetent baseline. Each agent walks toward the nearest
stocked resource, gathers, and walks toward the nearest workshop /
forge depending on what's in inventory. No phase routing, no
specialisation, no coordinated handoff. The seed exists so the
meta-harness search has somewhere to climb from.
"""

from __future__ import annotations

from collections import deque


def policy(env, agent_id: int) -> int:
    NOOP = 0
    MOVE_N, MOVE_S, MOVE_E, MOVE_W = 1, 2, 3, 4
    GATHER = 5
    CRAFT = 6
    CRAFT_TOOL = 7
    DROP_PLANK = 11
    DROP_BRICK = 12
    PICKUP_PLANK = 15
    PICKUP_BRICK = 16
    W_I, S_I, P_I, B_I = 0, 1, 2, 3

    r = int(env.agent_pos[agent_id, 0])
    c = int(env.agent_pos[agent_id, 1])
    pos = (r, c)
    inv = env.inventory[agent_id]
    iw = int(inv[W_I]); is_ = int(inv[S_I]); ip = int(inv[P_I]); ib = int(inv[B_I])
    inv_total = iw + is_ + ip + ib
    inv_free = int(env.inventory_capacity) - inv_total

    sawmill = env.sawmill_cells_set
    masonry = env.masonry_cells_set
    forge = env.forge_cells_set
    forest = env.forest_cells_set
    quarry = env.quarry_cells_set

    stocked_f, stocked_q = set(), set()
    for idx in range(env.n_resources):
        if env.resource_stocked[idx]:
            rr = int(env.resource_pos[idx, 0])
            cc = int(env.resource_pos[idx, 1])
            (stocked_f if int(env.resource_type[idx]) == 0 else stocked_q).add((rr, cc))

    def mv(dr, dc):
        if dr == -1 and dc == 0: return MOVE_N
        if dr == 1 and dc == 0: return MOVE_S
        if dr == 0 and dc == 1: return MOVE_E
        if dr == 0 and dc == -1: return MOVE_W
        return NOOP

    def go(target_set):
        if not target_set or pos in target_set:
            return None
        visited = {pos}
        q = deque()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < env.height and 0 <= nc < env.width and not env.walls[nr, nc]:
                cell = (nr, nc)
                if cell in visited: continue
                visited.add(cell)
                if cell in target_set:
                    return mv(dr, dc)
                q.append((nr, nc, dr, dc))
        while q:
            r0, c0, fdr, fdc = q.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r0 + dr, c0 + dc
                if 0 <= nr < env.height and 0 <= nc < env.width and not env.walls[nr, nc]:
                    cell = (nr, nc)
                    if cell in visited: continue
                    visited.add(cell)
                    if cell in target_set:
                        return mv(fdr, fdc)
                    q.append((nr, nc, fdr, fdc))
        return None

    def near(target_set):
        if pos in target_set:
            return True
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            if (r + dr, c + dc) in target_set:
                return True
        return False

    if pos in forge and ip >= 2 and ib >= 1 and not env.has_tool[agent_id]:
        return CRAFT_TOOL

    if iw >= 1 and near(sawmill):
        return CRAFT
    if is_ >= 1 and near(masonry):
        return CRAFT

    if ip >= 2 and ib >= 1:
        a = go(forge)
        if a is not None: return a

    if pos in stocked_f and inv_free > 0 and iw < 3:
        return GATHER
    if pos in stocked_q and inv_free > 0 and is_ < 3:
        return GATHER

    if iw >= 1:
        a = go(sawmill)
        if a is not None: return a
    if is_ >= 1:
        a = go(masonry)
        if a is not None: return a

    if inv_free > 0:
        a = go(stocked_f)
        if a is not None: return a
        a = go(stocked_q)
        if a is not None: return a

    return NOOP
