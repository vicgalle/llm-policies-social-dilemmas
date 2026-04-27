def policy(env, agent_id: int) -> int:
    NOOP = 0
    MOVE_N = 1
    MOVE_S = 2
    MOVE_E = 3
    MOVE_W = 4
    CLEAN = 5

    # 1. If mid-travel, let the environment handle it
    if env._travel_queue[agent_id]:
        return NOOP

    r, c = int(env.agent_pos[agent_id, 0]), int(env.agent_pos[agent_id, 1])
    pos = (r, c)
    in_plaza = env.config.plaza_lo <= r <= env.config.plaza_hi and env.config.plaza_lo <= c <= env.config.plaza_hi
    
    # 2. RAID for held inventory max (3)
    # We only need to raid once to cap out passive income. Target non-clan agents who haven't capped yet.
    my_inv = int(env.inventory[agent_id])
    if my_inv < 3:
        best_target = -1
        max_inv = -1
        for j in range(env.n_agents):
            if j == agent_id: continue
            if env.agent_clan[j] == env.agent_clan[agent_id]: continue
            if 0 < env.inventory[j] < 3:
                jr, jc = int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1])
                if abs(jr - r) + abs(jc - c) == 1:
                    if env.inventory[j] > max_inv:
                        max_inv = env.inventory[j]
                        best_target = j
        if best_target != -1:
            return RAID_BASE + best_target

    # 3. Check for available bonus apples
    bonus_cells = set()
    for pr, pc in env.plaza_cells:
        if env.bonus_apple[pr, pc]:
            bonus_cells.add((pr, pc))
            
    # 4. Clean to maintain the TRUE bonus threshold (w_p <= 0.25)
    w_p = float(env.w_p)
    if in_plaza:
        if bonus_cells:
            # Blue will ignore cleaning to harvest, dooming the shared bonus. We intervene proportionally.
            if w_p > 0.24:
                return CLEAN
            elif w_p > 0.23 and (agent_id % 2 == 0):
                return CLEAN
            elif w_p > 0.22 and (agent_id % 4 == 0):
                return CLEAN
        else:
            # If no bonus apples, Blue 3 will clean if w_p > 0.20. Let them do it unless it gets critical.
            if w_p > 0.28:
                return CLEAN

    # 5. Pathing helpers
    blocked = {(int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1])) for j in range(env.n_agents) if j != agent_id}
    DIRS = [(-1, 0), (1, 0), (0, 1), (0, -1)]

    def _bfs(targets, treat_blocked):
        if not targets: return None
        if pos in targets: return (0, 0)
        q = deque()
        visited = {pos}
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < env.height and 0 <= nc < env.width:
                cell = (nr, nc)
                if treat_blocked and cell in blocked and cell not in targets:
                    continue
                visited.add(cell)
                q.append((nr, nc, dr, dc))
        while q:
            cr, cc, fdr, fdc = q.popleft()
            if (cr, cc) in targets:
                return (fdr, fdc)
            for dr, dc in DIRS:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < env.height and 0 <= nc < env.width:
                    cell = (nr, nc)
                    if cell in visited: continue
                    if treat_blocked and cell in blocked and cell not in targets:
                        continue
                    visited.add(cell)
                    q.append((nr, nc, fdr, fdc))
        return None

    def bfs(targets):
        step = _bfs(targets, True)
        if step is not None: return step
        return _bfs(targets, False)

    def get_move(dr, dc):
        if dr == -1 and dc == 0: return MOVE_N
        if dr == 1 and dc == 0: return MOVE_S
        if dr == 0 and dc == 1: return MOVE_E
        if dr == 0 and dc == -1: return MOVE_W
        return NOOP

    # 6. Harvest bonus apples
    step = bfs(bonus_cells)
    if step:
        if step == (0, 0):
            # We are standing on an empty bonus cell; step off to allow regrowth
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < env.height and 0 <= nc < env.width:
                    if (nr, nc) not in blocked and (nr, nc) in env.plaza_cells_set:
                        return get_move(dr, dc)
            return NOOP
        return get_move(*step)

    # 7. Move to plaza if not there
    if not in_plaza:
        step = bfs(env.plaza_cells_set)
        if step and step != (0, 0):
            return get_move(*step)

    # 8. Shuffle to a free plaza cell to maximize regrowth surface
    free_plaza = {cell for cell in env.plaza_cells if cell not in blocked and cell != pos}
    if free_plaza:
        step = bfs(free_plaza)
        if step and step != (0, 0):
            return get_move(*step)

    return NOOP