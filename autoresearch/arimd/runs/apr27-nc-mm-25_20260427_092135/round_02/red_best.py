def policy(env, agent_id: int) -> int:
    if env._travel_queue[agent_id]:
        return 0

    r, c = int(env.agent_pos[agent_id, 0]), int(env.agent_pos[agent_id, 1])
    pos = (r, c)
    plaza = env.plaza_cells_set
    in_plaza = pos in plaza

    blocked = {(int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1]))
               for j in range(env.n_agents) if j != agent_id}

    DIRS = ((-1, 0), (1, 0), (0, 1), (0, -1))
    def mv(dr: int, dc: int) -> int:
        if dr == -1 and dc == 0: return 1
        if dr ==  1 and dc == 0: return 2
        if dr ==  0 and dc == 1: return 3
        if dr ==  0 and dc == -1: return 4
        return 0

    def bfs(target_set, treat_blocked=True):
        if not target_set: return None
        if pos in target_set: return (0, 0)
        from collections import deque
        visited = {pos}
        queue = deque()
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < env.height and 0 <= nc < env.width:
                cell = (nr, nc)
                if treat_blocked and cell in blocked and cell not in target_set:
                    continue
                visited.add(cell)
                if cell in target_set: return (dr, dc)
                queue.append((nr, nc, dr, dc))
        while queue:
            cr, cc, fdr, fdc = queue.popleft()
            for dr, dc in DIRS:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < env.height and 0 <= nc < env.width:
                    cell = (nr, nc)
                    if cell not in visited:
                        if treat_blocked and cell in blocked and cell not in target_set:
                            continue
                        visited.add(cell)
                        if cell in target_set: return (fdr, fdc)
                        queue.append((nr, nc, fdr, fdc))
        return None

    # Priority 1: Grab high-value plaza apples!
    bonus_set = {cell for cell in env.plaza_cells if env.bonus_apple[cell[0], cell[1]]}
    if bonus_set:
        s = bfs(bonus_set, treat_blocked=True)
        if s is None: s = bfs(bonus_set, treat_blocked=False)
        if s and s != (0, 0): return mv(*s)

    # Priority 2: Staggered emergency w_P cleaning (protects the 10.0 team bonus)
    w_p = float(env.w_p)
    clean_threshold = 0.5 + 0.02 * (agent_id % 4)
    if w_p > clean_threshold:
        if in_plaza:
            return 5  # CLEAN
        s = bfs(plaza, treat_blocked=True)
        if s is None: s = bfs(plaza, treat_blocked=False)
        if s and s != (0, 0): return mv(*s)

    # Priority 3: Grab our OWN clan's orchard apples
    clan = int(env.agent_clan[agent_id])
    own_orchard = set(env.orchard_cells_per_q[clan])
    own_apples = {cell for cell in own_orchard if env.orchard_apple[cell[0], cell[1]]}
    if own_apples:
        s = bfs(own_apples, treat_blocked=True)
        if s is None: s = bfs(own_apples, treat_blocked=False)
        if s and s != (0, 0): return mv(*s)

    # Priority 4: Wait perfectly perched on empty boundary cells (blocking 0 regrowth)
    WAIT_CELLS = {
        0: [(7, 8), (8, 7)],
        1: [(7, 11), (8, 12)],
        2: [(11, 7), (12, 8)],
        3: [(11, 12), (12, 11)],
    }
    wait_cells = set(WAIT_CELLS[clan])
    if pos not in wait_cells:
        s = bfs(wait_cells, treat_blocked=True)  # Don't force into blocked wait cells
        if s is not None and s != (0, 0): 
            return mv(*s)
        
        # Fallback 1: If wait cells are occupied, idle gracefully in own orchard
        if pos not in own_orchard:
            s2 = bfs(own_orchard, treat_blocked=True)
            if s2 is None: s2 = bfs(own_orchard, treat_blocked=False)
            if s2 and s2 != (0, 0): return mv(*s2)

    return 0