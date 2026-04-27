def policy(env, agent_id: int) -> int:
    if env._travel_queue[agent_id]:
        return 0

    clan = int(env.agent_clan[agent_id])
    my_role = int(agent_id) % 4
    H, W = env.height, env.width
    r = int(env.agent_pos[agent_id, 0])
    c = int(env.agent_pos[agent_id, 1])
    pos = (r, c)

    own_orchard = set(env.orchard_cells_per_q[clan])
    own_river = set(env.river_cells_per_q[clan])
    rivers = env.river_cells_set
    plaza = env.plaza_cells_set
    w_q_own = float(env.w_q[clan])
    w_p = float(env.w_p)

    blocked = {(int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1]))
               for j in range(env.n_agents) if j != agent_id}

    DIRS = ((-1, 0), (1, 0), (0, 1), (0, -1))

    def mv(dr, dc):
        if dr == -1 and dc == 0: return 1
        if dr ==  1 and dc == 0: return 2
        if dr ==  0 and dc == 1: return 3
        if dr ==  0 and dc == -1: return 4
        return 0

    def bfs(target_set):
        if not target_set: return None
        if pos in target_set: return (0, 0)
        
        visited = {pos}
        queue = deque()
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                cell = (nr, nc)
                if cell not in blocked or cell in target_set:
                    visited.add(cell)
                    if cell in target_set: return (dr, dc)
                    queue.append((nr, nc, dr, dc))
        
        while queue:
            cr, cc, fdr, fdc = queue.popleft()
            for dr, dc in DIRS:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < H and 0 <= nc < W:
                    cell = (nr, nc)
                    if cell not in visited and (cell not in blocked or cell in target_set):
                        visited.add(cell)
                        if cell in target_set: return (fdr, fdc)
                        queue.append((nr, nc, fdr, fdc))
                        
        visited = {pos}
        queue = deque()
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                cell = (nr, nc)
                visited.add(cell)
                if cell in target_set: return (dr, dc)
                queue.append((nr, nc, dr, dc))
        while queue:
            cr, cc, fdr, fdc = queue.popleft()
            for dr, dc in DIRS:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < H and 0 <= nc < W:
                    cell = (nr, nc)
                    if cell not in visited:
                        visited.add(cell)
                        if cell in target_set: return (fdr, fdc)
                        queue.append((nr, nc, fdr, fdc))
        return None

    # 1. Initial Raid Logic - max out inventory (passive points generator)
    inventory = int(env.inventory[agent_id])
    if inventory < 3:
        # Prioritize hunting the stationary Blue plaza specialists (j % 4 == 3)
        targets = [j for j in range(env.n_agents) 
                   if int(env.agent_clan[j]) != clan 
                   and int(env.inventory[j]) > 0 
                   and (j % 4 == 3)]
        if not targets:
            # Fallback to any non-clanmate with inventory
            targets = [j for j in range(env.n_agents) 
                       if int(env.agent_clan[j]) != clan 
                       and int(env.inventory[j]) > 0]
        
        if targets:
            for j in targets:
                jr, jc = int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1])
                if abs(jr - r) + abs(jc - c) == 1:
                    return 6 + j  # RAID_j
            
            target_cells = {(int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1])) for j in targets}
            step = bfs(target_cells)
            if step is not None and step != (0, 0):
                return mv(*step)

    # 2. Plaza Emergency Fallback - Hysteresis cleanup if ALL plaza specialists were unseated
    if my_role == 3:
        if w_p > 0.35 or (pos in plaza and w_p > 0.20):
            if pos not in plaza:
                step = bfs(plaza)
                if step is not None and step != (0, 0):
                    return mv(*step)
            else:
                return 5  # CLEAN

    # 3. River Cleanliness - cleanly mirror Blue Role 0 if we replaced them
    clean_trigger = 0.08 if my_role == 0 else 0.30
    if w_q_own > clean_trigger:
        adj_river = False
        for dr, dc in DIRS:
            if (r + dr, c + dc) in own_river:
                adj_river = True
                break
        if adj_river:
            return 5  # CLEAN
        
        approach = set()
        for cell in own_river:
            for dr, dc in DIRS:
                nbr = (cell[0] + dr, cell[1] + dc)
                if 0 <= nbr[0] < H and 0 <= nbr[1] < W:
                    if nbr not in rivers and nbr not in plaza:
                        approach.add(nbr)
        step = bfs(approach)
        if step is not None and step != (0, 0):
            return mv(*step)

    # 4. Harvest Orchard Apples
    apple_cells = {cell for cell in own_orchard if env.orchard_apple[cell[0], cell[1]]}
    if apple_cells:
        step = bfs(apple_cells)
        if step is not None and step != (0, 0):
            return mv(*step)

    # 5. Idle strictly at the mathematical center of the orchard for the geometric advantage
    if clan == 0: center = (5, 5)
    elif clan == 1: center = (5, 14)
    elif clan == 2: center = (14, 5)
    else: center = (14, 14)

    if pos != center:
        step = bfs({center})
        if step is not None and step != (0, 0):
            return mv(*step)

    return 0