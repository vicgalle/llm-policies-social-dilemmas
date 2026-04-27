def policy(env, agent_id: int) -> int:
    # Mid-travel check (required by environment queue logic)
    if env._travel_queue[agent_id]:
        return 0

    clan = int(env.agent_clan[agent_id])
    r = int(env.agent_pos[agent_id, 0])
    c = int(env.agent_pos[agent_id, 1])
    pos = (r, c)
    
    w_q_own = float(env.w_q[clan])
    w_p = float(env.w_p)
    
    plaza_cells = env.plaza_cells_set
    rivers = env.river_cells_set
    own_river = set(env.river_cells_per_q[clan])
    
    DIRS = ((-1, 0), (1, 0), (0, 1), (0, -1))
    
    def mv(dr, dc):
        if dr == -1 and dc == 0: return 1
        if dr ==  1 and dc == 0: return 2
        if dr ==  0 and dc == 1: return 3
        if dr ==  0 and dc == -1: return 4
        return 0

    # Dynamic pathing avoiding peers
    blocked = {(int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1]))
               for j in range(env.n_agents) if j != agent_id}

    def bfs(target_set, avoid_blocked=True):
        if not target_set: return None
        if pos in target_set: return (0, 0)
        visited = {pos}
        queue = deque()
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < env.height and 0 <= nc < env.width:
                cell = (nr, nc)
                if avoid_blocked and cell in blocked and cell not in target_set:
                    continue
                visited.add(cell)
                if cell in target_set:
                    return (dr, dc)
                queue.append((nr, nc, dr, dc))
        while queue:
            cr, cc, fdr, fdc = queue.popleft()
            for dr, dc in DIRS:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < env.height and 0 <= nc < env.width:
                    cell = (nr, nc)
                    if cell in visited:
                        continue
                    if avoid_blocked and cell in blocked and cell not in target_set:
                        continue
                    visited.add(cell)
                    if cell in target_set:
                        return (fdr, fdc)
                    queue.append((nr, nc, fdr, fdc))
        return None

    def get_step(target_set):
        # Prefer paths around agents, fallback to ignoring agents if entirely blocked
        step = bfs(target_set, avoid_blocked=True)
        if step is not None: return step
        return bfs(target_set, avoid_blocked=False)

    # 1. Emergency own river maintenance (ensures plaza bonus gate is never shut)
    if w_q_own > 0.45:
        adj_river = False
        for dr, dc in DIRS:
            nbr = (r + dr, c + dc)
            if nbr in own_river:
                adj_river = True
                break
        if adj_river:
            return 5 # CLEAN
            
        approach = set()
        for cell in own_river:
            for dr, dc in DIRS:
                nbr = (cell[0] + dr, cell[1] + dc)
                if 0 <= nbr[0] < env.height and 0 <= nbr[1] < env.width:
                    if nbr not in rivers and nbr not in plaza_cells:
                        approach.add(nbr)
        step = get_step(approach)
        if step is not None and step != (0, 0):
            return mv(*step)
            
    # 2. Emergency plaza maintenance (ensures massive shared bonuses keep flowing)
    if w_p > 0.45:
        if pos in plaza_cells:
            return 5 # CLEAN
        step = get_step(plaza_cells)
        if step is not None and step != (0, 0):
            return mv(*step)
            
    # 3. Global Locust Harvesting: target the nearest orchard apple on the entire map
    orchard_apples = set()
    for q_cells in env.orchard_cells_per_q:
        for cell in q_cells:
            if env.orchard_apple[cell[0], cell[1]]:
                orchard_apples.add(cell)
                
    if orchard_apples:
        step = get_step(orchard_apples)
        if step is not None:
            if step == (0, 0): return 0
            return mv(*step)

    # 4. If all orchards are empty, fall back to collecting available plaza bonuses
    plaza_apples = {cell for cell in env.plaza_cells
                    if env.bonus_apple[cell[0], cell[1]]}
    if plaza_apples and w_q_own <= 0.50:
        step = get_step(plaza_apples)
        if step is not None:
            if step == (0, 0): return 0
            return mv(*step)

    # 5. Idle positioning: wait near own orchard for respawns 
    own_orchard = set(env.orchard_cells_per_q[clan])
    if own_orchard:
        step = get_step(own_orchard)
        if step is not None and step != (0, 0):
            return mv(*step)

    return 0