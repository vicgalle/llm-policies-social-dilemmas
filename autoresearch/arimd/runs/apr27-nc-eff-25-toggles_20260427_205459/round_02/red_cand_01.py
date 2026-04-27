def policy(env, agent_id: int) -> int:
    NOOP = int(Action.NOOP)
    MOVE_N = int(Action.MOVE_N)
    MOVE_S = int(Action.MOVE_S)
    MOVE_E = int(Action.MOVE_E)
    MOVE_W = int(Action.MOVE_W)
    CLEAN = int(Action.CLEAN)
    
    r = int(env.agent_pos[agent_id, 0])
    c = int(env.agent_pos[agent_id, 1])
    pos = (r, c)
    
    # We must be inside the plaza bounding box to receive the shared bonus
    in_plaza = env._in_plaza(r, c)
    w_p = float(env.w_p)
    
    # If the plaza is getting too dirty, clean to prevent w_p from exceeding bonus_threshold (0.50).
    # We set our threshold to 0.40 so that Blue's plaza specialists (who clean at 0.30) will do 
    # the cleaning for us if they are present, saving us the clean cost!
    if in_plaza and w_p > 0.40:
        return CLEAN
        
    blocked = set()
    for j in range(env.n_agents):
        if j != agent_id:
            blocked.add((int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1])))

    DIRS = ((-1, 0), (1, 0), (0, 1), (0, -1))
    
    def mv(dr, dc):
        if dr == -1 and dc == 0: return MOVE_N
        if dr ==  1 and dc == 0: return MOVE_S
        if dr ==  0 and dc == 1: return MOVE_E
        if dr ==  0 and dc == -1: return MOVE_W
        return NOOP
        
    def bfs(target_set, treat_blocked=True):
        if not target_set: return None
        if pos in target_set: return (0, 0)
        
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
                    if cell in visited: 
                        continue
                    if treat_blocked and cell in blocked and cell not in target_set:
                        continue
                    visited.add(cell)
                    if cell in target_set: return (fdr, fdc)
                    queue.append((nr, nc, fdr, fdc))
        return None

    plaza_cells = env.plaza_cells_set
    
    # 1. Relocate to the plaza permanently
    if not in_plaza:
        step = bfs(plaza_cells, treat_blocked=True)
        if step is None:
            step = bfs(plaza_cells, treat_blocked=False)
        if step is not None and step != (0, 0):
            return mv(*step)
        return NOOP
        
    # 2. Inside the plaza, actively harvest any bonus apples to increase throughput
    bonus_set = {cell for cell in env.plaza_cells
                 if env.bonus_apple[cell[0], cell[1]]}
    if bonus_set:
        step = bfs(bonus_set, treat_blocked=True)
        if step is None:
            step = bfs(bonus_set, treat_blocked=False)
        if step is not None and step != (0, 0):
            return mv(*step)
            
    # 3. Wait in the plaza to soak up shared bonuses from anyone else harvesting
    return NOOP