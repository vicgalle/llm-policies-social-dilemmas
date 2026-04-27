def policy(env, agent_id: int) -> int:
    NOOP, MOVE_N, MOVE_S, MOVE_E, MOVE_W, CLEAN = 0, 1, 2, 3, 4, 5
    
    r = int(env.agent_pos[agent_id, 0])
    c = int(env.agent_pos[agent_id, 1])
    pos = (r, c)
    
    plaza = env.plaza_cells_set
    in_plaza = pos in plaza
    w_p = float(env.w_p)
    
    # Avoid routing through cells currently occupied by other agents
    blocked = {(int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1]))
               for j in range(env.n_agents) if j != agent_id}
               
    DIRS = ((-1, 0), (1, 0), (0, 1), (0, -1))
    
    def mv(dr: int, dc: int) -> int:
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
            if not (0 <= nr < env.height and 0 <= nc < env.width): 
                continue
            cell = (nr, nc)
            if treat_blocked and cell in blocked: 
                continue
            visited.add(cell)
            if cell in target_set: 
                return (dr, dc)
            queue.append((nr, nc, dr, dc))
            
        while queue:
            cr, cc, fdr, fdc = queue.popleft()
            for dr, dc in DIRS:
                nr, nc = cr + dr, cc + dc
                if not (0 <= nr < env.height and 0 <= nc < env.width): 
                    continue
                cell = (nr, nc)
                if cell in visited: 
                    continue
                if treat_blocked and cell in blocked: 
                    continue
                visited.add(cell)
                if cell in target_set: 
                    return (fdr, fdc)
                queue.append((nr, nc, fdr, fdc))
        return None

    # Priority 1: Move into the plaza to ensure we receive the shared occupant bonus
    if not in_plaza:
        step = bfs(plaza, treat_blocked=True)
        if step is None:  # Fallback if entirely surrounded by other agents
            step = bfs(plaza, treat_blocked=False)
        if step is not None and step != (0, 0):
            return mv(*step)
        return NOOP

    # Priority 2: Fail-safe cleanup to keep the shared bonus active (w_p <= 0.5).
    # Stagger thresholds by agent_id to prevent multiple Red agents paying the cost simultaneously.
    clean_threshold = 0.40 + agent_id * 0.005
    if w_p > clean_threshold:
        return CLEAN
        
    # Priority 3: Grab any active bonus apples to secure the +2.0 collector premium.
    bonus_set = {cell for cell in env.plaza_cells if env.bonus_apple[cell[0], cell[1]]}
    if bonus_set:
        step = bfs(bonus_set, treat_blocked=True)
        if step is None:
            step = bfs(bonus_set, treat_blocked=False)
        if step is not None and step != (0, 0):
            return mv(*step)
            
    # Priority 4: Sit tight to avoid accidentally stepping off the plaza and missing payouts.
    return NOOP