def policy(env, agent_id) -> int:
    r = int(env.agent_pos[agent_id, 0])
    c = int(env.agent_pos[agent_id, 1])
    pos = (r, c)
    
    H, W = env.height, env.width
    DIRS = ((-1, 0), (1, 0), (0, 1), (0, -1))
    
    def mv(dr, dc):
        if dr == -1 and dc == 0: return int(Action.MOVE_N)
        if dr ==  1 and dc == 0: return int(Action.MOVE_S)
        if dr ==  0 and dc == 1: return int(Action.MOVE_E)
        if dr ==  0 and dc == -1: return int(Action.MOVE_W)
        return int(Action.NOOP)

    # Treat other agents as obstacles
    blocked = {(int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1])) 
               for j in range(env.n_agents) if j != agent_id}

    def bfs(targets, treat_blocked=True):
        if not targets:
            return None
        target_set = set(targets)
        if pos in target_set:
            return (0, 0)
            
        visited = {pos}
        queue = deque()
        
        # Initial expansion
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                if not treat_blocked or (nr, nc) not in blocked or (nr, nc) in target_set:
                    visited.add((nr, nc))
                    if (nr, nc) in target_set:
                        return (dr, dc)
                    queue.append((nr, nc, dr, dc))
                    
        # BFS traversal
        while queue:
            cr, cc, fdr, fdc = queue.popleft()
            for dr, dc in DIRS:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < H and 0 <= nc < W:
                    if (nr, nc) not in visited:
                        if not treat_blocked or (nr, nc) not in blocked or (nr, nc) in target_set:
                            visited.add((nr, nc))
                            if (nr, nc) in target_set:
                                return (fdr, fdc)
                            queue.append((nr, nc, fdr, fdc))
        return None

    def get_step(targets):
        # First try to find a path avoiding other agents
        s = bfs(targets, treat_blocked=True)
        if s is not None:
            return s
        # Fallback: find path ignoring agents (if path is temporarily blocked)
        return bfs(targets, treat_blocked=False)

    # 1. Emergency Public Good Maintenance
    # If the plaza gets too dirty, the massive shared bonus is lost.
    w_p = float(env.w_p)
    if w_p > 0.50:
        plaza = env.plaza_cells_set
        if pos in plaza:
            return int(Action.CLEAN)
        else:
            step_dir = get_step(plaza)
            if step_dir is not None and step_dir != (0, 0):
                return mv(*step_dir)

    # 2. Greedily free-ride on all Orchard Apples (from ANY clan)
    orchard_apples = []
    for cell in env.orchard_cells_list:
        if env.orchard_apple[cell[0], cell[1]]:
            orchard_apples.append(cell)
            
    step_dir = get_step(orchard_apples)
    if step_dir is not None:
        if step_dir == (0, 0):
            return int(Action.NOOP)
        return mv(*step_dir)
        
    # 3. If no Orchard Apples exist, help collect Plaza Apples to speed up their regrowth
    plaza_apples = []
    for cell in env.plaza_cells:
        if env.bonus_apple[cell[0], cell[1]]:
            plaza_apples.append(cell)
            
    step_dir = get_step(plaza_apples)
    if step_dir is not None:
        if step_dir == (0, 0):
            # If we just collected the plaza apple, step off so it can regrow
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in blocked:
                    return mv(dr, dc)
            return int(Action.NOOP)
        return mv(*step_dir)
        
    # 4. If the map is completely devoid of apples, camp in the cleanest clan's orchard
    best_clan = int(np.argmin(env.w_q))
    own_orchard = env.orchard_cells_per_q[best_clan]
    if pos in set(own_orchard):
        return int(Action.NOOP)
        
    step_dir = get_step(own_orchard)
    if step_dir is not None:
        return mv(*step_dir)
        
    return int(Action.NOOP)