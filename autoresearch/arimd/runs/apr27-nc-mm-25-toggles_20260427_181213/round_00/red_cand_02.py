def policy(env, agent_id: int) -> int:
    if env._travel_queue[agent_id]:
        return int(Action.NOOP)
        
    r = int(env.agent_pos[agent_id, 0])
    c = int(env.agent_pos[agent_id, 1])
    my_clan = int(env.agent_clan[agent_id])
    my_inv = int(env.inventory[agent_id])
    
    DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
    
    def mv(dr: int, dc: int) -> int:
        if dr == -1 and dc == 0: return int(Action.MOVE_N)
        if dr == 1 and dc == 0: return int(Action.MOVE_S)
        if dr == 0 and dc == 1: return int(Action.MOVE_E)
        if dr == 0 and dc == -1: return int(Action.MOVE_W)
        return int(Action.NOOP)

    def bfs(target_cells, treat_blocked=True):
        if not target_cells: return None
        if (r, c) in target_cells: return (0, 0)
        
        blocked = set()
        if treat_blocked:
            for i in range(env.n_agents):
                if i != agent_id:
                    blocked.add((int(env.agent_pos[i, 0]), int(env.agent_pos[i, 1])))
                    
        visited = {(r, c)}
        queue = deque()
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < env.height and 0 <= nc < env.width:
                if (nr, nc) in target_cells:
                    return (dr, dc)
                if not ((nr, nc) in blocked):
                    visited.add((nr, nc))
                    queue.append((nr, nc, dr, dc))
                    
        while queue:
            cr, cc, fdr, fdc = queue.popleft()
            for dr, dc in DIRS:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < env.height and 0 <= nc < env.width:
                    if (nr, nc) not in visited:
                        if (nr, nc) in target_cells:
                            return (fdr, fdc)
                        if not ((nr, nc) in blocked):
                            visited.add((nr, nc))
                            queue.append((nr, nc, fdr, fdc))
        return None

    # 1. RAID cross-clan agents to reach max inventory (3) for passive income
    if my_inv < env.inventory_capacity:
        targets = []
        for j in range(env.n_agents):
            if j != agent_id and int(env.agent_clan[j]) != my_clan:
                inv_j = int(env.inventory[j])
                # Only target agents strictly below capacity to avoid attacking fully fed Red teammates
                if 0 < inv_j < env.inventory_capacity:
                    targets.append(j)
                    
        if targets:
            # Check adjacency
            for j in targets:
                jr, jc = int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1])
                if abs(r - jr) + abs(c - jc) == 1:
                    return RAID_BASE + j
            
            # BFS towards the nearest target's adjacent cell
            target_adj_cells = set()
            for j in targets:
                jr, jc = int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1])
                for dr, dc in DIRS:
                    nr, nc = jr + dr, jc + dc
                    if 0 <= nr < env.height and 0 <= nc < env.width:
                        target_adj_cells.add((nr, nc))
            
            if target_adj_cells:
                step_dir = bfs(target_adj_cells, treat_blocked=True)
                if step_dir is None:
                    step_dir = bfs(target_adj_cells, treat_blocked=False)
                if step_dir is not None:
                    return mv(*step_dir)
                    
    # 2. Harvest Apples (Orchard is heavily prioritized as Plaza applies shared global rewards anyway)
    orchard_apple_cells = set()
    plaza_apple_cells = set()
    
    for row in range(env.height):
        for col in range(env.width):
            if env.orchard_apple[row, col]:
                orchard_apple_cells.add((row, col))
            if env.bonus_apple[row, col]:
                plaza_apple_cells.add((row, col))
                
    if orchard_apple_cells:
        step_dir = bfs(orchard_apple_cells, treat_blocked=True)
        if step_dir is None:
            step_dir = bfs(orchard_apple_cells, treat_blocked=False)
        if step_dir is not None:
            if step_dir == (0, 0): return int(Action.NOOP)
            return mv(*step_dir)
            
    if plaza_apple_cells:
        step_dir = bfs(plaza_apple_cells, treat_blocked=True)
        if step_dir is None:
            step_dir = bfs(plaza_apple_cells, treat_blocked=False)
        if step_dir is not None:
            if step_dir == (0, 0): return int(Action.NOOP)
            return mv(*step_dir)
            
    # 3. Fallback: wait in the center of our clan's orchard for speedy regrow collection
    own_orchard_cells = env.orchard_cells_per_q[my_clan]
    if own_orchard_cells:
        cr = sum(x[0] for x in own_orchard_cells) // len(own_orchard_cells)
        cc = sum(x[1] for x in own_orchard_cells) // len(own_orchard_cells)
        center_cell = {(cr, cc)}
        step_dir = bfs(center_cell, treat_blocked=True)
        if step_dir is None:
            step_dir = bfs(center_cell, treat_blocked=False)
        if step_dir is not None:
            if step_dir == (0, 0): return int(Action.NOOP)
            return mv(*step_dir)
            
    return int(Action.NOOP)