def policy(env, agent_id: int) -> int:
    NOOP = 0
    MOVE_N = 1
    MOVE_S = 2
    MOVE_E = 3
    MOVE_W = 4
    
    # If mid-travel (e.g. from an env-forced queue), we must yield
    if env._travel_queue[agent_id]:
        return NOOP
        
    r = int(env.agent_pos[agent_id, 0])
    c = int(env.agent_pos[agent_id, 1])
    pos = (r, c)
    clan = int(env.agent_clan[agent_id])
    
    DIRS = ((-1, 0), (1, 0), (0, 1), (0, -1))
    
    def mv(dr, dc):
        if dr == -1 and dc == 0: return MOVE_N
        if dr == 1 and dc == 0: return MOVE_S
        if dr == 0 and dc == 1: return MOVE_E
        if dr == 0 and dc == -1: return MOVE_W
        return NOOP

    # Treat all other agents as obstacles
    blocked = {(int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1]))
               for j in range(env.n_agents) if j != agent_id}

    def bfs(target_set):
        """Standard Manhattan BFS to the nearest target."""
        if not target_set: return None
        if pos in target_set: return (0, 0)
        
        # Pass 1: Strict pathing avoiding other agents
        visited = {pos}
        queue = deque([(r, c, 0, 0)])
        
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < env.height and 0 <= nc < env.width:
                cell = (nr, nc)
                if cell in target_set:
                    return (dr, dc)
                if cell not in blocked:
                    visited.add(cell)
                    queue.append((nr, nc, dr, dc))
                    
        while queue:
            cr, cc, fdr, fdc = queue.popleft()
            for dr, dc in DIRS:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < env.height and 0 <= nc < env.width:
                    cell = (nr, nc)
                    if cell in target_set:
                        return (fdr, fdc)
                    if cell not in visited and cell not in blocked:
                        visited.add(cell)
                        queue.append((nr, nc, fdr, fdc))
        
        # Pass 2: Fallback ignoring agents (allows stepping toward blocked goals)
        visited = {pos}
        queue = deque([(r, c, 0, 0)])
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < env.height and 0 <= nc < env.width:
                cell = (nr, nc)
                if cell in target_set:
                    return (dr, dc)
                visited.add(cell)
                queue.append((nr, nc, dr, dc))
                
        while queue:
            cr, cc, fdr, fdc = queue.popleft()
            for dr, dc in DIRS:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < env.height and 0 <= nc < env.width:
                    cell = (nr, nc)
                    if cell in target_set:
                        return (fdr, fdc)
                    if cell not in visited:
                        visited.add(cell)
                        queue.append((nr, nc, fdr, fdc))
                        
        return None

    # PHASE 1: Raid adjacent out-of-clan Blue targets if our inventory is below capacity (3).
    # Red maxes out at 3; Blue stays at 2. Filter isolates Blue and ignores maxed-out Red.
    if env.inventory[agent_id] < env.inventory_capacity:
        valid_targets = []
        for j in range(env.n_agents):
            if env.agent_clan[j] != clan and 0 < env.inventory[j] < 3:
                valid_targets.append(j)
        
        if valid_targets:
            adj_targets = []
            for j in valid_targets:
                dist = abs(env.agent_pos[j, 0] - r) + abs(env.agent_pos[j, 1] - c)
                if dist == 1:
                    adj_targets.append(j)
            
            # If anyone is adjacent, safely strike (cost only applied if they stay adjacent)
            if adj_targets:
                return int(RAID_BASE + adj_targets[0])
            
            # Otherwise, use standard 0-cost movement to chase a target
            target_cells = {(int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1])) for j in valid_targets}
            step = bfs(target_cells)
            if step is not None and step != (0, 0):
                return mv(*step)

    # PHASE 2: Greedily hunt the globally nearest private orchard apple across ALL quadrants
    apple_cells = set()
    for q_cells in env.orchard_cells_per_q:
        for cell in q_cells:
            if env.orchard_apple[cell[0], cell[1]]:
                apple_cells.add((cell[0], cell[1]))
                
    if apple_cells:
        step = bfs(apple_cells)
        if step is not None and step != (0, 0):
            return mv(*step)

    # PHASE 3: If no visible apples, identify the cleanest orchard and migrate to its center to wait
    best_q = None
    best_wq = 999.0
    for q in range(NUM_CLANS):
        if env.w_q[q] < best_wq:
            best_wq = env.w_q[q]
            best_q = q
            
    if best_q is not None:
        orchard_cells = env.orchard_cells_per_q[best_q]
        r_c = sum(c[0] for c in orchard_cells) // len(orchard_cells)
        c_c = sum(c[1] for c in orchard_cells) // len(orchard_cells)
        
        if pos != (r_c, c_c):
            step = bfs({(r_c, c_c)})
            if step is not None and step != (0, 0):
                return mv(*step)

    return NOOP