def policy(env, agent_id: int) -> int:
    NOOP = int(Action.NOOP)
    
    # If mid-travel queue is populated, our action is overridden anyway
    if env._travel_queue[agent_id]:
        return NOOP

    r, c = int(env.agent_pos[agent_id, 0]), int(env.agent_pos[agent_id, 1])
    my_clan = int(env.agent_clan[agent_id])
    my_inv = int(env.inventory[agent_id])
    inv_cap = int(env.inventory_capacity)

    DIRS = ((-1, 0), (1, 0), (0, 1), (0, -1))

    def mv(dr: int, dc: int) -> int:
        if dr == -1 and dc == 0: return int(Action.MOVE_N)
        if dr ==  1 and dc == 0: return int(Action.MOVE_S)
        if dr ==  0 and dc == 1: return int(Action.MOVE_E)
        if dr ==  0 and dc == -1: return int(Action.MOVE_W)
        return NOOP

    def bfs(target_set):
        if not target_set: 
            return None
        if (r, c) in target_set: 
            return (0, 0)
        
        blocked = {(int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1])) 
                   for j in range(env.n_agents) if j != agent_id}
        
        # Pass 1: route strictly around other agents. Pass 2: fallback, ignore agents.
        for treat_blocked in (True, False):
            visited = {(r, c)}
            queue = deque([(r, c, 0, 0)])
            while queue:
                cr, cc, fdr, fdc = queue.popleft()
                for dr, dc in DIRS:
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < env.height and 0 <= nc < env.width):
                        continue
                    cell = (nr, nc)
                    if cell in visited:
                        continue
                    if treat_blocked and cell in blocked and cell not in target_set:
                        continue
                    
                    if (cr, cc) == (r, c):
                        next_fdr, next_fdc = dr, dc
                    else:
                        next_fdr, next_fdc = fdr, fdc
                        
                    if cell in target_set:
                        return (next_fdr, next_fdc)
                        
                    visited.add(cell)
                    queue.append((nr, nc, next_fdr, next_fdc))
        return None

    # Goal 1: Maximize passive income by reaching inventory capacity
    if my_inv < inv_cap:
        valid_targets = []
        for j in range(env.n_agents):
            if j == agent_id: 
                continue
            if int(env.agent_clan[j]) == my_clan: 
                continue  # Same-clan raids do nothing
            # Target agents who have apples, but avoid maxed-out (successful) Red agents
            if 0 < int(env.inventory[j]) < inv_cap:
                valid_targets.append(j)
        
        if valid_targets:
            adj_targets = []
            for j in valid_targets:
                tr, tc = int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1])
                if abs(tr - r) + abs(tc - c) == 1:
                    adj_targets.append(j)
            
            if adj_targets:
                # Prioritize taking the raid immediately
                return int(RAID_BASE) + adj_targets[0]
            
            # Move towards the closest valid target
            target_cells = {(int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1])) for j in valid_targets}
            step = bfs(target_cells)
            if step is not None and step != (0, 0):
                return mv(*step)

    # Goal 2: Free-ride the plaza and ruthlessly farm the cleanest orchard
    cur_q = env.quadrant_of(r, c)
    best_clan = cur_q
    min_score = float(env.w_q[cur_q])
    
    # Identify which quadrant has the lowest w_q (highest regrowth probability)
    for q in range(NUM_CLANS):
        score = float(env.w_q[q])
        if q != cur_q:
            score += 0.10  # Apply hysteresis penalty to avoid oscillating mid-travel
        if score < min_score:
            min_score = score
            best_clan = q
            
    # Target existing fresh apples in the chosen clan's optimal orchard
    target_cells = set()
    for cell in env.orchard_cells_per_q[best_clan]:
        if env.orchard_apple[cell[0], cell[1]]:
            target_cells.add(cell)
            
    if target_cells:
        step = bfs(target_cells)
        if step is not None and step != (0, 0):
            return mv(*step)
            
    # If no apples are currently grown, loiter on an unoccupied tile in that orchard
    best_orchard = set(env.orchard_cells_per_q[best_clan])
    blocked = {(int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1])) 
               for j in range(env.n_agents) if j != agent_id}
               
    free_best_orchard = {cell for cell in best_orchard if cell not in blocked}
    if not free_best_orchard:
        free_best_orchard = best_orchard  # Fallback: squeeze in anyway
        
    step = bfs(free_best_orchard)
    if step is not None and step != (0, 0):
        return mv(*step)
        
    return NOOP