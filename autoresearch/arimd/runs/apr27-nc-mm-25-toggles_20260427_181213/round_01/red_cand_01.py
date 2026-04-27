def policy(env, agent_id: int) -> int:
    NOOP = 0
    RAID_BASE = 6
    
    my_inv = int(env.inventory[agent_id])
    my_clan = int(env.agent_clan[agent_id])
    r, c = int(env.agent_pos[agent_id, 0]), int(env.agent_pos[agent_id, 1])
    
    DIRS = [(-1, 0), (1, 0), (0, 1), (0, -1)]
    DIR_TO_ACTION = {
        (-1, 0): 1,
        (1, 0): 2,
        (0, 1): 3,
        (0, -1): 4
    }
    
    def move_to(target_cells):
        if not target_cells: return 0
        if (r, c) in target_cells: return 0
        
        queue = deque([(r, c, 0)])
        visited = set([(r, c)])
        while queue:
            cr, cc, first_move = queue.popleft()
            for dr, dc in DIRS:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < env.height and 0 <= nc < env.width:
                    if (nr, nc) not in visited:
                        act = first_move if first_move != 0 else DIR_TO_ACTION[(dr, dc)]
                        if (nr, nc) in target_cells:
                            return act
                        visited.add((nr, nc))
                        queue.append((nr, nc, act))
        return 0

    # Phase A: Hunt for 3rd held apple if we don't have it to max out passive income
    if my_inv < 3:
        best_target = -1
        # Try to raid adjacent valid targets first
        for j in range(env.n_agents):
            if j == agent_id: continue
            if int(env.agent_clan[j]) == my_clan: continue
            inv_j = int(env.inventory[j])
            if inv_j > 0:
                jr, jc = int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1])
                dist = abs(jr - r) + abs(jc - c)
                if dist == 1:
                    # Prioritize hitting Blue cooperators (who stay at 2 inventory)
                    if inv_j == 2:
                        best_target = j
                        break
                    else:
                        best_target = j
        if best_target != -1:
            return RAID_BASE + best_target
                    
        # Otherwise, track down the nearest valid target (if early enough to pay off)
        steps_left = getattr(env, 'max_steps', 1000) - getattr(env, '_step_count', 0)
        if steps_left > 20:
            targets = []
            for j in range(env.n_agents):
                if j == agent_id: continue
                if int(env.agent_clan[j]) == my_clan: continue
                if int(env.inventory[j]) > 0:
                    targets.append((int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1])))
            
            if targets:
                m = move_to(targets)
                if m != 0:
                    return m

    # Phase B: Aggressive Free-Riding (Vacuum up private Orchard apples)
    apple_cells = []
    for q in range(4):
        for cell in env.orchard_cells_per_q[q]:
            if env.orchard_apple[cell[0], cell[1]]:
                apple_cells.append(cell)
                
    if apple_cells:
        m = move_to(apple_cells)
        if m != 0:
            return m
            
    # Phase C: Waiting for apples
    # Return to our own clan's orchard so we are positioned for regrowth
    my_orchard = env.orchard_cells_per_q[my_clan]
    if (r, c) not in my_orchard:
        m = move_to(my_orchard)
        if m != 0: 
            return m
            
    return NOOP