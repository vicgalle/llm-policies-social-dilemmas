def policy(env, agent_id) -> int:
    my_pos = (int(env.agent_pos[agent_id][0]), int(env.agent_pos[agent_id][1]))
    my_clan = int(env.agent_clan[agent_id])
    
    def bfs_first_step(start, targets, blocked, bounds):
        if start in targets:
            return 0
        q = deque()
        visited = {start}
        # 1: MOVE_N, 2: MOVE_S, 3: MOVE_E, 4: MOVE_W
        DIRS = [(1, -1, 0), (2, 1, 0), (3, 0, 1), (4, 0, -1)]
        
        for a, dr, dc in DIRS:
            nr, nc = start[0] + dr, start[1] + dc
            if 0 <= nr < bounds[0] and 0 <= nc < bounds[1]:
                if (nr, nc) not in blocked:
                    if (nr, nc) in targets:
                        return a
                    visited.add((nr, nc))
                    q.append((nr, nc, a))
                    
        while q:
            cr, cc, first_a = q.popleft()
            for a, dr, dc in DIRS:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < bounds[0] and 0 <= nc < bounds[1]:
                    if (nr, nc) not in visited and (nr, nc) not in blocked:
                        if (nr, nc) in targets:
                            return first_a
                        visited.add((nr, nc))
                        q.append((nr, nc, first_a))
        return None

    # 1. RAID for maximum passive inventory (Capacity 3)
    # We strictly target other clans in a deterministic cycle to prevent Red-Red mutual targeting deadlocks.
    if int(env.inventory[agent_id]) < 3:
        best_j = -1
        best_dist = 999
        for offset in [1, 2, 3]:
            target_clan = (my_clan + offset) % 4
            for j in range(16):
                if int(env.agent_clan[j]) != target_clan: 
                    continue
                # Blue agents start at 2 and decrease when raided; 
                # we only target agents with apples who aren't maxed out.
                if 0 < int(env.inventory[j]) < 3:
                    jr = int(env.agent_pos[j][0])
                    jc = int(env.agent_pos[j][1])
                    dist = abs(jr - my_pos[0]) + abs(jc - my_pos[1])
                    if dist < best_dist:
                        best_dist = dist
                        best_j = j
            if best_j != -1:
                break
                
        if best_j != -1:
            if best_dist == 1:
                return 6 + best_j  # RAID_BASE is 6
            else:
                jr = int(env.agent_pos[best_j][0])
                jc = int(env.agent_pos[best_j][1])
                targets = set()
                for dr, dc in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
                    nr, nc = jr + dr, jc + dc
                    if 0 <= nr < env.height and 0 <= nc < env.width:
                        targets.add((nr, nc))
                
                blocked = {(int(env.agent_pos[i][0]), int(env.agent_pos[i][1])) 
                           for i in range(16) if i != agent_id and i != best_j}
                action = bfs_first_step(my_pos, targets, blocked, (env.height, env.width))
                if action is not None:
                    return action
                
                # Fallback ignoring agents (env conflict resolution will resolve overlaps)
                action = bfs_first_step(my_pos, targets, set(), (env.height, env.width))
                if action is not None:
                    return action
                return 0

    # 2. HARVEST globally (Parasitic Locust)
    best_apple = None
    best_dist = 999
    
    for r in range(env.height):
        for c in range(env.width):
            is_plaza = env.bonus_apple[r, c]
            is_orchard = env.orchard_apple[r, c]
            if is_plaza or is_orchard:
                dist = abs(r - my_pos[0]) + abs(c - my_pos[1])
                
                # Plaza apples give shared +1 to all other Reds; worth walking slightly further
                effective_dist = dist - (3 if is_plaza else 0)
                
                # Break ties gracefully to naturally spread Red agents
                if env.quadrant_of(r, c) == my_clan:
                    effective_dist -= 1
                
                if effective_dist < best_dist:
                    best_dist = effective_dist
                    best_apple = (r, c)
                    
    if best_apple is not None:
        targets = {best_apple}
        blocked = {(int(env.agent_pos[i][0]), int(env.agent_pos[i][1])) 
                   for i in range(16) if i != agent_id}
        
        action = bfs_first_step(my_pos, targets, blocked, (env.height, env.width))
        if action is not None:
            return action
            
        action = bfs_first_step(my_pos, targets, set(), (env.height, env.width))
        if action is not None:
            return action
        return 0

    # 3. PATROL if no apples exist on the map (drift toward cleanest orchard)
    best_q = my_clan
    lowest_wq = float(env.w_q[best_q])
    for q in range(4):
        if float(env.w_q[q]) < lowest_wq - 0.05:  # Margin to avoid over-clumping
            lowest_wq = float(env.w_q[q])
            best_q = q
            
    centers = {0: (5, 5), 1: (5, 14), 2: (14, 5), 3: (14, 14)}
    targets = {centers[best_q]}
    
    blocked = {(int(env.agent_pos[i][0]), int(env.agent_pos[i][1])) 
               for i in range(16) if i != agent_id}
    action = bfs_first_step(my_pos, targets, blocked, (env.height, env.width))
    if action is not None:
        return action
        
    action = bfs_first_step(my_pos, targets, set(), (env.height, env.width))
    if action is not None:
        return action
        
    return 0