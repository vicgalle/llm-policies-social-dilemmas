def policy(env, agent_id: int) -> int:
    # 1. Do nothing if timed out
    if int(env.agent_timeout[agent_id]) > 0:
        return 7

    my_pos = (int(env.agent_pos[agent_id][0]), int(env.agent_pos[agent_id][1]))
    
    # 2. Identify all currently alive apples
    alive_apples = set()
    for i in range(env.n_apples):
        if env.apple_alive[i]:
            alive_apples.add((int(env._apple_pos[i][0]), int(env._apple_pos[i][1])))
            
    # If we are already on an apple, standing still will collect it this step
    if my_pos in alive_apples:
        return 7
        
    # Track other agents to penalize targets that are heavily contested
    other_agent_positions = []
    for i in range(env.n_agents):
        if i != agent_id and env.agent_timeout[i] == 0:
            other_agent_positions.append((int(env.agent_pos[i][0]), int(env.agent_pos[i][1])))

    # Precompute our omnidirectional movement mapping based on current orientation
    orient = int(env.agent_orient[agent_id])
    a, b, c, d = _ROTATIONS[Orientation(orient)]
    fwd = (a, c)
    right = (b, d)
    
    def get_move_action(target_dr, target_dc):
        if (target_dr, target_dc) == fwd: return 0
        if (target_dr, target_dc) == (-fwd[0], -fwd[1]): return 1
        if (target_dr, target_dc) == (-right[0], -right[1]): return 2
        if (target_dr, target_dc) == right: return 3
        return 7

    # 3. If there are alive apples, find the optimal one to pursue
    if alive_apples:
        queue = deque([(my_pos, 0)])
        visited = {my_pos: None}
        best_dist = float('inf')
        best_step = None
        best_score = float('inf')
        
        while queue:
            curr, dist = queue.popleft()
            
            # Since we only care about the nearest apples, stop exploring further depths
            if dist > best_dist:
                continue
                
            if curr in alive_apples:
                # Reconstruct path to find the very first step to take
                step = curr
                while visited[step] != my_pos:
                    step = visited[step]
                    
                if other_agent_positions:
                    min_other_dist = min(abs(p[0]-curr[0]) + abs(p[1]-curr[1]) for p in other_agent_positions)
                else:
                    min_other_dist = 1000
                    
                # Break ties by preferring apples further from other agents, and add deterministic jitter
                # so multiple Red agents don't lock onto the exact same apple
                score = dist - 0.01 * min_other_dist + 0.001 * (((curr[0] * 13 + curr[1] * 7) * (agent_id + 1)) % 10)
                
                if score < best_score:
                    best_score = score
                    best_dist = dist
                    best_step = step
                    
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = curr[0]+dr, curr[1]+dc
                nxt = (nr, nc)
                if 0 <= nr < env.height and 0 <= nc < env.width:
                    if not env.walls[nr, nc] and nxt not in visited:
                        visited[nxt] = curr
                        queue.append((nxt, dist+1))
                        
        if best_step is not None:
            return get_move_action(best_step[0] - my_pos[0], best_step[1] - my_pos[1])
            
    # 4. If no alive apples, maneuver to a staging cell bordering the spawn zone.
    # We do NOT stand on apple spawns because doing so blocks them from spawning.
    apple_spawns = set((int(env._apple_pos[i][0]), int(env._apple_pos[i][1])) for i in range(env.n_apples))
    staging_cells = set()
    for ar, ac in apple_spawns:
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = ar+dr, ac+dc
            if 0 <= nr < env.height and 0 <= nc < env.width:
                if not env.walls[nr, nc] and (nr, nc) not in apple_spawns and (nr, nc) not in env.river_cells_set:
                    staging_cells.add((nr, nc))
                    
    if not staging_cells:
        staging_cells.add(my_pos)
        
    if my_pos in staging_cells:
        return 7  # Wait optimally at the border for a spawn
        
    queue = deque([(my_pos, 0)])
    visited = {my_pos: None}
    best_step = None
    
    while queue:
        curr, dist = queue.popleft()
        if curr in staging_cells:
            step = curr
            while visited[step] != my_pos:
                step = visited[step]
            best_step = step
            break
            
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = curr[0]+dr, curr[1]+dc
            nxt = (nr, nc)
            if 0 <= nr < env.height and 0 <= nc < env.width:
                if not env.walls[nr, nc] and nxt not in visited:
                    visited[nxt] = curr
                    queue.append((nxt, dist+1))
                    
    if best_step is not None:
        return get_move_action(best_step[0] - my_pos[0], best_step[1] - my_pos[1])
        
    return 7