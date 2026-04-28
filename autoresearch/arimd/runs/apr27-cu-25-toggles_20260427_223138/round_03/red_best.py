def policy(env, agent_id) -> int:
    my_pos = tuple(env.agent_pos[agent_id])
    orient = int(env.agent_orient[agent_id])
    
    # 1. Collect all alive apples deterministically
    alive_apples = []
    for i in range(env.n_apples):
        if env.apple_alive[i]:
            alive_apples.append(tuple(env._apple_pos[i]))
            
    # 2. Fast BFS to compute shortest path distances
    def get_all_distances(start_pos):
        distances = {start_pos: 0}
        queue = deque([start_pos])
        while queue:
            curr = queue.popleft()
            dist = distances[curr]
            r, c = curr
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < env.height and 0 <= nc < env.width and not env.walls[nr, nc]:
                    if (nr, nc) not in distances:
                        distances[(nr, nc)] = dist + 1
                        queue.append((nr, nc))
        return distances

    my_dists = get_all_distances(my_pos)
    
    # 3. Predict other agents' intended apple targets
    waste_count = sum(1 for (r, c) in env.river_cells_list if env.waste[r, c])
    waste_d = waste_count / len(env.river_cells_list) if env.river_cells_list else 0.0
    
    n_cleaners_base = max(1, int(round(0.2 * env.n_agents)))
    if waste_d >= 0.32:
        n_cleaners = min(env.n_agents, n_cleaners_base + max(2, env.n_agents // 4))
    else:
        n_cleaners = n_cleaners_base
        
    other_targets = {}
    for i in range(env.n_agents):
        if i == agent_id or env.agent_timeout[i] > 0:
            continue
            
        is_cleaner = (i < n_cleaners)
        if is_cleaner and waste_d <= 0.05:
            is_cleaner = False
            
        if is_cleaner:
            # We assume active cleaners won't compete for apples
            continue
            
        o_pos = tuple(env.agent_pos[i])
        o_dists = get_all_distances(o_pos)
        
        best_d = 1e9
        best_a = None
        for a in alive_apples:
            if a in o_dists and o_dists[a] < best_d:
                best_d = o_dists[a]
                best_a = a
                
        if best_a is not None:
            other_targets[i] = (best_a, best_d)
            
    # 4. Evaluate all apples to find the best uncontested or winnable target
    best_apple = None
    best_score = -1e9
    
    for apple in alive_apples:
        if apple not in my_dists:
            continue
        d_me = my_dists[apple]
        
        d_others_targeting = 1e9
        should_yield = False
        
        for i, (tgt_a, tgt_d) in other_targets.items():
            if tgt_a == apple:
                if tgt_d < d_others_targeting:
                    d_others_targeting = tgt_d
                # Tie-breaker: let lower ID agents take it (matches env logic)
                if tgt_d <= d_me and i < agent_id:
                    should_yield = True
                    
        if d_me < d_others_targeting:
            score = 1000 - d_me
        elif d_me == d_others_targeting:
            if should_yield:
                score = -1000 - d_me
            else:
                score = 500 - d_me
        else:
            score = -1000 - d_me
            
        if score > best_score:
            best_score = score
            best_apple = apple
            
    target_pos = best_apple
    
    # 5. If no valid targets, move to the best staging cell right outside the orchard
    if target_pos is None or best_score < 0:
        apple_points = []
        for i in range(env.n_apples):
            apple_points.append(tuple(env._apple_pos[i]))
            
        if apple_points:
            center_r = sum(r for r, c in apple_points) / len(apple_points)
            center_c = sum(c for r, c in apple_points) / len(apple_points)
        else:
            center_r, center_c = env.height / 2, env.width / 2
            
        best_cell = None
        best_d = 1e9
        apple_set = set(apple_points)
        for r in range(env.height):
            for c in range(env.width):
                if not env.walls[r, c] and (r, c) not in apple_set:
                    dist_to_center = abs(r - center_r) + abs(c - center_c)
                    if dist_to_center < best_d:
                        best_d = dist_to_center
                        best_cell = (r, c)
                        
        if best_cell is not None and best_cell in my_dists:
            target_pos = best_cell
        else:
            target_pos = my_pos
            
    # 6. Step toward the target using backtracking
    if target_pos == my_pos:
        return int(CleanupAction.STAND)
        
    target_dists = get_all_distances(target_pos)
    
    best_next_step = None
    best_step_d = 1e9
    
    r, c = my_pos
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if (nr, nc) in target_dists:
            if target_dists[(nr, nc)] < best_step_d:
                best_step_d = target_dists[(nr, nc)]
                best_next_step = (dr, dc)
                
    if best_next_step is None:
        return int(CleanupAction.STAND)
        
    target_dr, target_dc = best_next_step
    
    # 7. Execute purely via strafing / forward-backward to save turns not rotating
    a, b, c_rot, d_rot = _ROTATIONS[Orientation(orient)]
    actions = {
        int(CleanupAction.FORWARD): (a, c_rot),
        int(CleanupAction.BACKWARD): (-a, -c_rot),
        int(CleanupAction.STEP_LEFT): (-b, -d_rot),
        int(CleanupAction.STEP_RIGHT): (b, d_rot)
    }
    
    for act, (adr, adc) in actions.items():
        if adr == target_dr and adc == target_dc:
            return act
            
    return int(CleanupAction.STAND)