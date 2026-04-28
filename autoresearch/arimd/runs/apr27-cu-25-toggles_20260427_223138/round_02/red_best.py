def policy(env, agent_id) -> int:
    if env.agent_timeout[agent_id] > 0:
        return int(CleanupAction.STAND)
        
    H, W = env.height, env.width
    walls = env.walls
    
    # 1. Fast BFS to compute distances from a given cell
    def get_distances(start_r, start_c):
        dist = np.full((H, W), 9999)
        dist[start_r, start_c] = 0
        q = deque([(start_r, start_c)])
        while q:
            r, c = q.popleft()
            d = dist[r, c]
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and not walls[nr, nc]:
                    if dist[nr, nc] == 9999:
                        dist[nr, nc] = d + 1
                        q.append((nr, nc))
        return dist

    agent_positions = [tuple(env.agent_pos[i]) for i in range(env.n_agents)]
    
    # Precompute distances from all active agents
    dists = []
    for i in range(env.n_agents):
        r, c = agent_positions[i]
        dists.append(get_distances(r, c))
        
    my_dists = dists[agent_id]
    
    # 2. Score apples to find the optimal target
    best_apple = None
    best_score = -999999
    
    for idx in range(env.n_apples):
        if not env.apple_alive[idx]:
            continue
        ar, ac = env._apple_pos[idx]
        my_d = my_dists[ar, ac]
        
        if my_d == 9999:
            continue
            
        # Distance of the closest competitor
        other_ds = [dists[i][ar, ac] for i in range(env.n_agents) 
                    if i != agent_id and env.agent_timeout[i] == 0]
        min_other = min(other_ds) if other_ds else 9999
        
        # Prioritize apples we can definitively claim
        if my_d < min_other:
            score = 10000 - my_d
        elif my_d == min_other:
            score = 5000 - my_d
        else:
            score = -my_d
            
        # Deterministic pseudo-random tie-breaker to spread out Red agents
        score += ((agent_id * 17 + idx * 31) % 10) * 0.1
            
        if score > best_score:
            best_score = score
            best_apple = (ar, ac)
            
    # 3. If no apples are currently spawned, move to a strategic waiting spot
    if best_apple is None:
        candidates = []
        apple_set = set(env.apple_points)
        river_set = env.river_cells_set
        stream_set = env.stream_cells_set
        
        for r in range(env.height):
            for c in range(env.width):
                if not walls[r, c] and (r, c) not in apple_set and (r, c) not in river_set and (r, c) not in stream_set:
                    candidates.append((r, c))
        
        if candidates and len(env.apple_points) > 0:
            avg_r = sum(p[0] for p in env.apple_points) / len(env.apple_points)
            avg_c = sum(p[1] for p in env.apple_points) / len(env.apple_points)
            
            # Spread waiting targets vertically based on agent ID
            offset_r = ((agent_id % 3) - 1) * 3
            target_r = avg_r + offset_r
            target_c = avg_c
            
            best_cand = None
            best_cand_dist = 999999
            for cand in candidates:
                dist_to_center = abs(cand[0] - target_r) + abs(cand[1] - target_c)
                if dist_to_center < best_cand_dist:
                    best_cand_dist = dist_to_center
                    best_cand = cand
            
            best_apple = best_cand
        else:
            best_apple = agent_positions[agent_id]
            
    # 4. Extract the shortest path backwards from the target
    start_r, start_c = agent_positions[agent_id]
    if start_r == best_apple[0] and start_c == best_apple[1]:
        return int(CleanupAction.STAND)
        
    target_dists = get_distances(best_apple[0], best_apple[1])
    
    best_moves = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = start_r + dr, start_c + dc
        if 0 <= nr < H and 0 <= nc < W and not walls[nr, nc]:
            d = target_dists[nr, nc]
            best_moves.append((d, dr, dc))
            
    if best_moves:
        best_moves.sort()
        if best_moves[0][0] == 9999:
            return int(CleanupAction.STAND)
        best_dr, best_dc = best_moves[0][1], best_moves[0][2]
    else:
        return int(CleanupAction.STAND)
        
    # 5. Translate delta to a cardinal strafe action, bypassing rotations entirely
    orient = int(env.agent_orient[agent_id])
    a, b, c, d_val = _ROTATIONS[Orientation(orient)]
    
    if best_dr == a and best_dc == c: 
        return int(CleanupAction.FORWARD)
    if best_dr == -a and best_dc == -c: 
        return int(CleanupAction.BACKWARD)
    if best_dr == -b and best_dc == -d_val: 
        return int(CleanupAction.STEP_LEFT)
    if best_dr == b and best_dc == d_val: 
        return int(CleanupAction.STEP_RIGHT)
    
    return int(CleanupAction.STAND)