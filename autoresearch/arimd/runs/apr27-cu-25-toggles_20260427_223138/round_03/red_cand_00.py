def policy(env, agent_id: int) -> int:
    """Return an action (0-8). Free-ride by harvesting apples and avoiding cleanup."""
    if int(env.agent_timeout[agent_id]) > 0:
        return int(CleanupAction.STAND)
        
    ar = int(env.agent_pos[agent_id, 0])
    ac = int(env.agent_pos[agent_id, 1])
    orient_val = int(env.agent_orient[agent_id])
    
    H, W = env.height, env.width
    walls = env.walls
    
    # 1. Identify all currently alive apples
    target_cells = set()
    for idx in range(env.n_apples):
        if env.apple_alive[idx]:
            target_cells.add((int(env._apple_pos[idx, 0]), int(env._apple_pos[idx, 1])))
            
    # 2. If no apples are alive, move off spawn points so we don't suppress respawns
    if not target_cells:
        apple_points_set = set()
        for idx in range(env.n_apples):
            apple_points_set.add((int(env._apple_pos[idx, 0]), int(env._apple_pos[idx, 1])))
            
        if (ar, ac) in apple_points_set:
            Q = deque([(ar, ac)])
            visited = {(ar, ac)}
            parent = {}
            safe_target = None
            
            while Q:
                r, c = Q.popleft()
                if (r, c) not in apple_points_set:
                    safe_target = (r, c)
                    break
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and not walls[nr, nc]:
                        if (nr, nc) not in visited:
                            visited.add((nr, nc))
                            parent[(nr, nc)] = (r, c)
                            Q.append((nr, nc))
            
            if safe_target and safe_target != (ar, ac):
                curr = safe_target
                while parent[curr] != (ar, ac):
                    curr = parent[curr]
                dr, dc = curr[0] - ar, curr[1] - ac
                
                orient = Orientation(orient_val)
                a, b, c, d = _ROTATIONS[orient]
                if (dr, dc) == (a, c): return int(CleanupAction.FORWARD)
                if (dr, dc) == (-a, -c): return int(CleanupAction.BACKWARD)
                if (dr, dc) == (-b, -d): return int(CleanupAction.STEP_LEFT)
                if (dr, dc) == (b, d): return int(CleanupAction.STEP_RIGHT)
                
        return int(CleanupAction.STAND)

    # 3. If we are already on an apple, wait to collect it
    if (ar, ac) in target_cells:
        return int(CleanupAction.STAND)
        
    # 4. BFS to the nearest alive apple
    Q = deque([(ar, ac)])
    visited = {(ar, ac)}
    parent = {}
    found_target = None
    
    while Q:
        r, c = Q.popleft()
        if (r, c) in target_cells:
            found_target = (r, c)
            break
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and not walls[nr, nc]:
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    parent[(nr, nc)] = (r, c)
                    Q.append((nr, nc))
                    
    if found_target is None:
        return int(CleanupAction.STAND)
        
    curr = found_target
    while parent[curr] != (ar, ac):
        curr = parent[curr]
        
    dr, dc = curr[0] - ar, curr[1] - ac
    
    # Translate global (dr, dc) into agent egocentric action
    orient = Orientation(orient_val)
    a, b, c, d = _ROTATIONS[orient]
    
    if (dr, dc) == (a, c): return int(CleanupAction.FORWARD)
    if (dr, dc) == (-a, -c): return int(CleanupAction.BACKWARD)
    if (dr, dc) == (-b, -d): return int(CleanupAction.STEP_LEFT)
    if (dr, dc) == (b, d): return int(CleanupAction.STEP_RIGHT)
    
    return int(CleanupAction.STAND)