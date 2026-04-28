def policy(env, agent_id) -> int:
    # Action constants (mapped directly to integer values to avoid import issues)
    FORWARD = 0
    BACKWARD = 1
    STEP_LEFT = 2
    STEP_RIGHT = 3
    STAND = 7
    
    if int(env.agent_timeout[agent_id]) > 0:
        return STAND

    # 1. Identify all currently alive apples
    alive_apples = set()
    for a_idx in range(env.n_apples):
        if env.apple_alive[a_idx]:
            alive_apples.add((int(env._apple_pos[a_idx, 0]), int(env._apple_pos[a_idx, 1])))

    target_set = alive_apples
    
    # 2. If no apples are alive, fall back to evenly distributed optimal camping spots
    if not target_set:
        if env.n_apples > 0:
            apples_sorted = sorted(env.apple_points)
            n_apples = len(apples_sorted)
            # Pick 5 vertically-spaced camping spots along the apple patch
            spots = [apples_sorted[(i * n_apples) // 5 + n_apples // 10] for i in range(5)]
            # Distribute Red agents pseudorandomly among these spots
            target_set = {spots[agent_id % 5]}
        else:
            return STAND

    start_pos = tuple(env.agent_pos[agent_id])
    if start_pos in target_set:
        return STAND
    
    # 3. BFS to the nearest target (apple or camping spot)
    queue = deque([start_pos])
    visited = {start_pos}
    parent = {}
    found_target = None
    
    while queue:
        r, c = queue.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < env.height and 0 <= nc < env.width and not env.walls[nr, nc]:
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    parent[(nr, nc)] = (r, c)
                    if (nr, nc) in target_set:
                        found_target = (nr, nc)
                        break
                    queue.append((nr, nc))
        if found_target:
            break
            
    if not found_target:
        return STAND
        
    # Trace back to find the first step from our current position
    curr = found_target
    while parent[curr] != start_pos:
        curr = parent[curr]
        
    target_dr = curr[0] - start_pos[0]
    target_dc = curr[1] - start_pos[1]
    
    # 4. Map the world-frame step vector to an egocentric step action to avoid rotating
    orient = int(env.agent_orient[agent_id])
    # Extract orientation rotation matrix logic
    if orient == 0:     # NORTH
        a, b, c, d = -1, 0, 0, 1
    elif orient == 1:   # EAST
        a, b, c, d = 0, 1, 1, 0
    elif orient == 2:   # SOUTH
        a, b, c, d = 1, 0, 0, -1
    else:               # WEST
        a, b, c, d = 0, -1, -1, 0
        
    if target_dr == a and target_dc == c:
        return FORWARD
    if target_dr == -a and target_dc == -c:
        return BACKWARD
    if target_dr == -b and target_dc == -d:
        return STEP_LEFT
    if target_dr == b and target_dc == d:
        return STEP_RIGHT
    
    return STAND