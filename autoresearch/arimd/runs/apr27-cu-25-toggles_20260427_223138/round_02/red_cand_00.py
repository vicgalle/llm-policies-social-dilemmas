def policy(env, agent_id) -> int:
    # 0. Check timeout (even though beam is disabled, good practice)
    if int(env.agent_timeout[agent_id]) > 0:
        return 7  # STAND
        
    # Translate our current orientation into the movement action map
    orient = int(env.agent_orient[agent_id])
    a, b, c, d = _ROTATIONS[Orientation(orient)]
    action_map = {
        (a, c): 0,        # FORWARD
        (-a, -c): 1,      # BACKWARD
        (-b, -d): 2,      # STEP_LEFT
        (b, d): 3         # STEP_RIGHT
    }
    
    start = (int(env.agent_pos[agent_id, 0]), int(env.agent_pos[agent_id, 1]))
    
    # 1. Parse current apple states
    alive_apples = set()
    empty_apples = set()
    for i in range(env.n_apples):
        pos = (int(env._apple_pos[i, 0]), int(env._apple_pos[i, 1]))
        if env.apple_alive[i]:
            alive_apples.add(pos)
        else:
            empty_apples.add(pos)
            
    if start in alive_apples:
        return 7
        
    # Break ties in BFS differently depending on our agent parity 
    # to encourage Red agents to naturally target different apples.
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if agent_id % 2 == 1:
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
    def get_action_towards(targets):
        if start in targets:
            return 7
            
        queue = deque([start])
        came_from = {start: None}
        target = None
        
        while queue:
            curr = queue.popleft()
            if curr in targets:
                target = curr
                break
                
            r, c = curr
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < env.height and 0 <= nc < env.width:
                    if not env.walls[nr, nc] and (nr, nc) not in came_from:
                        came_from[(nr, nc)] = curr
                        queue.append((nr, nc))
                        
        if target is not None:
            # Backtrack to the first step
            curr = target
            while came_from[curr] != start:
                curr = came_from[curr]
            dr = curr[0] - start[0]
            dc = curr[1] - start[1]
            return action_map.get((dr, dc), 7)
            
        return None

    # 2. Aggressively path to the closest alive apple
    if alive_apples:
        act = get_action_towards(alive_apples)
        if act is not None:
            return act

    # 3. If no apples are currently alive, prepare by camping adjacent to an empty spawn
    # We must stand *adjacent* to an empty apple cell, not ON it, so we don't block it from spawning.
    adj_empty = set()
    for er, ec in empty_apples:
        for dr, dc in dirs:
            nr, nc = er + dr, ec + dc
            if 0 <= nr < env.height and 0 <= nc < env.width:
                # Target is walkable and NOT an empty apple itself
                if not env.walls[nr, nc] and (nr, nc) not in empty_apples:
                    adj_empty.add((nr, nc))
                    
    # Look at where others are standing so we don't bunch up
    agent_positions = set()
    for j in range(env.n_agents):
        if j != agent_id:
            agent_positions.add((int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1])))
            
    # Try to grab a waiting spot that no one else is currently occupying
    valid_wait_cells = set()
    for cell in adj_empty:
        if cell not in agent_positions:
            valid_wait_cells.add(cell)
            
    # Fallback if every wait cell is somehow occupied
    if not valid_wait_cells:
        valid_wait_cells = adj_empty
        
    if valid_wait_cells:
        act = get_action_towards(valid_wait_cells)
        if act is not None:
            return act
            
    # Fallback to resting
    return 7