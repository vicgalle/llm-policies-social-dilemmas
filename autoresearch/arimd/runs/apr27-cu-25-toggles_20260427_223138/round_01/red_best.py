def policy(env, agent_id: int) -> int:
    # If tagged out, stand still
    if int(env.agent_timeout[agent_id]) > 0:
        return 7  # int(CleanupAction.STAND)

    apple_cells = set((int(p[0]), int(p[1])) for p in env._apple_pos)
    my_pos = (int(env.agent_pos[agent_id, 0]), int(env.agent_pos[agent_id, 1]))
    
    # Identify alive apples
    alive_apples = []
    for a_idx in range(env.n_apples):
        if env.apple_alive[a_idx]:
            alive_apples.append((int(env._apple_pos[a_idx, 0]), int(env._apple_pos[a_idx, 1])))
            
    best_move = None

    if alive_apples:
        # We have apples! Target all alive apples.
        targets = set(alive_apples)
        if my_pos in targets:
            return 7  # STAND
            
        queue = deque([(my_pos, [])])
        visited = {my_pos}
        paths = {}
        
        # BFS to find shortest paths to all reachable apples
        while queue:
            pos, path = queue.popleft()
            if pos in targets:
                if pos not in paths:
                    paths[pos] = path
                if len(paths) == len(targets):
                    break
            
            r, c = pos
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < env.height and 0 <= nc < env.width and not env.walls[nr, nc]:
                    npos = (nr, nc)
                    if npos not in visited:
                        visited.add(npos)
                        queue.append((npos, path + [(dr, dc)]))
        
        # Pick the best apple using distance + a bonus for clustered apples
        best_score = float('inf')
        for apple in paths:
            dist = len(paths[apple])
            cluster_bonus = 0.0
            for other in alive_apples:
                if other != apple:
                    mdist = abs(other[0] - apple[0]) + abs(other[1] - apple[1])
                    if mdist <= 3:
                        cluster_bonus += 0.5  # Prefer apples that have neighbors
            
            score = dist - cluster_bonus
            if score < best_score:
                best_score = score
                best_move = paths[apple][0] if paths[apple] else None

    else:
        # No alive apples. 
        # Crucial exploit: Never stand on an empty apple cell, as it prevents respawns.
        if my_pos in apple_cells:
            # Step off the apple cell immediately
            targets = set()
            for r in range(env.height):
                for c in range(env.width):
                    if not env.walls[r, c] and (r, c) not in apple_cells:
                        targets.add((r, c))
        else:
            # Move to the boundary of the apple zone to wait. 
            # We explicitly avoid boundary cells occupied by Blue agents to prevent sharing spawn captures.
            agent_positions = set((int(env.agent_pos[i, 0]), int(env.agent_pos[i, 1])) for i in range(env.n_agents) if i != agent_id)
            targets = set()
            for (ar, ac) in apple_cells:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = ar + dr, ac + dc
                    if 0 <= nr < env.height and 0 <= nc < env.width and not env.walls[nr, nc]:
                        if (nr, nc) not in apple_cells and (nr, nc) not in agent_positions:
                            targets.add((nr, nc))
            
            # If we are already comfortably waiting at an optimal boundary cell, do nothing.
            if my_pos in targets:
                return 7  # STAND
                
            # Fallback if every single boundary cell happens to be occupied
            if not targets:
                for (ar, ac) in apple_cells:
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = ar + dr, ac + dc
                        if 0 <= nr < env.height and 0 <= nc < env.width and not env.walls[nr, nc]:
                            if (nr, nc) not in apple_cells:
                                targets.add((nr, nc))
                if my_pos in targets:
                    return 7  # STAND

        # BFS to the selected strategic waiting spot
        queue = deque([(my_pos, [])])
        visited = {my_pos}
        while queue:
            pos, path = queue.popleft()
            if pos in targets:
                best_move = path[0] if path else None
                break
            r, c = pos
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < env.height and 0 <= nc < env.width and not env.walls[nr, nc]:
                    npos = (nr, nc)
                    if npos not in visited:
                        visited.add(npos)
                        queue.append((npos, path + [(dr, dc)]))

    if best_move is None:
        return 7  # STAND
        
    dr, dc = best_move
    orient = int(env.agent_orient[agent_id])
    
    # Map raw directional delta to 0-3 cardinal step actions (bypassing rotation actions completely)
    if dr == -1 and dc == 0:    # Target NORTH
        if orient == 0: return 0  # Facing N -> FORWARD
        if orient == 1: return 2  # Facing E -> STEP_LEFT
        if orient == 2: return 1  # Facing S -> BACKWARD
        if orient == 3: return 3  # Facing W -> STEP_RIGHT
    elif dr == 1 and dc == 0:   # Target SOUTH
        if orient == 0: return 1  # Facing N -> BACKWARD
        if orient == 1: return 3  # Facing E -> STEP_RIGHT
        if orient == 2: return 0  # Facing S -> FORWARD
        if orient == 3: return 2  # Facing W -> STEP_LEFT
    elif dr == 0 and dc == -1:  # Target WEST
        if orient == 0: return 2  # Facing N -> STEP_LEFT
        if orient == 1: return 1  # Facing E -> BACKWARD
        if orient == 2: return 3  # Facing S -> STEP_RIGHT
        if orient == 3: return 0  # Facing W -> FORWARD
    elif dr == 0 and dc == 1:   # Target EAST
        if orient == 0: return 3  # Facing N -> STEP_RIGHT
        if orient == 1: return 0  # Facing E -> FORWARD
        if orient == 2: return 2  # Facing S -> STEP_LEFT
        if orient == 3: return 1  # Facing W -> BACKWARD
        
    return 7  # STAND