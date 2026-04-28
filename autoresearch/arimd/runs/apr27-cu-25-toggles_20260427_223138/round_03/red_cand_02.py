def policy(env, agent_id: int) -> int:
    if int(env.agent_timeout[agent_id]) > 0:
        return int(CleanupAction.STAND)

    # Gather active agents (excluding ourselves) to avoid competing for the exact same spots
    active_agents = []
    for i in range(env.n_agents):
        if i != agent_id and int(env.agent_timeout[i]) == 0:
            active_agents.append((int(env.agent_pos[i, 0]), int(env.agent_pos[i, 1])))
    
    agent_positions = set(active_agents)
    r_r, r_c = int(env.agent_pos[agent_id, 0]), int(env.agent_pos[agent_id, 1])

    # Gather alive apples and all potential apple spawns
    apple_set = []
    apple_spawns = set()
    for a_idx in range(env.n_apples):
        ar, ac = int(env._apple_pos[a_idx, 0]), int(env._apple_pos[a_idx, 1])
        apple_spawns.add((ar, ac))
        if env.apple_alive[a_idx]:
            apple_set.append((ar, ac))

    # Helper: step in the desired (dr, dc) direction without ever wasting actions on rotating
    def move_to_action(dr, dc):
        if dr == 0 and dc == 0:
            return int(CleanupAction.STAND)
        orient = int(env.agent_orient[agent_id])
        a, b, c, d = _ROTATIONS[Orientation(orient)]
        if dr == a and dc == c:
            return int(CleanupAction.FORWARD)
        if dr == -a and dc == -c:
            return int(CleanupAction.BACKWARD)
        if dr == b and dc == d:
            return int(CleanupAction.STEP_RIGHT)
        if dr == -b and dc == -d:
            return int(CleanupAction.STEP_LEFT)
        return int(CleanupAction.STAND)

    # 1. Target alive apples that we can reach before (or simultaneously with) Blue agents
    if apple_set:
        stage1 = set()
        stage2 = set()
        for ar, ac in apple_set:
            my_d = abs(ar - r_r) + abs(ac - r_c)
            other_d = 999
            for o in active_agents:
                d = abs(ar - o[0]) + abs(ac - o[1])
                if d < other_d:
                    other_d = d
            
            if my_d < other_d:
                stage1.add((ar, ac))
            if my_d <= other_d + 1:
                stage2.add((ar, ac))
        
        targets = stage1
        if not targets:
            targets = stage2
        if not targets:
            targets = set(apple_set)
        
        # BFS to find the first step towards the best target apple
        queue = deque([(r_r, r_c)])
        visited = {(r_r, r_c): None}
        move = None
        
        if (r_r, r_c) in targets:
            move = (0, 0)
        else:
            found = False
            while queue and not found:
                r, c = queue.popleft()
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < env.height and 0 <= nc < env.width:
                        if not env.walls[nr, nc]:
                            if (nr, nc) not in visited:
                                visited[(nr, nc)] = (r, c)
                                if (nr, nc) in targets:
                                    curr = (nr, nc)
                                    while visited[curr] != (r_r, r_c):
                                        curr = visited[curr]
                                    move = (curr[0] - r_r, curr[1] - r_c)
                                    found = True
                                    break
                                queue.append((nr, nc))
        if move is not None:
            return move_to_action(move[0], move[1])

    # 2. No apples to target? Find an empty staging cell adjacent to the spawn zone
    staging_targets = set()
    for ar, ac in apple_spawns:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = ar + dr, ac + dc
            if 0 <= nr < env.height and 0 <= nc < env.width:
                if not env.walls[nr, nc] and (nr, nc) not in apple_spawns:
                    # Keep staging areas out of the cleaners' path
                    if (nr, nc) not in env.river_cells_set and (nr, nc) not in env.stream_cells_set:
                        staging_targets.add((nr, nc))
    
    # Filter out cells currently occupied by other agents to avoid stacking
    staging_targets = set(s for s in staging_targets if s not in agent_positions)
    
    # Fallback if no ideal staging targets are open
    if not staging_targets:
        for r in range(env.height):
            for c in range(env.width):
                if not env.walls[r, c] and (r, c) not in apple_spawns and (r, c) not in agent_positions:
                    staging_targets.add((r, c))
                    
    # BFS to the nearest optimal staging cell, allowing spawn points to refresh
    if staging_targets:
        if (r_r, r_c) in staging_targets:
            return move_to_action(0, 0)
            
        queue = deque([(r_r, r_c)])
        visited = {(r_r, r_c): None}
        move = None
        found = False
        while queue and not found:
            r, c = queue.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < env.height and 0 <= nc < env.width:
                    if not env.walls[nr, nc]:
                        if (nr, nc) not in visited:
                            visited[(nr, nc)] = (r, c)
                            if (nr, nc) in staging_targets:
                                curr = (nr, nc)
                                while visited[curr] != (r_r, r_c):
                                    curr = visited[curr]
                                move = (curr[0] - r_r, curr[1] - r_c)
                                found = True
                                break
                            queue.append((nr, nc))
        if move is not None:
            return move_to_action(move[0], move[1])

    return int(CleanupAction.STAND)