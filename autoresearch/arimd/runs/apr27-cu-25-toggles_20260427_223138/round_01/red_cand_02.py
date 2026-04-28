def policy(env, agent_id: int) -> int:
    if int(env.agent_timeout[agent_id]) > 0:
        return 7
        
    ar, ac = int(env.agent_pos[agent_id, 0]), int(env.agent_pos[agent_id, 1])
    
    # 1. Gather all currently active apples
    alive_apples = set()
    for a_idx in range(env.n_apples):
        if env.apple_alive[a_idx]:
            alive_apples.add((int(env._apple_pos[a_idx, 0]), int(env._apple_pos[a_idx, 1])))
            
    # 2. If no apples are currently active, spread out deep inside the spawn area
    if not alive_apples:
        apple_set = set((int(env._apple_pos[i, 0]), int(env._apple_pos[i, 1])) for i in range(env.n_apples))
        if apple_set:
            apple_rs = [r for (r, c) in apple_set]
            apple_cs = [c for (r, c) in apple_set]
            min_r, max_r = min(apple_rs), max(apple_rs)
            min_c, max_c = min(apple_cs), max(apple_cs)
            
            center_r = (min_r + max_r) // 2
            diff = max(1, max_r - min_r)
            
            # Distribute Red agents evenly across the height of the spawn block
            offset = (agent_id * 7) % 5 - 2
            target_r = center_r + offset * (diff // 4)
            target_c = max_c - 1  # Deep inside the cluster
            
            target_r = max(0, min(env.height - 1, target_r))
            target_c = max(0, min(env.width - 1, target_c))
            
            # If we are within a couple of tiles of our ideal wait spot, stand still and wait
            if abs(ar - target_r) + abs(ac - target_c) <= 2:
                return 7
                
            alive_apples.add((target_r, target_c))
            
    # 3. BFS pathfinding strictly based on walls (ignoring other agents to take optimal overlaps)
    if alive_apples:
        if (ar, ac) in alive_apples:
            return 7
            
        queue = deque([(ar, ac)])
        visited = set([(ar, ac)])
        parent = {}
                
        step_dir = None
        while queue:
            r, c = queue.popleft()
            if (r, c) in alive_apples:
                curr = (r, c)
                while parent[curr] != (ar, ac):
                    curr = parent[curr]
                step_dir = (curr[0] - ar, curr[1] - ac)
                break
                
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < env.height and 0 <= nc < env.width:
                    if not env.walls[nr, nc] and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        parent[(nr, nc)] = (r, c)
                        queue.append((nr, nc))
                                
        # 4. Map movement directly to pure translation to bypass slow rotations
        if step_dir is not None:
            dr, dc = step_dir
            orient = int(env.agent_orient[agent_id])
            
            if orient == 0:    # NORTH
                if dr == -1 and dc == 0: return 0  # FORWARD
                if dr == 1 and dc == 0: return 1   # BACKWARD
                if dr == 0 and dc == -1: return 2  # STEP_LEFT
                if dr == 0 and dc == 1: return 3   # STEP_RIGHT
            elif orient == 1:  # EAST
                if dr == 0 and dc == 1: return 0
                if dr == 0 and dc == -1: return 1
                if dr == -1 and dc == 0: return 2
                if dr == 1 and dc == 0: return 3
            elif orient == 2:  # SOUTH
                if dr == 1 and dc == 0: return 0
                if dr == -1 and dc == 0: return 1
                if dr == 0 and dc == 1: return 2
                if dr == 0 and dc == -1: return 3
            elif orient == 3:  # WEST
                if dr == 0 and dc == -1: return 0
                if dr == 0 and dc == 1: return 1
                if dr == 1 and dc == 0: return 2
                if dr == -1 and dc == 0: return 3
                
    return 7