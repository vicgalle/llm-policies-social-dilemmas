def policy(env, agent_id: int) -> int:
    r, c = int(env.agent_pos[agent_id, 0]), int(env.agent_pos[agent_id, 1])
    pos = (r, c)
    
    blocked = set()
    for j in range(env.n_agents):
        if j != agent_id:
            blocked.add((int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1])))

    DIRS = ((-1, 0), (1, 0), (0, 1), (0, -1))

    def mv(dr, dc):
        if dr == -1 and dc == 0: return 1
        if dr ==  1 and dc == 0: return 2
        if dr ==  0 and dc == 1: return 3
        if dr ==  0 and dc == -1: return 4
        return 0

    def bfs(targets):
        if not targets: return None
        if pos in targets: return (0, 0)
        
        queue = deque([pos])
        visited = {pos}
        parent = {}
        
        while queue:
            curr = queue.popleft()
            if curr in targets:
                step = curr
                while parent[step] != pos:
                    step = parent[step]
                return (step[0] - pos[0], step[1] - pos[1])
                
            for dr, dc in DIRS:
                nr, nc = curr[0] + dr, curr[1] + dc
                if 0 <= nr < env.height and 0 <= nc < env.width:
                    if (nr, nc) not in visited:
                        if (nr, nc) not in blocked:
                            visited.add((nr, nc))
                            parent[(nr, nc)] = curr
                            queue.append((nr, nc))
                        elif (nr, nc) in targets:
                            # We can path toward a target even if it's currently occupied
                            visited.add((nr, nc))
                            parent[(nr, nc)] = curr
                            queue.append((nr, nc))
        return None

    # 1. Hunt / Raid for 3rd apple (maximize passive holding reward)
    if env.inventory[agent_id] < 3:
        for j in range(env.n_agents):
            if env.agent_clan[j] != env.agent_clan[agent_id] and 0 < env.inventory[j] <= 2:
                jr, jc = int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1])
                if abs(r - jr) + abs(c - jc) == 1:
                    return 6 + int(j)  # RAID_BASE = 6
                    
        target_agents = set()
        for j in range(env.n_agents):
            if env.agent_clan[j] != env.agent_clan[agent_id] and 0 < env.inventory[j] <= 2:
                target_agents.add((int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1])))
        if target_agents:
            step_dir = bfs(target_agents)
            if step_dir is not None:
                return mv(*step_dir)

    # 2. Harvest Orchard Apples (ignore Plaza completely to free-ride on Blue's collection)
    apple_cells = set()
    for row in range(env.height):
        for col in range(env.width):
            if env.orchard_apple[row, col]:
                apple_cells.add((row, col))
                
    if apple_cells:
        step_dir = bfs(apple_cells)
        if step_dir is not None:
            return mv(*step_dir)
            
    # 3. Drift to a healthy orchard if no apples are currently spawned
    healthy_cells = set()
    for q in range(4):
        if env.w_q[q] < 0.25:
            for cell in env.orchard_cells_per_q[q]:
                healthy_cells.add(cell)
                
    if not healthy_cells:
        best_q = 0
        for q in range(1, 4):
            if env.w_q[q] < env.w_q[best_q]:
                best_q = q
        for cell in env.orchard_cells_per_q[best_q]:
            healthy_cells.add(cell)
            
    if healthy_cells:
        step_dir = bfs(healthy_cells)
        if step_dir is not None:
            if step_dir == (0, 0):
                # Step to an adjacent empty cell in the same orchard to allow apples to spawn
                for dr, dc in DIRS:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in healthy_cells and (nr, nc) not in blocked:
                        return mv(dr, dc)
                return 0
            return mv(*step_dir)
            
    return 0