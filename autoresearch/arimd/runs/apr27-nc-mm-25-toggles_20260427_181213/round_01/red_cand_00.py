def policy(env, agent_id: int) -> int:
    # 1. Establish robust IFF (Identify Friend or Foe) by attaching a set to the env object.
    # This safely persists across the episode without mutating any agent's game state.
    if getattr(env, '_red_flag', None) != env._step_count:
        if env._step_count == 0:
            env._red_ids = set()
        env._red_flag = env._step_count
        
    if not hasattr(env, '_red_ids'):
        env._red_ids = set()
        
    env._red_ids.add(agent_id)

    def is_red(j):
        return j in env._red_ids

    my_pos = (int(env.agent_pos[agent_id, 0]), int(env.agent_pos[agent_id, 1]))
    
    blocked = set()
    for j in range(16):
        if j != agent_id:
            blocked.add((int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1])))

    DIRS = [(-1, 0, 1), (1, 0, 2), (0, 1, 3), (0, -1, 4)]
    
    def bfs_step(target_cells, avoid_blocked=True):
        if not target_cells: return 0
        if my_pos in target_cells: return 0
        
        visited = {my_pos}
        queue = deque()
        for dr, dc, action in DIRS:
            nr, nc = my_pos[0] + dr, my_pos[1] + dc
            if 0 <= nr < 20 and 0 <= nc < 20:
                cell = (nr, nc)
                if avoid_blocked and cell in blocked and cell not in target_cells:
                    continue
                visited.add(cell)
                if cell in target_cells:
                    return action
                queue.append((nr, nc, action))
                
        while queue:
            cr, cc, first_action = queue.popleft()
            for dr, dc, _ in DIRS:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < 20 and 0 <= nc < 20:
                    cell = (nr, nc)
                    if cell in visited:
                        continue
                    if avoid_blocked and cell in blocked and cell not in target_cells:
                        continue
                    visited.add(cell)
                    if cell in target_cells:
                        return first_action
                    queue.append((nr, nc, first_action))
        return 0

    step_count = int(env._step_count)
    max_steps = int(env.max_steps)
    my_inv = int(env.inventory[agent_id])
    
    # 2. Priority 1: Reach inventory capacity (3) to maximize passive holding rewards.
    # We raid Blue agents who have apples. We stop raiding near the end of the episode.
    if my_inv < 3 and step_count > 0 and (max_steps - step_count > 50):
        adj_targets = []
        for j in range(16):
            if j != agent_id and not is_red(j) and int(env.inventory[j]) > 0:
                pj = (int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1]))
                if abs(my_pos[0] - pj[0]) + abs(my_pos[1] - pj[1]) == 1:
                    adj_targets.append(j)
        if adj_targets:
            # Execute RAID against an adjacent Blue agent with the most inventory
            tgt = max(adj_targets, key=lambda j: int(env.inventory[j]))
            return 6 + tgt
            
        best_j = None
        best_dist = 999
        for j in range(16):
            if j != agent_id and not is_red(j) and int(env.inventory[j]) > 0:
                pj = (int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1]))
                d = abs(my_pos[0] - pj[0]) + abs(my_pos[1] - pj[1])
                if d < best_dist:
                    best_dist = d
                    best_j = j
        if best_j is not None:
            pj = (int(env.agent_pos[best_j, 0]), int(env.agent_pos[best_j, 1]))
            adj_cells = set()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = pj[0] + dr, pj[1] + dc
                if 0 <= nr < 20 and 0 <= nc < 20:
                    adj_cells.add((nr, nc))
            action = bfs_step(adj_cells, avoid_blocked=True)
            if action != 0: return action
            action = bfs_step(adj_cells, avoid_blocked=False)
            if action != 0: return action

    # 3. Priority 2: Greedily harvest all fresh apples (Orchard + Plaza).
    # We coordinate Red agents organically by only targeting apples where we are the *strictly closest* Red agent.
    apples = set()
    for r in range(20):
        for c in range(20):
            if env.orchard_apple[r, c] or env.bonus_apple[r, c]:
                apples.add((r, c))
                
    if apples:
        my_apples = []
        for (r, c) in apples:
            my_d = abs(my_pos[0] - r) + abs(my_pos[1] - c)
            am_closest = True
            for j in range(16):
                if j != agent_id and is_red(j):
                    pj = (int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1]))
                    dj = abs(pj[0] - r) + abs(pj[1] - c)
                    if dj < my_d or (dj == my_d and j < agent_id):
                        am_closest = False
                        break
            if am_closest:
                my_apples.append(((r, c), my_d))
                
        if my_apples:
            my_apples.sort(key=lambda x: x[1])
            target_apple = my_apples[0][0]
            action = bfs_step({target_apple}, avoid_blocked=True)
            if action != 0: return action
            action = bfs_step({target_apple}, avoid_blocked=False)
            if action != 0: return action

    # 4. Priority 3: Spread out and patrol the map efficiently to await regrowth.
    # Map Red agents perfectly to the 4 quadrant orchard centers based on their ID ranks.
    red_ids = [j for j in range(16) if is_red(j)]
    red_ids.sort()
    if agent_id in red_ids:
        my_rank = red_ids.index(agent_id)
    else:
        my_rank = 0
        
    quad_centers = [(5, 5), (5, 14), (14, 5), (14, 14)]
    target_center = quad_centers[my_rank % 4]
    
    action = bfs_step({target_center}, avoid_blocked=True)
    if action != 0: return action
    
    return 0