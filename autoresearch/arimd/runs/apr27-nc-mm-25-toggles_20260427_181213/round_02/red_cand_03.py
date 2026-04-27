def policy(env, agent_id: int) -> int:
    NOOP = 0
    MOVE_N = 1
    MOVE_S = 2
    MOVE_E = 3
    MOVE_W = 4
    CLEAN = 5
    RAID_BASE = 6
    
    if env._travel_queue[agent_id]:
        return NOOP

    r = int(env.agent_pos[agent_id][0])
    c = int(env.agent_pos[agent_id][1])
    
    blocked = set()
    for j in range(env.n_agents):
        if j != agent_id:
            blocked.add((int(env.agent_pos[j][0]), int(env.agent_pos[j][1])))
            
    def bfs(targets, treat_blocked=True):
        if not targets: return NOOP
        target_set = set(targets)
        if (r, c) in target_set: return NOOP
        
        queue = deque()
        visited = set([(r, c)])
        
        for dr, dc, a in [(-1, 0, MOVE_N), (1, 0, MOVE_S), (0, 1, MOVE_E), (0, -1, MOVE_W)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < env.height and 0 <= nc < env.width:
                if treat_blocked and (nr, nc) in blocked and (nr, nc) not in target_set:
                    continue
                if (nr, nc) in target_set:
                    return a
                visited.add((nr, nc))
                queue.append((nr, nc, a))
                
        while queue:
            cr, cc, first_a = queue.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < env.height and 0 <= nc < env.width and (nr, nc) not in visited:
                    if treat_blocked and (nr, nc) in blocked and (nr, nc) not in target_set:
                        continue
                    if (nr, nc) in target_set:
                        return first_a
                    visited.add((nr, nc))
                    queue.append((nr, nc, first_a))
        return NOOP

    # 1. Fill inventory to 3 for passive per-step income
    # Only target agents with 0 < inventory < 3 to avoid chasing other Red agents
    if env.inventory[agent_id] < 3:
        adj = []
        for j in range(env.n_agents):
            if j != agent_id and 0 < env.inventory[j] < 3:
                jr, jc = int(env.agent_pos[j][0]), int(env.agent_pos[j][1])
                if abs(jr - r) + abs(jc - c) == 1:
                    adj.append(j)
        if adj:
            adj.sort(key=lambda j: env.inventory[j], reverse=True)
            return RAID_BASE + adj[0]
            
        agent_targets = []
        for j in range(env.n_agents):
            if j != agent_id and 0 < env.inventory[j] < 3:
                agent_targets.append((int(env.agent_pos[j][0]), int(env.agent_pos[j][1])))
        if agent_targets:
            act = bfs(agent_targets, treat_blocked=False)
            if act != NOOP:
                return act

    # 2. Greedily collect any available apples (plaza or orchard)
    apples = []
    for r_ in range(env.height):
        for c_ in range(env.width):
            if env.orchard_apple[r_, c_] or env.bonus_apple[r_, c_]:
                if (r_, c_) not in blocked:
                    apples.append((r_, c_))
                    
    if apples:
        act = bfs(apples, treat_blocked=True)
        if act != NOOP: return act
        act = bfs(apples, treat_blocked=False)
        if act != NOOP: return act

    # 3. If no apples exist, path toward the most viable spawning region
    best_q = int(np.argmin(env.w_q))
    targets = []
    if env.w_p < env.w_q[best_q]:
        targets = env.plaza_cells
    else:
        targets = env.orchard_cells_per_q[best_q]
        
    free_targets = [t for t in targets if t not in blocked]
    if free_targets:
        act = bfs(free_targets, treat_blocked=True)
        if act != NOOP: return act
        act = bfs(free_targets, treat_blocked=False)
        if act != NOOP: return act
        
    # 4. Random drift if completely path-blocked
    valid_moves = []
    for dr, dc, a in [(-1, 0, MOVE_N), (1, 0, MOVE_S), (0, 1, MOVE_E), (0, -1, MOVE_W)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < env.height and 0 <= nc < env.width and (nr, nc) not in blocked:
            valid_moves.append(a)
            
    if valid_moves:
        return valid_moves[(int(env._step_count) + agent_id) % len(valid_moves)]
        
    return NOOP