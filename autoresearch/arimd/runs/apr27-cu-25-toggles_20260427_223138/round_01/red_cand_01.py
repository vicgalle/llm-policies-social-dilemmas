def policy(env, agent_id) -> int:
    """Return the next action for the given agent_id."""
    if env.agent_timeout[agent_id] > 0:
        return int(CleanupAction.STAND)
        
    H, W = env.height, env.width
    
    # 1. Compute all-pairs shortest paths using BFS for all agents
    dists = {}
    next_steps = {}
    for i in range(env.n_agents):
        d = np.full((H, W), 1000)
        nxt = {}
        if env.agent_timeout[i] > 0:
            dists[i] = d
            next_steps[i] = nxt
            continue
            
        sr, sc = int(env.agent_pos[i, 0]), int(env.agent_pos[i, 1])
        d[sr, sc] = 0
        q = deque()
        
        # Initial expansion to capture the very first move to take
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = sr + dr, sc + dc
            if 0 <= nr < H and 0 <= nc < W and not env.walls[nr, nc]:
                d[nr, nc] = 1
                nxt[(nr, nc)] = (dr, dc)
                q.append((nr, nc, dr, dc))
                
        # Full BFS expansion
        while q:
            cr, cc, f_dr, f_dc = q.popleft()
            cd = d[cr, cc]
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < H and 0 <= nc < W and not env.walls[nr, nc]:
                    if d[nr, nc] > cd + 1:
                        d[nr, nc] = cd + 1
                        nxt[(nr, nc)] = (f_dr, f_dc)
                        q.append((nr, nc, f_dr, f_dc))
                        
        dists[i] = d
        next_steps[i] = nxt

    my_dist = dists[agent_id]
    
    # 2. Gather alive apples
    alive_apples = []
    for a_idx in range(env.n_apples):
        if env.apple_alive[a_idx]:
            alive_apples.append((int(env._apple_pos[a_idx, 0]), int(env._apple_pos[a_idx, 1])))
            
    # 3. Simulate Blue's greedy targeting to know which apples are lost causes
    blue_targets = {}
    for j in range(env.n_agents):
        if j == agent_id or env.agent_timeout[j] > 0:
            continue
        j_min_d = 1000
        j_closest = []
        for ar, ac in alive_apples:
            if dists[j][ar, ac] < j_min_d:
                j_min_d = dists[j][ar, ac]
                j_closest = [(ar, ac)]
            elif dists[j][ar, ac] == j_min_d:
                j_closest.append((ar, ac))
        for t in j_closest:
            if t not in blue_targets or j_min_d < blue_targets[t]:
                blue_targets[t] = j_min_d
                
    # 4. Find the best apple for us, heavily penalizing if a Blue is closer
    best_score = 1000000
    best_target = None
    for ar, ac in alive_apples:
        md = my_dist[ar, ac]
        if md >= 1000:
            continue
        score = md
        if (ar, ac) in blue_targets:
            blue_d = blue_targets[(ar, ac)]
            if blue_d < md:
                score += 100  # Will definitely lose the race
            elif blue_d == md:
                score += 5    # Tie; penalize slightly to prefer open apples
        
        if score < best_score:
            best_score = score
            best_target = (ar, ac)
            
    target = None
    if best_score < 100:
        target = best_target
    else:
        # 5. If no uncontested apples are found, move to an optimal wait spot
        # Avoid standing ON spawn points (blocks spawns). Just wait right outside them!
        apple_set = set(env.apple_points)
        adj_cells = set()
        for ar, ac in apple_set:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = ar + dr, ac + dc
                if 0 <= nr < H and 0 <= nc < W:
                    if not env.walls[nr, nc] and (nr, nc) not in apple_set:
                        adj_cells.add((nr, nc))
                        
        best_w_score = 1000000
        best_wait = None
        for wr, wc in adj_cells:
            md = my_dist[wr, wc]
            if md >= 1000:
                continue
            # Spread out from other harvesters to cover more spawn surface area
            nearby = sum(1 for j in range(env.n_agents) if j != agent_id and dists[j][wr, wc] <= 2)
            score = md + 5 * nearby
            if score < best_w_score:
                best_w_score = score
                best_wait = (wr, wc)
        target = best_wait
        
    # 6. Execute movement mapping
    if target is None or target not in next_steps[agent_id]:
        return int(CleanupAction.STAND)
        
    dr, dc = next_steps[agent_id][target]
    orient = int(env.agent_orient[agent_id])
    a, b, c, d = _ROTATIONS[Orientation(orient)]
    
    # 7. Strafe bypass: Match the world trajectory using step offsets, bypassing rotations
    if (dr, dc) == (a, c): return int(CleanupAction.FORWARD)
    if (dr, dc) == (-a, -c): return int(CleanupAction.BACKWARD)
    if (dr, dc) == (-b, -d): return int(CleanupAction.STEP_LEFT)
    if (dr, dc) == (b, d): return int(CleanupAction.STEP_RIGHT)
    
    return int(CleanupAction.STAND)