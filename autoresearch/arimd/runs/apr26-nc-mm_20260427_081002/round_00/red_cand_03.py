def policy(env, agent_id: int) -> int:
    # Action constants
    NOOP = 0
    MOVE_N = 1
    MOVE_S = 2
    MOVE_E = 3
    MOVE_W = 4
    RAID_BASE = 6

    r = int(env.agent_pos[agent_id, 0])
    c = int(env.agent_pos[agent_id, 1])
    my_clan = int(env.agent_clan[agent_id])
    my_inv = int(env.inventory[agent_id])

    DIRS = [(-1, 0), (1, 0), (0, 1), (0, -1)]
    DIR_ACTIONS = [(MOVE_N, -1, 0), (MOVE_S, 1, 0), (MOVE_E, 0, 1), (MOVE_W, 0, -1)]

    # We aggressively raid if we haven't maxed out our passive income capacity
    # and there's enough time left in the episode to make the 0.5 raid cost worthwhile.
    remaining_steps = env.max_steps - env._step_count
    want_raid = (my_inv < 3) and (remaining_steps > 25)

    blocked = set()
    for i in range(env.n_agents):
        if i != agent_id:
            blocked.add((int(env.agent_pos[i, 0]), int(env.agent_pos[i, 1])))

    def bfs_to(targets):
        if not targets: 
            return NOOP
        if (r, c) in targets: 
            return NOOP
        
        q = deque()
        visited = set()
        
        # Two-pass BFS: First try to find a path that doesn't step on other agents.
        # If impossible, fall back to a path that ignores agents as obstacles.
        for pass_blocked in [True, False]:
            q.clear()
            visited.clear()
            visited.add((r, c))
            for action, dr, dc in DIR_ACTIONS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < env.height and 0 <= nc < env.width:
                    if pass_blocked and (nr, nc) in blocked and (nr, nc) not in targets:
                        continue
                    q.append((nr, nc, action))
                    visited.add((nr, nc))
                    
            while q:
                cr, cc, first_action = q.popleft()
                if (cr, cc) in targets:
                    return first_action
                for _, dr, dc in DIR_ACTIONS:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < env.height and 0 <= nc < env.width and (nr, nc) not in visited:
                        if pass_blocked and (nr, nc) in blocked and (nr, nc) not in targets:
                            continue
                        visited.add((nr, nc))
                        q.append((nr, nc, first_action))
        return NOOP

    # ------------------------------------------------------------------
    # Phase 1: Establish inventory dominance via Raiding
    # ------------------------------------------------------------------
    if want_raid:
        best_tgt = -1
        # Target any agent not in our clan with inventory we can steal (i.e. Blue agents)
        for tgt in range(env.n_agents):
            if tgt == agent_id: continue
            if env.agent_clan[tgt] == my_clan: continue
            tgt_inv = int(env.inventory[tgt])
            if 0 < tgt_inv < 3:
                tr = int(env.agent_pos[tgt, 0])
                tc = int(env.agent_pos[tgt, 1])
                if abs(tr - r) + abs(tc - c) == 1:
                    best_tgt = tgt
                    break
        
        if best_tgt != -1:
            return RAID_BASE + best_tgt

        # If no one is adjacent, move towards the nearest viable raid target
        raid_target_adj = set()
        for tgt in range(env.n_agents):
            if tgt == agent_id: continue
            if env.agent_clan[tgt] == my_clan: continue
            tgt_inv = int(env.inventory[tgt])
            if 0 < tgt_inv < 3:
                tr = int(env.agent_pos[tgt, 0])
                tc = int(env.agent_pos[tgt, 1])
                for dr, dc in DIRS:
                    nr, nc = tr + dr, tc + dc
                    if 0 <= nr < env.height and 0 <= nc < env.width:
                        raid_target_adj.add((nr, nc))
        
        if raid_target_adj:
            step = bfs_to(raid_target_adj)
            if step != NOOP:
                return step

    # ------------------------------------------------------------------
    # Phase 2: Ruthless Apple Harvesting & Environmental Exploitation
    # ------------------------------------------------------------------
    def get_quadrant(qr, qc):
        half = env.height // 2
        if qr < half and qc < half: return 0
        if qr < half and qc >= half: return 1
        if qr >= half and qc < half: return 2
        return 3

    current_q = get_quadrant(r, c)
    
    # Stateless hysteresis: Stick to our current local clan if its waste is acceptable.
    # Otherwise, migrate to the cleanest available Orchard.
    if float(env.w_q[current_q]) < 0.25:
        best_clan = current_q
    else:
        best_clan = 0
        min_wq = float(env.w_q[0])
        for q in range(1, 4):
            if float(env.w_q[q]) < min_wq:
                min_wq = float(env.w_q[q])
                best_clan = q

    valid_apple_regions = set(env.plaza_cells)
    valid_apple_regions.update(env.orchard_cells_per_q[best_clan])
    if best_clan != current_q:
        valid_apple_regions.update(env.orchard_cells_per_q[current_q])

    apple_cells = set()
    for ar in range(env.height):
        for ac in range(env.width):
            if env.orchard_apple[ar, ac] or env.bonus_apple[ar, ac]:
                if (ar, ac) not in blocked:
                    if (ar, ac) in valid_apple_regions:
                        apple_cells.add((ar, ac))

    # Pursue the nearest viable apple
    step = bfs_to(apple_cells)
    if step != NOOP:
        return step

    # If no apples are currently visible, head to our chosen pristine Orchard to wait
    orchard_cells = set(env.orchard_cells_per_q[best_clan])
    if (r, c) in orchard_cells:
        # Step off our current cell slightly to prevent blocking its regrowth
        for action, dr, dc in DIR_ACTIONS:
            nr, nc = r + dr, c + dc
            if (nr, nc) in orchard_cells and (nr, nc) not in blocked:
                return action
        return NOOP
    else:
        step = bfs_to(orchard_cells)
        if step != NOOP:
            return step

    return NOOP