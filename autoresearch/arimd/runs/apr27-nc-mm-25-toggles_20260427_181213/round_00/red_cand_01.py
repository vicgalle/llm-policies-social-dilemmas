def policy(env, agent_id) -> int:
    NOOP = 0
    MOVE_N = 1
    MOVE_S = 2
    MOVE_E = 3
    MOVE_W = 4
    CLEAN = 5
    RAID_BASE = 6

    # Mid-travel agents cannot act.
    if env._travel_queue[agent_id]:
        return NOOP

    r, c = int(env.agent_pos[agent_id, 0]), int(env.agent_pos[agent_id, 1])
    pos = (r, c)
    my_clan = int(env.agent_clan[agent_id])

    DIRS = ((-1, 0), (1, 0), (0, 1), (0, -1))
    
    def mv(dr, dc):
        if dr == -1 and dc == 0: return MOVE_N
        if dr == 1 and dc == 0: return MOVE_S
        if dr == 0 and dc == 1: return MOVE_E
        if dr == 0 and dc == -1: return MOVE_W
        return NOOP

    def bfs_to(targets):
        """Robust BFS that returns the first step towards the nearest target."""
        if not targets: return None
        if pos in targets: return (0, 0)
        
        blocked = set()
        for j in range(env.n_agents):
            if j != agent_id:
                blocked.add((int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1])))
                
        visited = set([pos])
        queue = deque([(pos[0], pos[1], None)])
        
        # Pass 1: Try pathing while treating other agents as obstacles
        while queue:
            cr, cc, first_move = queue.popleft()
            for dr, dc in DIRS:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < env.height and 0 <= nc < env.width:
                    if (nr, nc) not in visited:
                        if (nr, nc) not in blocked or (nr, nc) in targets:
                            visited.add((nr, nc))
                            fm = first_move if first_move is not None else (dr, dc)
                            if (nr, nc) in targets:
                                return fm
                            queue.append((nr, nc, fm))
                            
        # Pass 2: Fallback ignoring agents (lets env resolve collisions if crowded)
        visited = set([pos])
        queue = deque([(pos[0], pos[1], None)])
        while queue:
            cr, cc, first_move = queue.popleft()
            for dr, dc in DIRS:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < env.height and 0 <= nc < env.width:
                    if (nr, nc) not in visited:
                        visited.add((nr, nc))
                        fm = first_move if first_move is not None else (dr, dc)
                        if (nr, nc) in targets:
                            return fm
                        queue.append((nr, nc, fm))
        return None

    # Priority 1: Top up held apples to max capacity (3) via opportunistic cross-clan raids
    if env.inventory[agent_id] < 3:
        valid_targets = []
        for j in range(env.n_agents):
            if j == agent_id: continue
            if env.inventory[j] == 0: continue
            if env.agent_clan[j] == my_clan: continue  # Same clan raids are invalid
            
            # Target neighboring clan intentionally to limit accidental Red-on-Red clustering
            is_target_clan = (env.agent_clan[j] == (my_clan + 1) % 4)
            jr, jc = int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1])
            dist = abs(jr - r) + abs(jc - c)
            
            valid_targets.append((not is_target_clan, dist, (j + agent_id) % env.n_agents, j, (jr, jc)))
            
        if valid_targets:
            valid_targets.sort()
            _, dist, _, target_id, target_pos = valid_targets[0]
            if dist == 1:
                return RAID_BASE + target_id
            else:
                move = bfs_to({target_pos})
                if move:
                    if move == (0, 0): return NOOP
                    return mv(*move)

    # Priority 2: Universal Harvesting — steal any visible fresh apple (Plaza or Orchard)
    apples = set()
    for r_i in range(env.height):
        for c_i in range(env.width):
            if env.bonus_apple[r_i, c_i] or env.orchard_apple[r_i, c_i]:
                apples.add((r_i, c_i))
                
    if apples:
        move = bfs_to(apples)
        if move:
            if move == (0, 0): return NOOP
            return mv(*move)
            
    # Priority 3: Wait in the Plaza (central hub) for apples to regrow
    if pos not in env.plaza_cells_set:
        move = bfs_to(env.plaza_cells_set)
        if move:
            if move == (0, 0): return NOOP
            return mv(*move)
            
    return NOOP