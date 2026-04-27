def policy(env, agent_id: int) -> int:
    NOOP = 0; MOVE_N = 1; MOVE_S = 2; MOVE_E = 3; MOVE_W = 4
    
    # If mid-travel, we must NOOP
    if env._travel_queue[agent_id]:
        return NOOP

    r = int(env.agent_pos[agent_id, 0])
    c = int(env.agent_pos[agent_id, 1])
    pos = (r, c)
    clan = int(env.agent_clan[agent_id])
    
    # 1. Maintain max held inventory (3) by opportunistically raiding adjacent targets
    if env.inventory[agent_id] < 3:
        for j in range(env.n_agents):
            if j == agent_id:
                continue
            # Avoid same-clan raids as they automatically resolve to NOOP
            if env.agent_clan[j] == clan:
                continue
            if env.inventory[j] > 0:
                jr, jc = int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1])
                if abs(jr - r) + abs(jc - c) == 1:
                    return 6 + j  # RAID_BASE + target_id
            
    # 2. Identify all currently available apples across the entire map
    apple_cells = set()
    for tr in range(env.height):
        for tc in range(env.width):
            if env.orchard_apple[tr, tc] and (tr, tc) in env.orchard_cells_set:
                apple_cells.add((tr, tc))
                
    plaza_apples = set()
    for tr in range(env.height):
        for tc in range(env.width):
            if env.bonus_apple[tr, tc] and (tr, tc) in env.plaza_cells_set:
                plaza_apples.add((tr, tc))
                
    blocked = {(int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1])) 
               for j in range(env.n_agents) if j != agent_id}
               
    DIRS = ((-1, 0), (1, 0), (0, 1), (0, -1))
    
    def mv(dr, dc):
        if dr == -1 and dc == 0: return MOVE_N
        if dr == 1 and dc == 0: return MOVE_S
        if dr == 0 and dc == 1: return MOVE_E
        if dr == 0 and dc == -1: return MOVE_W
        return NOOP

    def bfs(target_set, treat_blocked):
        if not target_set: 
            return None
        if pos in target_set: 
            return (0, 0)
        
        q = deque()
        visited = {pos}
        
        # Enqueue first steps
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < env.height and 0 <= nc < env.width:
                if treat_blocked and (nr, nc) in blocked and (nr, nc) not in target_set:
                    continue
                visited.add((nr, nc))
                if (nr, nc) in target_set:
                    return (dr, dc)
                q.append((nr, nc, dr, dc))
                
        # Propagate the first-step direction (fdr, fdc)
        while q:
            cr, cc, fdr, fdc = q.popleft()
            for dr, dc in DIRS:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < env.height and 0 <= nc < env.width:
                    if (nr, nc) in visited: 
                        continue
                    if treat_blocked and (nr, nc) in blocked and (nr, nc) not in target_set:
                        continue
                    visited.add((nr, nc))
                    if (nr, nc) in target_set:
                        return (fdr, fdc)
                    q.append((nr, nc, fdr, fdc))
        return None

    # 3. Target prioritization
    # Priority A: Nearest Orchard apple globally (private reward)
    step = bfs(apple_cells, treat_blocked=True)
    if step is None: 
        step = bfs(apple_cells, treat_blocked=False)
    if step is not None:
        if step == (0, 0): return NOOP
        return mv(*step)
        
    # Priority B: Nearest Plaza bonus apple
    step = bfs(plaza_apples, treat_blocked=True)
    if step is None: 
        step = bfs(plaza_apples, treat_blocked=False)
    if step is not None:
        if step == (0, 0): return NOOP
        return mv(*step)
        
    # Priority C: Wait in our own orchard for regrowth
    own_orchard = set(env.orchard_cells_per_q[clan])
    step = bfs(own_orchard, treat_blocked=True)
    if step is None: 
        step = bfs(own_orchard, treat_blocked=False)
    if step is not None:
        if step == (0, 0): return NOOP
        return mv(*step)

    return NOOP