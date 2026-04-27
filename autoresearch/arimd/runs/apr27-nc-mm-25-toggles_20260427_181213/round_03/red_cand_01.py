def policy(env, agent_id: int) -> int:
    NOOP = 0
    MOVE_N = 1
    MOVE_S = 2
    MOVE_E = 3
    MOVE_W = 4

    if env._travel_queue[agent_id]:
        return NOOP

    H, W = env.height, env.width
    r = int(env.agent_pos[agent_id, 0])
    c = int(env.agent_pos[agent_id, 1])
    pos = (r, c)

    my_clan = int(env.agent_clan[agent_id])
    my_inv = int(env.inventory[agent_id])
    inv_cap = env.inventory_capacity

    blocked = {(int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1]))
               for j in range(env.n_agents) if j != agent_id}

    DIRS = ((-1, 0), (1, 0), (0, 1), (0, -1))

    def mv(dr, dc):
        if dr == -1 and dc == 0: return MOVE_N
        if dr ==  1 and dc == 0: return MOVE_S
        if dr ==  0 and dc == 1: return MOVE_E
        if dr ==  0 and dc == -1: return MOVE_W
        return NOOP

    def get_bfs_step(target_set, treat_blocked=True):
        if not target_set:
            return None
        if pos in target_set:
            return (0, 0)
        
        visited = {pos}
        queue = deque([(r, c, None, None)])
        
        while queue:
            cr, cc, fdr, fdc = queue.popleft()
            for dr, dc in DIRS:
                nr, nc = cr + dr, cc + dc
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                cell = (nr, nc)
                if cell in visited:
                    continue
                
                # Cannot move through blocked cells unless it's our ultimate destination
                if treat_blocked and cell in blocked and cell not in target_set:
                    continue
                    
                visited.add(cell)
                first_dr = dr if fdr is None else fdr
                first_dc = dc if fdc is None else fdc
                
                if cell in target_set:
                    return (first_dr, first_dc)
                    
                queue.append((nr, nc, first_dr, first_dc))
        return None

    def bfs(target_set):
        step = get_bfs_step(target_set, treat_blocked=True)
        if step is not None:
            return step
        # Fallback ignoring agents if totally jammed
        return get_bfs_step(target_set, treat_blocked=False)

    # Phase A: Maximize Inventory (Raid until max capacity)
    if my_inv < inv_cap:
        plaza_set = env.plaza_cells_set
        
        def get_targets(require_plaza):
            targets = []
            for j in range(env.n_agents):
                if j == agent_id: continue
                # Same-clan raids resolve as NOOP, so only target other clans
                if int(env.agent_clan[j]) == my_clan: continue
                if int(env.inventory[j]) <= 0: continue
                
                if require_plaza:
                    pos_j = (int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1]))
                    if pos_j not in plaza_set:
                        continue
                targets.append(j)
            return targets
            
        # Target the Plaza primarily to hit the Blue 'P' roles and minimize friendly fire
        valid_targets = get_targets(require_plaza=True)
        if not valid_targets:
            valid_targets = get_targets(require_plaza=False)
            
        if valid_targets:
            # Check for immediate adjacency to raid
            for j in valid_targets:
                tr, tc = int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1])
                if abs(tr - r) + abs(tc - c) == 1:
                    return raid(j)
                    
            # Path to the closest target
            target_cells = {(int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1])) for j in valid_targets}
            step = bfs(target_cells)
            if step is not None:
                if step == (0, 0):
                    return NOOP
                return mv(*step)

    # Phase B: Free-ride and Farm Orchard Apples
    target_clan = my_clan
    # Structural fail-safe: if our home orchard is virtually dead, seek greener pastures
    if float(env.w_q[my_clan]) >= 0.48:
        best_c = int(np.argmin(env.w_q))
        if float(env.w_q[best_c]) < 0.45:
            target_clan = best_c
            
    target_orchard = set(env.orchard_cells_per_q[target_clan])
    apple_cells = {cell for cell in target_orchard if env.orchard_apple[cell[0], cell[1]]}
    
    # Immediately grab available private apples
    if apple_cells:
        step = bfs(apple_cells)
        if step is not None:
            if step == (0, 0):
                return NOOP
            return mv(*step)
            
    # Position centrally inside the orchard to statistically beat Blue harvesters to spawns
    if target_orchard:
        center_r = sum(c[0] for c in target_orchard) // len(target_orchard)
        center_c = sum(c[1] for c in target_orchard) // len(target_orchard)
        
        central_cells = {
            (r, c) for r, c in target_orchard 
            if abs(r - center_r) <= 1 and abs(c - center_c) <= 1
        }
        
        if pos in central_cells:
            return NOOP
            
        unoccupied_central = central_cells - blocked
        if unoccupied_central:
            step = bfs(unoccupied_central)
            if step is not None:
                if step == (0, 0):
                    return NOOP
                return mv(*step)
                
        # If the center is full, settle for any unblocked spot inside the orchard
        if pos in target_orchard:
            return NOOP
            
        unoccupied = target_orchard - blocked
        if unoccupied:
            step = bfs(unoccupied)
            if step is not None:
                if step == (0, 0):
                    return NOOP
                return mv(*step)
                
    return NOOP