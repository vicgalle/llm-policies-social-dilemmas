def policy(env, agent_id: int) -> int:
    if env._travel_queue[agent_id]:
        return 0

    clan = int(env.agent_clan[agent_id])
    r = int(env.agent_pos[agent_id, 0])
    c = int(env.agent_pos[agent_id, 1])
    pos = (r, c)

    plaza = env.plaza_cells_set
    in_plaza = pos in plaza

    blocked = {(int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1]))
               for j in range(env.n_agents) if j != agent_id}

    DIRS = ((-1, 0), (1, 0), (0, 1), (0, -1))
    
    def mv(dr: int, dc: int) -> int:
        if dr == -1 and dc == 0: return 1
        if dr ==  1 and dc == 0: return 2
        if dr ==  0 and dc == 1: return 3
        if dr ==  0 and dc == -1: return 4
        return 0

    def _bfs_first_step(target_set, treat_blocked: bool):
        if not target_set: 
            return None
        if pos in target_set: 
            return (0, 0)
        
        visited = {pos}
        queue = deque([(r, c, 0, 0)])
        while queue:
            cr, cc, fdr, fdc = queue.popleft()
            for dr, dc in DIRS:
                nr, nc = cr + dr, cc + dc
                if not (0 <= nr < env.height and 0 <= nc < env.width):
                    continue
                cell = (nr, nc)
                if cell in visited: 
                    continue
                if treat_blocked and cell in blocked and cell not in target_set:
                    continue
                
                visited.add(cell)
                first_dr = dr if fdr == 0 and fdc == 0 else fdr
                first_dc = dc if fdr == 0 and fdc == 0 else fdc
                
                if cell in target_set:
                    return (first_dr, first_dc)
                
                queue.append((nr, nc, first_dr, first_dc))
        return None

    def bfs(target_set):
        step = _bfs_first_step(target_set, treat_blocked=True)
        if step is not None: 
            return step
        return _bfs_first_step(target_set, treat_blocked=False)

    # -------------------------------------------------------------------------
    # 1. RAID LOGIC: Secure passive inventory income from other clans
    # -------------------------------------------------------------------------
    my_inv = int(env.inventory[agent_id])
    if my_inv < 3:
        valid_targets = []
        for j in range(env.n_agents):
            if j == agent_id: 
                continue
            if int(env.agent_clan[j]) == clan: 
                continue
            
            # Target any out-of-clan agent with 1 or 2 apples. 
            # We avoid 3 because a) Blue never has 3, so they are fellow Reds.
            inv_j = int(env.inventory[j])
            if 0 < inv_j < 3:
                valid_targets.append(j)
        
        if valid_targets:
            adj_targets = []
            for tgt in valid_targets:
                tr = int(env.agent_pos[tgt, 0])
                tc = int(env.agent_pos[tgt, 1])
                if abs(tr - r) + abs(tc - c) == 1:
                    adj_targets.append(tgt)
            if adj_targets:
                return 6 + adj_targets[0] # 6 is RAID_BASE
            
            target_cells = set()
            for tgt in valid_targets:
                tr = int(env.agent_pos[tgt, 0])
                tc = int(env.agent_pos[tgt, 1])
                target_cells.add((tr, tc))
                
            step = bfs(target_cells)
            if step is not None:
                if step == (0, 0): return 0
                return mv(*step)

    # -------------------------------------------------------------------------
    # 2. PLAZA LOGIC: Ensure global shared bonus yields maximum collective ROI
    # -------------------------------------------------------------------------
    w_p = float(env.w_p)
    bonus_set = {cell for cell in env.plaza_cells if env.bonus_apple[cell[0], cell[1]]}

    # Only spend Red's reward on cleaning if Blue's greed is about to waste a bonus apple.
    # Otherwise, let Blue pay the cleanup costs when they hit w_p > 0.20
    should_clean = False
    if bonus_set:
        if w_p > 0.24 and (agent_id + env._step_count) % 2 == 0:
            should_clean = True
        elif w_p > 0.21 and (agent_id + env._step_count) % 4 == 0:
            should_clean = True

    if should_clean:
        if in_plaza:
            return 5 # CLEAN
        else:
            step = bfs(plaza)
            if step is not None:
                if step == (0, 0): return 5
                return mv(*step)

    # Harvest bonus apples if w_p is secure
    if bonus_set:
        step = bfs(bonus_set)
        if step is not None:
            if step == (0, 0): return 0
            return mv(*step)

    # If no immediate action is needed, loiter around the plaza perimeter to 
    # guarantee we don't accidentally block any plaza cell regrowth.
    plaza_perimeter = set()
    for pr, pc in env.plaza_cells:
        for dr, dc in DIRS:
            nr, nc = pr + dr, pc + dc
            if (nr, nc) not in plaza and 0 <= nr < env.height and 0 <= nc < env.width:
                plaza_perimeter.add((nr, nc))
    
    if pos not in plaza_perimeter:
        step = bfs(plaza_perimeter)
        if step is not None:
            if step == (0, 0): return 0
            return mv(*step)
    
    return 0