def policy(env, agent_id: int, _state: dict = {}) -> int:
    step = int(env._step_count)
    
    # 0. Safely identify the Red agents within the episode without leaking across resets
    if "last_step" not in _state or _state["last_step"] > step:
        _state.clear()
        _state["red_ids"] = set()
    _state["last_step"] = step
    
    if "red_ids" not in _state:
        _state["red_ids"] = set()
    _state["red_ids"].add(agent_id)
    
    # Dynamically assign exactly one Red agent to manage the plaza
    reds = sorted(list(_state["red_ids"]))
    is_plaza_cleaner = (agent_id == reds[0])

    r = int(env.agent_pos[agent_id, 0])
    c = int(env.agent_pos[agent_id, 1])
    pos = (r, c)
    H, W = env.height, env.width
    clan = int(env.agent_clan[agent_id])
    inventory = int(env.inventory[agent_id])
    w_p = float(env.w_p)

    blocked = {(int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1]))
               for j in range(env.n_agents) if j != agent_id}

    DIRS = ((-1, 0), (1, 0), (0, 1), (0, -1))
    
    def mv(dr: int, dc: int) -> int:
        if dr == -1 and dc ==  0: return 1
        if dr ==  1 and dc ==  0: return 2
        if dr ==  0 and dc ==  1: return 3
        if dr ==  0 and dc == -1: return 4
        return 0

    def bfs(target_set: set, treat_blocked: bool = True):
        if not target_set: return None
        if pos in target_set: return (0, 0)
        
        visited = {pos}
        queue = deque()
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                cell = (nr, nc)
                if treat_blocked and cell in blocked and cell not in target_set:
                    continue
                visited.add(cell)
                if cell in target_set: return (dr, dc)
                queue.append((nr, nc, dr, dc))
                
        while queue:
            cr, cc, fdr, fdc = queue.popleft()
            for dr, dc in DIRS:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < H and 0 <= nc < W:
                    cell = (nr, nc)
                    if cell in visited: continue
                    if treat_blocked and cell in blocked and cell not in target_set:
                        continue
                    visited.add(cell)
                    if cell in target_set: return (fdr, fdc)
                    queue.append((nr, nc, fdr, fdc))
        return None

    def get_move(target_set: set) -> int:
        s = bfs(target_set, treat_blocked=True)
        if s is None:
            s = bfs(target_set, treat_blocked=False)
        if s is not None and s != (0, 0):
            return mv(*s)
        return 0

    in_plaza = env._in_plaza(r, c)
    plaza_cells = set(env.plaza_cells)

    # 1. Plaza Cleaner Urgent Clean (Priority #1 overrides everything to save the shared bonus)
    if is_plaza_cleaner and w_p > 0.22:
        if in_plaza:
            return 5  # CLEAN
        else:
            move = get_move(plaza_cells)
            if move != 0: return move

    # 2. Opportunistic Raiding to cap inventory at 3
    if inventory < 3:
        # Check adjacent for an immediate raid
        for tgt in range(env.n_agents):
            if tgt in _state["red_ids"]: continue
            if env.inventory[tgt] == 0: continue
            if env.agent_clan[tgt] == clan: continue  # Same clan resolves as NOOP
            r2, c2 = int(env.agent_pos[tgt, 0]), int(env.agent_pos[tgt, 1])
            if abs(r - r2) + abs(c - c2) == 1:
                return 6 + tgt  # RAID
        
        # Hunt nearest valid Blue agent
        targets = set()
        for tgt in range(env.n_agents):
            if tgt in _state["red_ids"]: continue
            if env.inventory[tgt] == 0: continue
            if env.agent_clan[tgt] == clan: continue
            r2, c2 = int(env.agent_pos[tgt, 0]), int(env.agent_pos[tgt, 1])
            # Plaza cleaner limits hunt to the plaza to avoid wandering off
            if is_plaza_cleaner and not env._in_plaza(r2, c2):
                continue
            targets.add((r2, c2))
            
        if targets:
            move = get_move(targets)
            if move != 0: return move

    # 3. Standard Roles Execution
    if is_plaza_cleaner:
        # Collect Plaza bonus apples
        bonus_set = {cell for cell in env.plaza_cells if env.bonus_apple[cell[0], cell[1]]}
        if bonus_set:
            move = get_move(bonus_set)
            if move != 0: return move

        if not in_plaza:
            move = get_move(plaza_cells)
            if move != 0: return move
        else:
            # Stand near the center of the plaza to quickly react to regrows
            cr = sum(pr for pr, pc in plaza_cells) // len(plaza_cells)
            cc = sum(pc for pr, pc in plaza_cells) // len(plaza_cells)
            if pos != (cr, cc):
                move = get_move({(cr, cc)})
                if move != 0: return move
        return 0

    else:
        # Harvester
        own_orchard = set(env.orchard_cells_per_q[clan])
        apple_cells = {cell for cell in own_orchard if env.orchard_apple[cell[0], cell[1]]}
        if apple_cells:
            move = get_move(apple_cells)
            if move != 0: return move

        if pos not in own_orchard:
            move = get_move(own_orchard)
            if move != 0: return move
        else:
            # Stand near the center of the orchard to maximize gathering potential
            cr = sum(r_cell for r_cell, c_cell in own_orchard) // len(own_orchard)
            cc = sum(c_cell for r_cell, c_cell in own_orchard) // len(own_orchard)
            if pos != (cr, cc):
                move = get_move({(cr, cc)})
                if move != 0: return move
        return 0