def policy(env, agent_id: int) -> int:
    if env._travel_queue[agent_id]:
        return int(Action.NOOP)

    my_clan = int(env.agent_clan[agent_id])
    my_inv = int(env.inventory[agent_id])
    max_inv = int(env.inventory_capacity)
    r = int(env.agent_pos[agent_id, 0])
    c = int(env.agent_pos[agent_id, 1])
    pos = (r, c)
    H, W = env.height, env.width

    plaza = env.plaza_cells_set
    rivers = env.river_cells_set
    river_to_q = env._river_to_q
    own_orchard = set(env.orchard_cells_per_q[my_clan])
    own_river = set(env.river_cells_per_q[my_clan])
    w_q_own = float(env.w_q[my_clan])
    
    blocked = {(int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1]))
               for j in range(env.n_agents) if j != agent_id}

    DIRS = ((-1, 0), (1, 0), (0, 1), (0, -1))

    def mv(dr: int, dc: int) -> int:
        if dr == -1 and dc == 0: return int(Action.MOVE_N)
        if dr ==  1 and dc == 0: return int(Action.MOVE_S)
        if dr ==  0 and dc == 1: return int(Action.MOVE_E)
        if dr ==  0 and dc == -1: return int(Action.MOVE_W)
        return int(Action.NOOP)

    def _bfs_first_step(target_set, treat_blocked: bool):
        if not target_set: return None
        if pos in target_set: return (0, 0)
        visited = {pos}
        queue = deque()
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W): continue
            cell = (nr, nc)
            if cell in visited: continue
            if treat_blocked and cell in blocked and cell not in target_set: continue
            visited.add(cell)
            if cell in target_set: return (dr, dc)
            queue.append((nr, nc, dr, dc))
        while queue:
            cr, cc, fdr, fdc = queue.popleft()
            for dr, dc in DIRS:
                nr, nc = cr + dr, cc + dc
                if not (0 <= nr < H and 0 <= nc < W): continue
                cell = (nr, nc)
                if cell in visited: continue
                if treat_blocked and cell in blocked and cell not in target_set: continue
                visited.add(cell)
                if cell in target_set: return (fdr, fdc)
                queue.append((nr, nc, fdr, fdc))
        return None

    def bfs(target_set):
        s = _bfs_first_step(target_set, treat_blocked=True)
        if s is not None: return s
        return _bfs_first_step(target_set, treat_blocked=False)

    # 1. High priority: Maximize passive inventory holding reward via cross-clan raids
    if my_inv < max_inv:
        valid_targets = []
        for j in range(env.n_agents):
            if j == agent_id: continue
            if int(env.agent_clan[j]) == my_clan: continue
            if int(env.inventory[j]) <= 0: continue
            if int(env.inventory[j]) >= max_inv: continue # Don't raid maxed-out Red teammates
            
            jr, jc = int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1])
            dist = abs(jr - r) + abs(jc - c)
            if dist == 1:
                return int(RAID_BASE) + j
            valid_targets.append((dist, j, (jr, jc)))
            
        if valid_targets:
            valid_targets.sort(key=lambda x: x[0])
            best_target_pos = valid_targets[0][2]
            step_dir = bfs({best_target_pos})
            if step_dir is not None and step_dir != (0, 0):
                return mv(*step_dir)

    # 2. Free-ride the Plaza 'P' role but fulfill 'C' clan duty when required to keep orchards alive
    within = int(agent_id) % 4
    step_count = int(env._step_count)
    K = env.max_steps // 4
    slot = min(step_count // max(K, 1), 3)
    
    ROLE_TABLE = (
        ('P', 'C', 'H', 'H'),
        ('H', 'P', 'C', 'H'),
        ('H', 'H', 'P', 'C'),
        ('C', 'H', 'H', 'P'),
    )
    role = ROLE_TABLE[within][slot]

    if role == 'C' and w_q_own > 0.08:
        def adjacent_to_own_river():
            for dr, dc in DIRS:
                nbr = (r + dr, c + dc)
                if nbr in rivers and river_to_q.get(nbr) == my_clan:
                    return True
            return False

        if adjacent_to_own_river():
            return int(Action.CLEAN)
        
        approach = set()
        for cell in own_river:
            for dr, dc in DIRS:
                nbr = (cell[0] + dr, cell[1] + dc)
                if not (0 <= nbr[0] < H and 0 <= nbr[1] < W): continue
                if nbr in rivers or nbr in plaza: continue
                approach.add(nbr)
        
        step_dir = bfs(approach)
        if step_dir is not None and step_dir != (0, 0):
            return mv(*step_dir)

    # 3. Super-harvester focus: Consume private orchard apples entirely instead of the public Plaza bonuses
    apple_cells = {cell for cell in own_orchard if env.orchard_apple[cell[0], cell[1]]}
    if apple_cells:
        step_dir = bfs(apple_cells)
        if step_dir is not None and step_dir != (0, 0):
            return mv(*step_dir)

    if own_orchard:
        step_dir = bfs(own_orchard)
        if step_dir is not None and step_dir != (0, 0):
            return mv(*step_dir)

    return int(Action.NOOP)