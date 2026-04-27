def policy(env, agent_id: int) -> int:
    # If the agent is currently locked in a travel queue, do nothing to let it resolve
    if hasattr(env, "_travel_queue") and env._travel_queue[agent_id]:
        return 0

    r, c = int(env.agent_pos[agent_id, 0]), int(env.agent_pos[agent_id, 1])
    pos = (r, c)
    H, W = env.height, env.width

    # Track blocked cells to avoid bumping into stationary agents
    blocked = set()
    for j in range(env.n_agents):
        if j != agent_id:
            blocked.add((int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1])))

    DIRS = ((-1, 0), (1, 0), (0, 1), (0, -1))

    def mv(dr: int, dc: int) -> int:
        if dr == -1 and dc == 0: return 1  # MOVE_N
        if dr ==  1 and dc == 0: return 2  # MOVE_S
        if dr ==  0 and dc == 1: return 3  # MOVE_E
        if dr ==  0 and dc == -1: return 4 # MOVE_W
        return 0                           # NOOP

    def get_bfs_move(target_set, treat_blocked: bool):
        if not target_set: 
            return None
        if pos in target_set: 
            return (0, 0)
        
        queue = deque([(pos[0], pos[1], 0, 0)])
        visited = {pos}
        
        while queue:
            cr, cc, fdr, fdc = queue.popleft()
            for dr, dc in DIRS:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < H and 0 <= nc < W:
                    cell = (nr, nc)
                    if cell not in visited:
                        # Avoid passing through blocked cells unless they are the target
                        if treat_blocked and cell in blocked and cell not in target_set:
                            continue
                        visited.add(cell)
                        ndr, ndc = (dr, dc) if fdr == 0 and fdc == 0 else (fdr, fdc)
                        if cell in target_set:
                            return (ndr, ndc)
                        queue.append((nr, nc, ndr, ndc))
        return None

    def bfs(target_set):
        # First attempt to find a path strictly avoiding other agents
        s = get_bfs_move(target_set, treat_blocked=True)
        if s is not None:
            return s
        # Fallback to a naive path if completely blocked
        return get_bfs_move(target_set, treat_blocked=False)

    # Priority 1: Exclusive Orchard apples (zero-sum advantage over Blue harvesters)
    orchard_apples = set()
    for cell in env.orchard_cells_list:
        if env.orchard_apple[cell[0], cell[1]]:
            orchard_apples.add(cell)
            
    if orchard_apples:
        move = bfs(orchard_apples)
        if move is not None:
            return mv(*move)

    # Priority 2: Plaza bonus apples (positive-sum, but worth grabbing if no orchard apples)
    plaza_apples = set()
    for cell in env.plaza_cells:
        if env.bonus_apple[cell[0], cell[1]]:
            plaza_apples.add(cell)
            
    if plaza_apples:
        move = bfs(plaza_apples)
        if move is not None:
            return mv(*move)

    # Priority 3: No apples currently exist -> migrate to the cleanest orchard to await regrowth
    best_q = env.quadrant_of(r, c)
    best_wq = float(env.w_q[best_q])
    
    # Check if another clan has significantly lower waste (faster regrowth rate)
    for q in range(4):
        wq = float(env.w_q[q])
        if wq < best_wq - 0.05:
            best_wq = wq
            best_q = q

    # If we are already in the optimal orchard, just stay put
    if env.quadrant_of(r, c) == best_q and pos in env.orchard_cells_per_q[best_q]:
        return 0

    # Otherwise, move into an unoccupied cell of the target clan's orchard
    target_orchard = {
        cell for cell in env.orchard_cells_per_q[best_q] 
        if cell not in blocked and cell != pos
    }
    
    if target_orchard:
        move = bfs(target_orchard)
        if move is not None:
            return mv(*move)

    return 0