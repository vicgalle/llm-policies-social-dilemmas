def policy(env, agent_id: int) -> int:
    NOOP, MOVE_N, MOVE_S, MOVE_E, MOVE_W, CLEAN = 0, 1, 2, 3, 4, 5
    
    # If mid-travel (from an older action), must return NOOP to let the queue process
    if env._travel_queue[agent_id]:
        return NOOP

    r = int(env.agent_pos[agent_id, 0])
    c = int(env.agent_pos[agent_id, 1])
    pos = (r, c)
    
    # Treat other agents as obstacles so we can route around them
    blocked = {(int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1])) 
               for j in range(env.n_agents) if j != agent_id}
               
    H, W = env.height, env.width
    DIRS = ((-1, 0), (1, 0), (0, 1), (0, -1))
    
    def mv(dr, dc):
        if dr == -1 and dc == 0: return MOVE_N
        if dr == 1 and dc == 0: return MOVE_S
        if dr == 0 and dc == 1: return MOVE_E
        if dr == 0 and dc == -1: return MOVE_W
        return NOOP

    def get_bfs_step(targets, avoid_blocked):
        if not targets:
            return None
        if pos in targets:
            return (0, 0)
            
        visited = {pos}
        queue = deque()
        # Seed initial steps
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                cell = (nr, nc)
                # Allow stepping onto a target even if it's currently occupied
                if avoid_blocked and cell in blocked and cell not in targets:
                    continue
                visited.add(cell)
                if cell in targets:
                    return (dr, dc)
                queue.append((nr, nc, dr, dc))
                
        # BFS expansion
        while queue:
            cr, cc, fdr, fdc = queue.popleft()
            for dr, dc in DIRS:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < H and 0 <= nc < W:
                    cell = (nr, nc)
                    if cell in visited:
                        continue
                    if avoid_blocked and cell in blocked and cell not in targets:
                        continue
                    visited.add(cell)
                    if cell in targets:
                        return (fdr, fdc)
                    queue.append((nr, nc, fdr, fdc))
        return None

    # 1. Global w_p fail-safe: Ensure the shared bonus isn't lost if Blue 'P' agents die out
    w_p = float(env.w_p)
    in_plaza = pos in env.plaza_cells_set
    
    if w_p > 0.50:
        if in_plaza:
            return CLEAN
        else:
            step_dir = get_bfs_step(env.plaza_cells_set, avoid_blocked=True)
            if step_dir is None:
                step_dir = get_bfs_step(env.plaza_cells_set, avoid_blocked=False)
            if step_dir is not None and step_dir != (0, 0):
                return mv(*step_dir)
            return NOOP

    # 2. Free-ride and monopolize private rewards: Harvest own clan's orchard apples
    clan = int(env.agent_clan[agent_id])
    own_orchard = set(env.orchard_cells_per_q[clan])
    own_apples = {cell for cell in own_orchard if env.orchard_apple[cell[0], cell[1]]}
    
    step_dir = get_bfs_step(own_apples, avoid_blocked=True)
    if step_dir is None:
        step_dir = get_bfs_step(own_apples, avoid_blocked=False)

    # 3. Locust mode: If our orchard is dead/empty, steal from other clans' orchards globally
    if step_dir is None:
        all_apples = {cell for cell in env.orchard_cells_list if env.orchard_apple[cell[0], cell[1]]}
        step_dir = get_bfs_step(all_apples, avoid_blocked=True)
        if step_dir is None:
            step_dir = get_bfs_step(all_apples, avoid_blocked=False)

    # 4. Fallback: Help with bonus apples if ALL orchards are entirely empty
    if step_dir is None:
        bonus_apples = {cell for cell in env.plaza_cells if env.bonus_apple[cell[0], cell[1]]}
        step_dir = get_bfs_step(bonus_apples, avoid_blocked=True)
        if step_dir is None:
            step_dir = get_bfs_step(bonus_apples, avoid_blocked=False)

    # 5. Drift and wait: Move back towards our own clan's orchard center to await regrowth
    if step_dir is None:
        step_dir = get_bfs_step(own_orchard, avoid_blocked=True)
        if step_dir is None:
            step_dir = get_bfs_step(own_orchard, avoid_blocked=False)

    if step_dir is not None and step_dir != (0, 0):
        return mv(*step_dir)
        
    return NOOP