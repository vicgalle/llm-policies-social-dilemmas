def policy(env, agent_id) -> int:
    from collections import deque
    
    NOOP = 0
    MOVE_N, MOVE_S, MOVE_E, MOVE_W = 1, 2, 3, 4
    CLEAN = 5
    
    # Do not interrupt an ongoing travel commitment.
    if env._travel_queue[agent_id]:
        return NOOP
        
    my_pos = tuple(env.agent_pos[agent_id])
    my_q = int(env.agent_clan[agent_id])
    
    # We treat other agents as walls to avoid movement collisions dropping our actions.
    blocked = {tuple(env.agent_pos[j]) for j in range(env.n_agents) if j != agent_id}
    DIRS = ((-1, 0), (1, 0), (0, 1), (0, -1))
    
    def mv(dr, dc):
        if dr == -1 and dc == 0: return MOVE_N
        if dr ==  1 and dc == 0: return MOVE_S
        if dr ==  0 and dc == 1: return MOVE_E
        if dr ==  0 and dc == -1: return MOVE_W
        return NOOP

    def _bfs_first_step(target_set, treat_blocked):
        if not target_set: return None
        if my_pos in target_set: return (0, 0)
        visited = {my_pos}
        queue = deque()
        for dr, dc in DIRS:
            nr, nc = my_pos[0] + dr, my_pos[1] + dc
            if not (0 <= nr < env.height and 0 <= nc < env.width): continue
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
                if not (0 <= nr < env.height and 0 <= nc < env.width): continue
                cell = (nr, nc)
                if cell in visited: continue
                if treat_blocked and cell in blocked and cell not in target_set: continue
                visited.add(cell)
                if cell in target_set: return (fdr, fdc)
                queue.append((nr, nc, fdr, fdc))
        return None

    # 1. Emergency Clean Reflex
    # If we replaced our clan's 'C' (cleaner) agent, our orchard will eventually die (w_q -> 1.0).
    # Since clean_cost=0.3 enables multiple +1.0 apples, we must step in to clean if w_q hits 0.25.
    w_q_own = float(env.w_q[my_q])
    if w_q_own > 0.25:
        adjacent = False
        for dr, dc in DIRS:
            nbr = (my_pos[0] + dr, my_pos[1] + dc)
            if nbr in env.river_cells_set and env._river_to_q.get(nbr) == my_q:
                adjacent = True
                break
        if adjacent:
            return CLEAN
        else:
            approach = set()
            for cell in env.river_cells_per_q[my_q]:
                for dr, dc in DIRS:
                    nbr = (cell[0] + dr, cell[1] + dc)
                    if 0 <= nbr[0] < env.height and 0 <= nbr[1] < env.width:
                        if nbr not in env.river_cells_set and nbr not in env.plaza_cells_set:
                            approach.add(nbr)
            step_dir = _bfs_first_step(approach, treat_blocked=True)
            if step_dir is None:
                step_dir = _bfs_first_step(approach, treat_blocked=False)
            if step_dir is not None and step_dir != (0, 0):
                return mv(*step_dir)

    # 2. Global Apple Harvesting Analysis
    apples = []
    for r in range(env.height):
        for c in range(env.width):
            if (r, c) in env.orchard_cells_set and env.orchard_apple[r, c]:
                apples.append((r, c, 'orchard'))
            elif (r, c) in env.plaza_cells_set and env.bonus_apple[r, c]:
                apples.append((r, c, 'plaza'))
                
    # Identify apples that Blue agents are likely already targeting so we don't duplicate effort or collide.
    targeted_by_others = set()
    for j in range(env.n_agents):
        if j == agent_id: 
            continue
            
        j_pos = tuple(env.agent_pos[j])
        q_j = int(env.agent_clan[j])
        
        best_dist = 999999
        best_apple = None
        for r, c, typ in apples:
            # Assume other agents only harvest their own clan's orchard, or any plaza
            if typ == 'orchard' and env.quadrant_of(r, c) != q_j:
                continue
            dist = abs(r - j_pos[0]) + abs(c - j_pos[1])
            if dist < best_dist:
                best_dist = dist
                best_apple = (r, c)
                
        # If agent j is closest to this apple, they've "claimed" it UNLESS we are closer!
        if best_apple:
            my_dist = abs(best_apple[0] - my_pos[0]) + abs(best_apple[1] - my_pos[1])
            if best_dist < my_dist or (best_dist == my_dist and j < agent_id):
                targeted_by_others.add(best_apple)
                
    free_orchard = set()
    free_plaza = set()
    all_orchard = set()
    all_plaza = set()
    
    for r, c, typ in apples:
        if typ == 'orchard':
            all_orchard.add((r, c))
            if (r, c) not in targeted_by_others:
                free_orchard.add((r, c))
        else:
            all_plaza.add((r, c))
            if (r, c) not in targeted_by_others:
                free_plaza.add((r, c))
                
    # 3. Pathing & Priority Cascade
    # Always prioritize free orchards (+1 to us). If none, help sweep the plaza. 
    # Fallback to occupied apples (we might out-path them), then finally wait in our own orchard.
    targets_to_try = [
        free_orchard,
        free_plaza,
        all_orchard,
        all_plaza,
        set(env.orchard_cells_per_q[my_q])
    ]
    
    for target_set in targets_to_try:
        if not target_set: 
            continue
            
        step_dir = _bfs_first_step(target_set, treat_blocked=True)
        if step_dir is None:
            step_dir = _bfs_first_step(target_set, treat_blocked=False)
            
        if step_dir is not None:
            if step_dir != (0, 0):
                return mv(*step_dir)
            else:
                return NOOP
                
    return NOOP