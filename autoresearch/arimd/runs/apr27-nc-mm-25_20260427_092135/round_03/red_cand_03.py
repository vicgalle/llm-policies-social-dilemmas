def policy(env, agent_id) -> int:
    NOOP, MOVE_N, MOVE_S, MOVE_E, MOVE_W, CLEAN = 0, 1, 2, 3, 4, 5
    
    if env._travel_queue[agent_id]:
        return NOOP
        
    r = int(env.agent_pos[agent_id, 0])
    c = int(env.agent_pos[agent_id, 1])
    pos = (r, c)
    w_p = float(env.w_p)
    
    plaza_cells = set(env.plaza_cells)
    in_plaza = pos in plaza_cells
    
    # Emergency backup: if Red happens to replace the Blue 'P' agents, Red must step in
    # to clean the plaza before w_P passes the shared-bonus threshold (0.6).
    if in_plaza and w_p > 0.45:
        return CLEAN
        
    plaza_emergency = w_p > 0.45
    num_plaza_apples = sum(1 for pr, pc in plaza_cells if env.bonus_apple[pr, pc])
    plaza_apple_emergency = num_plaza_apples > 5
    
    # Stay out of the plaza to prevent blocking regrowth, unless Blue is failing.
    avoid_plaza = not (plaza_emergency or plaza_apple_emergency)
    
    blocked = {(int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1]))
               for j in range(env.n_agents) if j != agent_id}
               
    targets = {}
    
    # Target all orchard apples map-wide
    for cell in env.orchard_cells_list:
        if env.orchard_apple[cell[0], cell[1]]:
            targets[cell] = 1.0
            
    # Target plaza apples only if it's an emergency or we're already inside
    if not avoid_plaza or in_plaza:
        for cell in plaza_cells:
            if env.bonus_apple[cell[0], cell[1]]:
                targets[cell] = 2.5
                
    # If plaza emergency, target entry into empty plaza cells to enable CLEAN
    if plaza_emergency and not in_plaza:
        for cell in plaza_cells:
            if not env.bonus_apple[cell[0], cell[1]]:
                targets[cell] = 2.0
                
    cur_q = env.quadrant_of(r, c)
    total_apples = sum(1 for cell in env.orchard_cells_list if env.orchard_apple[cell[0], cell[1]]) + num_plaza_apples
    
    # Emergency backup: if the ecosystem dies (Blue 'C' missing), jumpstart the river
    if total_apples < 3 and env.w_q[cur_q] > 0.4:
        adj_river = False
        for dr, dc in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
            nr, nc = r + dr, c + dc
            if (nr, nc) in env.river_cells_set and env._river_to_q[(nr, nc)] == cur_q:
                adj_river = True
                break
        if adj_river:
            return CLEAN
            
        for cell in env.river_cells_per_q[cur_q]:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = cell[0] + dr, cell[1] + dc
                if 0 <= nr < env.height and 0 <= nc < env.width:
                    if (nr, nc) not in env.river_cells_set and (nr, nc) not in plaza_cells:
                        targets[(nr, nc)] = 2.0
                        
    # Fallback if no targets anywhere: drift toward the healthiest orchard
    if not targets:
        best_q = -1
        lowest_wq = 999.0
        for q in range(4):
            if env.w_q[q] < lowest_wq:
                lowest_wq = float(env.w_q[q])
                best_q = q
        for cell in env.orchard_cells_per_q[best_q]:
            targets[cell] = 0.1
            
    def get_bfs_move(use_blocked=True):
        queue = deque([(pos[0], pos[1], 0, None)])
        visited = {pos}
        
        best_score = -1.0
        best_move = None
        best_dist = 9999
        
        while queue:
            cr, cc, d, fm = queue.popleft()
            
            if (cr, cc) in targets:
                val = targets[(cr, cc)]
                score = val / max(1, d)
                if score > best_score:
                    best_score = score
                    best_move = fm
                    best_dist = d
                elif score == best_score and d < best_dist:
                    best_move = fm
                    best_dist = d
                    
            if d > 30:
                continue
                
            for dr, dc, act in [(-1, 0, 1), (1, 0, 2), (0, 1, 3), (0, -1, 4)]:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < env.height and 0 <= nc < env.width:
                    if (nr, nc) not in visited:
                        if use_blocked and (nr, nc) in blocked:
                            continue
                        # Never step INTO the plaza if avoiding, but allow stepping OUT if inside
                        if avoid_plaza and (nr, nc) in plaza_cells and pos not in plaza_cells:
                            continue
                        visited.add((nr, nc))
                        queue.append((nr, nc, d + 1, fm if fm is not None else act))
        return best_move
        
    move = get_bfs_move(use_blocked=True)
    if move is None:
        move = get_bfs_move(use_blocked=False)
        
    return move if move is not None else NOOP