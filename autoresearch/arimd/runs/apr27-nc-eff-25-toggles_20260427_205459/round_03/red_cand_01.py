def policy(env, agent_id: int) -> int:
    if env._travel_queue[agent_id]:
        return 0

    clan = int(env.agent_clan[agent_id])
    H, W = env.height, env.width
    r = int(env.agent_pos[agent_id, 0])
    c = int(env.agent_pos[agent_id, 1])
    pos = (r, c)
    
    plaza_cells_set = env.plaza_cells_set
    rivers = env.river_cells_set
    own_river = set(env.river_cells_per_q[clan])
    
    w_q_own = float(env.w_q[clan])
    w_p = float(env.w_p)
    
    blocked = {(int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1])) for j in range(env.n_agents) if j != agent_id}
    
    def is_adjacent_to_own_river(rr, cc):
        for dr, dc in ((-1, 0), (1, 0), (0, 1), (0, -1)):
            if (rr + dr, cc + dc) in own_river:
                return True
        return False

    target_set = None
    best_score = -1.0
    
    # 1. Clean river (acts as failsafe, lets Blue roles clean first at 0.30)
    if w_q_own > 0.32:
        if is_adjacent_to_own_river(r, c):
            return 5  # CLEAN
        own_river_approach_cells = set()
        for cell in own_river:
            for dr, dc in ((-1, 0), (1, 0), (0, 1), (0, -1)):
                nbr = (cell[0] + dr, cell[1] + dc)
                if 0 <= nbr[0] < H and 0 <= nbr[1] < W:
                    if nbr not in rivers and nbr not in plaza_cells_set:
                        own_river_approach_cells.add(nbr)
        target_set = own_river_approach_cells
        best_score = 1000.0

    # 2. Clean plaza (acts as failsafe, lets Blue specialists clean first at 0.30)
    elif w_p > 0.32:
        if pos in plaza_cells_set:
            return 5  # CLEAN
        target_set = plaza_cells_set
        best_score = 900.0

    else:
        plaza_apples = [cell for cell in env.plaza_cells if env.bonus_apple[cell[0], cell[1]]]
        own_orchard_apples = [cell for cell in env.orchard_cells_per_q[clan] if env.orchard_apple[cell[0], cell[1]]]
        
        # 3. Plaza apples
        for apple in plaza_apples:
            d = abs(r - apple[0]) + abs(c - apple[1])
            competitor_dist = 999
            competitor_id = 999
            # Verify if anyone else is closer to this apple. If they are, let them have it;
            # Red still gets the massive +8.0 shared bonus passively.
            for j in range(env.n_agents):
                if j == agent_id:
                    continue
                dj = abs(env.agent_pos[j, 0] - apple[0]) + abs(env.agent_pos[j, 1] - apple[1])
                if dj < competitor_dist or (dj == competitor_dist and j < competitor_id):
                    competitor_dist = dj
                    competitor_id = j
                    
            if d < competitor_dist or (d == competitor_dist and agent_id < competitor_id):
                score = 8.0 / (d + 1)
            else:
                score = 0.0
                
            if score > best_score:
                best_score = score
                target_set = {apple}
                
        # 4. Own orchard apples
        for apple in own_orchard_apples:
            d = abs(r - apple[0]) + abs(c - apple[1])
            score = 1.0 / (d + 1)
            if score > best_score:
                best_score = score
                target_set = {apple}
                
        # 5. Idle targets (rest right outside the plaza boundaries to maximize plaza empty spaces for regrowth)
        if target_set is None:
            idle_cells = {
                0: {(7, 7), (7, 8), (8, 7)},
                1: {(7, 12), (7, 11), (8, 12)},
                2: {(12, 7), (12, 8), (11, 7)},
                3: {(12, 12), (12, 11), (11, 12)}
            }
            target_set = idle_cells[clan]

    if not target_set or pos in target_set:
        return 0  # NOOP

    def get_bfs_step():
        # Try with block awareness
        visited = {pos}
        queue = deque([(r, c, 0, 0)])
        while queue:
            cr, cc, fdr, fdc = queue.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, 1), (0, -1)):
                nr, nc = cr + dr, cc + dc
                if not (0 <= nr < H and 0 <= nc < W): continue
                cell = (nr, nc)
                if cell in visited: continue
                if cell in blocked and cell not in target_set: continue
                visited.add(cell)
                n_fdr = dr if (cr, cc) == pos else fdr
                n_fdc = dc if (cr, cc) == pos else fdc
                if cell in target_set: return (n_fdr, n_fdc)
                queue.append((nr, nc, n_fdr, n_fdc))
                
        # Fallback: Try without block awareness
        visited = {pos}
        queue = deque([(r, c, 0, 0)])
        while queue:
            cr, cc, fdr, fdc = queue.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, 1), (0, -1)):
                nr, nc = cr + dr, cc + dc
                if not (0 <= nr < H and 0 <= nc < W): continue
                cell = (nr, nc)
                if cell in visited: continue
                visited.add(cell)
                n_fdr = dr if (cr, cc) == pos else fdr
                n_fdc = dc if (cr, cc) == pos else fdc
                if cell in target_set: return (n_fdr, n_fdc)
                queue.append((nr, nc, n_fdr, n_fdc))
        return None

    step = get_bfs_step()
    if step:
        dr, dc = step
        if dr == -1 and dc == 0: return 1  # MOVE_N
        if dr == 1 and dc == 0:  return 2  # MOVE_S
        if dr == 0 and dc == 1:  return 3  # MOVE_E
        if dr == 0 and dc == -1: return 4  # MOVE_W
        
    return 0  # NOOP