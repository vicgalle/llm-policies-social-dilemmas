def policy(env, agent_id: int) -> int:
    if env._travel_queue[agent_id]:
        return 0

    r = int(env.agent_pos[agent_id, 0])
    c = int(env.agent_pos[agent_id, 1])
    pos = (r, c)
    my_clan = int(env.agent_clan[agent_id])
    my_inv = int(env.inventory[agent_id])
    
    blocked = {(int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1])) 
               for j in range(env.n_agents) if j != agent_id}
               
    DIRS = ((-1, 0), (1, 0), (0, 1), (0, -1))
    
    def mv(dr, dc):
        if dr == -1 and dc == 0: return 1  # MOVE_N
        if dr == 1 and dc == 0: return 2   # MOVE_S
        if dr == 0 and dc == 1: return 3   # MOVE_E
        if dr == 0 and dc == -1: return 4  # MOVE_W
        return 0                           # NOOP

    def get_bfs_step(target_set):
        if not target_set: return None
        if pos in target_set: return (0, 0)
        
        def _bfs(treat_blocked):
            visited = {pos}
            queue = deque()
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < env.height and 0 <= nc < env.width:
                    cell = (nr, nc)
                    if treat_blocked and cell in blocked and cell not in target_set:
                        continue
                    visited.add(cell)
                    if cell in target_set: 
                        return (dr, dc)
                    queue.append((nr, nc, dr, dc))
            while queue:
                cr, cc, fdr, fdc = queue.popleft()
                for dr, dc in DIRS:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < env.height and 0 <= nc < env.width:
                        cell = (nr, nc)
                        if cell in visited: 
                            continue
                        if treat_blocked and cell in blocked and cell not in target_set:
                            continue
                        visited.add(cell)
                        if cell in target_set: 
                            return (fdr, fdc)
                        queue.append((nr, nc, fdr, fdc))
            return None

        step = _bfs(True)
        if step is not None: return step
        return _bfs(False)

    # Priority 1: Reach inventory capacity (3 apples) by raiding unprotected other clans
    if my_inv < 3:
        enemies = [j for j in range(env.n_agents) 
                   if int(env.agent_clan[j]) != my_clan and int(env.inventory[j]) > 0]
        if enemies:
            adj_enemies = []
            for j in enemies:
                jr, jc = int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1])
                if abs(jr - r) + abs(jc - c) == 1:
                    adj_enemies.append(j)
            if adj_enemies:
                return 6 + adj_enemies[0]  # RAID_BASE + target
            
            enemy_pos_to_id = {(int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1])): j for j in enemies}
            step = get_bfs_step(set(enemy_pos_to_id.keys()))
            if step is not None and step != (0,0):
                return mv(*step)

    # Priority 2: Free-ride on Blue's river cleaning, stepping in only if waste threatens regrowth
    # Blue triggers at 0.08. We wait until 0.15 to ensure we only pay to clean if Blue is absent.
    w_q_own = float(env.w_q[my_clan])
    if w_q_own > 0.15:
        own_river = set(env.river_cells_per_q[my_clan])
        adj_to_river = False
        for dr, dc in DIRS:
            if (r + dr, c + dc) in own_river:
                adj_to_river = True
                break
        if adj_to_river:
            return 5  # CLEAN
        
        approach = set()
        for cell in own_river:
            for dr, dc in DIRS:
                nbr = (cell[0]+dr, cell[1]+dc)
                if 0 <= nbr[0] < env.height and 0 <= nbr[1] < env.width:
                    if nbr not in env.river_cells_set and nbr not in env.plaza_cells_set:
                        approach.add(nbr)
        step = get_bfs_step(approach)
        if step is not None and step != (0,0):
            return mv(*step)

    # Priority 3: Aggressively harvest fresh apples for immediate reward consumption
    apples = set()
    for cell in env.orchard_cells_list:
        if env.orchard_apple[cell[0], cell[1]]:
            apples.add(cell)
    for cell in env.plaza_cells:
        if env.bonus_apple[cell[0], cell[1]]:
            apples.add(cell)
            
    if apples:
        step = get_bfs_step(apples)
        if step is not None and step != (0,0):
            return mv(*step)
            
    # Priority 4: Default fallback position — wait for regrowth in own orchard
    own_orchard = set(env.orchard_cells_per_q[my_clan])
    if own_orchard:
        step = get_bfs_step(own_orchard)
        if step is not None and step != (0,0):
            return mv(*step)
            
    return 0