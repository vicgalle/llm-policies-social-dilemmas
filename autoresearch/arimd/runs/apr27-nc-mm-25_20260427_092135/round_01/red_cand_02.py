def policy(env, agent_id: int) -> int:
    # 1. Continue mid-travel action if any
    if env._travel_queue[agent_id]:
        return 0
        
    # 2. Maintain a persistent, deterministic roster of Red agents to assign roles
    if not hasattr(env, '_red_roster'):
        env._red_roster = []
    if agent_id not in env._red_roster:
        env._red_roster.append(agent_id)
        env._red_roster.sort()
        
    red_rank = env._red_roster.index(agent_id)
    
    # 3. Pathfinding Helpers
    H, W = env.height, env.width
    r = int(env.agent_pos[agent_id, 0])
    c = int(env.agent_pos[agent_id, 1])
    pos = (r, c)
    
    blocked = {(int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1]))
               for j in range(env.n_agents) if j != agent_id}
               
    DIRS = ((-1, 0), (1, 0), (0, 1), (0, -1))
    
    def mv(dr, dc):
        if dr == -1 and dc == 0: return 1
        if dr ==  1 and dc == 0: return 2
        if dr ==  0 and dc == 1: return 3
        if dr ==  0 and dc == -1: return 4
        return 0
        
    def get_step(target_set, avoid_blocked):
        if not target_set: return None
        if pos in target_set: return (0, 0)
        visited = {pos}
        queue = deque([(r, c, 0, 0)])
        while queue:
            cr, cc, fdr, fdc = queue.popleft()
            for dr, dc in DIRS:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < H and 0 <= nc < W:
                    cell = (nr, nc)
                    if cell not in visited:
                        if avoid_blocked and cell in blocked and cell not in target_set:
                            continue
                        visited.add(cell)
                        next_fdr = dr if fdr == 0 and fdc == 0 else fdr
                        next_fdc = dc if fdr == 0 and fdc == 0 else fdc
                        if cell in target_set:
                            return (next_fdr, next_fdc)
                        queue.append((nr, nc, next_fdr, next_fdc))
        return None

    def move_towards(target_set):
        step = get_step(target_set, avoid_blocked=True)
        if step is None:
            step = get_step(target_set, avoid_blocked=False)
        if step is not None and step != (0, 0):
            return mv(*step)
        return 0

    # 4. Agent Roles
    if red_rank == 0:
        # ROLE: PLAZA MANAGER 
        # Keeps w_P strictly <= 0.25 (by cleaning at 0.21) so the shared bonus is ALWAYS active.
        plaza = env.plaza_cells_set
        in_plaza = pos in plaza
        
        if not in_plaza:
            return move_towards(plaza)
            
        if float(env.w_p) > 0.21:
            return 5 # CLEAN
            
        bonus_set = {cell for cell in env.plaza_cells if env.bonus_apple[cell[0], cell[1]]}
        if bonus_set:
            action = move_towards(bonus_set)
            if action != 0: return action
            
        free_plaza = {cell for cell in env.plaza_cells if cell not in blocked and cell != pos}
        if free_plaza:
            action = move_towards(free_plaza)
            if action != 0: return action
            
        return 0
        
    else:
        # ROLE: ORCHARD HARVESTER 
        # Free-rides on Blue's intra-clan cleaning to farm private apples unconditionally.
        clan = int(env.agent_clan[agent_id])
        own_orchard = set(env.orchard_cells_per_q[clan])
        own_river = set(env.river_cells_per_q[clan])
        rivers = env.river_cells_set
        river_to_q = env._river_to_q
        
        # Fallback River Cleanup (Blue usually keeps w_q < 0.08, but we step in if needed to save the orchard)
        if float(env.w_q[clan]) > 0.15:
            adj_river = False
            for dr, dc in DIRS:
                nbr = (r + dr, c + dc)
                if nbr in rivers and river_to_q.get(nbr) == clan:
                    adj_river = True
                    break
            if adj_river:
                return 5 # CLEAN
                
            approach = set()
            for cell in own_river:
                for dr, dc in DIRS:
                    nbr = (cell[0] + dr, cell[1] + dc)
                    if 0 <= nbr[0] < H and 0 <= nbr[1] < W:
                        if nbr not in rivers and nbr not in env.plaza_cells_set:
                            approach.add(nbr)
            action = move_towards(approach)
            if action != 0: return action
            
        # Farm Orchard
        apple_cells = {cell for cell in own_orchard if env.orchard_apple[cell[0], cell[1]]}
        if apple_cells:
            action = move_towards(apple_cells)
            if action != 0: return action
            
        if own_orchard:
            action = move_towards(own_orchard)
            if action != 0: return action
            
        return 0