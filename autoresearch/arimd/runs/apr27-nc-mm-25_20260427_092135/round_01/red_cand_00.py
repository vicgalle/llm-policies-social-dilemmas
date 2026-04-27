def policy(env, agent_id: int) -> int:
    NOOP, MOVE_N, MOVE_S, MOVE_E, MOVE_W, CLEAN = 0, 1, 2, 3, 4, 5
    
    # Honor any mid-execution travel queues (we won't queue any, but good practice)
    if env._travel_queue[agent_id]:
        return NOOP
        
    r = int(env.agent_pos[agent_id, 0])
    c = int(env.agent_pos[agent_id, 1])
    pos = (r, c)
    
    clan = int(env.agent_clan[agent_id])
    w_p = float(env.w_p)
    
    plaza = env.plaza_cells_set
    in_plaza = pos in plaza
    
    own_orchard = set(env.orchard_cells_per_q[clan])
    
    # Cells currently occupied by other agents
    blocked = {(int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1]))
               for j in range(env.n_agents) if j != agent_id}
               
    DIRS = ((-1, 0), (1, 0), (0, 1), (0, -1))
    
    def mv(dr: int, dc: int) -> int:
        if dr == -1 and dc == 0: return MOVE_N
        if dr ==  1 and dc == 0: return MOVE_S
        if dr ==  0 and dc == 1: return MOVE_E
        if dr ==  0 and dc == -1: return MOVE_W
        return NOOP

    def bfs(target_set):
        if not target_set: return None
        if pos in target_set: return (0, 0)
        
        # 1st pass: BFS respecting agents as obstacles
        visited = {pos}
        queue = deque([ (r, c, 0, 0) ])
        while queue:
            cr, cc, fdr, fdc = queue.popleft()
            for dr, dc in DIRS:
                nr, nc = cr + dr, cc + dc
                if not (0 <= nr < env.height and 0 <= nc < env.width):
                    continue
                cell = (nr, nc)
                if cell in visited:
                    continue
                if cell in blocked and cell not in target_set:
                    continue
                visited.add(cell)
                if cell in target_set:
                    return (fdr if fdr != 0 else dr, fdc if fdc != 0 else dc)
                queue.append((nr, nc, fdr if fdr != 0 else dr, fdc if fdc != 0 else dc))
        
        # 2nd pass: Fallback BFS ignoring agents (they might move)
        visited = {pos}
        queue = deque([ (r, c, 0, 0) ])
        while queue:
            cr, cc, fdr, fdc = queue.popleft()
            for dr, dc in DIRS:
                nr, nc = cr + dr, cc + dc
                if not (0 <= nr < env.height and 0 <= nc < env.width):
                    continue
                cell = (nr, nc)
                if cell in visited:
                    continue
                visited.add(cell)
                if cell in target_set:
                    return (fdr if fdr != 0 else dr, fdc if fdc != 0 else dc)
                queue.append((nr, nc, fdr if fdr != 0 else dr, fdc if fdc != 0 else dc))
        return None

    # Priority 1: Emergency clean. If w_p is teetering on the edge of destroying our 6.0 
    # shared-bonus payout, we step in and clean it ourselves.
    if in_plaza and w_p >= 0.240:
        return CLEAN
        
    bonus_set = {cell for cell in env.plaza_cells if env.bonus_apple[cell[0], cell[1]]}
    
    # Priority 2: Staggered thresholds ensure only one Red agent usually abandons the orchard.
    my_threshold = 0.220 + 0.001 * agent_id
    if w_p > my_threshold and not in_plaza:
        step_dir = bfs(plaza)
        if step_dir is not None and step_dir != (0, 0):
            return mv(*step_dir)
            
    # Priority 3: If in Plaza and there are apples, devour them!
    # This empties the bonus_set and forces Blue to trigger its cleaning fallback.
    if in_plaza and bonus_set:
        step_dir = bfs(bonus_set)
        if step_dir is not None and step_dir != (0, 0):
            return mv(*step_dir)
            
    # Priority 3.5: In Plaza, waiting for Blue's clean to drop w_p or for apples to spawn
    if in_plaza and w_p > my_threshold:
        return NOOP
        
    # Priority 4: In Plaza, w_p is beautifully low, job is done -> return to orchard
    if in_plaza and w_p <= my_threshold:
        step_dir = bfs(own_orchard)
        if step_dir is not None and step_dir != (0, 0):
            return mv(*step_dir)
            
    # Priority 5: Harvest own orchard apples greedily
    apple_cells = {cell for cell in own_orchard if env.orchard_apple[cell[0], cell[1]]}
    if apple_cells:
        step_dir = bfs(apple_cells)
        if step_dir is not None and step_dir != (0, 0):
            return mv(*step_dir)
            
    # Priority 6: Drift to own orchard if nowhere else to be
    if pos not in own_orchard:
        step_dir = bfs(own_orchard)
        if step_dir is not None and step_dir != (0, 0):
            return mv(*step_dir)
            
    return NOOP