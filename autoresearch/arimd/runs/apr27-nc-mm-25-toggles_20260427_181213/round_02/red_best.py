def policy(env, agent_id: int) -> int:
    # If mid-travel, we must NOOP to let the environment dequeue our next move
    if env._travel_queue[agent_id]:
        return 0

    r = int(env.agent_pos[agent_id, 0])
    c = int(env.agent_pos[agent_id, 1])
    pos = (r, c)
    clan = int(env.agent_clan[agent_id])
    
    plaza = env.plaza_cells_set
    own_orchard = set(env.orchard_cells_per_q[clan])
    
    DIRS = ((-1, 0), (1, 0), (0, 1), (0, -1))
    
    def mv(dr, dc):
        if dr == -1 and dc == 0: return 1 # MOVE_N
        if dr ==  1 and dc == 0: return 2 # MOVE_S
        if dr ==  0 and dc == 1: return 3 # MOVE_E
        if dr ==  0 and dc == -1: return 4 # MOVE_W
        return 0

    # Locations of all other agents
    blocked = {(int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1])) 
               for j in range(env.n_agents) if j != agent_id}
               
    def bfs(target_set, treat_blocked=True):
        if not target_set: return None
        if pos in target_set: return (0, 0)
        
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

    # 1. RAID for passive income
    # If inventory < max (3), and there's enough time to pay off the -0.5 raid cost, 
    # try to hit an adjacent non-clan agent who actually has apples.
    steps_left = env.max_steps - env._step_count
    if env.inventory[agent_id] < 3 and steps_left > 20:
        for j in range(env.n_agents):
            if j == agent_id: continue
            if env.agent_clan[j] == clan: continue
            if env.inventory[j] <= 0: continue
            jr, jc = int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1])
            if abs(jr - r) + abs(jc - c) == 1:
                # Successfully found target — executing action keeps us stationary
                # so if we are simultaneously standing on a fresh apple, we'll eat it!
                return raid(j)

    # 2. Harvest highest value public goods (Plaza bonus)
    # Ignore apples currently being stood upon by others (they'll be eaten this tick)
    bonus_set = {cell for cell in env.plaza_cells if env.bonus_apple[cell[0], cell[1]]}
    free_bonus = {c for c in bonus_set if c not in blocked}
    
    if pos in free_bonus:
        return 0 # NOOP to stay and eat
        
    if free_bonus:
        step_dir = bfs(free_bonus, treat_blocked=True)
        if step_dir is None:
            step_dir = bfs(free_bonus, treat_blocked=False)
        if step_dir is not None and step_dir != (0, 0):
            return mv(*step_dir)

    # 3. Harvest secondary public goods (Own clan orchard)
    apple_cells = {cell for cell in own_orchard if env.orchard_apple[cell[0], cell[1]]}
    free_apples = {c for c in apple_cells if c not in blocked}
    
    if pos in free_apples:
        return 0
        
    if free_apples:
        step_dir = bfs(free_apples, treat_blocked=True)
        if step_dir is None:
            step_dir = bfs(free_apples, treat_blocked=False)
        if step_dir is not None and step_dir != (0, 0):
            return mv(*step_dir)

    # 4. Wait strategically
    # Position right on the outer boundary of the Plaza to be 1 move away 
    # from new Plaza spawns without actively blocking them.
    wait_cells = set()
    for pr, pc in env.plaza_cells:
        for dr, dc in DIRS:
            nr, nc = pr + dr, pc + dc
            if 0 <= nr < env.height and 0 <= nc < env.width:
                if (nr, nc) not in plaza:
                    wait_cells.add((nr, nc))
                    
    # Distribute agents among their respective clan's perimeter sections
    my_wait_cells = {cell for cell in wait_cells if env.quadrant_of(cell[0], cell[1]) == clan}
    if not my_wait_cells:
        my_wait_cells = wait_cells

    free_wait_cells = {c for c in my_wait_cells if c not in blocked}
    if not free_wait_cells:
        free_wait_cells = my_wait_cells

    if pos in free_wait_cells:
        return 0

    step_dir = bfs(free_wait_cells, treat_blocked=True)
    if step_dir is None:
        step_dir = bfs(free_wait_cells, treat_blocked=False)
    if step_dir is not None and step_dir != (0, 0):
        return mv(*step_dir)

    return 0