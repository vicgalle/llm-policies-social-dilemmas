def policy(env, agent_id: int) -> int:
    NOOP, MOVE_N, MOVE_S, MOVE_E, MOVE_W, CLEAN = 0, 1, 2, 3, 4, 5
    
    # If mid-travel queue exists, we must NOOP to let it process
    if env._travel_queue[agent_id]:
        return NOOP

    clan = int(env.agent_clan[agent_id])
    r = int(env.agent_pos[agent_id, 0])
    c = int(env.agent_pos[agent_id, 1])
    pos = (r, c)
    
    H, W = env.height, env.width
    DIRS = ((-1, 0), (1, 0), (0, 1), (0, -1))
    
    def mv(dr, dc):
        if dr == -1 and dc == 0: return MOVE_N
        if dr == 1 and dc == 0: return MOVE_S
        if dr == 0 and dc == 1: return MOVE_E
        if dr == 0 and dc == -1: return MOVE_W
        return NOOP

    plaza = env.plaza_cells_set
    rivers = env.river_cells_set
    own_river = set(env.river_cells_per_q[clan])
    own_orchard = set(env.orchard_cells_per_q[clan])
    
    w_q_own = float(env.w_q[clan])
    w_p = float(env.w_p)
    in_plaza = pos in plaza
    
    adj_own_river = False
    for dr, dc in DIRS:
        if (r + dr, c + dc) in own_river:
            adj_own_river = True
            break

    # We manually pathfind to avoid the 0.1 cost of the TRAVEL action
    blocked = {(int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1]))
               for j in range(env.n_agents) if j != agent_id}
               
    def bfs(target_set, avoid_blocked=True):
        if not target_set: return None
        if pos in target_set: return (0, 0)
        
        queue = deque([pos])
        visited = {pos}
        parent = {}
        
        while queue:
            curr = queue.popleft()
            if curr in target_set:
                step = curr
                while parent[step] != pos:
                    step = parent[step]
                return (step[0] - pos[0], step[1] - pos[1])
                
            for dr, dc in DIRS:
                nr, nc = curr[0] + dr, curr[1] + dc
                if 0 <= nr < H and 0 <= nc < W:
                    nbr = (nr, nc)
                    if nbr not in visited:
                        if avoid_blocked and nbr in blocked and nbr not in target_set:
                            continue
                        visited.add(nbr)
                        parent[nbr] = curr
                        queue.append(nbr)
        return None

    def get_step(target_set):
        step = bfs(target_set, avoid_blocked=True)
        if step is None:
            step = bfs(target_set, avoid_blocked=False)
        return step

    # 1. Maintain own river (Free-ride unless approaching failure threshold)
    if w_q_own > 0.40 or (w_q_own > 0.10 and adj_own_river):
        if adj_own_river:
            return CLEAN
        approach = set()
        for cell in own_river:
            for dr, dc in DIRS:
                nbr = (cell[0]+dr, cell[1]+dc)
                if 0 <= nbr[0] < H and 0 <= nbr[1] < W and nbr not in rivers:
                    approach.add(nbr)
        step = get_step(approach)
        if step and step != (0, 0):
            return mv(*step)

    # 2. Maintain plaza (Free-ride unless approaching failure threshold)
    if w_p > 0.45:
        if in_plaza:
            return CLEAN
        else:
            step = get_step(plaza)
            if step and step != (0, 0):
                return mv(*step)

    # 3. Harvest Plaza Apples (Highest margin points)
    bonus_apples = {cell for cell in env.plaza_cells 
                    if env.bonus_apple[cell[0], cell[1]]}
    if bonus_apples:
        step = get_step(bonus_apples)
        if step and step != (0, 0):
            return mv(*step)

    # 4. Hawk Strategy: Wait on the exact boundary of the plaza in our own clan's area.
    # This prevents us from blocking plaza spawn locations while staying 1 step away.
    plaza_border = set()
    for pr, pc in env.plaza_cells:
        for dr, dc in DIRS:
            nr, nc = pr+dr, pc+dc
            if (nr, nc) not in plaza and 0 <= nr < H and 0 <= nc < W:
                if env.quadrant_of(nr, nc) == clan:
                    plaza_border.add((nr, nc))
                    
    # If we are currently inside the plaza and there's no apple, step off to unblock spawns.
    if in_plaza:
        step = get_step(plaza_border)
        if step and step != (0, 0):
            return mv(*step)
            
    # If we are perched on the border, opportunistically grab adjacent orchard apples 
    # without taking a multi-step detour.
    if pos in plaza_border:
        own_apples = {cell for cell in own_orchard if env.orchard_apple[cell[0], cell[1]]}
        for dr, dc in DIRS:
            nbr = (r + dr, c + dc)
            if nbr in own_apples:
                return mv(dr, dc)
        return NOOP

    # Otherwise, pathfind towards our designated hawk perch on the border.
    step = get_step(plaza_border)
    if step and step != (0, 0):
        return mv(*step)
        
    return NOOP