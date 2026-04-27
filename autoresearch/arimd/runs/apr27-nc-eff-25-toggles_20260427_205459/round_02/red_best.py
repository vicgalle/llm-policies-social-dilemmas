def policy(env, agent_id) -> int:
    NOOP = 0; MOVE_N = 1; MOVE_S = 2; MOVE_E = 3; MOVE_W = 4; CLEAN = 5
    r = int(env.agent_pos[agent_id, 0])
    c = int(env.agent_pos[agent_id, 1])
    pos = (r, c)
    
    plaza = env.plaza_cells_set
    in_plaza = pos in plaza
    
    # Identify all currently available bonus apples in the plaza
    bonus_cells = {cell for cell in env.plaza_cells if env.bonus_apple[cell[0], cell[1]]}
    
    # Track cells currently occupied by other agents to avoid traffic jams
    blocked = {(int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1]))
               for j in range(env.n_agents) if j != agent_id}
    
    DIRS = ((-1, 0), (1, 0), (0, 1), (0, -1))
    
    def mv(dr, dc):
        if dr == -1 and dc == 0: return MOVE_N
        if dr ==  1 and dc == 0: return MOVE_S
        if dr ==  0 and dc == 1: return MOVE_E
        if dr ==  0 and dc == -1: return MOVE_W
        return NOOP

    def bfs(target_set, avoid_blocked=True):
        if not target_set: return None
        if pos in target_set: return (0, 0)
        visited = {pos}
        queue = deque([(r, c, None, None)])
        
        while queue:
            cr, cc, fdr, fdc = queue.popleft()
            for dr, dc in DIRS:
                nr, nc = cr + dr, cc + dc
                if not (0 <= nr < env.height and 0 <= nc < env.width):
                    continue
                cell = (nr, nc)
                if cell in visited:
                    continue
                # Do not route through blocked cells unless they are our destination
                if avoid_blocked and cell in blocked and cell not in target_set:
                    continue
                visited.add(cell)
                if cell in target_set:
                    return (fdr if fdr is not None else dr, fdc if fdc is not None else dc)
                queue.append((nr, nc, fdr if fdr is not None else dr, fdc if fdc is not None else dc))
        return None

    if bonus_cells:
        if not in_plaza:
            # Step in IMMEDIATELY to secure the shared occupant bonus. 
            # We ignore blocks on the boundary cell because the occupant is likely to move, 
            # and bumping is strictly better than routing away from the only entry point.
            step = bfs(plaza, avoid_blocked=False)
            if step is not None and step != (0, 0):
                return mv(*step)
            return NOOP
        else:
            # We are already inside. Move towards the apple to speed up collection and free the cell.
            step = bfs(bonus_cells, avoid_blocked=True)
            if step is None: step = bfs(bonus_cells, avoid_blocked=False)
            if step is not None and step != (0, 0):
                return mv(*step)
            return NOOP
            
    else:
        if in_plaza:
            # No apples exist. Step OUT to maximize empty cells and boost regrowth probability.
            non_plaza = {(nr, nc) for nr in range(env.height) for nc in range(env.width)
                         if (nr, nc) not in plaza}
            step = bfs(non_plaza, avoid_blocked=True)
            if step is None: step = bfs(non_plaza, avoid_blocked=False)
            if step is not None and step != (0, 0):
                return mv(*step)
            return NOOP
        else:
            # Wait precisely 1 step away from the plaza.
            is_adjacent = False
            for dr, dc in DIRS:
                if (r + dr, c + dc) in plaza:
                    is_adjacent = True
                    break
            
            if not is_adjacent:
                adj_plaza = set()
                for pr, pc in plaza:
                    for dr, dc in DIRS:
                        nbr = (pr + dr, pc + dc)
                        if 0 <= nbr[0] < env.height and 0 <= nbr[1] < env.width and nbr not in plaza:
                            adj_plaza.add(nbr)
                step = bfs(adj_plaza, avoid_blocked=True)
                if step is None: step = bfs(adj_plaza, avoid_blocked=False)
                if step is not None and step != (0, 0):
                    return mv(*step)
            
            # Already holding position exactly on the boundary
            return NOOP