def policy(env, agent_id: int) -> int:
    r, c = int(env.agent_pos[agent_id, 0]), int(env.agent_pos[agent_id, 1])
    pos = (r, c)
    H, W = env.height, env.width
    DIRS = ((-1, 0), (1, 0), (0, 1), (0, -1))
    
    plaza_set = env.plaza_cells_set
    in_plaza = pos in plaza_set
    
    blocked = {(int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1])) 
               for j in range(env.n_agents) if j != agent_id}
               
    bonus_apples = {(pr, pc) for pr, pc in env.plaza_cells if env.bonus_apple[pr, pc]}
    
    def get_step(targets, avoid_plaza=False, treat_blocked=False):
        if not targets: 
            return None
        if pos in targets: 
            return (0, 0)
        
        visited = {pos}
        queue = deque([(r, c, 0, 0)])
        
        while queue:
            cr, cc, fdr, fdc = queue.popleft()
            for dr, dc in DIRS:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < H and 0 <= nc < W:
                    cell = (nr, nc)
                    if avoid_plaza and cell in plaza_set and cell not in targets:
                        continue
                    if treat_blocked and cell in blocked and cell not in targets:
                        continue
                    if cell not in visited:
                        visited.add(cell)
                        first_dir = (fdr, fdc) if (fdr, fdc) != (0, 0) else (dr, dc)
                        if cell in targets:
                            return first_dir
                        queue.append((nr, nc, first_dir[0], first_dir[1]))
        return None

    def best_step(targets, avoid_plaza=False):
        s = get_step(targets, avoid_plaza=avoid_plaza, treat_blocked=True)
        if s is not None: 
            return s
        return get_step(targets, avoid_plaza=avoid_plaza, treat_blocked=False)

    def mv(dr, dc):
        if dr == -1 and dc == 0: return int(Action.MOVE_N)
        if dr == 1 and dc == 0: return int(Action.MOVE_S)
        if dr == 0 and dc == 1: return int(Action.MOVE_E)
        if dr == 0 and dc == -1: return int(Action.MOVE_W)
        return int(Action.NOOP)

    # Fail-safe: if w_p gets dangerously close to disabling the shared bonus, clean it.
    if env.w_p > 0.5:
        if in_plaza:
            return int(Action.CLEAN)
        else:
            step_dir = best_step(plaza_set, avoid_plaza=False)
            if step_dir is not None and step_dir != (0, 0):
                return mv(*step_dir)
            return int(Action.NOOP)

    if in_plaza:
        # If there are apples, grab the nearest one
        if bonus_apples:
            step_dir = best_step(bonus_apples, avoid_plaza=False)
            if step_dir is not None and step_dir != (0, 0):
                return mv(*step_dir)
            if step_dir == (0, 0):
                return int(Action.NOOP)
                
        # No apples: get out immediately to let the plaza regrow
        out_cells = {(nr, nc) for nr in range(H) for nc in range(W) if (nr, nc) not in plaza_set}
        avail_out = {c for c in out_cells if c not in blocked}
        if not avail_out: 
            avail_out = out_cells
            
        step_dir = best_step(avail_out, avoid_plaza=False)
        if step_dir is not None and step_dir != (0, 0):
            return mv(*step_dir)
        return int(Action.NOOP)
        
    else:
        # Outside plaza: prioritize bonus apples
        if bonus_apples:
            step_dir = best_step(bonus_apples, avoid_plaza=False)
            if step_dir is not None and step_dir != (0, 0):
                return mv(*step_dir)
                
        # Secondary: orchard apples without stepping into the plaza
        orchard_apples = {(or_r, or_c) for or_r, or_c in env.orchard_cells_list if env.orchard_apple[or_r, or_c]}
        if orchard_apples:
            step_dir = best_step(orchard_apples, avoid_plaza=True)
            if step_dir is not None and step_dir != (0, 0):
                return mv(*step_dir)
                
        # Wait on the plaza perimeter, distributing cleanly around it
        perimeter = set()
        for pr, pc in env.plaza_cells:
            for dr, dc in DIRS:
                nr, nc = pr + dr, pc + dc
                if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in plaza_set:
                    perimeter.add((nr, nc))
                    
        avail_perim = {p for p in perimeter if p not in blocked}
        if not avail_perim: 
            avail_perim = perimeter
        
        if pos not in avail_perim:
            step_dir = best_step(avail_perim, avoid_plaza=True)
            if step_dir is not None and step_dir != (0, 0):
                return mv(*step_dir)
                
        return int(Action.NOOP)