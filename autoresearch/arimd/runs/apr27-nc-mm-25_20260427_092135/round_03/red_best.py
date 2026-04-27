def policy(env, agent_id: int) -> int:
    # Mid-travel queue handling
    if env._travel_queue[agent_id]:
        return int(Action.NOOP)
        
    pos = (int(env.agent_pos[agent_id, 0]), int(env.agent_pos[agent_id, 1]))
    w_p = float(env.w_p)
    in_plaza = env._in_plaza(pos[0], pos[1])
    
    # Staggered failsafe thresholds so Red agents don't redundantly clean w_p on the same step
    clean_threshold = 0.40 + ((agent_id * 7) % 5) * 0.02
    
    def mv(dr, dc):
        if dr == -1 and dc == 0: return int(Action.MOVE_N)
        if dr == 1 and dc == 0: return int(Action.MOVE_S)
        if dr == 0 and dc == 1: return int(Action.MOVE_E)
        if dr == 0 and dc == -1: return int(Action.MOVE_W)
        return int(Action.NOOP)
        
    def bfs_to_set(target_set, treat_blocked=True):
        if not target_set: 
            return None
        if pos in target_set: 
            return (0, 0)
        
        blocked = set()
        if treat_blocked:
            for j in range(env.n_agents):
                if j != agent_id:
                    blocked.add((int(env.agent_pos[j,0]), int(env.agent_pos[j,1])))
                    
        queue = deque([pos])
        visited = {pos}
        first_move = {pos: None}
        
        # Consistent directional order
        DIRS = [(-1,0), (1,0), (0,1), (0,-1)]
        
        while queue:
            curr = queue.popleft()
            if curr in target_set:
                return first_move[curr]
                
            for dr, dc in DIRS:
                nr, nc = curr[0]+dr, curr[1]+dc
                if 0 <= nr < env.height and 0 <= nc < env.width:
                    nxt = (nr, nc)
                    if nxt not in visited:
                        # Allow pathing *to* an occupied target, but not *through* occupied non-targets
                        if treat_blocked and nxt in blocked and nxt not in target_set:
                            continue
                        visited.add(nxt)
                        queue.append(nxt)
                        if curr == pos:
                            first_move[nxt] = (dr, dc)
                        else:
                            first_move[nxt] = first_move[curr]
        return None

    def get_step(target_set):
        s = bfs_to_set(target_set, treat_blocked=True)
        if s is not None: 
            return s
        # Fallback to ignore agents if completely blocked
        return bfs_to_set(target_set, treat_blocked=False)

    # 1. Clean Plaza (Failsafe)
    # We let Blue clean if they are around, but clean it ourselves if w_p gets dangerously close to 0.6
    if w_p > clean_threshold:
        if in_plaza:
            return int(Action.CLEAN)
        else:
            step = get_step(env.plaza_cells_set)
            if step is not None:
                if step == (0,0): return int(Action.NOOP)
                return mv(*step)
                
    # 2. Harvest Plaza Bonus Apples (Highest Priority)
    bonus_cells = {
        (r, c) for r, c in env.plaza_cells
        if env.bonus_apple[r, c]
    }
    if bonus_cells:
        step = get_step(bonus_cells)
        if step is not None:
            if step == (0,0): return int(Action.NOOP)
            return mv(*step)
            
    # 3. Harvest Orchard Apples (Opportunistic Global Harvester)
    orchard_cells = {
        (r, c) for r, c in env.orchard_cells_list
        if env.orchard_apple[r, c]
    }
    if orchard_cells:
        step = get_step(orchard_cells)
        if step is not None:
            if step == (0,0): return int(Action.NOOP)
            return mv(*step)
            
    # 4. Idle perimeter boundary (Vacate the Plaza)
    # If there are no apples, we ensure we step OFF the plaza to maximize the cells available 
    # for the highly-lucrative bonus apples to spawn.
    plaza_adj = set()
    for r in range(env.height):
        for c in range(env.width):
            if not env._in_plaza(r, c):
                if (8 <= r <= 11 and (c == 7 or c == 12)) or (8 <= c <= 11 and (r == 7 or r == 12)):
                    plaza_adj.add((r, c))
                    
    if pos not in plaza_adj:
        step = get_step(plaza_adj)
        if step is not None:
            if step == (0,0): return int(Action.NOOP)
            return mv(*step)
            
    return int(Action.NOOP)