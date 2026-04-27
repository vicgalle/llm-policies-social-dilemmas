def policy(env, agent_id: int) -> int:
    NOOP = 0
    MOVE_N = 1
    MOVE_S = 2
    MOVE_E = 3
    MOVE_W = 4
    RAID_BASE = 6
    
    if env._travel_queue[agent_id]:
        return NOOP

    my_clan = int(env.agent_clan[agent_id])
    r = int(env.agent_pos[agent_id, 0])
    c = int(env.agent_pos[agent_id, 1])
    my_inv = int(env.inventory[agent_id])

    DIRS = [(-1, 0), (1, 0), (0, 1), (0, -1)]

    def get_mv(dr, dc):
        if dr == -1 and dc == 0: return MOVE_N
        if dr == 1 and dc == 0: return MOVE_S
        if dr == 0 and dc == 1: return MOVE_E
        if dr == 0 and dc == -1: return MOVE_W
        return NOOP

    blocked = set()
    for j in range(env.n_agents):
        if j != agent_id:
            blocked.add((int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1])))

    def bfs(target_cells, avoid_cells=None):
        if not target_cells: return None
        if (r, c) in target_cells: return (0, 0)
        
        queue = deque([(r, c, None)])
        visited = {(r, c)}
        if avoid_cells:
            visited |= avoid_cells
        
        while queue:
            curr_r, curr_c, first_step = queue.popleft()
            for dr, dc in DIRS:
                nr, nc = curr_r + dr, curr_c + dc
                if 0 <= nr < env.height and 0 <= nc < env.width:
                    if (nr, nc) not in visited:
                        visited.add((nr, nc))
                        step = first_step if first_step else (dr, dc)
                        if (nr, nc) in target_cells:
                            return step
                        queue.append((nr, nc, step))
        return None

    def get_path(target_cells):
        step = bfs(target_cells, avoid_cells=blocked)
        if step: return step
        return bfs(target_cells, avoid_cells=None)

    # Phase 1: Hoard inventory to max capacity (3) for the passive per-step holding reward
    if my_inv < 3:
        valid_targets = []
        for j in range(env.n_agents):
            if int(env.agent_clan[j]) == my_clan: continue
            inv_j = int(env.inventory[j])
            if 0 < inv_j < 3:
                j_r, j_c = int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1])
                dist = abs(j_r - r) + abs(j_c - c)
                valid_targets.append((dist, j, j_r, j_c))
        
        if valid_targets:
            valid_targets.sort()
            adj_targets = [t for t in valid_targets if t[0] == 1]
            if adj_targets:
                return RAID_BASE + adj_targets[0][1]
            
            for target in valid_targets:
                tj_r, tj_c = target[2], target[3]
                adj_cells = set()
                for dr, dc in DIRS:
                    nr, nc = tj_r + dr, tj_c + dc
                    if 0 <= nr < env.height and 0 <= nc < env.width:
                        adj_cells.add((nr, nc))
                step = get_path(adj_cells)
                if step:
                    if step != (0, 0):
                        return get_mv(*step)
                    else:
                        return NOOP

    # Phase 2: Snipe high-value plaza bonus apples
    plaza_apples = set()
    for pr, pc in env.plaza_cells:
        if env.bonus_apple[pr, pc]:
            plaza_apples.add((pr, pc))
    
    if plaza_apples:
        step = get_path(plaza_apples)
        if step:
            if step != (0, 0):
                return get_mv(*step)
            return NOOP

    # Phase 3: Harvest own-orchard apples if the plaza is empty
    own_orchard = env.orchard_cells_per_q[my_clan]
    orchard_apples = set()
    for o_r, o_c in own_orchard:
        if env.orchard_apple[o_r, o_c]:
            orchard_apples.add((o_r, o_c))
    
    if orchard_apples:
        step = get_path(orchard_apples)
        if step:
            if step != (0, 0):
                return get_mv(*step)
            return NOOP

    # Phase 4: Wait perfectly outside the plaza boundaries so we don't block regrowth, ready to pounce
    min_dist = 999
    wait_cells = set()
    for r_c in range(env.height):
        for c_c in range(env.width):
            if env.quadrant_of(r_c, c_c) == my_clan and not env._in_plaza(r_c, c_c):
                pr = max(env.config.plaza_lo, min(env.config.plaza_hi, r_c))
                pc = max(env.config.plaza_lo, min(env.config.plaza_hi, c_c))
                dist = abs(r_c - pr) + abs(c_c - pc)
                if dist < min_dist:
                    min_dist = dist
                    wait_cells = {(r_c, c_c)}
                elif dist == min_dist:
                    wait_cells.add((r_c, c_c))
    
    free_wait_cells = wait_cells - blocked
    if not free_wait_cells:
        free_wait_cells = wait_cells
    
    step = get_path(free_wait_cells)
    if step:
        if step != (0, 0):
            return get_mv(*step)
        return NOOP

    return NOOP