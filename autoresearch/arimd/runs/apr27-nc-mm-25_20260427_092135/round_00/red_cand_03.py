def policy(env, agent_id: int) -> int:
    # Handle optional framework timeouts and travel queues
    if getattr(env, 'agent_timeout', None) is not None and env.agent_timeout[agent_id] > 0:
        return 0
    if getattr(env, '_travel_queue', None) and env._travel_queue[agent_id]:
        return 0

    # `deque` is pre-loaded in the namespace
    
    # --- Helpers ---
    def get_best_move(target_cells):
        if not target_cells: return 0
        r, c = int(env.agent_pos[agent_id][0]), int(env.agent_pos[agent_id][1])
        if (r, c) in target_cells: return 0
        
        blocked = set((int(env.agent_pos[j][0]), int(env.agent_pos[j][1])) for j in range(env.n_agents) if j != agent_id)
        DIRS = [(-1, 0, 1), (1, 0, 2), (0, 1, 3), (0, -1, 4)]
        
        def bfs(treat_blocked):
            visited = set([(r, c)])
            queue = deque([(r, c, 0)])
            while queue:
                cr, cc, first_move = queue.popleft()
                for dr, dc, m in DIRS:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < env.height and 0 <= nc < env.width:
                        if (nr, nc) not in visited:
                            # Avoid other agents unless they are our exact target
                            if treat_blocked and (nr, nc) in blocked and (nr, nc) not in target_cells:
                                continue
                            visited.add((nr, nc))
                            fm = first_move if first_move != 0 else m
                            if (nr, nc) in target_cells:
                                return fm
                            queue.append((nr, nc, fm))
            return 0

        m = bfs(True)
        if m != 0: return m
        return bfs(False)

    # --- State ---
    my_inv = int(env.inventory[agent_id])
    my_clan = int(env.agent_clan[agent_id])
    my_pos = (int(env.agent_pos[agent_id][0]), int(env.agent_pos[agent_id][1]))
    
    plaza_lo = env.config.plaza_lo
    plaza_hi = env.config.plaza_hi
    plaza_cells = set()
    for pr in range(plaza_lo, plaza_hi + 1):
        for pc in range(plaza_lo, plaza_hi + 1):
            plaza_cells.add((pr, pc))
            
    in_plaza = my_pos in plaza_cells
    w_p = float(env.w_p)

    # --- Phase 1: Raiding to secure ongoing held-apple rewards ---
    if my_inv < 3:
        valid_victims = []
        for j in range(env.n_agents):
            if env.agent_clan[j] != my_clan:
                inv = int(env.inventory[j])
                if inv == 2 or inv == 1:
                    valid_victims.append(j)
        
        if valid_victims:
            def score(j):
                dist = abs(env.agent_pos[j][0] - my_pos[0]) + abs(env.agent_pos[j][1] - my_pos[1])
                inv_penalty = 0 if env.inventory[j] == 2 else 1000
                return inv_penalty + dist
            
            valid_victims.sort(key=score)
            best_target = valid_victims[0]
            best_dist = abs(env.agent_pos[best_target][0] - my_pos[0]) + abs(env.agent_pos[best_target][1] - my_pos[1])
            
            if best_dist == 1:
                return 6 + best_target  # RAID_BASE is 6
            else:
                target_cells = {(int(env.agent_pos[best_target][0]), int(env.agent_pos[best_target][1]))}
                m = get_best_move(target_cells)
                if m != 0: return m

    # --- Phase 2: Plaza Coordination & Harvesting ---
    # Coordinated cleaning: only clean if w_p crosses the shared-bonus threshold
    if w_p > 0.25 and in_plaza:
        cleaners = []
        for i in range(env.n_agents):
            if env.inventory[i] == 3:
                ir, ic = int(env.agent_pos[i][0]), int(env.agent_pos[i][1])
                if (ir, ic) in plaza_cells:
                    cleaners.append(i)
        cleaners.sort()
        cleans_needed = int((w_p - 0.25) / env.config.wp_clean_amount) + 1
        if agent_id in cleaners[:cleans_needed]:
            return 5  # CLEAN action

    # Collect high-value Plaza bonus apples
    bonus_apples = set()
    for pr in range(plaza_lo, plaza_hi + 1):
        for pc in range(plaza_lo, plaza_hi + 1):
            if env.bonus_apple[pr, pc]:
                bonus_apples.add((pr, pc))
                
    if bonus_apples:
        m = get_best_move(bonus_apples)
        if m != 0: return m
        
    # Collect regular Orchard apples if Plaza is empty
    orchard_apples = set()
    for q in range(4):
        for o_r, o_c in env.orchard_cells_per_q[q]:
            if env.orchard_apple[o_r, o_c]:
                orchard_apples.add((o_r, o_c))
                
    if orchard_apples:
        m = get_best_move(orchard_apples)
        if m != 0: return m
        
    # Prevent suppressing plaza regrowth: Step just outside the plaza if standing on an empty plaza cell
    if in_plaza:
        non_plaza = set()
        for tr in range(env.height):
            for tc in range(env.width):
                if (tr, tc) not in plaza_cells:
                    non_plaza.add((tr, tc))
        m = get_best_move(non_plaza)
        if m != 0: return m
        
    # Default NOOP
    return 0