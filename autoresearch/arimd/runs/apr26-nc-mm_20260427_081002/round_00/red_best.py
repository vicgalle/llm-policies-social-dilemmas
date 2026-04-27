def policy(env, agent_id: int) -> int:
    if env._travel_queue[agent_id]:
        return 0

    inventory = int(env.inventory[agent_id])
    clan = int(env.agent_clan[agent_id])
    pos = (int(env.agent_pos[agent_id, 0]), int(env.agent_pos[agent_id, 1]))

    # Valid victims: out-of-clan agents with 1 or 2 apples.
    # Filtering out agents with >= 3 apples naturally protects other Red agents 
    # who have already successfully raided their way to the cap.
    valid_victims = []
    for j in range(env.n_agents):
        if j != agent_id and int(env.agent_clan[j]) != clan:
            inv_j = int(env.inventory[j])
            if 0 < inv_j < 3:
                valid_victims.append(j)

    # 1. RAID if we need inventory and are adjacent to a victim
    if inventory < env.inventory_capacity:
        adjacent_victims = []
        for j in valid_victims:
            tgt_pos = (int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1]))
            dist = abs(pos[0] - tgt_pos[0]) + abs(pos[1] - tgt_pos[1])
            if dist == 1:
                adjacent_victims.append(j)
        
        if adjacent_victims:
            return 6 + adjacent_victims[0]  # RAID_BASE is 6

    # 2. Pathfinding targets: globally find all apples across the map
    targets = set()
    for r in range(env.height):
        for c in range(env.width):
            if env.orchard_apple[r, c] or env.bonus_apple[r, c]:
                targets.add((r, c))
                
    # If we still need inventory, add cells adjacent to valid victims as valid targets
    remaining_steps = env.max_steps - env._step_count
    chase_victims = (inventory < env.inventory_capacity) and (remaining_steps > 30)
    
    if chase_victims:
        for j in valid_victims:
            vr, vc = int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1])
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = vr + dr, vc + dc
                if 0 <= nr < env.height and 0 <= nc < env.width:
                    targets.add((nr, nc))
                    
    # Fallback to the cleanest orchard on the board if no apples or victims are currently available
    if not targets:
        w_q_list = [float(env.w_q[i]) for i in range(4)]
        best_q = w_q_list.index(min(w_q_list))
        targets = set(env.orchard_cells_per_q[best_q])
        
    # 3. BFS to the closest valid target
    blocked = {(int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1]))
               for j in range(env.n_agents) if j != agent_id}

    def bfs_step(target_set):
        if not target_set:
            return 0
        if pos in target_set:
            return 0
            
        DIRS = [(-1, 0), (1, 0), (0, 1), (0, -1)]
        
        def _search(treat_blocked: bool):
            queue = deque([pos])
            visited = {pos}
            first_step = {}
            
            while queue:
                curr = queue.popleft()
                if curr in target_set:
                    return first_step[curr]
                    
                for dr, dc in DIRS:
                    nr, nc = curr[0] + dr, curr[1] + dc
                    if 0 <= nr < env.height and 0 <= nc < env.width:
                        nxt = (nr, nc)
                        if nxt not in visited:
                            # Avoid pathing through occupied cells unless that cell is itself a target
                            if treat_blocked and nxt in blocked and nxt not in target_set:
                                continue
                            visited.add(nxt)
                            if curr == pos:
                                first_step[nxt] = (dr, dc)
                            else:
                                first_step[nxt] = first_step[curr]
                            queue.append(nxt)
            return None

        # First attempt to find a clean path around other agents
        step_dir = _search(treat_blocked=True)
        # Fallback to ghost-pathing (walking straight at the target, potentially bumping) if boxed in
        if step_dir is None:
            step_dir = _search(treat_blocked=False)
            
        if step_dir is not None:
            dr, dc = step_dir
            if dr == -1 and dc == 0: return 1
            if dr == 1 and dc == 0: return 2
            if dr == 0 and dc == 1: return 3
            if dr == 0 and dc == -1: return 4
            
        return 0

    return bfs_step(targets)