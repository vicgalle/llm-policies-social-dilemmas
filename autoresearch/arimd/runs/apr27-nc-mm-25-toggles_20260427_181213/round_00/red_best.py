def policy(env, agent_id: int) -> int:
    if env._travel_queue[agent_id]:
        return 0

    r = int(env.agent_pos[agent_id, 0])
    c = int(env.agent_pos[agent_id, 1])

    # 1. Raid immediately if adjacent to a valid target and inventory is not capped
    if env.inventory[agent_id] < 3:
        best_raid_target = -1
        best_inv = -1
        for j in range(env.n_agents):
            if j != agent_id and env.agent_clan[j] != env.agent_clan[agent_id] and env.inventory[j] > 0:
                od = abs(int(env.agent_pos[j,0]) - r) + abs(int(env.agent_pos[j,1]) - c)
                if od == 1:
                    if env.inventory[j] > best_inv:
                        best_inv = env.inventory[j]
                        best_raid_target = j
        if best_raid_target != -1:
            return int(6 + best_raid_target)  # 6 is RAID_BASE

    # 2. Score potential targets (Apples vs Hunting Victims)
    apples = []
    for r_idx in range(env.height):
        for c_idx in range(env.width):
            if env.orchard_apple[r_idx, c_idx] or env.bonus_apple[r_idx, c_idx]:
                apples.append((r_idx, c_idx))

    best_cost = float('inf')
    best_target = None
    agent_positions = [(int(env.agent_pos[j,0]), int(env.agent_pos[j,1])) for j in range(env.n_agents) if j != agent_id]

    # Evaluate apples
    for ar, ac in apples:
        d = abs(r - ar) + abs(c - ac)
        min_other = min([abs(or_ - ar) + abs(oc - ac) for or_, oc in agent_positions] + [float('inf')])
        
        cost = float(d)
        if d > min_other:
            cost += 50.0  # Abandon if someone else is strictly closer
        elif d == min_other:
            cost += 2.0   # Minor penalty for tie-breakers
            
        if d == 1 and min_other > 1:
            cost = -2000.0 # Absolute priority for adjacent uncontested apples
            
        if cost < best_cost:
            best_cost = cost
            best_target = (ar, ac)

    # Evaluate hunting if inventory is not capped
    if env.inventory[agent_id] < 3:
        for j in range(env.n_agents):
            if j != agent_id and env.agent_clan[j] != env.agent_clan[agent_id] and env.inventory[j] > 0:
                or_ = int(env.agent_pos[j,0])
                oc = int(env.agent_pos[j,1])
                od = abs(or_ - r) + abs(oc - c)
                
                min_other = min([abs(int(env.agent_pos[k,0]) - or_) + abs(int(env.agent_pos[k,1]) - oc) 
                                 for k in range(env.n_agents) if k != agent_id and k != j] + [float('inf')])
                
                # Highly prioritize tracking down victims to max out passive held reward
                cost = float(od) - 1000.0
                if od > min_other:
                    cost += 50.0 # Avoid crowded agents/competing with other Reds
                    
                if cost < best_cost:
                    best_cost = cost
                    best_target = (or_, oc)

    occupied = set(agent_positions)
    
    # 3. If no immediate goals, default to finding a free spot in the Plaza
    if best_target is None:
        if 8 <= r <= 11 and 8 <= c <= 11:
            best_target = (r, c)
        else:
            best_d = float('inf')
            for pr in range(8, 12):
                for pc in range(8, 12):
                    if (pr, pc) not in occupied:
                        d = abs(r - pr) + abs(c - pc)
                        if d < best_d:
                            best_d = float(d)
                            best_target = (pr, pc)
            if best_target is None:
                best_target = (r, c)

    if best_target == (r, c):
        return 0

    # 4. Shortest-path BFS towards target avoiding currently occupied cells
    target_set = {best_target}
    visited = set()
    queue = deque()
    visited.add((r, c))

    DIRS = [(-1, 0, 1), (1, 0, 2), (0, -1, 4), (0, 1, 3)]
    found_action = None

    for dr, dc, act in DIRS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < env.height and 0 <= nc < env.width:
            if (nr, nc) not in occupied:
                if (nr, nc) in target_set:
                    found_action = act
                    break
                visited.add((nr, nc))
                queue.append((nr, nc, act))

    if found_action is None:
        while queue:
            cr, cc, first_act = queue.popleft()
            for dr, dc, _ in DIRS:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < env.height and 0 <= nc < env.width:
                    if (nr, nc) not in occupied:
                        if (nr, nc) in target_set:
                            found_action = first_act
                            break
                        if (nr, nc) not in visited:
                            visited.add((nr, nc))
                            queue.append((nr, nc, first_act))
            if found_action is not None:
                break

    if found_action is not None:
        return int(found_action)
    else:
        # Fallback greedy chase if BFS fails (often helps when tailing a moving victim)
        dr = best_target[0] - r
        dc = best_target[1] - c
        if abs(dr) > abs(dc):
            return 2 if dr > 0 else 1
        elif dc != 0:
            return 3 if dc > 0 else 4
        else:
            return 0