def policy(env, agent_id) -> int:
    r, c = int(env.agent_pos[agent_id, 0]), int(env.agent_pos[agent_id, 1])
    clan = int(env.agent_clan[agent_id])
    inv = int(env.inventory[agent_id])
    w_p = float(env.w_p)
    
    # 0. If standing on an apple, NOOP to collect it instantly
    if (r, c) in env.orchard_cells_set and env.orchard_apple[r, c]:
        return 0
    if (r, c) in env.plaza_cells_set and env.bonus_apple[r, c] and w_p <= 0.25:
        return 0

    # 1. Patch Blue's shared-bonus leak: keep w_p safely below 0.25
    if (r, c) in env.plaza_cells_set and w_p > 0.22:
        return 5  # CLEAN
        
    # 2. Raid target selection to hit 3 inventory
    if inv < 3:
        # Check adjacent targets first
        for j in range(env.n_agents):
            if j == agent_id: continue
            if int(env.agent_clan[j]) == clan: continue
            if int(env.inventory[j]) <= 0: continue
            jr, jc = int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1])
            if abs(jr - r) + abs(jc - c) == 1:
                return 6 + j  # RAID action
                
        # Pathfind to nearest target
        targets = set()
        for j in range(env.n_agents):
            if j == agent_id: continue
            if int(env.agent_clan[j]) == clan: continue
            if int(env.inventory[j]) > 0:
                targets.add((int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1])))
        if targets:
            q = deque([(r, c, -1)])
            visited = {(r, c)}
            step = None
            while q:
                cr, cc, first_move = q.popleft()
                if (cr, cc) in targets:
                    step = first_move
                    break
                for dr, dc, act in [(-1, 0, 1), (1, 0, 2), (0, 1, 3), (0, -1, 4)]:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < env.height and 0 <= nc < env.width:
                        if (nr, nc) not in visited:
                            visited.add((nr, nc))
                            q.append((nr, nc, act if first_move == -1 else first_move))
            if step is not None and step != -1:
                return step

    # 3. Pathfind to nearest Orchard apple
    orchard_apples = set()
    for ar, ac in env.orchard_cells_list:
        if env.orchard_apple[ar, ac]:
            orchard_apples.add((ar, ac))
    if orchard_apples:
        q = deque([(r, c, -1)])
        visited = {(r, c)}
        step = None
        while q:
            cr, cc, first_move = q.popleft()
            if (cr, cc) in orchard_apples:
                step = first_move
                break
            for dr, dc, act in [(-1, 0, 1), (1, 0, 2), (0, 1, 3), (0, -1, 4)]:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < env.height and 0 <= nc < env.width:
                    if (nr, nc) not in visited:
                        visited.add((nr, nc))
                        q.append((nr, nc, act if first_move == -1 else first_move))
        if step is not None and step != -1:
            return step

    # 4. Pathfind to Plaza apples only if no orchard apples and w_p is safe
    if w_p <= 0.25:
        plaza_apples = set()
        for ar, ac in env.plaza_cells:
            if env.bonus_apple[ar, ac]:
                plaza_apples.add((ar, ac))
        if plaza_apples:
            q = deque([(r, c, -1)])
            visited = {(r, c)}
            step = None
            while q:
                cr, cc, first_move = q.popleft()
                if (cr, cc) in plaza_apples:
                    step = first_move
                    break
                for dr, dc, act in [(-1, 0, 1), (1, 0, 2), (0, 1, 3), (0, -1, 4)]:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < env.height and 0 <= nc < env.width:
                        if (nr, nc) not in visited:
                            visited.add((nr, nc))
                            q.append((nr, nc, act if first_move == -1 else first_move))
            if step is not None and step != -1:
                return step

    # 5. Wait safely exactly on the perimeter (never blocking spawns)
    blocked = set(env.orchard_cells_list) | set(env.plaza_cells)
    if (r, c) not in blocked:
        return 0  # We're already safely idling
        
    safe_spots = set()
    for o_r, o_c in env.orchard_cells_list:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = o_r + dr, o_c + dc
            if 0 <= nr < env.height and 0 <= nc < env.width:
                if (nr, nc) not in blocked:
                    safe_spots.add((nr, nc))
                    
    q = deque([(r, c, -1)])
    visited = {(r, c)}
    step = None
    while q:
        cr, cc, first_move = q.popleft()
        if (cr, cc) in safe_spots:
            step = first_move
            break
        for dr, dc, act in [(-1, 0, 1), (1, 0, 2), (0, 1, 3), (0, -1, 4)]:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < env.height and 0 <= nc < env.width:
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    q.append((nr, nc, act if first_move == -1 else first_move))
    if step is not None and step != -1:
        return step

    return 0