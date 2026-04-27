def policy(env, agent_id: int) -> int:
    NOOP, MOVE_N, MOVE_S, MOVE_E, MOVE_W, CLEAN = 0, 1, 2, 3, 4, 5
    clan = int(env.agent_clan[agent_id])
    pos = (int(env.agent_pos[agent_id, 0]), int(env.agent_pos[agent_id, 1]))
    r, c = pos
    DIRS = ((-1, 0), (1, 0), (0, 1), (0, -1))
    
    if env._travel_queue[agent_id]:
        return NOOP

    def get_mv(dr, dc):
        if dr == -1 and dc == 0: return MOVE_N
        if dr == 1 and dc == 0: return MOVE_S
        if dr == 0 and dc == 1: return MOVE_E
        if dr == 0 and dc == -1: return MOVE_W
        return NOOP

    blocked = set()
    for j in range(env.n_agents):
        if j != agent_id:
            blocked.add((int(env.agent_pos[j,0]), int(env.agent_pos[j,1])))

    def bfs(targets):
        if not targets: return None
        if pos in targets: return (0,0)
        q = deque([pos])
        visited = {pos}
        parent = {}
        while q:
            cr, cc = q.popleft()
            if (cr, cc) in targets:
                curr = (cr, cc)
                while parent[curr] != pos:
                    curr = parent[curr]
                return (curr[0]-r, curr[1]-c)
            for dr, dc in DIRS:
                nr, nc = cr+dr, cc+dc
                if 0 <= nr < env.height and 0 <= nc < env.width:
                    if (nr, nc) not in visited:
                        # Allow pathing through empty cells, but not other agents (unless they are targets)
                        if (nr, nc) in blocked and (nr, nc) not in targets:
                            continue
                        visited.add((nr, nc))
                        parent[(nr, nc)] = (cr, cc)
                        q.append((nr, nc))
        return None

    steps_left = env.max_steps - env._step_count

    # 1. Raid adjacent out-of-clan agents if inventory is not full (highly profitable early game)
    if env.inventory[agent_id] < 3 and steps_left > 20:
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            for j in range(env.n_agents):
                if env.agent_clan[j] != clan and env.inventory[j] > 0:
                    if int(env.agent_pos[j,0]) == nr and int(env.agent_pos[j,1]) == nc:
                        return 6 + j

    # 1.5. Clean plaza if danger approaches threshold (protect the shared bonus multiplier)
    in_plaza = pos in env.plaza_cells_set
    if in_plaza and env.w_p > 0.24:
        if (env._step_count + agent_id) % 3 == 0:
            return CLEAN

    # 2. Seek target to raid if inventory is below capacity
    if env.inventory[agent_id] < 3 and steps_left > 30:
        targets = []
        for j in range(env.n_agents):
            if env.agent_clan[j] != clan and env.inventory[j] > 0:
                targets.append((int(env.agent_pos[j,0]), int(env.agent_pos[j,1])))
        if targets:
            step_dir = bfs(set(targets))
            if step_dir and step_dir != (0,0):
                return get_mv(*step_dir)

    # 3. Hoover up Plaza Bonus apples (forces Blue to skip collection logic and clean the plaza instead)
    bonus_cells = {cell for cell in env.plaza_cells if env.bonus_apple[cell[0], cell[1]] and cell not in blocked}
    if bonus_cells:
        step_dir = bfs(bonus_cells)
        if step_dir and step_dir != (0,0):
            return get_mv(*step_dir)

    # 4. Harvest own clan's orchard when plaza is empty
    own_orchard = set(env.orchard_cells_per_q[clan])
    apple_cells = {cell for cell in own_orchard if env.orchard_apple[cell[0], cell[1]] and cell not in blocked}
    if apple_cells:
        step_dir = bfs(apple_cells)
        if step_dir and step_dir != (0,0):
            return get_mv(*step_dir)

    # 5. Move to staging cells on the plaza perimeter (avoids blocking regrowth inside the plaza)
    plaza_perimeter = set()
    for pr, pc in env.plaza_cells:
        for dr, dc in DIRS:
            nr, nc = pr+dr, pc+dc
            if (nr, nc) not in env.plaza_cells_set:
                if 0 <= nr < env.height and 0 <= nc < env.width:
                    plaza_perimeter.add((nr, nc))
                    
    staging = {cell for cell in plaza_perimeter if env.quadrant_of(cell[0], cell[1]) == clan and cell not in blocked}
    if not staging:
        staging = {cell for cell in plaza_perimeter if cell not in blocked}
    
    if staging and pos not in staging:
        step_dir = bfs(staging)
        if step_dir and step_dir != (0,0):
            return get_mv(*step_dir)
            
    # 6. Evacuate plaza if we have nothing to do
    if in_plaza:
        non_plaza = {(nr, nc) for nr in range(env.height) for nc in range(env.width) 
                     if (nr, nc) not in env.plaza_cells_set and (nr, nc) not in blocked}
        if non_plaza:
            step_dir = bfs(non_plaza)
            if step_dir and step_dir != (0,0):
                return get_mv(*step_dir)

    return NOOP