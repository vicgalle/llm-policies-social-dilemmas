def policy(env, agent_id) -> int:
    if env._travel_queue[agent_id]:
        return int(Action.NOOP)

    r = int(env.agent_pos[agent_id, 0])
    c = int(env.agent_pos[agent_id, 1])
    my_clan = int(env.agent_clan[agent_id])
    my_inv = int(env.inventory[agent_id])
    
    # Track positions of all other agents to avoid stepping on them
    blocked = set()
    for j in range(env.n_agents):
        if j != agent_id:
            blocked.add((int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1])))

    # Basic BFS routing
    def get_move(start, target_cells, blocked_cells):
        if start in target_cells:
            return int(Action.NOOP)
        
        queue = deque([start])
        visited = {start}
        parent = {}
        
        while queue:
            curr = queue.popleft()
            if curr in target_cells:
                path = []
                while curr != start:
                    path.append(curr)
                    curr = parent[curr]
                first_step = path[-1]
                dr = first_step[0] - start[0]
                dc = first_step[1] - start[1]
                if dr == -1: return int(Action.MOVE_N)
                if dr == 1: return int(Action.MOVE_S)
                if dc == 1: return int(Action.MOVE_E)
                if dc == -1: return int(Action.MOVE_W)
                return int(Action.NOOP)
                
            for dr, dc in [(-1,0), (1,0), (0,1), (0,-1)]:
                nr, nc = curr[0] + dr, curr[1] + dc
                if 0 <= nr < env.height and 0 <= nc < env.width:
                    nxt = (nr, nc)
                    if nxt not in visited and nxt not in blocked_cells:
                        visited.add(nxt)
                        parent[nxt] = curr
                        queue.append(nxt)
        return None

    # 1. Maximize passive income: Raid once to get inventory to 3
    if my_inv < 3:
        valid_raid_targets = []
        # Target cross-clan agents. Since Blue never raids, they stay at inventory == 2.
        # This brilliantly ensures Reds don't inadvertently keep raiding each other!
        for j in range(env.n_agents):
            if j != agent_id and int(env.agent_clan[j]) != my_clan:
                if int(env.inventory[j]) == 2:
                    valid_raid_targets.append(j)
        
        if not valid_raid_targets:
            for j in range(env.n_agents):
                if j != agent_id and int(env.agent_clan[j]) != my_clan:
                    if int(env.inventory[j]) > 0:
                        valid_raid_targets.append(j)
                        
        if valid_raid_targets:
            adjacent_targets = []
            for j in valid_raid_targets:
                jr, jc = int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1])
                if abs(jr - r) + abs(jc - c) == 1:
                    adjacent_targets.append(j)
            
            # Fire the raid if adjacent
            if adjacent_targets:
                return int(RAID_BASE + adjacent_targets[0])
                
            # Otherwise chase the nearest optimal target
            if env._step_count < 400:
                target_cells = set()
                for j in valid_raid_targets:
                    jr, jc = int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1])
                    for dr, dc in [(-1,0), (1,0), (0,1), (0,-1)]:
                        nr, nc = jr + dr, jc + dc
                        if 0 <= nr < env.height and 0 <= nc < env.width:
                            if (nr, nc) not in blocked:
                                target_cells.add((nr, nc))
                
                # Fallback to direct coords if surrounded
                if not target_cells:
                    for j in valid_raid_targets:
                        target_cells.add((int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1])))
                
                m = get_move((r, c), target_cells, blocked)
                if m is not None:
                    return m
                m = get_move((r, c), target_cells, set())
                if m is not None:
                    return m

    # 2. Perfect free-rider: Harvest continuously (never clean)
    apple_cells = set()
    for ar in range(env.height):
        for ac in range(env.width):
            if env.orchard_apple[ar, ac] or env.bonus_apple[ar, ac]:
                cell = (ar, ac)
                # Prioritize own orchard and the plaza to avoid unnecessary travel
                if cell in env.plaza_cells_set or cell in env.orchard_cells_per_q[my_clan]:
                    apple_cells.add(cell)
                    
    # Broaden to any apple anywhere if our preferred zones are empty
    if not apple_cells:
        for ar in range(env.height):
            for ac in range(env.width):
                if env.orchard_apple[ar, ac] or env.bonus_apple[ar, ac]:
                    apple_cells.add((ar, ac))
                    
    # Wait for regrowth in the middle of our own orchard if no apples are active
    if not apple_cells:
        cells = env.orchard_cells_per_q[my_clan]
        if cells:
            apple_cells.add(cells[len(cells) // 2])
            
    m = get_move((r, c), apple_cells, blocked)
    if m is not None:
        return m
    m = get_move((r, c), apple_cells, set())
    if m is not None:
        return m
        
    return int(Action.NOOP)