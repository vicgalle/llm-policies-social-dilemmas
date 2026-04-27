def policy(env, agent_id: int) -> int:
    my_pos = tuple(env.agent_pos[agent_id])
    my_clan = int(env.agent_clan[agent_id])
    
    # Generic BFS helper to find the first step of the shortest path to a target set.
    # We use a deque (available in the namespace) to ensure minimal path distance.
    def bfs_move(start, target_set, blocked_cells):
        if start in target_set:
            return 0  # NOOP if already there
        q = deque([(start[0], start[1], 0)])
        visited = {start}
        # Directions: 1=N, 2=S, 3=E, 4=W
        DIRS = [(-1, 0, 1), (1, 0, 2), (0, 1, 3), (0, -1, 4)]
        
        while q:
            r, c, first_action = q.popleft()
            for dr, dc, action in DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < env.height and 0 <= nc < env.width:
                    cell = (nr, nc)
                    if cell in target_set:
                        # Found target: return the first action taken on this path
                        return first_action if first_action != 0 else action
                    if cell not in visited and cell not in blocked_cells:
                        visited.add(cell)
                        q.append((nr, nc, first_action if first_action != 0 else action))
        return 0

    # 1. RAID Restraint Abandonment: 
    # If we are below inventory capacity, target the nearest out-of-clan agent with apples.
    if env.inventory[agent_id] < env.inventory_capacity:
        targets = []
        for j in range(env.n_agents):
            if int(env.agent_clan[j]) != my_clan and env.inventory[j] > 0:
                targets.append(j)
        
        if targets:
            adjacent_targets = []
            for j in targets:
                tj_pos = tuple(env.agent_pos[j])
                dist = abs(my_pos[0] - tj_pos[0]) + abs(my_pos[1] - tj_pos[1])
                if dist == 1:
                    adjacent_targets.append(j)
            
            # If already adjacent to a target, execute the RAID action.
            # Even if they move, an invalid raid safely defaults to NOOP with 0 cost.
            if adjacent_targets:
                return 6 + adjacent_targets[0]  # RAID_BASE is 6
            
            # Otherwise, path toward the adjacent cells of the nearest target.
            target_cells = set()
            for j in targets:
                tj_pos = tuple(env.agent_pos[j])
                for dr, dc in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
                    nr, nc = tj_pos[0] + dr, tj_pos[1] + dc
                    if 0 <= nr < env.height and 0 <= nc < env.width:
                        target_cells.add((nr, nc))
            
            blocked = {tuple(env.agent_pos[j]) for j in range(env.n_agents) if j != agent_id}
            move = bfs_move(my_pos, target_cells, blocked)
            if move != 0:
                return move

    # 2. Greedy Apple Gathering:
    # We are either full on inventory or no raid targets remain. Purely free-ride and gather.
    apple_cells = set()
    for r, c in env.orchard_cells_list:
        if env.orchard_apple[r, c]:
            apple_cells.add((r, c))
    for r, c in env.plaza_cells:
        if env.bonus_apple[r, c]:
            apple_cells.add((r, c))
            
    if apple_cells:
        blocked = {tuple(env.agent_pos[j]) for j in range(env.n_agents) if j != agent_id}
        move = bfs_move(my_pos, apple_cells, blocked)
        if move != 0:
            return move

    # 3. Default Loitering:
    # If no apples are visible, wait directly in the Plaza. Blue's 'P' agents will maintain 
    # its cleanliness, allowing us to immediately snatch any shared-bonus regrowths.
    plaza_set = set(env.plaza_cells)
    blocked = {tuple(env.agent_pos[j]) for j in range(env.n_agents) if j != agent_id}
    move = bfs_move(my_pos, plaza_set, blocked)
    if move != 0:
        return move

    return 0  # NOOP