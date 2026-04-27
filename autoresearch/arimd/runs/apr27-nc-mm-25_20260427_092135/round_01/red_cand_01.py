def policy(env, agent_id: int) -> int:
    r = int(env.agent_pos[agent_id, 0])
    c = int(env.agent_pos[agent_id, 1])
    pos = (r, c)

    # 1. Identify all available apples and assign them a reward value
    targets = []
    for cell in env.orchard_cells_list:
        if env.orchard_apple[cell[0], cell[1]]:
            targets.append((cell, 1.0))
            
    for cell in env.plaza_cells:
        if env.bonus_apple[cell[0], cell[1]]:
            targets.append((cell, 1.5))
            
    # 2. Select the optimal target maximizing reward per step distance
    best_target = None
    best_score = -1.0
    for cell, val in targets:
        dist = abs(pos[0] - cell[0]) + abs(pos[1] - cell[1])
        score = val / (dist + 1.0)
        if score > best_score:
            best_score = score
            best_target = cell

    # 3. Setup BFS to navigate around other agents
    blocked = set()
    for j in range(env.n_agents):
        if j != agent_id:
            blocked.add((int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1])))

    DIRS = ((-1, 0), (1, 0), (0, 1), (0, -1))
    
    def mv(dr, dc):
        if dr == -1 and dc == 0: return 1
        if dr ==  1 and dc == 0: return 2
        if dr ==  0 and dc == 1: return 3
        if dr ==  0 and dc == -1: return 4
        return 0

    def bfs(target_set):
        if pos in target_set:
            return (0, 0)
        
        # First attempt respects blocked agents. If no path, ignore blocked agents 
        # (they move, so walking into their cell just delays us a step rather than failing).
        for treat_blocked in (True, False):
            visited = {pos}
            queue = deque([(pos[0], pos[1], 0, 0)])
            while queue:
                cr, cc, fdr, fdc = queue.popleft()
                for dr, dc in DIRS:
                    nr = cr + dr
                    nc = cc + dc
                    if not (0 <= nr < env.height and 0 <= nc < env.width):
                        continue
                    cell = (nr, nc)
                    if cell in visited:
                        continue
                    if treat_blocked and cell in blocked and cell not in target_set:
                        continue
                    visited.add(cell)
                    
                    # Track the very first step taken from the origin
                    next_fdr = dr if fdr == 0 and fdc == 0 else fdr
                    next_fdc = dc if fdr == 0 and fdc == 0 else fdc
                    
                    if cell in target_set:
                        return (next_fdr, next_fdc)
                    queue.append((nr, nc, next_fdr, next_fdc))
        return None

    # 4. Execute step towards the best apple
    if best_target:
        step_dir = bfs({best_target})
        if step_dir is not None and step_dir != (0, 0):
            return mv(step_dir[0], step_dir[1])
        return 0

    # 5. If there are no apples anywhere, default to waiting in our own clan's orchard.
    #    If our clan is experiencing an ecological collapse, migrate to the healthiest clan.
    my_clan = int(env.agent_clan[agent_id])
    best_q = my_clan
    if float(env.w_q[my_clan]) >= 0.5:
        best_q = int(np.argmin(env.w_q))
        
    wait_cells = set(env.orchard_cells_per_q[best_q])
    step_dir = bfs(wait_cells)
    if step_dir is not None and step_dir != (0, 0):
        return mv(step_dir[0], step_dir[1])
        
    return 0