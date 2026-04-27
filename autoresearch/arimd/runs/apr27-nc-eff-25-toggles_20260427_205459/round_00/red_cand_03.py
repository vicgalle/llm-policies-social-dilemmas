def policy(env, agent_id: int) -> int:
    # 1. Mid-travel block
    if env._travel_queue[agent_id]:
        return 0

    H, W = env.height, env.width
    my_inv = int(env.inventory[agent_id])
    my_pos = (int(env.agent_pos[agent_id, 0]), int(env.agent_pos[agent_id, 1]))
    
    DIRS = ((-1, 0), (1, 0), (0, 1), (0, -1))
    
    def mv(dr, dc):
        if dr == -1 and dc == 0: return 1
        if dr == 1 and dc == 0: return 2
        if dr == 0 and dc == 1: return 3
        if dr == 0 and dc == -1: return 4
        return 0

    def bfs(target_set, avoid_agents=True):
        if not target_set: return None
        if my_pos in target_set: return (0, 0)
        
        blocked = set()
        if avoid_agents:
            for j in range(env.n_agents):
                if j != agent_id:
                    blocked.add((int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1])))
                    
        queue = deque()
        visited = set([my_pos])
        
        for dr, dc in DIRS:
            nr, nc = my_pos[0]+dr, my_pos[1]+dc
            if 0 <= nr < H and 0 <= nc < W:
                cell = (nr, nc)
                if cell not in blocked:
                    if cell in target_set:
                        return (dr, dc)
                    visited.add(cell)
                    queue.append((nr, nc, dr, dc))
                    
        while queue:
            cr, cc, fdr, fdc = queue.popleft()
            for dr, dc in DIRS:
                nr, nc = cr+dr, cc+dc
                if 0 <= nr < H and 0 <= nc < W:
                    cell = (nr, nc)
                    if cell not in visited and cell not in blocked:
                        if cell in target_set:
                            return (fdr, fdc)
                        visited.add(cell)
                        queue.append((nr, nc, fdr, fdc))
        return None

    # Priority 1: Max out held inventory to secure the 0.05/step baseline dividend.
    # We hunt the closest agent with apples and raid them until we hit cap 3.
    if my_inv < 3:
        best_dist = 9999
        best_target = -1
        for j in range(env.n_agents):
            if j == agent_id: continue
            if int(env.inventory[j]) > 0:
                r, c = int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1])
                dist = abs(my_pos[0] - r) + abs(my_pos[1] - c)
                if dist < best_dist:
                    best_dist = dist
                    best_target = j
                    
        if best_target != -1:
            if best_dist == 1:
                return RAID_BASE + best_target
            else:
                tr, tc = int(env.agent_pos[best_target, 0]), int(env.agent_pos[best_target, 1])
                adj_cells = set()
                for dr, dc in DIRS:
                    nr, nc = tr+dr, tc+dc
                    if 0 <= nr < H and 0 <= nc < W:
                        adj_cells.add((nr, nc))
                step = bfs(adj_cells, avoid_agents=True)
                if step is None:
                    step = bfs(adj_cells, avoid_agents=False)
                if step is not None and step != (0, 0):
                    return mv(*step)

    # Priority 2: Plaza safeguard. If all Blue plaza specialists happen to be wiped out, 
    # step in to prevent the global w_p threshold from totally stopping regrowth.
    in_plaza = my_pos in env.plaza_cells_set
    if in_plaza and float(env.w_p) > 0.34:
        return 5  # CLEAN

    # Priority 3: Be a super-harvester. We ignore quadrant boundaries, 
    # completely refuse to clean, and just vacuum up all the closest apples. 
    apples = set()
    for r in range(H):
        for c in range(W):
            if env.orchard_apple[r, c] or env.bonus_apple[r, c]:
                apples.add((r, c))
                
    if apples:
        step = bfs(apples, avoid_agents=True)
        if step is None:
            step = bfs(apples, avoid_agents=False)
        if step is not None and step != (0, 0):
            return mv(*step)
            
    # Priority 4: No apples available. Drift into the nearest orchard and wait.
    orchard_cells = set(env.orchard_cells_list)
    step = bfs(orchard_cells, avoid_agents=True)
    if step is None:
        step = bfs(orchard_cells, avoid_agents=False)
    if step is not None and step != (0, 0):
        return mv(*step)
        
    return 0  # NOOP