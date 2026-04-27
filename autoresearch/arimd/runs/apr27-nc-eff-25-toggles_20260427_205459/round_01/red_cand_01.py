def policy(env, agent_id: int) -> int:
    # If the environment forces a travel step, we must NOOP to let it resolve.
    if env._travel_queue[agent_id]:
        return 0

    NOOP, MOVE_N, MOVE_S, MOVE_E, MOVE_W, CLEAN = 0, 1, 2, 3, 4, 5
    
    clan = int(env.agent_clan[agent_id])
    inventory = int(env.inventory[agent_id])
    r, c = int(env.agent_pos[agent_id][0]), int(env.agent_pos[agent_id][1])
    
    # Track all other agents to pathfind around them.
    blocked = set()
    for j in range(env.n_agents):
        if j != agent_id:
            blocked.add((int(env.agent_pos[j][0]), int(env.agent_pos[j][1])))
            
    def move_action(dr, dc):
        if dr == -1 and dc == 0: return MOVE_N
        if dr == 1 and dc == 0: return MOVE_S
        if dr == 0 and dc == 1: return MOVE_E
        if dr == 0 and dc == -1: return MOVE_W
        return NOOP

    def bfs(targets):
        """Find the first step of the shortest path to any cell in `targets`."""
        if (r, c) in targets:
            return 0, 0
        if not targets:
            return None
        visited = {(r, c)}
        q = deque([(r, c, 0, 0)])
        while q:
            cr, cc, fdr, fdc = q.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < env.height and 0 <= nc < env.width:
                    if (nr, nc) not in visited:
                        if (nr, nc) in targets:
                            return (fdr if fdr != 0 else dr), (fdc if fdc != 0 else dc)
                        if (nr, nc) not in blocked:
                            visited.add((nr, nc))
                            q.append((nr, nc, fdr if fdr != 0 else dr, fdc if fdc != 0 else dc))
        return None

    # PHASE 1: Maximize held inventory (requires exactly 1 successful raid)
    if inventory < 3:
        # Prefer targeting Role 3 plaza specialists (easier to intercept and guaranteed cross-clan).
        t_plaza = []
        t_other = []
        for j in range(env.n_agents):
            if int(env.agent_clan[j]) == clan: continue
            if int(env.inventory[j]) <= 0: continue
            
            tr, tc = int(env.agent_pos[j][0]), int(env.agent_pos[j][1])
            if 8 <= tr <= 11 and 8 <= tc <= 11:
                t_plaza.append(j)
            else:
                t_other.append(j)
                
        targets = t_plaza if t_plaza else t_other
        
        if targets:
            # If we're already next to a valid victim, RAID them
            best_adj = -1
            for j in targets:
                tr, tc = int(env.agent_pos[j][0]), int(env.agent_pos[j][1])
                if abs(r - tr) + abs(c - tc) == 1:
                    best_adj = j
                    break
            
            if best_adj != -1:
                return 6 + best_adj  # 6 is RAID_BASE
                
            # Otherwise, pathfind towards the available adjacency slots next to the targets
            target_cells = set()
            for j in targets:
                tr, tc = int(env.agent_pos[j][0]), int(env.agent_pos[j][1])
                for dr, dc in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
                    nr, nc = tr + dr, tc + dc
                    if 0 <= nr < env.height and 0 <= nc < env.width:
                        if (nr, nc) not in blocked:
                            target_cells.add((nr, nc))
            
            step = bfs(target_cells)
            if step:
                return move_action(*step)

    # PHASE 2: Aggressive Orchard Harvesting (let Blue handle all the public clean-up)
    own_orchard = env.orchard_cells_per_q[clan]
    apple_cells = {
        (cell[0], cell[1]) 
        for cell in own_orchard 
        if env.orchard_apple[cell[0], cell[1]]
    }
    
    if apple_cells:
        step = bfs(apple_cells)
        if step:
            return move_action(*step)
            
    # When empty, do not wait on the edges like Blue!
    # Wait precisely at the center of the 5x5 orchard to minimize expected traversal distance.
    centers = [(5, 5), (5, 14), (14, 5), (14, 14)]
    center = centers[clan]
    
    if (r, c) != center:
        step = bfs({center})
        if step:
            return move_action(*step)
            
    return NOOP