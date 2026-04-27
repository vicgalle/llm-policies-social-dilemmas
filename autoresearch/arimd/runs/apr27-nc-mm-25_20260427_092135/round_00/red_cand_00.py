def policy(env, agent_id) -> int:
    if env._travel_queue[agent_id]:
        return 0

    def get_bfs_move(env, start_pos, target_cells, avoid_positions):
        if start_pos in target_cells:
            return 0, 0
        if not target_cells:
            return 0, 0
            
        H, W = env.height, env.width
        r, c = start_pos
        visited = {start_pos}
        if avoid_positions:
            visited = visited.union(avoid_positions)
        
        DIRS = [(-1, 0), (1, 0), (0, 1), (0, -1)]
        queue = deque()
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                if (nr, nc) not in visited:
                    if (nr, nc) in target_cells:
                        return dr, dc
                    visited.add((nr, nc))
                    queue.append((nr, nc, dr, dc))
                    
        while queue:
            cr, cc, fdr, fdc = queue.popleft()
            for dr, dc in DIRS:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < H and 0 <= nc < W:
                    if (nr, nc) not in visited:
                        if (nr, nc) in target_cells:
                            return fdr, fdc
                        visited.add((nr, nc))
                        queue.append((nr, nc, fdr, fdc))
                        
        return 0, 0

    def dir_to_action(dr, dc):
        if dr == -1 and dc == 0: return 1
        if dr == 1 and dc == 0: return 2
        if dr == 0 and dc == 1: return 3
        if dr == 0 and dc == -1: return 4
        return 0

    my_inv = int(env.inventory[agent_id])
    my_clan = int(env.agent_clan[agent_id])
    pos = (int(env.agent_pos[agent_id, 0]), int(env.agent_pos[agent_id, 1]))
    r, c = pos
    
    remaining_steps = env.max_steps - env._step_count
    
    # 1. RAID Logic (Get to max inventory safely)
    if my_inv < 3 and remaining_steps > 20:
        valid_raid_targets = []
        for j in range(env.n_agents):
            if int(env.agent_clan[j]) == my_clan: 
                continue
            # Target exactly 2 inventory to hit Blue (who never raid)
            # and to avoid looping on a Red agent who has already dropped to 1.
            if int(env.inventory[j]) == 2:
                rj, cj = int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1])
                if abs(rj - r) + abs(cj - c) == 1:
                    valid_raid_targets.append(j)
                    
        if valid_raid_targets:
            # 6 is RAID_BASE
            return 6 + valid_raid_targets[0]
            
        target_cells = set()
        for j in range(env.n_agents):
            if int(env.agent_clan[j]) == my_clan: 
                continue
            if int(env.inventory[j]) == 2:
                rj, cj = int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1])
                for dr_adj, dc_adj in [(-1,0), (1,0), (0,-1), (0,1)]:
                    nr, nc = rj + dr_adj, cj + dc_adj
                    if 0 <= nr < env.height and 0 <= nc < env.width:
                        target_cells.add((nr, nc))
                        
        if target_cells:
            avoid = {tuple(int(x) for x in env.agent_pos[j]) for j in range(env.n_agents) if j != agent_id}
            dr, dc = get_bfs_move(env, pos, target_cells, avoid)
            if (dr, dc) == (0, 0):
                dr, dc = get_bfs_move(env, pos, target_cells, set())
            if (dr, dc) != (0, 0):
                return dir_to_action(dr, dc)

    # 2. HARVEST Logic (Prioritize private Orchard apples; Plaza apples are a shared public good)
    tier1_apples = set()
    for r_c in env.orchard_cells_list:
        if env.orchard_apple[r_c[0], r_c[1]]:
            tier1_apples.add(r_c)
            
    # If w_p is high enough that the shared Plaza bonus is disabled, they act as private goods too
    if float(env.w_p) > 0.25:
        for r_c in env.plaza_cells:
            if env.bonus_apple[r_c[0], r_c[1]]:
                tier1_apples.add(r_c)
                
    dr, dc = 0, 0
    avoid = {tuple(int(x) for x in env.agent_pos[j]) for j in range(env.n_agents) if j != agent_id}
    
    if tier1_apples:
        dr, dc = get_bfs_move(env, pos, tier1_apples, avoid)
        if (dr, dc) == (0, 0):
            dr, dc = get_bfs_move(env, pos, tier1_apples, set())
            
    if (dr, dc) == (0, 0):
        # Tier 2: Plaza apples when shared bonus is active (let Blue get them if they are busy)
        tier2_apples = set()
        if float(env.w_p) <= 0.25:
            for r_c in env.plaza_cells:
                if env.bonus_apple[r_c[0], r_c[1]]:
                    tier2_apples.add(r_c)
        if tier2_apples:
            dr, dc = get_bfs_move(env, pos, tier2_apples, avoid)
            if (dr, dc) == (0, 0):
                dr, dc = get_bfs_move(env, pos, tier2_apples, set())
                
    if (dr, dc) == (0, 0):
        # Move to wait in the healthiest orchard across the entire map
        best_q = my_clan
        best_wq = float(env.w_q[my_clan])
        for q in range(4):
            wq = float(env.w_q[q])
            if wq < best_wq - 0.05:
                best_wq = wq
                best_q = q
                
        best_orchard = set(env.orchard_cells_per_q[best_q])
        dr, dc = get_bfs_move(env, pos, best_orchard, avoid)
        if (dr, dc) == (0, 0):
            dr, dc = get_bfs_move(env, pos, best_orchard, set())

    if (dr, dc) != (0, 0):
        return dir_to_action(dr, dc)
        
    return 0