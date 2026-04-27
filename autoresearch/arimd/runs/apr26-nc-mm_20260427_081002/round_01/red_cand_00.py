def policy(env, agent_id) -> int:
    # Safely identify fellow Red agents using a function attribute to avoid cross-fire 
    # during the initial apple acquisition phase.
    if not hasattr(policy, 'red_agents'):
        policy.red_agents = set()
    policy.red_agents.add(agent_id)
    
    r, c = int(env.agent_pos[agent_id, 0]), int(env.agent_pos[agent_id, 1])
    my_clan = int(env.agent_clan[agent_id])
    my_inv = int(env.inventory[agent_id])
    
    def move_towards(tr, tc) -> int:
        """Returns the best movement action to reach (tr, tc) while avoiding stationary agents."""
        if r == tr and c == tc:
            return 0
            
        all_moves = [
            (1, r - 1, c),  # MOVE_N
            (2, r + 1, c),  # MOVE_S
            (3, r, c + 1),  # MOVE_E
            (4, r, c - 1)   # MOVE_W
        ]
        # Sort moves to prioritize the ones that minimize Manhattan distance to target
        all_moves.sort(key=lambda x: abs(x[1] - tr) + abs(x[2] - tc))
        
        occupied = set()
        for i in range(env.n_agents):
            if i != agent_id:
                occupied.add((int(env.agent_pos[i, 0]), int(env.agent_pos[i, 1])))
                
        # Try best available move that does not bump into an occupied cell
        for m, nr, nc in all_moves:
            if 0 <= nr < env.height and 0 <= nc < env.width:
                if (nr, nc) not in occupied:
                    return m
                    
        # Fallback: if totally surrounded, return the preferred direction to queue a push
        for m, nr, nc in all_moves:
            if 0 <= nr < env.height and 0 <= nc < env.width:
                return m
        return 0

    # PRIORITY 1: Get inventory to the capacity of 3 for the continuous holding reward
    if my_inv < 3:
        target_candidates = []
        for tgt in range(env.n_agents):
            if tgt == agent_id: continue
            if tgt in policy.red_agents: continue
            if int(env.agent_clan[tgt]) == my_clan: continue
            if int(env.inventory[tgt]) <= 0: continue
            target_candidates.append(tgt)
            
        if target_candidates:
            # Target the closest valid agent
            target_candidates.sort(key=lambda t: abs(int(env.agent_pos[t, 0]) - r) + abs(int(env.agent_pos[t, 1]) - c))
            closest = target_candidates[0]
            cr, cc = int(env.agent_pos[closest, 0]), int(env.agent_pos[closest, 1])
            
            # If adjacent, RAID them
            if abs(cr - r) + abs(cc - c) == 1:
                return 6 + closest
            else:
                return move_towards(cr, cc)

    # PRIORITY 2: Harvest apples globally and efficiently
    best_apple = None
    best_score = -999999
    
    apples = []
    for ar in range(env.height):
        for ac in range(env.width):
            if env.orchard_apple[ar, ac] or env.bonus_apple[ar, ac]:
                apples.append((ar, ac))
                
    for ar, ac in apples:
        my_dist = abs(ar - r) + abs(ac - c)
        
        # Determine how close the nearest OTHER agent is to this apple
        min_other_dist = 999999
        for i in range(env.n_agents):
            if i != agent_id:
                oi_r, oi_c = int(env.agent_pos[i, 0]), int(env.agent_pos[i, 1])
                dist = abs(ar - oi_r) + abs(ac - oi_c)
                if dist < min_other_dist:
                    min_other_dist = dist
                    
        score = -my_dist
        # Apply strict heuristic penalties if another agent is poised to steal it
        if min_other_dist < my_dist:
            score -= 20
        elif min_other_dist == my_dist:
            score -= 5
            
        if score > best_score:
            best_score = score
            best_apple = (ar, ac)
            
    if best_apple:
        return move_towards(best_apple[0], best_apple[1])
        
    # PRIORITY 3: If no apples exist on the board, drift back to our clan's orchard center to await regrowth
    orchard = env.orchard_cells_per_q[my_clan]
    if orchard:
        cr = sum(x[0] for x in orchard) // len(orchard)
        cc = sum(x[1] for x in orchard) // len(orchard)
        return move_towards(cr, cc)
        
    return 0