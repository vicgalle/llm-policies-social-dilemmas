def policy(env, agent_id) -> int:
    # 1. Dynamically identify Red agents via the shared env instance
    if not hasattr(env, 'red_identified'):
        env.red_identified = set()
    env.red_identified.add(agent_id)
    
    # If currently tagged out, we can't act anyway
    if int(env.agent_timeout[agent_id]) > 0:
        return int(CleanupAction.STAND)
        
    # Wait for the first two steps to ensure all Red agents are registered
    if env._step_count < 2:
        return int(CleanupAction.STAND)
        
    ar = int(env.agent_pos[agent_id, 0])
    ac = int(env.agent_pos[agent_id, 1])
    
    # 2. Check if we have a safe and profitable shot at a Blue harvester
    best_orient = None
    best_score = 0
    
    curr_o = int(env.agent_orient[agent_id])
    # Prefer current orientation to avoid wasting a step rotating
    orient_choices = [curr_o] + [o for o in range(4) if o != curr_o]
    
    for o in orient_choices:
        a, b, c_rot, d_rot = _ROTATIONS[Orientation(o)]
        half_w = env.beam_width // 2
        
        blue_hits = 0
        red_hits = 0
        cleaner_hits = 0
        
        for dist in range(1, env.beam_length + 1):
            for w_off in range(-half_w, half_w + 1):
                br = ar + a * dist + b * w_off
                bc = ac + c_rot * dist + d_rot * w_off
                if 0 <= br < env.height and 0 <= bc < env.width and not env.walls[br, bc]:
                    # Check if an agent is in this cell of the beam
                    for j in range(env.n_agents):
                        if int(env.agent_timeout[j]) > 0:
                            continue
                        if int(env.agent_pos[j, 0]) == br and int(env.agent_pos[j, 1]) == bc:
                            if j in env.red_identified:
                                red_hits += 1
                            else:
                                # Harvesters are in the apple patch (col >= 11)
                                if bc >= 11:
                                    blue_hits += 1
                                # Cleaners stay near the river (col < 11); penalize hitting them
                                else:
                                    cleaner_hits += 1
                                    
        # Fire ONLY if it hits harvesters, spares cleaners, and spares Red teammates
        if red_hits == 0 and cleaner_hits == 0 and blue_hits > 0:
            if blue_hits > best_score:
                best_score = blue_hits
                best_orient = o
                
    if best_score > 0:
        if best_orient == curr_o:
            return int(CleanupAction.BEAM)
        else:
            diff = (best_orient - curr_o) % 4
            if diff == 1: return int(CleanupAction.ROTATE_RIGHT)
            if diff == 3: return int(CleanupAction.ROTATE_LEFT)
            return int(CleanupAction.ROTATE_LEFT)
            
    # 3. If no immediate threats, efficiently path to the nearest apple
    apples = set()
    for i in range(env.n_apples):
        if env.apple_alive[i]:
            apples.add((int(env._apple_pos[i, 0]), int(env._apple_pos[i, 1])))
            
    # If no apples exist yet, fall back to a patrol point just outside the spawn area
    if not apples:
        target_r = ar
        target_c = 11
        if 0 <= target_r < env.height and 0 <= target_c < env.width and not env.walls[target_r, target_c]:
            apples.add((target_r, target_c))
        else:
            for r in range(1, env.height - 1):
                for c in range(11, env.width - 1):
                    if not env.walls[r, c]:
                        apples.add((r, c))
                        
    start = (ar, ac)
    if start in apples:
        return int(CleanupAction.STAND)
        
    # Standard BFS to find nearest apple or patrol coordinate
    q = deque([start])
    came_from = {start: None}
    found = None
    
    while q:
        curr = q.popleft()
        if curr in apples:
            found = curr
            break
            
        r, c = curr
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < env.height and 0 <= nc < env.width and not env.walls[nr, nc]:
                if (nr, nc) not in came_from:
                    came_from[(nr, nc)] = curr
                    q.append((nr, nc))
                    
    # Backtrack to find the optimal first step
    if found:
        curr = found
        while came_from[curr] != start:
            curr = came_from[curr]
            
        dr = curr[0] - start[0]
        dc = curr[1] - start[1]
        
        a, b, c_rot, d_rot = _ROTATIONS[Orientation(curr_o)]
        
        # Translate absolute grid direction into egocentric strafe actions 
        # so we don't waste time rotating our view frame
        if (dr, dc) == (a, c_rot): 
            return int(CleanupAction.FORWARD)
        if (dr, dc) == (-a, -c_rot): 
            return int(CleanupAction.BACKWARD)
        if (dr, dc) == (-b, -d_rot): 
            return int(CleanupAction.STEP_LEFT)
        if (dr, dc) == (b, d_rot): 
            return int(CleanupAction.STEP_RIGHT)
        
    return int(CleanupAction.STAND)