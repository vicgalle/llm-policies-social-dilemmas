def policy(env, agent_id) -> int:
    from collections import deque

    if env.agent_timeout[agent_id] > 0:
        return int(CleanupAction.STAND)
        
    # Maintain a shared set of Red agents to avoid friendly fire, safely resetting each episode
    current_step = getattr(env, '_step_count', 0)
    if not hasattr(env, '_red_last_step') or env._red_last_step != current_step:
        env._red_last_step = current_step
        if current_step == 0:
            env._red_agents = set()
            env._red_targets = {}
            
    if not hasattr(env, '_red_agents'):
        env._red_agents = set()
    env._red_agents.add(agent_id)
    
    # Calculate the maximum number of cleaners Blue might draft. We NEVER zap these agents!
    max_cleaners = min(env.n_agents, max(1, int(round(0.2 * env.n_agents))) + max(2, env.n_agents // 4))
    restricted_targets = env._red_agents.union({j for j in range(env.n_agents) if j < max_cleaners})
    
    ar = int(env.agent_pos[agent_id, 0])
    ac = int(env.agent_pos[agent_id, 1])
    
    # 1. Zap Blue harvesters if they are encroaching on the apple area (columns >= 9)
    best_zap_act = None
    if current_step > 0:
        def get_agents_in_beam(orient_int):
            orient = Orientation(orient_int)
            a, b, c, d = _ROTATIONS[orient]
            half_w = env.beam_width // 2
            beam_cells = set()
            for dist in range(1, env.beam_length + 1):
                for w_off in range(-half_w, half_w + 1):
                    br = ar + a * dist + b * w_off
                    bc = ac + c * dist + d * w_off
                    if 0 <= br < env.height and 0 <= bc < env.width:
                        if not env.walls[br, bc]:
                            beam_cells.add((br, bc))
            hits = []
            for j in range(env.n_agents):
                if env.agent_timeout[j] > 0:
                    continue
                jr, jc = int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1])
                if (jr, jc) in beam_cells:
                    hits.append(j)
            return hits

        # Check standing fire, turn left, and turn right
        for act, turn in [(None, 0), (CleanupAction.ROTATE_LEFT, -1), (CleanupAction.ROTATE_RIGHT, 1)]:
            test_orient = (int(env.agent_orient[agent_id]) + turn) % 4
            hits = get_agents_in_beam(test_orient)
            
            bad_hit = False
            good_hit = False
            for j in hits:
                if j == agent_id:
                    continue
                if j in restricted_targets:
                    bad_hit = True
                elif int(env.agent_pos[j, 1]) >= 9:
                    good_hit = True
                    
            if good_hit and not bad_hit:
                best_zap_act = act if act is not None else CleanupAction.BEAM
                if act is None:
                    break
                    
    if best_zap_act is not None:
        return int(best_zap_act)
        
    # 2. Optimal Apple Harvesting (BFS)
    apples = []
    for idx in range(env.n_apples):
        if env.apple_alive[idx]:
            apples.append((int(env._apple_pos[idx, 0]), int(env._apple_pos[idx, 1])))
            
    if not hasattr(env, '_red_targets'):
        env._red_targets = {}
        
    def action_to_dr_dc(act, orient_int):
        orient = Orientation(orient_int)
        a, b, c, d = _ROTATIONS[orient]
        if act == CleanupAction.FORWARD: return a, c
        if act == CleanupAction.BACKWARD: return -a, -c
        if act == CleanupAction.STEP_LEFT: return -b, -d
        if act == CleanupAction.STEP_RIGHT: return b, d
        return 0, 0

    def get_step_action(dr, dc, orient_int):
        for act in [CleanupAction.FORWARD, CleanupAction.BACKWARD, CleanupAction.STEP_LEFT, CleanupAction.STEP_RIGHT]:
            test_dr, test_dc = action_to_dr_dc(act, orient_int)
            if test_dr == dr and test_dc == dc:
                return act
        return None

    def get_action_to(target_cells):
        if not target_cells: return None, None
        if (ar, ac) in target_cells:
            return CleanupAction.STAND, (ar, ac)
            
        queue = deque()
        queue.append((ar, ac))
        visited = set()
        visited.add((ar, ac))
        parent = {}
        
        target_found = None
        while queue:
            r, c = queue.popleft()
            if (r, c) in target_cells:
                target_found = (r, c)
                break
                
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < env.height and 0 <= nc < env.width:
                    # Treat agents as walkable (since mechanics allow overlapping)
                    if not env.walls[nr, nc] and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        parent[(nr, nc)] = (r, c)
                        queue.append((nr, nc))
                        
        if not target_found:
            return None, None
            
        curr = target_found
        while parent[curr] != (ar, ac):
            curr = parent[curr]
            
        dr = curr[0] - ar
        dc = curr[1] - ac
        
        act = get_step_action(dr, dc, int(env.agent_orient[agent_id]))
        return act, target_found

    # Coordinate targets with other Red agents to avoid double-booking
    other_red_targets = set()
    for r_id, t_pos in env._red_targets.items():
        if r_id != agent_id and r_id in env._red_agents:
            other_red_targets.add(t_pos)
            
    available_apples = [a for a in apples if a not in other_red_targets]
    
    # Try an uncontested apple first
    act, target = get_action_to(available_apples)
    if act is not None:
        env._red_targets[agent_id] = target
        return int(act)
        
    # Fallback to any alive apple
    act, target = get_action_to(apples)
    if act is not None:
        env._red_targets[agent_id] = target
        return int(act)
        
    # If no apples are currently alive, proceed to spawn area and wait
    if env.n_apples > 0:
        apple_spawns = [(int(env._apple_pos[i, 0]), int(env._apple_pos[i, 1])) for i in range(env.n_apples)]
        act, target = get_action_to(apple_spawns)
        if act is not None:
            return int(act)
        
    return int(CleanupAction.STAND)