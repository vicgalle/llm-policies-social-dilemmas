def policy(env, agent_id) -> int:
    from collections import deque
    
    # 0. Register Red agent
    if not hasattr(env, '_red_ids_set'):
        env._red_ids_set = set()
    env._red_ids_set.add(agent_id)
    
    if env.agent_timeout[agent_id] > 0:
        return int(CleanupAction.STAND)
        
    r, c = int(env.agent_pos[agent_id, 0]), int(env.agent_pos[agent_id, 1])
    orient = int(env.agent_orient[agent_id])
    
    # Initialize per-step coordination caches
    if not hasattr(env, '_red_targets_this_step'):
        env._red_targets_this_step = {'step': -1, 'targets': set()}
    if env._red_targets_this_step['step'] != env._step_count:
        env._red_targets_this_step = {'step': env._step_count, 'targets': set()}
        
    if not hasattr(env, '_red_claimed_apples'):
        env._red_claimed_apples = {'step': -1, 'apples': set()}
    if env._red_claimed_apples['step'] != env._step_count:
        env._red_claimed_apples = {'step': env._step_count, 'apples': set()}
        
    if not hasattr(env, '_red_claimed_camps'):
        env._red_claimed_camps = {'step': -1, 'camps': set()}
    if env._red_claimed_camps['step'] != env._step_count:
        env._red_claimed_camps = {'step': env._step_count, 'camps': set()}

    if not hasattr(env, '_red_camp_spots'):
        apple_set = set(env.apple_points)
        camp_spots = set()
        for ar, ac in apple_set:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = ar + dr, ac + dc
                if 0 <= nr < env.height and 0 <= nc < env.width:
                    if not env.walls[nr, nc] and (nr, nc) not in apple_set:
                        camp_spots.add((nr, nc))
        env._red_camp_spots = camp_spots

    # 1. Ambush Zapping Logic
    def get_beam_targets(test_orient):
        a, b, c_dir, d = _ROTATIONS[Orientation(test_orient)]
        half_w = env.beam_width // 2
        blue_targets = []
        red_count = 0
        for dist in range(1, env.beam_length + 1):
            for w_off in range(-half_w, half_w + 1):
                br = r + a * dist + b * w_off
                bc = c + c_dir * dist + d * w_off
                if 0 <= br < env.height and 0 <= bc < env.width:
                    if not env.walls[br, bc]:
                        for j in range(env.n_agents):
                            if env.agent_timeout[j] > 0: continue
                            if env.agent_pos[j, 0] == br and env.agent_pos[j, 1] == bc:
                                if j in env._red_ids_set:
                                    red_count += 1
                                else:
                                    # Zap if in apple half (harvester) AND not already targeted by other Red
                                    if env.agent_pos[j, 1] >= env.width // 2:
                                        if j not in env._red_targets_this_step['targets']:
                                            blue_targets.append(j)
        return blue_targets, red_count

    cb_targets, cr = get_beam_targets(orient)
    if len(cb_targets) > 0 and cr == 0:
        env._red_targets_this_step['targets'].update(cb_targets)
        return int(CleanupAction.BEAM)
        
    lb_targets, lr = get_beam_targets((orient - 1) % 4)
    rb_targets, rr = get_beam_targets((orient + 1) % 4)
    bb_targets, br = get_beam_targets((orient + 2) % 4)
    
    if len(lb_targets) > 0 and lr == 0:
        return int(CleanupAction.ROTATE_LEFT)
    if len(rb_targets) > 0 and rr == 0:
        return int(CleanupAction.ROTATE_RIGHT)
    if len(bb_targets) > 0 and br == 0:
        return int(CleanupAction.ROTATE_LEFT)

    # 2. Opportunistic cleaning (rarely triggers if we stay near apples)
    def count_waste(test_orient):
        a, b, c_dir, d = _ROTATIONS[Orientation(test_orient)]
        half_w = env.beam_width // 2
        w_count = 0
        for dist in range(1, env.beam_length + 1):
            for w_off in range(-half_w, half_w + 1):
                br = r + a * dist + b * w_off
                bc = c + c_dir * dist + d * w_off
                if 0 <= br < env.height and 0 <= bc < env.width:
                    if not env.walls[br, bc] and env.waste[br, bc]:
                        w_count += 1
        return w_count
        
    if count_waste(orient) >= 4:
        return int(CleanupAction.CLEAN)

    # Move helper using strafing to preserve our ambush orientation
    def get_move_action(dr, dc, o):
        a, b, c_dir, d = _ROTATIONS[Orientation(o)]
        if dr == a and dc == c_dir: return int(CleanupAction.FORWARD)
        if dr == -a and dc == -c_dir: return int(CleanupAction.BACKWARD)
        if dr == -b and dc == -d: return int(CleanupAction.STEP_LEFT)
        if dr == b and dc == d: return int(CleanupAction.STEP_RIGHT)
        return int(CleanupAction.STAND)

    # 3. Find nearest unclaimed apple
    alive_apples = set()
    for a_idx in range(env.n_apples):
        if env.apple_alive[a_idx]:
            alive_apples.add((int(env._apple_pos[a_idx, 0]), int(env._apple_pos[a_idx, 1])))

    def bfs_nearest_apple():
        if not alive_apples: return None
        queue = deque([(r, c, [])])
        visited = set([(r, c)])
        while queue:
            curr_r, curr_c, path = queue.popleft()
            if (curr_r, curr_c) in alive_apples:
                if (curr_r, curr_c) not in env._red_claimed_apples['apples']:
                    return (curr_r, curr_c), path
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = curr_r + dr, curr_c + dc
                if 0 <= nr < env.height and 0 <= nc < env.width:
                    if not env.walls[nr, nc] and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append((nr, nc, path + [(dr, dc)]))
        return None

    apple_res = bfs_nearest_apple()
    if apple_res is not None:
        (ar, ac), path = apple_res
        env._red_claimed_apples['apples'].add((ar, ac))
        if len(path) > 0:
            dr, dc = path[0]
            return get_move_action(dr, dc, orient)
        else:
            return int(CleanupAction.STAND)

    # 4. If no apples, go to an unclaimed camp spot to prepare for the next spawn
    def bfs_to_camp():
        queue = deque([(r, c, [])])
        visited = set([(r, c)])
        while queue:
            curr_r, curr_c, path = queue.popleft()
            if (curr_r, curr_c) in env._red_camp_spots:
                if (curr_r, curr_c) not in env._red_claimed_camps['camps']:
                    return (curr_r, curr_c), path
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = curr_r + dr, curr_c + dc
                if 0 <= nr < env.height and 0 <= nc < env.width:
                    if not env.walls[nr, nc] and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append((nr, nc, path + [(dr, dc)]))
        
        # Fallback to nearest even if claimed
        queue = deque([(r, c, [])])
        visited = set([(r, c)])
        while queue:
            curr_r, curr_c, path = queue.popleft()
            if (curr_r, curr_c) in env._red_camp_spots:
                return (curr_r, curr_c), path
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = curr_r + dr, curr_c + dc
                if 0 <= nr < env.height and 0 <= nc < env.width:
                    if not env.walls[nr, nc] and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append((nr, nc, path + [(dr, dc)]))
        return None

    camp_res = bfs_to_camp()
    if camp_res is not None:
        (cr_pos, cc_pos), path = camp_res
        if len(path) > 0:
            env._red_claimed_camps['camps'].add((cr_pos, cc_pos))
            dr, dc = path[0]
            return get_move_action(dr, dc, orient)
            
    # 5. Positioned at camp spot. Face WEST to watch the river/chokepoint.
    target_orient = int(Orientation.WEST)
    if orient != target_orient:
        diff = (target_orient - orient) % 4
        if diff == 1: return int(CleanupAction.ROTATE_RIGHT)
        else: return int(CleanupAction.ROTATE_LEFT)

    return int(CleanupAction.STAND)