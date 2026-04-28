# Reasoning:
# 1. Free-riding: We completely refuse to pay the public good cost (CLEAN) and never use the penalty BEAM. By slacking off, we force the Blue policy's adaptive role-drafting mechanism to reassign other Blue agents as cleaners when waste density rises, ensuring the river stays clean at their expense.
# 2. Optimal Hoarding via Strafing: Blue uses a standard rotation-based BFS, taking 1-2 extra steps to turn before moving. We exploit the environment's STEP_LEFT, STEP_RIGHT, and BACKWARD actions to strafe optimally, allowing us to move 1 cell per step and consistently outrun Blue to spawned apples.
# 3. Yield Maximization: Agents standing on spawn points block apples from appearing. When no apples are active, we camp on empty cells *adjacent* to the orchard rather than on the spawns themselves, maintaining 100% apple spawn capacity while being perfectly positioned.

def policy(env, agent_id: int) -> int:
    # 7 is STAND, 0 is FORWARD, 1 is BACKWARD, 2 is STEP_LEFT, 3 is STEP_RIGHT
    if int(env.agent_timeout[agent_id]) > 0:
        return 7

    start_r, start_c = int(env.agent_pos[agent_id, 0]), int(env.agent_pos[agent_id, 1])
    H, W = env.height, env.width
    
    # 1. Find all currently active apples
    active_apples = set()
    for idx in range(env.n_apples):
        if env.apple_alive[idx]:
            active_apples.add((int(env._apple_pos[idx, 0]), int(env._apple_pos[idx, 1])))
            
    # 2. If no apples, find safe camping spots adjacent to the orchard
    if not active_apples:
        safe_camps = []
        ap_set = set(env.apple_points)
        for r in range(H):
            for c in range(W):
                # Empty walkable cell that is not an apple spawn point
                if not env.walls[r, c] and (r, c) not in ap_set:
                    # Must be adjacent to an apple spawn
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        if (r+dr, c+dc) in ap_set:
                            safe_camps.append((r, c))
                            break
        
        if safe_camps:
            safe_camps.sort()
            my_camp = safe_camps[(agent_id * 17) % len(safe_camps)]
            target_cells = {my_camp}
        else:
            # Fallback if no safe camps found
            if not env.apple_points:
                return 7
            ap = sorted(list(env.apple_points))
            my_camp = ap[(agent_id * 17) % len(ap)]
            target_cells = {my_camp}
    else:
        target_cells = active_apples

    # 3. BFS to the nearest target (agents do not collide, so we ignore them as obstacles)
    queue = deque()
    queue.append((start_r, start_c))
    came_from = {(start_r, start_c): None}
    
    found_target = None
    
    while queue:
        r, c = queue.popleft()
        if (r, c) in target_cells:
            found_target = (r, c)
            break
            
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                if not env.walls[nr, nc]:
                    if (nr, nc) not in came_from:
                        came_from[(nr, nc)] = (r, c)
                        queue.append((nr, nc))
                        
    # 4. Reconstruct path and map to optimal strafing movement
    if found_target is not None and found_target != (start_r, start_c):
        curr = found_target
        while came_from[curr] != (start_r, start_c):
            curr = came_from[curr]
            
        dr = curr[0] - start_r
        dc = curr[1] - start_c
        
        # Absolute target direction: 0=NORTH, 1=EAST, 2=SOUTH, 3=WEST
        if dr == -1 and dc == 0: target_o = 0
        elif dr == 0 and dc == 1: target_o = 1
        elif dr == 1 and dc == 0: target_o = 2
        elif dr == 0 and dc == -1: target_o = 3
        else: return 7
        
        orient = int(env.agent_orient[agent_id])
        diff = (target_o - orient) % 4
        
        # Optimal strafing directly to the adjacent cell without rotating
        if diff == 0: return 0    # FORWARD
        elif diff == 1: return 3  # STEP_RIGHT
        elif diff == 2: return 1  # BACKWARD
        elif diff == 3: return 2  # STEP_LEFT

    return 7