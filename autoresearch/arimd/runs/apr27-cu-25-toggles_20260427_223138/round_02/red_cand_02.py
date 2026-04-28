def policy(env, agent_id: int) -> int:
    if env.agent_timeout[agent_id] > 0:
        return int(CleanupAction.STAND)
        
    def get_valid_neighbors(r, c):
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < env.height and 0 <= nc < env.width and not env.walls[nr, nc]:
                neighbors.append((nr, nc))
        return neighbors

    agent_pos = (int(env.agent_pos[agent_id, 0]), int(env.agent_pos[agent_id, 1]))
    
    def get_action_for_dr_dc(dr, dc):
        orient = Orientation(int(env.agent_orient[agent_id]))
        a, b, c, d = _ROTATIONS[orient]
        if dr == a and dc == c: return int(CleanupAction.FORWARD)
        if dr == -a and dc == -c: return int(CleanupAction.BACKWARD)
        if dr == -b and dc == -d: return int(CleanupAction.STEP_LEFT)
        if dr == b and dc == d: return int(CleanupAction.STEP_RIGHT)
        return int(CleanupAction.STAND)

    def get_action_for_path(target):
        if target == agent_pos:
            return int(CleanupAction.STAND)
        q = deque([agent_pos])
        visited = {agent_pos: None}
        found = False
        while q:
            curr = q.popleft()
            if curr == target:
                found = True
                break
            for nxt in get_valid_neighbors(*curr):
                if nxt not in visited:
                    visited[nxt] = curr
                    q.append(nxt)
        if not found:
            return int(CleanupAction.STAND)
        curr = target
        while visited[curr] != agent_pos:
            curr = visited[curr]
        dr = curr[0] - agent_pos[0]
        dc = curr[1] - agent_pos[1]
        return get_action_for_dr_dc(dr, dc)

    potential_waste = len(env.river_cells_list)
    current_waste = sum(1 for r, c in env.river_cells_list if env.waste[r, c])
    waste_density = current_waste / potential_waste if potential_waste > 0 else 0.0

    alive_apples = set()
    for idx in range(env.n_apples):
        if env.apple_alive[idx]:
            alive_apples.add((int(env._apple_pos[idx, 0]), int(env._apple_pos[idx, 1])))

    def count_waste_in_beam(test_orient=None):
        orient = Orientation(int(test_orient if test_orient is not None else env.agent_orient[agent_id]))
        a, b, c, d = _ROTATIONS[orient]
        ar, ac = agent_pos
        half_w = env.beam_width // 2
        count = 0
        for dist in range(1, env.beam_length + 1):
            for w_off in range(-half_w, half_w + 1):
                br = ar + a * dist + b * w_off
                bc = ac + c * dist + d * w_off
                if 0 <= br < env.height and 0 <= bc < env.width:
                    if not env.walls[br, bc] and env.waste[br, bc]:
                        count += 1
        return count

    # 1. Efficient Cleaning Check: 
    # If waste is becoming problematic and there are no apples, we are willing to take a 1-hit clean (-1 cost).
    # Otherwise, we prioritize apples but will opportunistically snap a shot if we can hit >=2 waste cells at once.
    min_hits_needed = 1 if (waste_density > 0.15 and not alive_apples) else 2
    best_act = None
    best_hits = 0

    current_clean_hits = count_waste_in_beam(None)
    if current_clean_hits >= min_hits_needed:
        best_hits = current_clean_hits
        best_act = int(CleanupAction.CLEAN)

    for rot_act, turn in [(CleanupAction.ROTATE_LEFT, -1), (CleanupAction.ROTATE_RIGHT, 1)]:
        test_o = (int(env.agent_orient[agent_id]) + turn) % 4
        hits = count_waste_in_beam(test_o)
        if hits > best_hits and hits >= min_hits_needed:
            best_hits = hits
            best_act = int(rot_act)

    if best_act is not None:
        return best_act

    # 2. Harvest Apples:
    if alive_apples:
        q = deque([agent_pos])
        visited = {agent_pos: None}
        target = None
        while q:
            curr = q.popleft()
            if curr in alive_apples:
                target = curr
                break
            for nxt in get_valid_neighbors(*curr):
                if nxt not in visited:
                    visited[nxt] = curr
                    q.append(nxt)
        if target:
            return get_action_for_path(target)

    # 3. Handle River Downtime (Desperate):
    # Only if waste density is high and no apples are available, we route ourselves to an advantageous spot near the waste.
    if waste_density > 0.15:
        waste_adj = set()
        for r, c in env.river_cells_list:
            if env.waste[r, c]:
                for nr, nc in get_valid_neighbors(r, c):
                    if not env.waste[nr, nc]:
                        waste_adj.add((nr, nc))
        if waste_adj:
            q = deque([agent_pos])
            visited = {agent_pos: None}
            target = None
            while q:
                curr = q.popleft()
                if curr in waste_adj:
                    target = curr
                    break
                for nxt in get_valid_neighbors(*curr):
                    if nxt not in visited:
                        visited[nxt] = curr
                        q.append(nxt)
            if target:
                return get_action_for_path(target)

    # 4. Idle Position (Optimal Waiting):
    # Stand on free empty spaces specifically adjacent to apple spawn locations to prevent spawn-blocking!
    apple_points_set = set((int(r), int(c)) for r, c in env.apple_points)
    waiting_spots = set()
    for r in range(env.height):
        for c in range(env.width):
            if not env.walls[r, c] and (r, c) not in apple_points_set:
                for nr, nc in get_valid_neighbors(r, c):
                    if (nr, nc) in apple_points_set:
                        waiting_spots.add((r, c))
                        break

    if agent_pos in waiting_spots:
        return int(CleanupAction.STAND)

    if waiting_spots:
        q = deque([agent_pos])
        visited = {agent_pos: None}
        target = None
        while q:
            curr = q.popleft()
            if curr in waiting_spots:
                target = curr
                break
            for nxt in get_valid_neighbors(*curr):
                if nxt not in visited:
                    visited[nxt] = curr
                    q.append(nxt)
        if target:
            return get_action_for_path(target)

    return int(CleanupAction.STAND)