def policy(env, agent_id: int) -> int:
    if int(env.agent_timeout[agent_id]) > 0:
        return 7

    start = (int(env.agent_pos[agent_id][0]), int(env.agent_pos[agent_id][1]))

    if not env.apple_points:
        return 7

    def get_first_step(targets):
        if start in targets:
            return (0, 0)
        q = deque([start])
        visited = {start: None}
        while q:
            curr = q.popleft()
            if curr in targets:
                path = curr
                while visited[path] != start:
                    path = visited[path]
                return (path[0] - start[0], path[1] - start[1])
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = curr[0] + dr, curr[1] + dc
                if 0 <= nr < env.height and 0 <= nc < env.width:
                    if not env.walls[nr, nc]:
                        if (nr, nc) not in visited:
                            visited[(nr, nc)] = curr
                            q.append((nr, nc))
        return None

    def get_holonomic_action(dr, dc, orient):
        # Maps a chosen grid step (dr, dc) to the optimal action (0-3) without rotating
        if dr == -1 and dc == 0:
            return [0, 2, 1, 3][orient]
        elif dr == 1 and dc == 0:
            return [1, 3, 0, 2][orient]
        elif dr == 0 and dc == 1:
            return [3, 0, 2, 1][orient]
        elif dr == 0 and dc == -1:
            return [2, 1, 3, 0][orient]
        return 7

    # 1. Harvest any alive apples
    apple_targets = set()
    for a_idx in range(env.n_apples):
        if env.apple_alive[a_idx]:
            apple_targets.add((int(env._apple_pos[a_idx][0]), int(env._apple_pos[a_idx][1])))

    if apple_targets:
        step = get_first_step(apple_targets)
        if step is not None and step != (0, 0):
            dr, dc = step
            return get_holonomic_action(dr, dc, int(env.agent_orient[agent_id]))

    # 2. If no apples exist, find optimal camping spots
    # We camp just inside the apple region, but strictly off the actual spawn points
    min_c = min(c for r, c in env.apple_points)
    max_c = max(c for r, c in env.apple_points)
    
    camping_cells = set()
    apple_points_set = set(env.apple_points)
    for r in range(env.height):
        for c in range(env.width):
            if not env.walls[r, c] and (r, c) not in apple_points_set:
                if min_c - 1 <= c <= max_c + 1:
                    camping_cells.add((r, c))

    if camping_cells:
        step = get_first_step(camping_cells)
        if step is not None and step != (0, 0):
            dr, dc = step
            return get_holonomic_action(dr, dc, int(env.agent_orient[agent_id]))

    return 7