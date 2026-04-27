def policy(env, agent_id: int) -> int:
    if env._travel_queue[agent_id]:
        return 0

    clan = int(env.agent_clan[agent_id])
    H, W = env.height, env.width
    r, c = int(env.agent_pos[agent_id, 0]), int(env.agent_pos[agent_id, 1])
    pos = (r, c)

    plaza = env.plaza_cells_set
    rivers = env.river_cells_set
    own_river = set(env.river_cells_per_q[clan])
    own_orchard = set(env.orchard_cells_per_q[clan])
    orchards = env.orchard_cells_set

    w_q_own = float(env.w_q[clan])
    w_p = float(env.w_p)
    in_plaza = pos in plaza

    blocked = {(int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1]))
               for j in range(env.n_agents) if j != agent_id}

    DIRS = ((-1, 0), (1, 0), (0, 1), (0, -1))

    def mv(dr: int, dc: int) -> int:
        if dr == -1 and dc == 0: return 1
        if dr ==  1 and dc == 0: return 2
        if dr ==  0 and dc == 1: return 3
        if dr ==  0 and dc == -1: return 4
        return 0

    # Phase 2 auto-collects if standing on an apple. Stay if we want it, vacate if it's trapped.
    if in_plaza and env.bonus_apple[r, c]:
        if w_q_own > 0.5:
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in blocked:
                    return mv(dr, dc)
            return 0
        else:
            return 0

    if pos in orchards and env.orchard_apple[r, c]:
        return 0

    from collections import deque

    def bfs_target(target_set):
        def _bfs(tb):
            if not target_set: return None
            if pos in target_set: return (0, 0)
            visited = {pos}
            queue = deque()
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    queue.append((nr, nc, dr, dc))
                    visited.add((nr, nc))
            while queue:
                cr, cc, fdr, fdc = queue.popleft()
                if (cr, cc) in target_set:
                    return (fdr, fdc)
                if tb and (cr, cc) in blocked:
                    continue
                for dr, dc in DIRS:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        if (nr, nc) not in visited:
                            visited.add((nr, nc))
                            queue.append((nr, nc, fdr, fdc))
            return None
        s = _bfs(True)
        if s is not None: return s
        return _bfs(False)

    def adjacent_to_own_river():
        for dr, dc in DIRS:
            nbr = (r + dr, c + dc)
            if nbr in own_river:
                return True
        return False

    # Emergency Fail-Safes: step in to clean only if Blue is failing to prevent threshold crossing.
    if w_p > 0.40:
        if in_plaza:
            return 5
        step = bfs_target(plaza)
        if step and step != (0, 0): return mv(*step)

    if w_q_own > 0.40:
        if adjacent_to_own_river() and not in_plaza:
            return 5
        approach = set()
        for cell in own_river:
            for dr, dc in DIRS:
                nbr = (cell[0] + dr, cell[1] + dc)
                if 0 <= nbr[0] < H and 0 <= nbr[1] < W:
                    if nbr not in rivers and nbr not in plaza:
                        approach.add(nbr)
        step = bfs_target(approach)
        if step and step != (0, 0): return mv(*step)

    # Global Harvester: weigh reward potential per step to path optimally.
    def find_apple(treat_blocked: bool):
        best_score = -1.0
        best_step = None
        visited = {pos}
        queue = deque()
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                queue.append((nr, nc, dr, dc, 1))
                visited.add((nr, nc))
        while queue:
            cr, cc, fdr, fdc, d = queue.popleft()
            
            score = -1.0
            if (cr, cc) in plaza and env.bonus_apple[cr, cc]:
                if w_q_own <= 0.5:
                    score = 2.0 / d
            elif (cr, cc) in orchards and env.orchard_apple[cr, cc]:
                score = 1.0 / d
                
            if score > best_score:
                best_score = score
                best_step = (fdr, fdc)
                
            if treat_blocked and (cr, cc) in blocked:
                continue
                
            for dr, dc in DIRS:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < H and 0 <= nc < W:
                    if (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append((nr, nc, fdr, fdc, d + 1))
        return best_step, best_score

    step, score = find_apple(True)
    if step is None:
        step, score = find_apple(False)
        
    if step is not None:
        return mv(*step)
        
    # Idle behavior: drift out of plaza to prevent regrowth blocking.
    if pos not in own_orchard:
        step = bfs_target(own_orchard)
        if step and step != (0, 0):
            return mv(*step)
            
    return 0