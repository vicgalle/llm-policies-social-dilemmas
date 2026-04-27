def policy(env, agent_id: int) -> int:
    my_r = int(env.agent_pos[agent_id, 0])
    my_c = int(env.agent_pos[agent_id, 1])
    my_pos = (my_r, my_c)
    my_clan = int(env.agent_clan[agent_id])
    my_inv = int(env.inventory[agent_id])
    
    # We treat all other agents as obstacles to avoid movement conflicts.
    blocked = {(int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1])) 
               for j in range(16) if j != agent_id}
    
    def get_step(start, targets, use_blocked_logic=True):
        if not targets: 
            return 0
        if start in targets: 
            return 0
        
        for use_blocked in ([True, False] if use_blocked_logic else [False]):
            queue = deque([(start, 0)])
            visited = {start}
            while queue:
                curr, first_action = queue.popleft()
                for dr, dc, a in [(-1, 0, 1), (1, 0, 2), (0, 1, 3), (0, -1, 4)]:
                    nr, nc = curr[0] + dr, curr[1] + dc
                    if 0 <= nr < 20 and 0 <= nc < 20:
                        nxt = (nr, nc)
                        if nxt in targets:
                            return first_action or a
                        if nxt not in visited:
                            if use_blocked and nxt in blocked:
                                continue
                            visited.add(nxt)
                            queue.append((nxt, first_action or a))
        return 0

    # Phase 1: Secure maximum passive income by stealing 1 apple (inventory 2 -> 3)
    if my_inv < 3:
        best_j = None
        best_dist = float('inf')
        
        for j in range(16):
            if j == agent_id: 
                continue
            if int(env.agent_clan[j]) == my_clan: 
                continue
            
            inv = int(env.inventory[j])
            # Target agents with 1 or 2 apples. Agents with 3 are guaranteed Red, so we never team-kill.
            if 0 < inv < 3:
                jr, jc = int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1])
                dist = abs(jr - my_r) + abs(jc - my_c)
                if dist < best_dist:
                    best_dist = dist
                    best_j = j
                    
        if best_j is not None:
            if best_dist == 1:
                return 6 + best_j  # RAID_BASE + best_j
            else:
                jr, jc = int(env.agent_pos[best_j, 0]), int(env.agent_pos[best_j, 1])
                target_cells = set()
                for dr, dc in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
                    nr, nc = jr + dr, jc + dc
                    if 0 <= nr < 20 and 0 <= nc < 20:
                        target_cells.add((nr, nc))
                step = get_step(my_pos, target_cells)
                if step:
                    return step

    # Phase 2: Harvest mode - find the absolute closest accessible apple globally
    apples = []
    for r, c in env.orchard_cells_list:
        if env.orchard_apple[r, c] and (r, c) not in blocked:
            apples.append((r, c))
    for r, c in env.plaza_cells:
        if env.bonus_apple[r, c] and (r, c) not in blocked:
            apples.append((r, c))
            
    if apples:
        targets = set(apples)
        step = get_step(my_pos, targets)
        if step:
            return step
            
    # Phase 3: Wait centrally in our own orchard for regrowth
    centers = {
        0: (5, 5),
        1: (5, 14),
        2: (14, 5),
        3: (14, 14)
    }
    cr, cc = centers[my_clan]
    if my_pos != (cr, cc):
        step = get_step(my_pos, {(cr, cc)})
        if step:
            return step
            
    return 0