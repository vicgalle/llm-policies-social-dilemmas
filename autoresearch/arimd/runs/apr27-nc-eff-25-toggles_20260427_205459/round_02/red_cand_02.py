def policy(env, agent_id: int) -> int:
    r = int(env.agent_pos[agent_id, 0])
    c = int(env.agent_pos[agent_id, 1])
    
    in_plaza = (8 <= r <= 11 and 8 <= c <= 11)
    # Check for bonus apples explicitly; these are the only ones collected in Phase 2
    bonus_cells = [(br, bc) for br in range(8, 12) for bc in range(8, 12) if env.bonus_apple[br, bc]]
    plaza_cells = [(pr, pc) for pr in range(8, 12) for pc in range(8, 12)]
    
    # Precompute occupied cells to route safely around stationary Blue agents
    occupied = set()
    for j in range(env.n_agents):
        if j != agent_id:
            occupied.add((int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1])))
            
    def get_best_step(r, c, targets, require_plaza=False):
        if (r, c) in targets: return 0
        current_min_d = min(abs(tr - r) + abs(tc - c) for tr, tc in targets)
        
        def is_valid(nr, nc):
            if not (0 <= nr < env.height and 0 <= nc < env.width): return False
            if require_plaza and not (8 <= nr <= 11 and 8 <= nc <= 11): return False
            return True

        # 1. Prefer empty cells that strictly reduce Manhattan distance
        best_move = None
        min_d = 999
        for dr, dc, action in [(-1, 0, 1), (1, 0, 2), (0, 1, 3), (0, -1, 4)]:
            nr, nc = r + dr, c + dc
            if not is_valid(nr, nc): continue
            d = min(abs(tr - nr) + abs(tc - nc) for tr, tc in targets)
            if d < current_min_d and (nr, nc) not in occupied:
                if d < min_d:
                    min_d = d
                    best_move = action
        if best_move is not None: return best_move
            
        # 2. If we are exactly 1 step away, forcefully push into the target even if occupied.
        # Phase 1 movement resolution allows "following" vacating agents safely.
        if current_min_d == 1:
            for dr, dc, action in [(-1, 0, 1), (1, 0, 2), (0, 1, 3), (0, -1, 4)]:
                nr, nc = r + dr, c + dc
                if not is_valid(nr, nc): continue
                d = min(abs(tr - nr) + abs(tc - nc) for tr, tc in targets)
                if d == 0: return action

        # 3. Take an empty sidestep (keeps distance the same) if blocked linearly
        min_d = 999
        for dr, dc, action in [(-1, 0, 1), (1, 0, 2), (0, 1, 3), (0, -1, 4)]:
            nr, nc = r + dr, c + dc
            if not is_valid(nr, nc): continue
            d = min(abs(tr - nr) + abs(tc - nc) for tr, tc in targets)
            if d == current_min_d and (nr, nc) not in occupied:
                if d < min_d:
                    min_d = d
                    best_move = action
        if best_move is not None: return best_move

        # 4. Push into an occupied cell to reduce distance as a last resort
        min_d = 999
        for dr, dc, action in [(-1, 0, 1), (1, 0, 2), (0, 1, 3), (0, -1, 4)]:
            nr, nc = r + dr, c + dc
            if not is_valid(nr, nc): continue
            d = min(abs(tr - nr) + abs(tc - nc) for tr, tc in targets)
            if d < current_min_d:
                if d < min_d:
                    min_d = d
                    best_move = action
        if best_move is not None: return best_move
            
        return 0

    if bonus_cells:
        if in_plaza:
            # We are inside and apples exist. Maneuver to them safely while ensuring we stay in the plaza.
            return get_best_step(r, c, bonus_cells, require_plaza=True)
        else:
            # Step in. Plunge into the nearest plaza cell so Phase 2 recognizes us as occupants.
            return get_best_step(r, c, plaza_cells, require_plaza=False)
    else:
        # Zero bonus apples exist globally. Evacuate the plaza onto un-harvested adjacent safe cells!
        # This expands the empty cell surface area for Phase 8 regrowth algorithms.
        SAFE_SPOTS = [
            (7, 8), (7, 9), (7, 10), (7, 11),
            (12, 8), (12, 9), (12, 10), (12, 11),
            (9, 7), (10, 7), (9, 12), (10, 12)
        ]
        # Shift preference dynamically to find a guaranteed unique, un-colliding spot per Red agent
        my_spots = [SAFE_SPOTS[(agent_id + i) % 12] for i in range(12)]
        target_spot = None
        for spot in my_spots:
            occ = False
            for j in range(env.n_agents):
                if j != agent_id and int(env.agent_pos[j, 0]) == spot[0] and int(env.agent_pos[j, 1]) == spot[1]:
                    occ = True
                    break
            if not occ:
                target_spot = spot
                break
        if not target_spot:
            target_spot = SAFE_SPOTS[agent_id % 12]
            
        if (r, c) == target_spot:
            return 0
        else:
            return get_best_step(r, c, [target_spot], require_plaza=False)