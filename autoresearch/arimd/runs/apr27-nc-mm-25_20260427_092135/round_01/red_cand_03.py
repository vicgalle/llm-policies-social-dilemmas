def policy(env, agent_id: int) -> int:
    r, c = int(env.agent_pos[agent_id, 0]), int(env.agent_pos[agent_id, 1])
    pos = (r, c)
    w_p = float(env.w_p)
    plaza = env.plaza_cells_set
    
    # Emergency safety: If w_p is dangerously high and we are already in the plaza, 
    # clean to prevent Blue from harvesting and destroying the shared bonus payout.
    if w_p >= 0.24 and pos in plaza:
        return 5  # CLEAN

    # Treat positions occupied by other agents as blocked to avoid traffic jams
    blocked = {(int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1]))
               for j in range(env.n_agents) if j != agent_id}
               
    DIRS = ((-1, 0), (1, 0), (0, 1), (0, -1))
    
    def mv(dr, dc):
        if dr == -1 and dc == 0: return 1  # MOVE_N
        if dr ==  1 and dc == 0: return 2  # MOVE_S
        if dr ==  0 and dc == 1: return 3  # MOVE_E
        if dr ==  0 and dc == -1: return 4 # MOVE_W
        return 0                           # NOOP

    def bfs(targets, treat_blocked=True):
        if pos in targets:
            return (0, 0)
        
        queue = deque([pos])
        visited = {pos}
        parent = {}
        
        while queue:
            curr = queue.popleft()
            for dr, dc in DIRS:
                nr, nc = curr[0] + dr, curr[1] + dc
                if 0 <= nr < env.height and 0 <= nc < env.width:
                    nbr = (nr, nc)
                    if nbr not in visited:
                        if treat_blocked and nbr in blocked and nbr not in targets:
                            continue
                        visited.add(nbr)
                        parent[nbr] = curr
                        if nbr in targets:
                            # Backtrack to find the first step of the path
                            trace = nbr
                            while parent[trace] != pos:
                                trace = parent[trace]
                            return (trace[0] - pos[0], trace[1] - pos[1])
                        queue.append(nbr)
        return None

    def get_step(targets):
        if not targets: return None
        s = bfs(targets, treat_blocked=True)
        if s is not None: return s
        # Fallback to traversing through blocked cells if no unblocked path exists
        return bfs(targets, treat_blocked=False)

    orchards = env.orchard_cells_set
    
    # Locate all currently existing apples across the entire map
    apples = {cell for cell in orchards if env.orchard_apple[cell[0], cell[1]]}
    bonus = {cell for cell in plaza if env.bonus_apple[cell[0], cell[1]]}
    
    all_apples = apples | bonus
    
    # Prefer apples that aren't actively being stood on by another agent
    free_apples = all_apples - blocked
    if not free_apples and all_apples:
        free_apples = all_apples
        
    step = get_step(free_apples)
    if step is not None and step != (0, 0):
        return mv(*step)
        
    # If no apples exist on the map, vacate the Plaza to maximize bonus spawn rates
    # and hover in the Orchards waiting for the next spawn.
    if pos not in orchards:
        step = get_step(orchards)
        if step is not None and step != (0, 0):
            return mv(*step)
            
    return 0