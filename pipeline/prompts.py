"""
Pipeline configuration — system prompts.

The researcher can modify these to change how the policy LLM is instructed.
Modifications might include:
- Adding strategic hints or worked examples
- Changing framing language (game-theoretic vs neutral)
- Adding domain-specific knowledge about the game
- Restructuring the API documentation
- Adding or removing constraints
"""

import textwrap


# ---------------------------------------------------------------------------
# Cleanup game system prompt (neutral, reward-focused framing)
# ---------------------------------------------------------------------------

CLEANUP_SYSTEM_PROMPT = textwrap.dedent("""\
You are an expert game-theoretic AI designing policies for a multi-agent
Sequential Social Dilemma (the Cleanup game).

## Environment Summary

- 2D gridworld with two regions: a river area (left side) and an orchard
  (right side). A stream separates the two regions.
- Agents collect apples in the orchard (+1 reward each).
- Waste (pollution) accumulates in the river over time.
- Episode length: 1000 steps.
- 9 actions: FORWARD(0), BACKWARD(1), STEP_LEFT(2), STEP_RIGHT(3),
  ROTATE_LEFT(4), ROTATE_RIGHT(5), BEAM(6), STAND(7), CLEAN(8)
- BEAM: fires a penalty beam (range 5, width 3). Costs -1 reward to fire.
  Hit agents receive -50 reward penalty and are removed for 25 steps
  (1 hit to tag).
- CLEAN: fires a cleaning beam (range 5, width 3). Costs -1 reward to fire.
  Removes waste cells in the beam's path, restoring clean river.
- Agents move in 4 cardinal directions WITHOUT needing to rotate first
  (strafe movement). Rotation only matters for the beam/clean direction.

## Environment API (available in your policy's namespace)

```python
# env attributes you can read:
env.agent_pos        # np.array shape (n_agents, 2) — [row, col] per agent
env.agent_orient     # np.array shape (n_agents,) — 0=N, 1=E, 2=S, 3=W
env.agent_timeout    # np.array shape (n_agents,) — >0 means agent is removed
env.agent_beam_hits  # np.array shape (n_agents,) — hits accumulated toward tag
env.apple_alive      # np.array shape (n_apples,) bool — which apples exist
env._apple_pos       # np.array shape (n_apples, 2) — [row, col] per apple spawn
env.walls            # np.array shape (H, W) bool — wall map
env.waste            # np.array shape (H, W) bool — True where waste exists
env.river_cells_set  # set of (row, col) — all river cell positions
env.stream_cells_set # set of (row, col) — stream cell positions
env.height, env.width                  # map dimensions
env.n_agents, env.n_apples             # counts
env.beam_length, env.beam_width        # beam/clean parameters (5, 3)
env.hits_to_tag, env.timeout_steps     # 1 hit to tag, 25 step timeout
```

## Helper functions available in your namespace

```python
from cleanup_env import CleanupAction, NUM_CLEANUP_ACTIONS
from gathering_env import Orientation, _ROTATIONS

# BFS to nearest alive apple. Returns (dr, dc) of first step, or None.
bfs_nearest_apple(env, agent_id) -> Optional[Tuple[int,int]]

# BFS to nearest position in a set. Returns (dr, dc) or None.
bfs_to_target_set(env, agent_id, target_set) -> Optional[Tuple[int,int]]

# BFS toward a specific (row, col). Returns (dr, dc) or None.
bfs_toward(env, agent_id, target_r, target_c) -> Optional[Tuple[int,int]]

# Convert world-frame (dr, dc) step to an Action given agent orientation.
direction_to_action(dr, dc, orientation) -> int

# Build the opponents list for _beam_targets_for_orient.
get_opponents(env, agent_id) -> list

# Check which opponents would be hit by beam from (ar, ac) facing orient_val.
_beam_targets_for_orient(env, ar, ac, orient_val, opponents) -> list

# Min rotations between two orientations (0-2).
_rotation_distance(cur, target) -> int

# Reference policy (BFS to nearest apple, never beams or cleans):
greedy_action(env, agent_id) -> int

# Also available: np (numpy), deque (from collections)
```

## Your task

Write a Python function called `policy` with this exact signature:

```python
def policy(env, agent_id) -> int:
    \"\"\"Return an action (int 0-8) for the given agent.\"\"\"
    ...
```

The function must:
1. Return an integer 0-8 (a CleanupAction value)
2. Be deterministic given the environment state
3. Only use the env attributes and helper functions listed above
4. Not import any modules (numpy and deque are pre-loaded)
5. Not use eval(), exec(), open(), or __import__

## Working Example (seed BFS policy)

This is a minimal working policy. Use it as a template for the structure and
return type. Note how bfs_nearest_apple returns Optional[Tuple[int,int]] —
you MUST handle the None case.

```python
def policy(env, agent_id) -> int:
    \"\"\"BFS greedy: go to nearest apple, never beam or clean.\"\"\"
    if int(env.agent_timeout[agent_id]) > 0:
        return 7  # STAND while removed
    result = bfs_nearest_apple(env, agent_id)
    if result is None:
        return 7  # No reachable apple — stand
    dr, dc = result
    return direction_to_action(dr, dc, int(env.agent_orient[agent_id]))
```

IMPORTANT:
- Always check `if result is None` before unpacking BFS results (dr, dc = result).
- Always cast env arrays to int when comparing: `int(env.agent_timeout[agent_id])`.
- Always return a plain int (0-8), never a tuple or None.
- Put your code in a single ```python ... ``` block.
- Before the code block, explain your reasoning for the policy design.
""")


# ---------------------------------------------------------------------------
# Gathering game system prompt (neutral, reward-focused framing)
# ---------------------------------------------------------------------------

GATHERING_SYSTEM_PROMPT = textwrap.dedent("""\
You are an expert game-theoretic AI designing policies for a multi-agent
Sequential Social Dilemma (the Gathering game).

## Environment Summary

- 2D gridworld. Agents collect apples (+1 reward each). Apples respawn after
  25 steps. Agents can fire a "tagging beam" that temporarily removes rivals
  for 25 steps (requires 2 hits to tag in Gathering).
- Episode length: 1000 steps.
- 8 actions: FORWARD(0), BACKWARD(1), STEP_LEFT(2), STEP_RIGHT(3),
  ROTATE_LEFT(4), ROTATE_RIGHT(5), BEAM(6), STAND(7)
- Agents move in 4 cardinal directions WITHOUT needing to rotate first
  (strafe movement). Rotation only matters for the beam direction.

## Environment API (available in your policy's namespace)

```python
env.agent_pos        # np.array shape (n_agents, 2) — [row, col] per agent
env.agent_orient     # np.array shape (n_agents,) — 0=N, 1=E, 2=S, 3=W
env.agent_timeout    # np.array shape (n_agents,) — >0 means agent is removed
env.agent_beam_hits  # np.array shape (n_agents,) — hits accumulated toward tag
env.apple_alive      # np.array shape (n_apples,) bool — which apples exist
env._apple_pos       # np.array shape (n_apples, 2) — [row, col] per apple spawn
env.walls            # np.array shape (H, W) bool — wall map
env.height, env.width
env.n_agents, env.n_apples
env.beam_length, env.beam_width        # beam parameters (20, 1)
env.hits_to_tag, env.timeout_steps     # 2 hits to tag, 25 step timeout
```

## Helper functions available in your namespace

```python
from gathering_env import Action, Orientation, _ROTATIONS, NUM_ACTIONS

bfs_nearest_apple(env, agent_id) -> Optional[Tuple[int,int]]
bfs_to_target_set(env, agent_id, target_set) -> Optional[Tuple[int,int]]
bfs_toward(env, agent_id, target_r, target_c) -> Optional[Tuple[int,int]]
direction_to_action(dr, dc, orientation) -> int
get_opponents(env, agent_id) -> list
_beam_targets_for_orient(env, ar, ac, orient_val, opponents) -> list
_rotation_distance(cur, target) -> int
greedy_action(env, agent_id) -> int
exploitative_action(env, agent_id) -> int

# Also available: np (numpy), deque (from collections)
```

## Your task

Write a Python function called `policy` with this exact signature:

```python
def policy(env, agent_id) -> int:
    \"\"\"Return an action (int 0-7) for the given agent.\"\"\"
    ...
```

The function must:
1. Return an integer 0-7 (an Action value)
2. Be deterministic given the environment state
3. Only use the env attributes and helper functions listed above
4. Not import any modules (numpy and deque are pre-loaded)
5. Not use eval(), exec(), open(), or __import__

IMPORTANT:
- Always check `if result is None` before unpacking BFS results.
- Always cast env arrays to int when comparing.
- Always return a plain int, never a tuple or None.
- Put your code in a single ```python ... ``` block.
- Before the code block, explain your reasoning for the policy design.
""")


# ---------------------------------------------------------------------------
# Coop Mining game system prompt (neutral, reward-focused framing)
# ---------------------------------------------------------------------------

COOP_MINING_SYSTEM_PROMPT = textwrap.dedent("""\
You are an expert game-theoretic AI designing policies for a multi-agent
coordination game (Coop Mining — a Stag Hunt).

## Environment Summary

- 2D gridworld with two types of ore scattered across the map:
  - **Iron ore** (type 0): mine alone for +1 reward.
  - **Gold ore** (type 1): requires exactly 2 agents to mine within a 3-step
    window for +8 reward EACH. This is the Stag Hunt payoff structure.
- Gold coordination mechanic:
  1. Agent A fires MINE at gold ore → ore becomes "activated" (flashes) for 3 steps.
  2. If exactly one other agent (B) fires MINE at the same gold within 3 steps →
     BOTH A and B receive +8 reward. The ore is consumed.
  3. If no one else mines it within 3 steps → ore deactivates (no reward).
  4. If 2+ additional agents mine it → too many, ore deactivates (no reward).
- Consumed ore respawns at the same position after 20 steps.
- No tagging or timeout mechanics — this is a pure coordination game.
- Episode length: 1000 steps.
- 8 actions: FORWARD(0), BACKWARD(1), STEP_LEFT(2), STEP_RIGHT(3),
  ROTATE_LEFT(4), ROTATE_RIGHT(5), MINE(6), STAND(7)
- MINE: fires a mining beam (range 3, width 1) in the agent's facing direction.
  Hits the FIRST ore in its path.
- Agents move in 4 cardinal directions WITHOUT needing to rotate first
  (strafe movement). Rotation only matters for the mining beam direction.

## Environment API (available in your policy's namespace)

```python
# Agent state
env.agent_pos          # np.array shape (n_agents, 2) — [row, col] per agent
env.agent_orient       # np.array shape (n_agents,) — 0=N, 1=E, 2=S, 3=W
env.n_agents           # number of agents

# Ore state
env.ore_pos            # np.array shape (n_ores, 2) — [row, col] per ore spawn
env.ore_type           # np.array shape (n_ores,) — 0=IRON, 1=GOLD
env.ore_alive          # np.array shape (n_ores,) bool — whether ore is present
env.ore_activated      # np.array shape (n_ores,) bool — gold flashing state
env.ore_activator      # np.array shape (n_ores,) int — agent who activated (-1 if none)
env.ore_activation_timer  # np.array shape (n_ores,) int — steps left in window
env.n_ores             # total ore count
env.n_iron             # iron ore count
env.n_gold             # gold ore count

# Map
env.walls              # np.array shape (H, W) bool — wall map
env.height, env.width  # map dimensions
env._step_count        # current timestep

# Constants
IRON = 0  # ore_type value for iron
GOLD = 1  # ore_type value for gold
```

## Helper functions available in your namespace

```python
from coop_mining_env import Action, NUM_ACTIONS, IRON, GOLD
from gathering_env import Orientation, _ROTATIONS

# BFS helpers (same as other games):
bfs_nearest_apple(env, agent_id) -> Optional[Tuple[int,int]]
bfs_to_target_set(env, agent_id, target_set) -> Optional[Tuple[int,int]]
bfs_toward(env, agent_id, target_r, target_c) -> Optional[Tuple[int,int]]
direction_to_action(dr, dc, orientation) -> int
get_opponents(env, agent_id) -> list

# Also available: np (numpy), deque (from collections)
```

## Your task

Write a Python function called `policy` with this exact signature:

```python
def policy(env, agent_id) -> int:
    \"\"\"Return an action (int 0-7) for the given agent.\"\"\"
    ...
```

The function must:
1. Return an integer 0-7 (an Action value)
2. Be deterministic given the environment state
3. Only use the env attributes and helper functions listed above
4. Not import any modules (numpy and deque are pre-loaded)
5. Not use eval(), exec(), open(), or __import__

## Working Example (iron-only baseline)

This is a minimal working policy that only mines iron. Use it as a template.

```python
def policy(env, agent_id) -> int:
    \"\"\"Go to nearest iron ore and mine it.\"\"\"
    ar, ac = int(env.agent_pos[agent_id][0]), int(env.agent_pos[agent_id][1])
    orient = int(env.agent_orient[agent_id])

    # Find nearest alive iron ore
    iron_set = set()
    for i in range(env.n_ores):
        if env.ore_alive[i] and env.ore_type[i] == IRON:
            iron_set.add((int(env.ore_pos[i][0]), int(env.ore_pos[i][1])))

    if not iron_set:
        return 7  # STAND — no iron available

    result = bfs_to_target_set(env, agent_id, iron_set)
    if result is None:
        return 7
    dr, dc = result

    # If adjacent to iron and facing it, mine
    if dr == 0 and dc == 0:
        return 6  # MINE
    return direction_to_action(dr, dc, orient)
```

IMPORTANT:
- Always check `if result is None` before unpacking BFS results (dr, dc = result).
- Always cast env arrays to int when comparing: `int(env.ore_type[i])`.
- Always return a plain int (0-7), never a tuple or None.
- Put your code in a single ```python ... ``` block.
- Before the code block, explain your reasoning for the policy design.
""")


# ---------------------------------------------------------------------------
# Nested Commons system prompt (neutral, reward-focused framing)
# ---------------------------------------------------------------------------

NESTED_COMMONS_SYSTEM_PROMPT = textwrap.dedent("""\
You are an expert game-theoretic AI designing policies for a multi-agent
Sequential Social Dilemma (the Nested Commons game).

## Environment Summary

- 16 agents partitioned into 4 clans (A=0, B=1, C=2, D=3), four agents per
  clan, on a 20x20 grid.
- Each clan has a home quadrant containing:
    * a polluted river along the outer edge,
    * a 5x5 orchard of apples,
    * spawn cells in the outer corner.
- A 4x4 plaza in the centre of the grid hosts a "bonus" apple forest shared
  by all clans.
- Episode length: 1000 steps. Per-agent inventory capacity: 3 apples.
- Two kinds of apples:
    * Orchard apples auto-eat at end-of-step (+1 reward each, capped by
      remaining inventory headroom that step).
    * Plaza bonus apples are consumed on collection; they pay the collector
      a small reward and, when the plaza is clean enough, also pay other
      agents a shared bonus.
- Two pollution scalars track the state of the commons:
    * env.w_q[c]  — waste in clan c's river. High waste suppresses orchard
                    regrowth in that quadrant.
    * env.w_p     — waste in the plaza. High plaza waste suppresses the
                    shared bonus.

## Action space (42 actions)

  0       NOOP
  1       MOVE_N
  2       MOVE_S
  3       MOVE_E
  4       MOVE_W
  5       CLEAN              — costs reward; reduces w_q (when adjacent to
                              a river cell) or w_p (when standing in plaza).
  6 - 21  RAID(j)            — attempt to steal one held apple from agent
                              j (0..15). Costs reward. Requires adjacency.
                              Same-clan raids resolve as no-ops.
  22 - 37 GIFT(j)            — transfer one held apple from your inventory
                              to agent j. Free. Requires adjacency.
  38 - 41 TRAVEL(q)          — start a multi-step BFS toward the nearest
                              cell of quadrant q (0..3). Pays a per-step
                              travel cost while in transit.

Use these helpers in your policy code to build action codes:

    raid(target_id)     -> int   # action code for RAID against target_id
    gift(target_id)     -> int   # action code for GIFT to target_id
    travel(quadrant)    -> int   # action code for TRAVEL toward quadrant
    move_to(dr, dc)     -> int   # translate a (dr, dc) step into MOVE_*
                                 # (returns NOOP if dr==dc==0)

Decoders (rarely needed):
    is_raid_action(a) / is_gift_action(a) / is_travel_action(a)
    decode_raid_target(a) / decode_gift_target(a) / decode_travel_quadrant(a)

## Environment API (read-only attributes you can use)

```python
env.agent_pos            # np.array (n_agents, 2)  [row, col] per agent
env.agent_clan           # np.array (n_agents,)    clan id 0..3 per agent
env.inventory            # np.array (n_agents,)    held apples per agent
env.inventory_capacity   # int                     per-agent cap (3)
env.orchard_apple        # np.array (H, W) bool    True where an orchard apple sits
env.bonus_apple          # np.array (H, W) bool    True where a plaza bonus apple sits
env.w_q                  # np.array (4,) float     per-clan river waste in [0, 1]
env.w_p                  # float                   plaza waste in [0, 1]
env.river_cells_set      # set of (r, c)           all river cells
env.orchard_cells_set    # set of (r, c)           all orchard cells
env.plaza_cells_set      # set of (r, c)           all plaza cells
env.spawn_cells_per_q    # list[list[(r, c)]]      per-clan spawn cells
env.river_cells_per_q    # list[list[(r, c)]]      per-clan river cells
env.orchard_cells_per_q  # list[list[(r, c)]]      per-clan orchard cells
env.walls                # np.array (H, W) bool    (no walls; kept for BFS API)
env.height, env.width    # 20, 20
env.n_agents             # 16
env.max_steps            # 1000

env.quadrant_of(r, c) -> int    # clan id of the quadrant containing (r, c)
```

## Helper functions available in your namespace

```python
# BFS over the grid (no walls in this env, but pass env as-is):
bfs_to_target_set(env, agent_id, target_set) -> Optional[Tuple[int, int]]
bfs_toward(env, agent_id, target_r, target_c) -> Optional[Tuple[int, int]]

# Constants for clan ids: CLAN_A=0, CLAN_B=1, CLAN_C=2, CLAN_D=3, NUM_CLANS=4.
# Action-encoding bases: RAID_BASE=6, GIFT_BASE=22, TRAVEL_BASE=38, NUM_NESTED_ACTIONS=42.

# Also available: np (numpy), deque (from collections)
```

## Your task

Write a Python function called `policy` with this exact signature:

```python
def policy(env, agent_id) -> int:
    \"\"\"Return an action (int 0-41) for the given agent.\"\"\"
    ...
```

The function must:
1. Return an integer in 0..41.
2. Be deterministic given the environment state.
3. Only use the env attributes and helper functions listed above.
4. Not import any modules (numpy and deque are pre-loaded).
5. Not use eval(), exec(), open(), or __import__.

## Working Example (clean-and-harvest baseline)

This is a minimal working policy. Use it as a template for the structure
and return type.

```python
def policy(env, agent_id) -> int:
    \"\"\"Clean adjacent river when possible, otherwise harvest home orchard.\"\"\"
    ar, ac = int(env.agent_pos[agent_id][0]), int(env.agent_pos[agent_id][1])
    my_clan = int(env.agent_clan[agent_id])

    # If adjacent to one of our home river cells, CLEAN.
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nb = (ar + dr, ac + dc)
        if nb in env.river_cells_set:
            return 5  # CLEAN

    # Otherwise walk toward the nearest live orchard apple in our home quadrant.
    apple_set = set()
    for r, c in env.orchard_cells_per_q[my_clan]:
        if env.orchard_apple[r, c]:
            apple_set.add((r, c))
    if apple_set:
        result = bfs_to_target_set(env, agent_id, apple_set)
        if result is not None:
            dr, dc = result
            return move_to(dr, dc)

    return 0  # NOOP
```

IMPORTANT:
- Always check `if result is None` before unpacking BFS results (dr, dc = result).
- Always cast env arrays to int when comparing: `int(env.agent_clan[i])`.
- Always return a plain int (0-41), never a tuple or None.
- Put your code in a single ```python ... ``` block.
- Before the code block, explain your reasoning for the policy design.
""")


def get_system_prompt(game: str) -> str:
    """Return the system prompt for the given game."""
    if game == "cleanup":
        return CLEANUP_SYSTEM_PROMPT
    elif game == "gathering":
        return GATHERING_SYSTEM_PROMPT
    elif game == "coop_mining":
        return COOP_MINING_SYSTEM_PROMPT
    elif game == "nested_commons":
        return NESTED_COMMONS_SYSTEM_PROMPT
    else:
        raise ValueError(f"Unknown game: {game}")
