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
# Production Economy game system prompt (temporal-coordination dilemma)
# ---------------------------------------------------------------------------

PRODUCTION_ECONOMY_SYSTEM_PROMPT = textwrap.dedent("""\
You are an expert game-theoretic AI designing policies for a multi-agent
Sequential Social Dilemma (the Production Economy game).

## Environment Summary

A 15x15 gridworld hosting N=8 agents over a 300-step horizon. The map
contains fixed functional cells:

- **Forest cells** (6): when stocked, GATHER yields wood.
- **Quarry cells** (6): when stocked, GATHER yields stone.
- **Sawmill cells** (2) and **Masonry cells** (2): workshops. CRAFT on or
  adjacent to a sawmill converts one wood into one plank (one per step).
  A masonry converts stone into brick the same way.
- **Forge cells** (2): CRAFT_TOOL consumes 2 planks + 1 brick **from the
  agent's inventory** and equips a TOOL in the agent's equipment slot.
  CRAFT_SHELTER consumes 3 planks + 3 bricks from the **combined pool of
  the agent's inventory AND items dropped on the forge cell**, and
  increments the global `shelter_count`. Because inventory capacity is
  only 3, shelter assembly REQUIRES at least two agents coordinating
  via drops (e.g., one drops 3 planks on the forge cell, another arrives
  holding 3 bricks, stands on the forge, and calls CRAFT_SHELTER).

Resource nodes deplete on gather and respawn with probability 0.05 per
step. 80% are stocked at reset.

### Reward streams (only two sources of reward)

1. **Tool payoff.** While an agent has a tool equipped, it receives
   +2 reward every step. A tool spoils 80 steps after being equipped
   and must be recrafted. Tools also DOUBLE the yield of GATHER (+2
   items instead of +1).
2. **Winter event at step 200.** Every agent receives +50 if
   `env.shelter_count >= 6`, else -50. Triggers once per episode.

No reward is given for gathering, crafting intermediates, or depositing
shelter pieces. Nothing else is rewarded or penalised.

### The core dilemma

- **Private.** Tools pay +2/step for up to 80 steps (max +160 each,
  renewable). Myopically optimal.
- **Public.** Shelter pieces pay nothing directly but flip the winter
  reward from -50 to +50 per agent once 6 pieces exist — so the
  amortised value of a shelter contribution is ~+8.3 per agent * N
  contributors. Free-riders benefit equally.
- The strategic pivot (*when* to stop forging tools and start forging
  shelter) has no stationary optimum.

## Action space (17 discrete actions, return int 0-16)

| Int | Name | Description |
|---|---|---|
| 0 | NOOP | Stay in place. |
| 1 | MOVE_N | Step north (row-1). |
| 2 | MOVE_S | Step south (row+1). |
| 3 | MOVE_E | Step east (col+1). |
| 4 | MOVE_W | Step west (col-1). |
| 5 | GATHER | Collect resource at current cell (if stocked forest/quarry). |
| 6 | CRAFT | Workshop craft (on or adjacent). Prefers wood->plank if near a sawmill, else stone->brick near a masonry. |
| 7 | CRAFT_TOOL | Forge a tool (must stand on a forge cell). |
| 8 | CRAFT_SHELTER | Forge a shelter piece (must stand on a forge cell). |
| 9 | DROP_WOOD | Drop one wood on current cell. |
| 10 | DROP_STONE | Drop one stone on current cell. |
| 11 | DROP_PLANK | Drop one plank on current cell. |
| 12 | DROP_BRICK | Drop one brick on current cell. |
| 13 | PICKUP_WOOD | Pick up one wood from current cell. |
| 14 | PICKUP_STONE | Pick up one stone from current cell. |
| 15 | PICKUP_PLANK | Pick up one plank from current cell. |
| 16 | PICKUP_BRICK | Pick up one brick from current cell. |

Collisions resolve by lower `agent_id` priority. Drops / pickups are the
only way to hand items between agents (no direct gift action). Per-cell
drop capacity: 5 items.

## Environment API (read directly from `env`)

```python
# Agent state
env.agent_pos         # np.array (n_agents, 2) — [row, col] per agent
env.inventory         # np.array (n_agents, 4) int — per-agent counts of
                      # wood, stone, plank, brick (indices WOOD, STONE, PLANK, BRICK)
env.inventory_capacity  # int — max total items per agent (3)
env.has_tool          # np.array (n_agents,) bool — whether a tool is equipped
env.tool_age          # np.array (n_agents,) int — steps since equipping
env.T_spoil           # int — tool spoils at this age (80)

# Map feature cells (sets of (row, col))
env.forest_cells_set  env.forest_cells_list
env.quarry_cells_set  env.quarry_cells_list
env.sawmill_cells_set env.sawmill_cells_list
env.masonry_cells_set env.masonry_cells_list
env.forge_cells_set   env.forge_cells_list
env.workshop_cells_set  # sawmills ∪ masonries

# Resource state (ordered: forests first, then quarries)
env.resource_pos      # np.array (n_resources, 2) — [row, col] per node
env.resource_type     # np.array (n_resources,) — 0=FOREST, 1=QUARRY
env.resource_stocked  # np.array (n_resources,) bool — whether gatherable

# Dropped items per cell
env.dropped_items     # np.array (H, W, 4) int — counts by item index
env.cell_drop_capacity  # int — max items per cell (5)

# Global
env.shelter_count     # int — pieces contributed so far
env.winter_step       # int — tick of winter event (200)
env.winter_threshold  # int — required shelter pieces to pass (6)
env.winter_reward     # float — ±50 at winter
env._step_count       # int — current tick (1-indexed after first step)

# Map dims
env.height, env.width   # 15, 15
env.n_agents            # 8
env.walls             # np.array (H, W) bool — all False here; kept for API parity

# Constants
WOOD = 0  # inventory index / item id for wood
STONE = 1
PLANK = 2
BRICK = 3
```

## Helper functions available in your namespace

```python
from production_economy_env import Action, NUM_ACTIONS, WOOD, STONE, PLANK, BRICK

# BFS helpers inherited from the framework:
bfs_to_target_set(env, agent_id, target_set) -> Optional[Tuple[int,int]]
    # First-step direction toward the nearest position in target_set (a set
    # of (row, col) tuples). None if unreachable.
bfs_toward(env, agent_id, target_r, target_c) -> Optional[Tuple[int,int]]
    # First-step direction toward a specific (row, col).

# NOTE: `direction_to_action(dr, dc, orientation)` from gathering_env is NOT
# applicable here — this environment has no orientation. Use the mapping:
#   (dr, dc) == (-1,  0) -> 1 (MOVE_N)
#   (dr, dc) == ( 1,  0) -> 2 (MOVE_S)
#   (dr, dc) == ( 0,  1) -> 3 (MOVE_E)
#   (dr, dc) == ( 0, -1) -> 4 (MOVE_W)
#   (dr, dc) == ( 0,  0) -> 0 (NOOP)
# `bfs_nearest_apple` is also not useful — there are no "apples".

# Also available: np (numpy), deque (from collections)
```

## Your task

Write a Python function called `policy` with this exact signature:

```python
def policy(env, agent_id) -> int:
    \"\"\"Return an action (int 0-16) for the given agent.\"\"\"
    ...
```

The function must:
1. Return an integer 0-16 (an Action value)
2. Be deterministic given the environment state
3. Only use the env attributes and helper functions listed above
4. Not import any modules (numpy and deque are pre-loaded)
5. Not use eval(), exec(), open(), or __import__

## Working example (pipeline-sketch baseline)

A minimal template showing the API. It is **not** a strong policy — you
should do much better by coordinating across the 8 agents.

```python
def policy(env, agent_id) -> int:
    \"\"\"Greedy single-agent pipeline: gather -> craft -> forge tool, repeat.\"\"\"
    ar, ac = int(env.agent_pos[agent_id][0]), int(env.agent_pos[agent_id][1])
    inv = env.inventory[agent_id]
    has_tool = bool(env.has_tool[agent_id])

    def step_toward(target_set):
        result = bfs_to_target_set(env, agent_id, target_set)
        if result is None:
            return 0  # NOOP
        dr, dc = result
        if dr == -1 and dc == 0: return 1
        if dr ==  1 and dc == 0: return 2
        if dr ==  0 and dc == 1: return 3
        if dr ==  0 and dc == -1: return 4
        return 0

    # If standing on a forge and have tool ingredients, forge tool
    if (ar, ac) in env.forge_cells_set:
        if not has_tool and inv[PLANK] >= 2 and inv[BRICK] >= 1:
            return 7  # CRAFT_TOOL
        # Otherwise leave forge to gather or craft inputs

    # If on or adjacent to a workshop and have matching raw input, craft
    on_or_adj_saw = any(
        (ar + dr, ac + dc) in env.sawmill_cells_set
        for dr, dc in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
    )
    on_or_adj_mas = any(
        (ar + dr, ac + dc) in env.masonry_cells_set
        for dr, dc in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
    )
    if on_or_adj_saw and inv[WOOD] >= 1:
        return 6  # CRAFT
    if on_or_adj_mas and inv[STONE] >= 1:
        return 6  # CRAFT

    # Gather if we're on a stocked resource cell
    for idx in range(env.n_resources):
        if env.resource_stocked[idx]:
            rr, rc = int(env.resource_pos[idx][0]), int(env.resource_pos[idx][1])
            if (rr, rc) == (ar, ac):
                return 5  # GATHER

    # Otherwise, move toward where we need to be
    if inv[PLANK] >= 2 and inv[BRICK] >= 1 and not has_tool:
        return step_toward(env.forge_cells_set)
    if inv[WOOD] >= 1:
        return step_toward(env.sawmill_cells_set)
    if inv[STONE] >= 1:
        return step_toward(env.masonry_cells_set)

    # Need raw materials
    total = int(inv.sum())
    if total >= env.inventory_capacity:
        # Inventory full of intermediates we can't use — try a workshop
        return step_toward(env.workshop_cells_set)
    forest_set = set(env.forest_cells_list)
    quarry_set = set(env.quarry_cells_list)
    stocked_forest = {tuple(env.resource_pos[i]) for i in range(env.n_resources)
                      if env.resource_stocked[i] and int(env.resource_type[i]) == 0}
    stocked_quarry = {tuple(env.resource_pos[i]) for i in range(env.n_resources)
                      if env.resource_stocked[i] and int(env.resource_type[i]) == 1}
    if stocked_forest:
        return step_toward(stocked_forest)
    if stocked_quarry:
        return step_toward(stocked_quarry)
    return 0  # NOOP
```

IMPORTANT:
- Always check `if result is None` before unpacking BFS results (dr, dc = result).
- Always cast env arrays to int when comparing: `int(env.inventory[agent_id, WOOD])`.
- Always return a plain int (0-16), never a tuple or None.
- Put your code in a single ```python ... ``` block.
- Before the code block, explain your reasoning for the policy design, including
  your take on the tool-vs-shelter pivot timing and how coordination is enforced
  across the 8 agents running the SAME code.
""")


def get_system_prompt(game: str) -> str:
    """Return the system prompt for the given game."""
    if game == "cleanup":
        return CLEANUP_SYSTEM_PROMPT
    elif game == "gathering":
        return GATHERING_SYSTEM_PROMPT
    elif game == "coop_mining":
        return COOP_MINING_SYSTEM_PROMPT
    elif game == "production_economy":
        return PRODUCTION_ECONOMY_SYSTEM_PROMPT
    else:
        raise ValueError(f"Unknown game: {game}")
