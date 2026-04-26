"""System prompt for the M1 synthesizer LLM.

Proposer-modifiable. The seed describes env mechanics + the policy API
contract. It does NOT include strategic hints about what to do — that
is what the synthesizer is asked to discover.
"""

from __future__ import annotations


PRODUCTION_ECONOMY_SYSTEM = """\
You are writing a Python policy for a multi-agent gridworld game called
'Production Economy'. All agents in self-play execute the same code; your
function is called once per agent per step.

## Env mechanics (public spec)

* 15x15 grid, 8 agents, 300-step horizon.
* Cell types: forest (yields WOOD), quarry (yields STONE), sawmill
  (converts WOOD->PLANK), masonry (converts STONE->BRICK), forge
  (converts intermediates -> TOOL or SHELTER_PIECE), open space.
* Resources stochastically respawn (per-cell prob 0.05 per step).
* Inventory cap: 3 items per agent.
* Cell drop cap: 5 items per cell. Items dropped by one agent can be
  picked up by another.
* Tools: equipped (don't occupy inventory), give +2 reward per step
  while equipped, double GATHER yield (+1 bonus). Spoil after 80
  equipped steps.
* Shelter pieces: contribute to a *global* shelter_count. At step 200
  the 'winter' event fires: every agent gets +50 if shelter_count >= 6,
  else -50.

## Actions (return an int 0..16)

  0  NOOP
  1  MOVE_N         5  GATHER          9  DROP_WOOD       13 PICKUP_WOOD
  2  MOVE_S         6  CRAFT           10 DROP_STONE      14 PICKUP_STONE
  3  MOVE_E         7  CRAFT_TOOL      11 DROP_PLANK      15 PICKUP_PLANK
  4  MOVE_W         8  CRAFT_SHELTER   12 DROP_BRICK      16 PICKUP_BRICK

* GATHER works on stocked forest/quarry cells.
* CRAFT works on workshop cells (sawmill/masonry; on or adjacent).
* CRAFT_TOOL needs 2 PLANK + 1 BRICK in inventory; agent must stand on
  a forge cell.
* CRAFT_SHELTER needs 3 PLANK + 3 BRICK from the combined pool of
  agent inventory + items dropped on the forge cell. Agent must stand
  on a forge cell.

## Env state your policy can read

  env.agent_pos[agent_id]              # [row, col]
  env.inventory[agent_id]              # [wood, stone, plank, brick] counts
  env.has_tool[agent_id]               # bool
  env.tool_age[agent_id]               # 0..80
  env.shelter_count                    # global int
  env.dropped_items[r, c, item_idx]    # cell-level drop counts
  env.resource_stocked[idx]            # bool, indexed by resource node
  env.resource_pos[idx]                # [row, col]; idx < env.n_resources
  env.resource_type[idx]               # 0=forest, 1=quarry
  env.forest_cells_set, env.quarry_cells_set
  env.sawmill_cells_set, env.masonry_cells_set, env.forge_cells_set
  env.forge_cells_list                 # list of (r, c)
  env.height, env.width, env.walls     # map dimensions; walls is bool array
  env.n_agents, env.inventory_capacity, env.cell_drop_capacity
  env.winter_step, env.winter_threshold
  env._step_count                      # current step (1-indexed)

## Output format

Output exactly one ```python``` code block containing:

```python
def policy(env, agent_id) -> int:
    \"\"\"Return one of the 17 actions for agent_id at the current step.\"\"\"
    ...
    return <int 0..16>
```

The `numpy as np` and `from collections import deque` names are
pre-bound. Do NOT define helper functions outside `policy` — keep
everything inside the function body.
"""


# Other games — minimal placeholders. Extend when running on those envs.
CLEANUP_SYSTEM = (
    "You are writing a Python policy for the Cleanup gridworld. "
    "All agents share the same code in self-play. Output one ```python``` "
    "block with `def policy(env, agent_id) -> int:` returning an action "
    "in 0..8. Read env attributes (env.agent_pos, env.apple_alive, ...) "
    "directly."
)
GATHERING_SYSTEM = (
    "You are writing a Python policy for the Gathering gridworld. "
    "All agents share the same code in self-play. Output one ```python``` "
    "block with `def policy(env, agent_id) -> int:` returning an action "
    "in 0..7. Read env attributes (env.agent_pos, env.apple_alive, ...) "
    "directly."
)
COOP_MINING_SYSTEM = (
    "You are writing a Python policy for the Coop Mining gridworld. "
    "All agents share the same code in self-play. Output one ```python``` "
    "block with `def policy(env, agent_id) -> int:` returning an action "
    "in 0..7. Read env attributes (env.agent_pos, env.gold_state, ...) "
    "directly."
)


def get_system_prompt(game: str) -> str:
    return {
        "production_economy": PRODUCTION_ECONOMY_SYSTEM,
        "cleanup": CLEANUP_SYSTEM,
        "gathering": GATHERING_SYSTEM,
        "coop_mining": COOP_MINING_SYSTEM,
    }.get(game, PRODUCTION_ECONOMY_SYSTEM)
