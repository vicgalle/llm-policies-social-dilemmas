"""
Production Economy — A Sequential Social Dilemma with Temporal Depth
=====================================================================

Implements the Production Economy game defined in production_economy.md.

Agents gather raw resources (wood from forests, stone from quarries),
craft intermediate goods at workshops (sawmill: wood→plank, masonry: stone→brick),
and produce final goods at forges:
  - TOOL (private, fast-payoff, renewable): +2 per step while equipped, spoils
    after 80 steps. Requires 2 planks + 1 brick.
  - SHELTER_PIECE (public, delayed, threshold-triggered): contributes to a
    global shelter_count. At step 200, all agents receive +50 if shelter_count
    >= 6, else -50. Requires 3 planks + 3 bricks.

Key features:
  - Temporal coordination: agents must pivot from private tool production to
    public shelter contribution before the winter deadline.
  - Pipeline balance: gathering / crafting / forging must remain in steady
    proportion; inventory cap (3 slots) and per-cell drop cap (5) stress
    handoff logistics.
  - Homogeneous self-play with 8 agents on a 15x15 grid.
"""

from __future__ import annotations

import numpy as np
from enum import IntEnum
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class Action(IntEnum):
    """Seventeen discrete actions for the Production Economy game."""
    NOOP = 0
    MOVE_N = 1
    MOVE_S = 2
    MOVE_E = 3
    MOVE_W = 4
    GATHER = 5
    CRAFT = 6           # workshop: wood->plank (sawmill) or stone->brick (masonry)
    CRAFT_TOOL = 7      # forge: 2 planks + 1 brick -> tool
    CRAFT_SHELTER = 8   # forge: 3 planks + 3 bricks -> shelter piece
    DROP_WOOD = 9
    DROP_STONE = 10
    DROP_PLANK = 11
    DROP_BRICK = 12
    PICKUP_WOOD = 13
    PICKUP_STONE = 14
    PICKUP_PLANK = 15
    PICKUP_BRICK = 16

NUM_ACTIONS = len(Action)
MAX_ACTION = NUM_ACTIONS - 1  # 16

# Item types (indices into inventory array)
WOOD = 0
STONE = 1
PLANK = 2
BRICK = 3
NUM_ITEM_TYPES = 4
ITEM_NAMES = ["wood", "stone", "plank", "brick"]

# Cell functional types (for internal bookkeeping)
FOREST = 0
QUARRY = 1
SAWMILL = 2
MASONRY = 3
FORGE = 4

# Movement offsets (row, col)
_MOVE_OFFSETS = {
    Action.MOVE_N: (-1, 0),
    Action.MOVE_S: (1, 0),
    Action.MOVE_E: (0, 1),
    Action.MOVE_W: (0, -1),
}

# Drop / pickup action → item index
_DROP_ITEM = {
    Action.DROP_WOOD: WOOD,
    Action.DROP_STONE: STONE,
    Action.DROP_PLANK: PLANK,
    Action.DROP_BRICK: BRICK,
}
_PICKUP_ITEM = {
    Action.PICKUP_WOOD: WOOD,
    Action.PICKUP_STONE: STONE,
    Action.PICKUP_PLANK: PLANK,
    Action.PICKUP_BRICK: BRICK,
}


# ---------------------------------------------------------------------------
# Fixed 15x15 layout
# ---------------------------------------------------------------------------
#
# Legend: . = open, F = forest, Q = quarry, S = sawmill, M = masonry,
#         G = forge, P = spawn point.
#
# Layout rationale:
#   - Forests scattered in rows 0-4 (upper half).
#   - Quarries scattered in rows 10-14 (lower half).
#   - Workshops in rows 6 and 8 (middle band), two of each type,
#     flanking the forges.
#   - Two forges centred on row 7.
#   - Eight spawn points at the corners + edge midpoints.

PRODUCTION_ECONOMY_MAP = """\
P.F....P....F.P
....F..........
....F..........
.........F.....
.F.............
...............
P..S...G...S..P
P......G......P
...M.......M...
...............
.....Q.........
....Q......Q...
...............
.Q......Q......
P......Q......P"""


def _parse_map(ascii_map: str):
    """Parse the fixed ASCII map into feature arrays.

    Returns
    -------
    height, width : int
    forest_cells, quarry_cells, sawmill_cells, masonry_cells, forge_cells,
    spawn_points : list of (row, col)
    """
    lines = ascii_map.strip().splitlines()
    height = len(lines)
    width = max(len(ln) for ln in lines)
    lines = [ln.ljust(width) for ln in lines]

    forest: List[Tuple[int, int]] = []
    quarry: List[Tuple[int, int]] = []
    sawmill: List[Tuple[int, int]] = []
    masonry: List[Tuple[int, int]] = []
    forge: List[Tuple[int, int]] = []
    spawn: List[Tuple[int, int]] = []

    for r, ln in enumerate(lines):
        for c, ch in enumerate(ln):
            if ch == "F":
                forest.append((r, c))
            elif ch == "Q":
                quarry.append((r, c))
            elif ch == "S":
                sawmill.append((r, c))
            elif ch == "M":
                masonry.append((r, c))
            elif ch == "G":
                forge.append((r, c))
            elif ch == "P":
                spawn.append((r, c))

    return height, width, forest, quarry, sawmill, masonry, forge, spawn


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class ProductionEconomyEnv:
    """Multi-agent Production Economy gridworld.

    Parameters
    ----------
    ascii_map : str
        Fixed 15x15 layout (see PRODUCTION_ECONOMY_MAP).
    n_agents : int
        Number of agents running the shared policy (default 8).
    max_steps : int
        Episode horizon (default 300).
    winter_step : int
        Tick at which the winter event fires (default 200).
    winter_threshold : int
        Minimum ``shelter_count`` at winter to pass (default 6).
    winter_reward : float
        Magnitude of winter reward (±, default 50.0).
    T_spoil : int
        Steps until an equipped tool spoils (default 80).
    tool_step_reward : float
        Reward per step per surviving equipped tool (default 2.0).
    tool_gather_bonus : int
        Extra units produced by GATHER when a tool is equipped (default 1,
        so tool-equipped GATHER yields 1 + 1 = 2 units).
    inventory_capacity : int
        Total inventory slots per agent (default 3).
    cell_drop_capacity : int
        Maximum items that may sit on a single cell (default 5).
    resource_respawn_prob : float
        Per-step probability of respawning a depleted resource cell (default 0.05).
    initial_stocked_fraction : float
        Fraction of resource nodes stocked at reset (default 0.8).
    seed : int or None
        Random seed.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        ascii_map: str = PRODUCTION_ECONOMY_MAP,
        n_agents: int = 8,
        max_steps: int = 300,
        # Winter event
        winter_step: int = 200,
        winter_threshold: int = 6,
        winter_reward: float = 50.0,
        # Tools
        T_spoil: int = 80,
        tool_step_reward: float = 2.0,
        tool_gather_bonus: int = 1,
        # Inventory / drops
        inventory_capacity: int = 3,
        cell_drop_capacity: int = 5,
        # Resources
        resource_respawn_prob: float = 0.05,
        initial_stocked_fraction: float = 0.8,
        # Misc
        seed: Optional[int] = None,
    ):
        (self.height, self.width,
         self.forest_cells_list, self.quarry_cells_list,
         self.sawmill_cells_list, self.masonry_cells_list,
         self.forge_cells_list, self.spawn_points) = _parse_map(ascii_map)

        self.n_agents = n_agents
        self.max_steps = max_steps

        self.winter_step = winter_step
        self.winter_threshold = winter_threshold
        self.winter_reward = winter_reward

        self.T_spoil = T_spoil
        self.tool_step_reward = tool_step_reward
        self.tool_gather_bonus = tool_gather_bonus

        self.inventory_capacity = inventory_capacity
        self.cell_drop_capacity = cell_drop_capacity

        self.resource_respawn_prob = resource_respawn_prob
        self.initial_stocked_fraction = initial_stocked_fraction

        # No walls in this layout; kept for API compatibility.
        self.walls = np.zeros((self.height, self.width), dtype=bool)

        # Set forms for quick membership tests
        self.forest_cells_set = set(self.forest_cells_list)
        self.quarry_cells_set = set(self.quarry_cells_list)
        self.sawmill_cells_set = set(self.sawmill_cells_list)
        self.masonry_cells_set = set(self.masonry_cells_list)
        self.forge_cells_set = set(self.forge_cells_list)
        self.workshop_cells_set = self.sawmill_cells_set | self.masonry_cells_set

        # Resource node arrays (forests first, then quarries)
        rpos = self.forest_cells_list + self.quarry_cells_list
        rtype = [FOREST] * len(self.forest_cells_list) + [QUARRY] * len(self.quarry_cells_list)
        self.n_resources = len(rpos)
        self.resource_pos = np.array(rpos, dtype=np.int32) if rpos else np.zeros((0, 2), dtype=np.int32)
        self.resource_type = np.array(rtype, dtype=np.int32)
        # Map (r,c) → index into resource arrays
        self._resource_idx = {tuple(p): i for i, p in enumerate(rpos)}

        # Random state
        self.rng = np.random.default_rng(seed)

        # --- Mutable state (initialised in reset) ---
        self._step_count: int = 0
        self.agent_pos = np.zeros((n_agents, 2), dtype=np.int32)
        # Inventory: one row per agent, columns are item counts (wood, stone, plank, brick)
        self.inventory = np.zeros((n_agents, NUM_ITEM_TYPES), dtype=np.int32)
        self.has_tool = np.zeros(n_agents, dtype=bool)
        self.tool_age = np.zeros(n_agents, dtype=np.int32)
        # Resource stock (True = stocked / gatherable)
        self.resource_stocked = np.zeros(self.n_resources, dtype=bool)
        # Dropped items: per-cell counts of each item type
        self.dropped_items = np.zeros((self.height, self.width, NUM_ITEM_TYPES), dtype=np.int32)
        # Global shelter counter
        self.shelter_count: int = 0
        # Winter trigger bookkeeping
        self._winter_triggered: bool = False

        # Aliases for framework compatibility (n_apples / apple_respawn_time are used
        # by feedback._env_description and gathering_policy.run_episode's verbose mode).
        self.n_apples = self.n_resources
        self.apple_respawn_time = int(round(1.0 / max(self.resource_respawn_prob, 1e-6)))

        self.action_space_n = NUM_ACTIONS

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> Dict[int, object]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._step_count = 0
        self._winter_triggered = False

        # Spawn agents: prefer marked spawn cells; if more agents than spawns,
        # fall back to open cells (not walls, not resource/workshop/forge).
        taken: set = set()
        feature_cells = (self.forest_cells_set | self.quarry_cells_set
                         | self.sawmill_cells_set | self.masonry_cells_set
                         | self.forge_cells_set)
        spawn_i = 0
        for i in range(self.n_agents):
            if spawn_i < len(self.spawn_points):
                p = self.spawn_points[spawn_i]
                spawn_i += 1
                if p not in taken:
                    self.agent_pos[i] = p
                    taken.add(p)
                    continue
            # Fallback: nearest unused open non-feature cell
            placed = False
            for r in range(self.height):
                for c in range(self.width):
                    p = (r, c)
                    if p in taken or p in feature_cells or self.walls[r, c]:
                        continue
                    self.agent_pos[i] = p
                    taken.add(p)
                    placed = True
                    break
                if placed:
                    break

        self.inventory[:] = 0
        self.has_tool[:] = False
        self.tool_age[:] = 0

        # Deterministically stock a random subset equal to round(fraction * N)
        # (keeps per-seed variance to the identity of which nodes are stocked,
        # not the total count).
        n_stocked = int(round(self.initial_stocked_fraction * self.n_resources))
        self.resource_stocked[:] = False
        if n_stocked > 0 and self.n_resources > 0:
            stocked_idx = self.rng.choice(self.n_resources, size=n_stocked, replace=False)
            self.resource_stocked[stocked_idx] = True

        self.dropped_items[:] = 0
        self.shelter_count = 0

        # Return empty per-agent obs dict (policies read env state directly).
        return {i: None for i in range(self.n_agents)}

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self, actions: Dict[int, int],
    ) -> Tuple[Dict[int, object], Dict[int, float], Dict[int, bool], Dict[int, bool], Dict[int, dict]]:
        self._step_count += 1
        rewards = {i: 0.0 for i in range(self.n_agents)}
        winter_triggered_this_step = False

        # --- 1. Apply movements with atomic collision resolution ---
        # Compute each agent's desired target. Resolve conflicts:
        #   (a) multiple agents wanting the same cell → lowest agent_id wins,
        #       others revert to current (priority rule from the spec).
        #   (b) agent wants a cell occupied by a stationary agent → blocked.
        #   (c) swaps and chains of moves into vacated cells → allowed
        #       (iterated to a stable set of moves).
        positions = [tuple(self.agent_pos[i]) for i in range(self.n_agents)]
        desired = list(positions)
        for i in range(self.n_agents):
            act = actions.get(i, Action.NOOP)
            if act not in _MOVE_OFFSETS:
                continue
            dr, dc = _MOVE_OFFSETS[act]
            cur = positions[i]
            tgt = (cur[0] + dr, cur[1] + dc)
            if not (0 <= tgt[0] < self.height and 0 <= tgt[1] < self.width):
                continue
            if self.walls[tgt[0], tgt[1]]:
                continue
            desired[i] = tgt

        # (a) Same-target conflicts: lowest agent_id wins, others revert.
        claims: Dict[Tuple[int, int], List[int]] = {}
        for i in range(self.n_agents):
            if desired[i] != positions[i]:
                claims.setdefault(desired[i], []).append(i)
        for tgt, ids in claims.items():
            if len(ids) > 1:
                ids.sort()
                for loser in ids[1:]:
                    desired[loser] = positions[loser]

        # (b,c) Iteratively block moves into cells held by stationary agents.
        # Converges in at most n_agents iterations.
        for _ in range(self.n_agents + 1):
            changed = False
            for i in range(self.n_agents):
                if desired[i] == positions[i]:
                    continue
                tgt = desired[i]
                for j in range(self.n_agents):
                    if i == j:
                        continue
                    if positions[j] == tgt and desired[j] == positions[j]:
                        desired[i] = positions[i]
                        changed = True
                        break
            if not changed:
                break

        for i in range(self.n_agents):
            self.agent_pos[i] = desired[i]

        # --- 2. Apply GATHER / CRAFT / DROP / PICKUP in agent_id order ---
        for i in range(self.n_agents):
            act = actions.get(i, Action.NOOP)
            if act == Action.GATHER:
                self._do_gather(i)
            elif act == Action.CRAFT:
                self._do_craft_workshop(i)
            elif act == Action.CRAFT_TOOL:
                self._do_craft_tool(i)
            elif act == Action.CRAFT_SHELTER:
                self._do_craft_shelter(i)
            elif act in _DROP_ITEM:
                self._do_drop(i, _DROP_ITEM[act])
            elif act in _PICKUP_ITEM:
                self._do_pickup(i, _PICKUP_ITEM[act])

        # --- 3. Age equipped tools; unequip spoiled ones ---
        for i in range(self.n_agents):
            if self.has_tool[i]:
                self.tool_age[i] += 1
                if self.tool_age[i] >= self.T_spoil:
                    self.has_tool[i] = False
                    self.tool_age[i] = 0

        # --- 4. Pay +tool_step_reward per surviving equipped tool ---
        for i in range(self.n_agents):
            if self.has_tool[i]:
                rewards[i] += self.tool_step_reward

        # --- 5. Winter event at step == winter_step ---
        if self._step_count == self.winter_step and not self._winter_triggered:
            self._winter_triggered = True
            winter_triggered_this_step = True
            if self.shelter_count >= self.winter_threshold:
                for i in range(self.n_agents):
                    rewards[i] += self.winter_reward
            else:
                for i in range(self.n_agents):
                    rewards[i] -= self.winter_reward

        # --- 6. Regrow depleted resource nodes ---
        depleted = ~self.resource_stocked
        n_depleted = int(depleted.sum())
        if n_depleted > 0:
            draws = self.rng.random(n_depleted)
            regrow_mask = draws < self.resource_respawn_prob
            dep_idx = np.where(depleted)[0]
            for k, idx in enumerate(dep_idx):
                if regrow_mask[k]:
                    self.resource_stocked[idx] = True

        # --- 7. Episode termination ---
        done = self._step_count >= self.max_steps
        terminated = {i: done for i in range(self.n_agents)}
        truncated = {i: False for i in range(self.n_agents)}

        # --- 8. Observations: policies read env state directly ---
        obs = {i: None for i in range(self.n_agents)}

        # --- 9. Info ---
        steps_until_winter = max(0, self.winter_step - self._step_count)
        info: Dict[int, dict] = {}
        for i in range(self.n_agents):
            info[i] = {
                "timeout": 0,  # no tagging/timeout mechanism (API compatibility)
                "inventory": self.inventory[i].copy(),
                "has_tool": bool(self.has_tool[i]),
                "tool_age": int(self.tool_age[i]),
                "shelter_count": int(self.shelter_count),
                "steps_until_winter": steps_until_winter,
                "winter_triggered": winter_triggered_this_step,
            }

        return obs, rewards, terminated, truncated, info

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _inventory_total(self, agent_id: int) -> int:
        return int(self.inventory[agent_id].sum())

    def _inventory_free(self, agent_id: int) -> int:
        return self.inventory_capacity - self._inventory_total(agent_id)

    def _do_gather(self, agent_id: int) -> None:
        pos = (int(self.agent_pos[agent_id, 0]), int(self.agent_pos[agent_id, 1]))
        idx = self._resource_idx.get(pos)
        if idx is None:
            return
        if not self.resource_stocked[idx]:
            return
        rtype = int(self.resource_type[idx])
        item = WOOD if rtype == FOREST else STONE
        yield_amount = 1 + (self.tool_gather_bonus if self.has_tool[agent_id] else 0)
        free = self._inventory_free(agent_id)
        take = min(free, yield_amount)
        if take <= 0:
            # Resource still depletes because the agent "used" it — but spec does not
            # mandate this. We'll NOT deplete if we can't store anything, to be
            # forgiving (no wasted action). Cost is an extra turn.
            return
        self.inventory[agent_id, item] += take
        self.resource_stocked[idx] = False

    def _do_craft_workshop(self, agent_id: int) -> None:
        """Sawmill: wood→plank. Masonry: stone→brick. Agent must be on or adjacent."""
        ar, ac = int(self.agent_pos[agent_id, 0]), int(self.agent_pos[agent_id, 1])

        # Check "on or adjacent" to each workshop type
        def _near(cellset):
            if (ar, ac) in cellset:
                return True
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                if (ar + dr, ac + dc) in cellset:
                    return True
            return False

        # Prefer the workshop whose input the agent has.
        if _near(self.sawmill_cells_set) and self.inventory[agent_id, WOOD] >= 1:
            self.inventory[agent_id, WOOD] -= 1
            self.inventory[agent_id, PLANK] += 1
            return
        if _near(self.masonry_cells_set) and self.inventory[agent_id, STONE] >= 1:
            self.inventory[agent_id, STONE] -= 1
            self.inventory[agent_id, BRICK] += 1
            return
        # Silent fail if neither condition met.

    def _do_craft_tool(self, agent_id: int) -> None:
        """Forge tool: 2 planks + 1 brick. Agent must be on a forge cell and
        have an empty tool slot."""
        pos = (int(self.agent_pos[agent_id, 0]), int(self.agent_pos[agent_id, 1]))
        if pos not in self.forge_cells_set:
            return
        if self.has_tool[agent_id]:
            return
        if self.inventory[agent_id, PLANK] < 2 or self.inventory[agent_id, BRICK] < 1:
            return
        self.inventory[agent_id, PLANK] -= 2
        self.inventory[agent_id, BRICK] -= 1
        self.has_tool[agent_id] = True
        self.tool_age[agent_id] = 0

    def _do_craft_shelter(self, agent_id: int) -> None:
        """Forge shelter: 3 planks + 3 bricks.

        Because inventory capacity (3) is smaller than the recipe (6 items),
        shelter assembly requires pooling inputs across agents via drops onto
        the forge cell. The recipe therefore consumes from the combined pool
        of (a) the calling agent's inventory and (b) items dropped on the
        forge cell itself. Inventory is drained first so the forge never
        dips into drops that are not strictly needed.
        """
        pos = (int(self.agent_pos[agent_id, 0]), int(self.agent_pos[agent_id, 1]))
        if pos not in self.forge_cells_set:
            return
        r, c = pos
        inv_p = int(self.inventory[agent_id, PLANK])
        inv_b = int(self.inventory[agent_id, BRICK])
        drop_p = int(self.dropped_items[r, c, PLANK])
        drop_b = int(self.dropped_items[r, c, BRICK])
        if inv_p + drop_p < 3 or inv_b + drop_b < 3:
            return
        # Consume 3 planks, inventory first then drops
        take_from_inv_p = min(inv_p, 3)
        take_from_drop_p = 3 - take_from_inv_p
        take_from_inv_b = min(inv_b, 3)
        take_from_drop_b = 3 - take_from_inv_b
        self.inventory[agent_id, PLANK] -= take_from_inv_p
        self.inventory[agent_id, BRICK] -= take_from_inv_b
        self.dropped_items[r, c, PLANK] -= take_from_drop_p
        self.dropped_items[r, c, BRICK] -= take_from_drop_b
        self.shelter_count += 1

    def _do_drop(self, agent_id: int, item: int) -> None:
        if self.inventory[agent_id, item] <= 0:
            return
        r, c = int(self.agent_pos[agent_id, 0]), int(self.agent_pos[agent_id, 1])
        cell_total = int(self.dropped_items[r, c].sum())
        if cell_total >= self.cell_drop_capacity:
            return
        self.inventory[agent_id, item] -= 1
        self.dropped_items[r, c, item] += 1

    def _do_pickup(self, agent_id: int, item: int) -> None:
        r, c = int(self.agent_pos[agent_id, 0]), int(self.agent_pos[agent_id, 1])
        if self.dropped_items[r, c, item] <= 0:
            return
        if self._inventory_free(agent_id) <= 0:
            return
        self.dropped_items[r, c, item] -= 1
        self.inventory[agent_id, item] += 1

    # ------------------------------------------------------------------
    # Rendering (god's-eye debug view)
    # ------------------------------------------------------------------

    def render(self, cell_size: int = 16) -> np.ndarray:
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        COLOUR = {
            "background": np.array([30, 30, 30]),
            "forest_stocked": np.array([30, 160, 30]),
            "forest_empty": np.array([70, 40, 0]),
            "quarry_stocked": np.array([160, 160, 160]),
            "quarry_empty": np.array([90, 90, 90]),
            "sawmill": np.array([160, 100, 40]),
            "masonry": np.array([120, 50, 200]),
            "forge": np.array([255, 120, 40]),
            "dropped": np.array([255, 220, 80]),
            "agent": np.array([50, 100, 255]),
            "agent_tool": np.array([120, 200, 255]),
        }

        img[:] = COLOUR["background"]

        for idx, (r, c) in enumerate(self.forest_cells_list):
            img[r, c] = COLOUR["forest_stocked"] if self.resource_stocked[idx] else COLOUR["forest_empty"]
        offset = len(self.forest_cells_list)
        for idx, (r, c) in enumerate(self.quarry_cells_list):
            img[r, c] = (COLOUR["quarry_stocked"] if self.resource_stocked[offset + idx]
                         else COLOUR["quarry_empty"])
        for (r, c) in self.sawmill_cells_list:
            img[r, c] = COLOUR["sawmill"]
        for (r, c) in self.masonry_cells_list:
            img[r, c] = COLOUR["masonry"]
        for (r, c) in self.forge_cells_list:
            img[r, c] = COLOUR["forge"]

        # Dropped items: overlay with yellowish tint where items exist
        for r in range(self.height):
            for c in range(self.width):
                if int(self.dropped_items[r, c].sum()) > 0:
                    img[r, c] = (img[r, c].astype(int) + COLOUR["dropped"]) // 2

        # Agents on top
        for i in range(self.n_agents):
            r, c = int(self.agent_pos[i, 0]), int(self.agent_pos[i, 1])
            img[r, c] = COLOUR["agent_tool"] if self.has_tool[i] else COLOUR["agent"]

        if cell_size > 1:
            img = np.repeat(np.repeat(img, cell_size, axis=0), cell_size, axis=1)
        return img

    # ------------------------------------------------------------------
    # Social outcome metrics (Perolat et al., matching existing framework)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_metrics(
        episode_rewards: Dict[int, List[float]],
        episode_timeouts: Dict[int, List[bool]],
    ) -> Dict[str, float]:
        n = len(episode_rewards)
        T = len(next(iter(episode_rewards.values())))

        returns = {i: sum(episode_rewards[i]) for i in episode_rewards}
        R = np.array(list(returns.values()))

        # Efficiency — rewards per step (collective). Matches other envs' formula.
        efficiency = float(R.sum() / T) if T > 0 else 0.0

        # Equality via Gini coefficient. Requires non-negative totals; shift if
        # necessary so the metric is well-defined when reward vectors include
        # large negative winter penalties.
        R_shift = R - R.min() if R.min() < 0 else R
        total = R_shift.sum()
        if total > 0:
            gini_num = sum(abs(R_shift[i] - R_shift[j]) for i in range(n) for j in range(n))
            equality = 1.0 - gini_num / (2 * n * total)
        else:
            equality = 1.0

        # Sustainability — mean time of reward-producing events per agent.
        mean_times = []
        for i in episode_rewards:
            rews = episode_rewards[i]
            times = [t for t, r in enumerate(rews) if r > 0]
            if times:
                mean_times.append(np.mean(times))
        sustainability = float(np.mean(mean_times)) if mean_times else 0.0

        # Peace — no aggression in this game; fix to n for API parity
        # (same convention as coop_mining).
        peace = float(n)

        # Maximin — worst-off agent's total return
        maximin = float(R.min()) if len(R) > 0 else 0.0

        return {
            "efficiency": efficiency,
            "equality": equality,
            "sustainability": sustainability,
            "peace": peace,
            "maximin": maximin,
        }


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def make_production_economy(n_agents: int = 8, **kwargs) -> ProductionEconomyEnv:
    """Create a Production Economy environment with default parameters."""
    defaults = dict(
        ascii_map=PRODUCTION_ECONOMY_MAP,
        n_agents=n_agents,
        max_steps=300,
        winter_step=200,
        winter_threshold=6,
        winter_reward=50.0,
        T_spoil=80,
        tool_step_reward=2.0,
        tool_gather_bonus=1,
        inventory_capacity=3,
        cell_drop_capacity=5,
        resource_respawn_prob=0.05,
        initial_stocked_fraction=0.8,
    )
    defaults.update(kwargs)
    return ProductionEconomyEnv(**defaults)


# ---------------------------------------------------------------------------
# Quick self-test / calibration demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Production Economy smoke test ===")
    env = make_production_economy(n_agents=8, seed=0)
    obs = env.reset(seed=42)
    print(f"Map size:             {env.height} x {env.width}")
    print(f"Forest cells:         {len(env.forest_cells_list)}")
    print(f"Quarry cells:         {len(env.quarry_cells_list)}")
    print(f"Sawmill cells:        {len(env.sawmill_cells_list)}")
    print(f"Masonry cells:        {len(env.masonry_cells_list)}")
    print(f"Forge cells:          {len(env.forge_cells_list)}")
    print(f"Spawn points:         {len(env.spawn_points)}")
    print(f"Resource nodes:       {env.n_resources} (stocked: {int(env.resource_stocked.sum())})")
    print(f"Horizon:              {env.max_steps}")
    print(f"Num actions:          {env.action_space_n}")
    print()

    # ---- Random baseline ----
    from collections import defaultdict
    ep_rewards = {i: [] for i in range(env.n_agents)}
    ep_timeouts = {i: [] for i in range(env.n_agents)}
    for step in range(env.max_steps):
        actions = {i: int(env.rng.integers(NUM_ACTIONS)) for i in range(env.n_agents)}
        obs, rewards, *_ = env.step(actions)
        for i in range(env.n_agents):
            ep_rewards[i].append(rewards[i])
            ep_timeouts[i].append(False)
    totals = {i: sum(ep_rewards[i]) for i in range(env.n_agents)}
    print("Random policy:")
    print(f"  Per-agent totals:   {[round(totals[i], 1) for i in range(env.n_agents)]}")
    print(f"  Mean per-agent:     {np.mean(list(totals.values())):.2f}")
    print(f"  Final shelter:      {env.shelter_count}")
    metrics = ProductionEconomyEnv.compute_metrics(ep_rewards, ep_timeouts)
    print(f"  Metrics:            {metrics}")
    print()

    # ---- Render snapshot ----
    img = env.render(cell_size=16)
    try:
        from PIL import Image
        Image.fromarray(img).save("production_economy_render.png")
        print("Saved render to production_economy_render.png")
    except ImportError:
        print("PIL not installed; skipping image save.")
    print(f"Render shape:         {img.shape}")
    print("\nAll smoke tests passed.")
