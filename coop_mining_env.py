"""
Coop Mining — A Stag Hunt Gridworld Environment
================================================

Implements the Coop Mining game from Melting Pot 2.0 (Agapiou et al., 2023).

Two types of ore appear on the map:
  - Iron (gray): can be mined by a single player for +1 reward
  - Gold (yellow): requires exactly 2 players mining within a 3-step window
    for +8 reward each

Mining iron is like playing Hare in a Stag Hunt (safe, low reward).
Mining gold is like playing Stag (risky, requires coordination, high reward).

Key features:
  - N agents with position and orientation on a 2D grid
  - 8 discrete actions: move (4 dirs), rotate (2 dirs), mine, stand
  - Mining beam: range 3, width 1 (hits first ore in path)
  - Gold activation signal: mined gold "flashes" for 3 steps as a
    coordination cue for a second miner
  - No tagging/timeout mechanism (pure coordination game)
  - Gymnasium-style API: reset() / step(actions)
"""

from __future__ import annotations

import numpy as np
from collections import defaultdict
from enum import IntEnum
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class Action(IntEnum):
    FORWARD = 0
    BACKWARD = 1
    STEP_LEFT = 2
    STEP_RIGHT = 3
    ROTATE_LEFT = 4
    ROTATE_RIGHT = 5
    MINE = 6
    STAND = 7

NUM_ACTIONS = len(Action)


class Orientation(IntEnum):
    NORTH = 0  # -row
    EAST = 1   # +col
    SOUTH = 2  # +row
    WEST = 3   # -col


# Rotation matrices: (forward, right) offsets → (dr, dc) world offsets
_ROTATIONS = {
    Orientation.NORTH: (-1, 0, 0, 1),
    Orientation.EAST:  (0, 1, 1, 0),
    Orientation.SOUTH: (1, 0, 0, -1),
    Orientation.WEST:  (0, -1, -1, 0),
}

IRON = 0
GOLD = 1

# ---------------------------------------------------------------------------
# Default maps (ASCII)
#   @ = wall   I = iron spawn   G = gold spawn   . = empty   P = player spawn
# ---------------------------------------------------------------------------

# Small map: 15×18, ~4 agents
COOP_MINING_MAP_SMALL = """\
@@@@@@@@@@@@@@@@@@
@P..I..I..I..I.P@
@..I..I..I..I...@
@.I..I..G..I..I.@
@..I..G..G..I...@
@.I..I..G..I..I.@
@..I..I..I..I...@
@.I..G..I..G..I.@
@..I..I..I..I...@
@.I..I..G..I..I.@
@..I..G..G..I...@
@.I..I..G..I..I.@
@..I..I..I..I...@
@P..I..I..I..I.P@
@@@@@@@@@@@@@@@@@@"""

# Large map: 20×30, ~6 agents
# Gold concentrated in 3 "veins" (clusters) for natural meeting points
COOP_MINING_MAP_LARGE = """\
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@P..I..I..I..I..I..I..I..I.P@
@.I..I..I..I..I..I..I..I..I.@
@..I..I..G.G.I..I.G.G..I..I.@
@.I..I..G.G.G.I.G.G.G..I..I.@
@..I..I..G.G.I..I.G.G..I..I.@
@.I..I..I..I..I..I..I..I..I.@
@..I..I..I..I..I..I..I..I..I@
@.I..I..I..I..I..I..I..I..I.@
@..I.G.G.I..I..I..I.G.G.I..@
@.I.G.G.G.I..I..I.G.G.G.I..@
@..I.G.G.I..I..I..I.G.G.I..@
@.I..I..I..I..I..I..I..I..I.@
@..I..I..I..I..I..I..I..I..I@
@.I..I..I.G.G.G.G.G.I..I..I.@
@..I..I..G.G.G.G.G.G..I..I.@
@.I..I..I.G.G.G.G.G.I..I..I.@
@..I..I..I..I..I..I..I..I..I@
@P..I..I..I..I..I..I..I..I.P@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"""


# ---------------------------------------------------------------------------
# Map parser
# ---------------------------------------------------------------------------

def parse_mining_map(
    ascii_map: str,
) -> Tuple[np.ndarray, List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Parse an ASCII map into walls, iron spawns, gold spawns, and player spawns."""
    lines = [l for l in ascii_map.strip().splitlines()]
    height = len(lines)
    width = max(len(l) for l in lines)
    lines = [l.ljust(width) for l in lines]

    walls = np.zeros((height, width), dtype=bool)
    iron_points: List[Tuple[int, int]] = []
    gold_points: List[Tuple[int, int]] = []
    spawn_points: List[Tuple[int, int]] = []

    for r, line in enumerate(lines):
        for c, ch in enumerate(line):
            if ch == "@":
                walls[r, c] = True
            elif ch == "I":
                iron_points.append((r, c))
            elif ch == "G":
                gold_points.append((r, c))
            elif ch == "P":
                spawn_points.append((r, c))

    if not spawn_points:
        for r in range(height):
            for c in range(width):
                if not walls[r, c] and (r, c) not in iron_points and (r, c) not in gold_points:
                    if r <= 1 or r >= height - 2 or c <= 1 or c >= width - 2:
                        spawn_points.append((r, c))

    return walls, iron_points, gold_points, spawn_points


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class CoopMiningEnv:
    """Multi-agent Coop Mining (Stag Hunt) gridworld.

    Parameters
    ----------
    ascii_map : str
        ASCII map. ``@`` = wall, ``I`` = iron spawn, ``G`` = gold spawn,
        ``P`` = player spawn, ``.`` or space = empty.
    n_agents : int
        Number of agents.
    max_steps : int
        Episode length.
    mine_range : int
        Mining beam range in cells ahead of the agent.
    gold_reward : float
        Reward to EACH of the two miners on successful gold extraction.
    iron_reward : float
        Reward for mining iron ore.
    gold_window : int
        Number of timesteps a gold ore stays activated (flashing) after
        first mine, waiting for a second miner.
    ore_respawn_time : int
        Steps until consumed ore respawns at its original position.
    seed : int or None
        Random seed.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        ascii_map: str = COOP_MINING_MAP_LARGE,
        n_agents: int = 6,
        max_steps: int = 1000,
        # Mining
        mine_range: int = 3,
        gold_reward: float = 8.0,
        iron_reward: float = 1.0,
        gold_window: int = 3,
        # Ore respawn
        ore_respawn_time: int = 20,
        # Misc
        seed: Optional[int] = None,
    ):
        self.walls, iron_pts, gold_pts, self.spawn_points = parse_mining_map(ascii_map)
        self.height, self.width = self.walls.shape

        self.n_agents = n_agents
        self.max_steps = max_steps
        self.mine_range = mine_range
        self.gold_reward = gold_reward
        self.iron_reward = iron_reward
        self.gold_window = gold_window
        self.ore_respawn_time = ore_respawn_time

        # Build ore arrays: iron first, then gold
        all_ore_pos = iron_pts + gold_pts
        all_ore_type = [IRON] * len(iron_pts) + [GOLD] * len(gold_pts)
        self.n_ores = len(all_ore_pos)
        self.n_iron = len(iron_pts)
        self.n_gold = len(gold_pts)
        self.ore_pos = np.array(all_ore_pos, dtype=np.int32)      # (M, 2)
        self.ore_type = np.array(all_ore_type, dtype=np.int32)     # (M,)

        # Random state
        self.rng = np.random.default_rng(seed)

        # --- Mutable state (initialised in reset) ---
        self._step_count: int = 0
        # Agent state
        self.agent_pos = np.zeros((n_agents, 2), dtype=np.int32)
        self.agent_orient = np.zeros(n_agents, dtype=np.int32)
        # Ore state
        self.ore_alive = np.ones(self.n_ores, dtype=bool)
        self.ore_respawn_timer = np.zeros(self.n_ores, dtype=np.int32)
        # Gold activation state
        self.ore_activated = np.zeros(self.n_ores, dtype=bool)
        self.ore_activation_timer = np.zeros(self.n_ores, dtype=np.int32)
        self.ore_activator = np.full(self.n_ores, -1, dtype=np.int32)

        # For rendering / debugging
        self._beam_cells: set = set()

        # Compatibility with run_episode (gathering_policy.py verbose printing)
        self.n_apples = self.n_ores  # alias for verbose output
        self.apple_respawn_time = self.ore_respawn_time  # alias

        # Gymnasium-like metadata
        self.action_space_n = NUM_ACTIONS

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> Dict[int, np.ndarray]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._step_count = 0

        # Spawn agents
        available = list(self.spawn_points)
        self.rng.shuffle(available)
        ore_set = set(map(tuple, self.ore_pos.tolist()))
        open_cells = [
            (r, c) for r in range(self.height) for c in range(self.width)
            if not self.walls[r, c] and (r, c) not in ore_set
        ]
        for i in range(self.n_agents):
            if i < len(available):
                self.agent_pos[i] = available[i]
            else:
                idx = self.rng.integers(len(open_cells))
                self.agent_pos[i] = open_cells[idx]
            self.agent_orient[i] = self.rng.integers(4)

        # All ore starts alive
        self.ore_alive[:] = True
        self.ore_respawn_timer[:] = 0
        self.ore_activated[:] = False
        self.ore_activation_timer[:] = 0
        self.ore_activator[:] = -1
        self._beam_cells = set()

        return {}  # No pixel observations needed; policies read env state directly

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self, actions: Dict[int, int],
    ) -> Tuple[Dict, Dict[int, float], Dict[int, bool], Dict[int, bool], Dict[int, dict]]:
        self._step_count += 1
        rewards = {i: 0.0 for i in range(self.n_agents)}
        self._beam_cells = set()

        # --- 1. Process movement and rotation ---
        for i in range(self.n_agents):
            act = actions.get(i, Action.STAND)
            if act == Action.FORWARD:
                self._move_agent(i, "forward")
            elif act == Action.BACKWARD:
                self._move_agent(i, "backward")
            elif act == Action.STEP_LEFT:
                self._move_agent(i, "left")
            elif act == Action.STEP_RIGHT:
                self._move_agent(i, "right")
            elif act == Action.ROTATE_LEFT:
                self.agent_orient[i] = (self.agent_orient[i] - 1) % 4
            elif act == Action.ROTATE_RIGHT:
                self.agent_orient[i] = (self.agent_orient[i] + 1) % 4
            # MINE and STAND are handled below / do nothing for movement

        # --- 2. Collect mining actions ---
        # For each agent that used MINE, find the first ore in their beam path
        mine_targets: Dict[int, int] = {}  # agent_id → ore_idx
        for i in range(self.n_agents):
            if actions.get(i, Action.STAND) == Action.MINE:
                ore_idx = self._cast_mine_beam(i)
                if ore_idx is not None:
                    mine_targets[i] = ore_idx

        # Group miners by ore
        ore_miners: Dict[int, List[int]] = defaultdict(list)
        for agent_id, ore_idx in mine_targets.items():
            ore_miners[ore_idx].append(agent_id)

        # --- 3. Resolve mining ---
        for ore_idx, miners in ore_miners.items():
            if not self.ore_alive[ore_idx]:
                continue

            if self.ore_type[ore_idx] == IRON:
                # Iron: first miner gets the reward, ore consumed
                winner = miners[0]
                rewards[winner] += self.iron_reward
                self._consume_ore(ore_idx)

            elif self.ore_type[ore_idx] == GOLD:
                if not self.ore_activated[ore_idx]:
                    # Gold not yet activated
                    if len(miners) == 1:
                        # Single miner: activate (flash)
                        self.ore_activated[ore_idx] = True
                        self.ore_activation_timer[ore_idx] = self.gold_window
                        self.ore_activator[ore_idx] = miners[0]
                    elif len(miners) == 2:
                        # Two miners in same step: instant success
                        for m in miners:
                            rewards[m] += self.gold_reward
                        self._consume_ore(ore_idx)
                    else:
                        # 3+ miners: too many, ore stays normal (fail)
                        pass
                else:
                    # Gold already activated — new miners joining
                    activator = self.ore_activator[ore_idx]
                    new_miners = [m for m in miners if m != activator]
                    if len(new_miners) == 0:
                        # Same activator re-mining: no effect
                        pass
                    elif len(new_miners) == 1:
                        # Exactly one new miner: success!
                        rewards[activator] += self.gold_reward
                        rewards[new_miners[0]] += self.gold_reward
                        self._consume_ore(ore_idx)
                    else:
                        # 2+ new miners: too many, deactivate
                        self._deactivate_ore(ore_idx)

        # --- 4. Tick activation timers ---
        for idx in range(self.n_ores):
            if self.ore_activated[idx]:
                self.ore_activation_timer[idx] -= 1
                if self.ore_activation_timer[idx] <= 0:
                    # Timer expired without second miner: deactivate
                    self._deactivate_ore(idx)

        # --- 5. Respawn consumed ore ---
        self._respawn_ores()

        # --- 6. Episode termination ---
        done = self._step_count >= self.max_steps
        terminated = {i: done for i in range(self.n_agents)}
        truncated = {i: False for i in range(self.n_agents)}

        # --- 7. Info ---
        info = {i: {"timeout": 0} for i in range(self.n_agents)}

        return {}, rewards, terminated, truncated, info

    # ------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------

    def _move_agent(self, agent_id: int, direction: str) -> None:
        orient = Orientation(self.agent_orient[agent_id])
        a, b, c, d = _ROTATIONS[orient]
        if direction == "forward":
            dr, dc = a, c
        elif direction == "backward":
            dr, dc = -a, -c
        elif direction == "left":
            dr, dc = -b, -d
        elif direction == "right":
            dr, dc = b, d
        else:
            return

        nr = self.agent_pos[agent_id, 0] + dr
        nc = self.agent_pos[agent_id, 1] + dc

        if 0 <= nr < self.height and 0 <= nc < self.width and not self.walls[nr, nc]:
            self.agent_pos[agent_id] = (nr, nc)

    # ------------------------------------------------------------------
    # Mining beam
    # ------------------------------------------------------------------

    def _cast_mine_beam(self, agent_id: int) -> Optional[int]:
        """Cast mining beam and return index of first ore hit, or None."""
        orient = Orientation(self.agent_orient[agent_id])
        a, _, c, _ = _ROTATIONS[orient]  # forward direction only (width 1)
        ar, ac = self.agent_pos[agent_id]

        # Build fast lookup: position → ore index (only alive ores)
        ore_at: Dict[Tuple[int, int], int] = {}
        for idx in range(self.n_ores):
            if self.ore_alive[idx]:
                pos = (int(self.ore_pos[idx, 0]), int(self.ore_pos[idx, 1]))
                ore_at[pos] = idx

        # Cast ray forward
        for dist in range(1, self.mine_range + 1):
            br = ar + a * dist
            bc = ac + c * dist
            if br < 0 or br >= self.height or bc < 0 or bc >= self.width:
                break
            if self.walls[br, bc]:
                break
            self._beam_cells.add((br, bc))
            pos = (br, bc)
            if pos in ore_at:
                return ore_at[pos]

        return None

    # ------------------------------------------------------------------
    # Ore management
    # ------------------------------------------------------------------

    def _consume_ore(self, ore_idx: int) -> None:
        self.ore_alive[ore_idx] = False
        self.ore_respawn_timer[ore_idx] = self.ore_respawn_time
        self.ore_activated[ore_idx] = False
        self.ore_activation_timer[ore_idx] = 0
        self.ore_activator[ore_idx] = -1

    def _deactivate_ore(self, ore_idx: int) -> None:
        self.ore_activated[ore_idx] = False
        self.ore_activation_timer[ore_idx] = 0
        self.ore_activator[ore_idx] = -1

    def _respawn_ores(self) -> None:
        dead = ~self.ore_alive
        self.ore_respawn_timer[dead] -= 1
        respawned = dead & (self.ore_respawn_timer <= 0)
        self.ore_alive[respawned] = True
        self.ore_respawn_timer[respawned] = 0

    # ------------------------------------------------------------------
    # Social outcome metrics
    # ------------------------------------------------------------------

    @staticmethod
    def compute_metrics(
        episode_rewards: Dict[int, List[float]],
        episode_timeouts: Dict[int, List[bool]],
    ) -> Dict[str, float]:
        """Compute social-outcome metrics from one episode.

        Same definitions as GatheringEnv.compute_metrics for consistency.
        """
        n = len(episode_rewards)
        T = len(next(iter(episode_rewards.values())))

        returns = {i: sum(episode_rewards[i]) for i in episode_rewards}
        R = np.array(list(returns.values()))

        # Efficiency (U)
        efficiency = float(R.sum() / T) if T > 0 else 0.0

        # Equality (E) via Gini coefficient
        total = R.sum()
        if total > 0:
            gini_num = sum(abs(R[i] - R[j]) for i in range(n) for j in range(n))
            equality = 1.0 - gini_num / (2 * n * total)
        else:
            equality = 1.0

        # Sustainability (S)
        mean_times = []
        for i in episode_rewards:
            rews = episode_rewards[i]
            times = [t for t, r in enumerate(rews) if r > 0]
            if times:
                mean_times.append(np.mean(times))
        sustainability = float(np.mean(mean_times)) if mean_times else 0.0

        # Peace (P) — no timeout mechanism, always perfect
        peace = float(n)

        # Maximin (Rawlsian welfare)
        maximin = float(R.min()) if len(R) > 0 else 0.0

        return {
            "efficiency": efficiency,
            "equality": equality,
            "sustainability": sustainability,
            "peace": peace,
            "maximin": maximin,
        }


# ---------------------------------------------------------------------------
# Convenience factory functions
# ---------------------------------------------------------------------------

def make_coop_mining(
    n_agents: int = 4,
    small: bool = False,
    **kwargs,
) -> CoopMiningEnv:
    """Create a Coop Mining environment with standard parameters."""
    if small:
        return CoopMiningEnv(
            ascii_map=COOP_MINING_MAP_SMALL,
            n_agents=n_agents,
            **kwargs,
        )
    return CoopMiningEnv(
        ascii_map=COOP_MINING_MAP_SMALL,
        n_agents=n_agents,
        **kwargs,
    )


def make_coop_mining_large(
    n_agents: int = 6,
    **kwargs,
) -> CoopMiningEnv:
    """Create a large Coop Mining environment."""
    return CoopMiningEnv(
        ascii_map=COOP_MINING_MAP_LARGE,
        n_agents=n_agents,
        **kwargs,
    )
