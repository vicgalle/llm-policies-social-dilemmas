"""
Nested Commons — A Sequential Social Dilemma with Compositional Mechanism Design
==================================================================================

Implements the Nested Commons game defined in nested_commons.md.

Sixteen agents live on a 20x20 grid divided into four 10x10 quadrants
(clan homes A, B, C, D), with a 4x4 plaza superimposed at the inner
corners (rows 8-11, cols 8-11).  Each quadrant has a polluted river,
a 5x5 orchard, and four spawn points.  The plaza hosts a shared
bonus-apple forest whose payoff is gated by a global cleanliness
threshold.

Three nested dilemmas:
  1. Intra-clan public good (river cleaning per quadrant).
  2. Inter-clan public good (plaza cleaning).
  3. Raid restraint between clans.

Action space (42 discrete actions):
    0  NOOP
    1  MOVE_N
    2  MOVE_S
    3  MOVE_E
    4  MOVE_W
    5  CLEAN              cost -1; reduces w_q (river-adjacent) or w_P (in plaza)
    6-21   RAID_j         RAID target agent j (j in 0..15); cost -0.5; p_succ=0.6
    22-37  GIFT_j         GIFT 1 apple to target agent j; free
    38-41  TRAVEL_q       BFS toward nearest cell of quadrant q (q in 0..3);
                          3-step commitment; -0.1 per travel step

Encode parameterized actions with the helper functions:
    raid(target_id), gift(target_id), travel(quadrant)

Inventory model:
  - Capacity 3.
  - Orchard apples enter inventory as "fresh".  At end-of-step, fresh
    apples auto-eat for reward, capped at (capacity - held_at_start).
    Excess fresh is discarded.  Fresh apples never persist past a step.
  - Gifts received and successful raid-wins enter inventory as "held".
    Held apples persist; capacity caps the total held.
  - Held apples reduce the next step's eat budget — the anti-hoarding
    pressure called for by the spec.
  - Plaza bonus apples never enter inventory (immediate consume; +2 to
    collector, +2 to each other agent if w_P ≤ bonus_threshold).
"""

from __future__ import annotations

import numpy as np
from collections import deque
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------

class Action(IntEnum):
    """Base actions; parameterized actions live in numeric ranges below."""
    NOOP = 0
    MOVE_N = 1
    MOVE_S = 2
    MOVE_E = 3
    MOVE_W = 4
    CLEAN = 5

# Parameterized action ranges (16 agents, 4 quadrants).
RAID_BASE = 6        # RAID target j  =  RAID_BASE + j   (j ∈ [0, 15])
GIFT_BASE = 22       # GIFT target j  =  GIFT_BASE + j
TRAVEL_BASE = 38     # TRAVEL quad q  =  TRAVEL_BASE + q (q ∈ [0, 3])
NUM_ACTIONS = 42
MAX_ACTION = NUM_ACTIONS - 1  # 41

# Clan / quadrant indices
CLAN_A, CLAN_B, CLAN_C, CLAN_D = 0, 1, 2, 3
NUM_CLANS = 4
CLAN_NAMES = ("A", "B", "C", "D")

# Movement offsets (row, col)
_MOVE_OFFSETS = {
    int(Action.MOVE_N): (-1, 0),
    int(Action.MOVE_S): (1, 0),
    int(Action.MOVE_E): (0, 1),
    int(Action.MOVE_W): (0, -1),
}


def raid(target_id: int) -> int:
    """Encode a RAID action against `target_id` (0..15)."""
    return RAID_BASE + int(target_id)


def gift(target_id: int) -> int:
    """Encode a GIFT action toward `target_id` (0..15)."""
    return GIFT_BASE + int(target_id)


def travel(quadrant: int) -> int:
    """Encode a TRAVEL action toward `quadrant` (0..3 = A,B,C,D)."""
    return TRAVEL_BASE + int(quadrant)


def is_raid_action(a: int) -> bool:
    return RAID_BASE <= int(a) < RAID_BASE + 16


def is_gift_action(a: int) -> bool:
    return GIFT_BASE <= int(a) < GIFT_BASE + 16


def is_travel_action(a: int) -> bool:
    return TRAVEL_BASE <= int(a) < TRAVEL_BASE + 4


def decode_raid_target(a: int) -> int:
    return int(a) - RAID_BASE


def decode_gift_target(a: int) -> int:
    return int(a) - GIFT_BASE


def decode_travel_quadrant(a: int) -> int:
    return int(a) - TRAVEL_BASE


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NestedCommonsConfig:
    """All numeric constants of the Nested Commons environment.

    Per spec §8 the researcher MUST NOT modify this dataclass; only the
    pipeline (prompts/feedback/helpers/config) around it.
    """

    # Grid / population
    grid_size: int = 20
    plaza_lo: int = 8        # inclusive
    plaza_hi: int = 11       # inclusive (plaza is rows/cols 8..11)
    n_agents: int = 16
    n_clans: int = 4
    inventory_capacity: int = 3
    max_steps: int = 1000

    # Quadrant waste dynamics
    wq_growth: float = 0.005
    wq_growth_per_apple: float = 0.04        # was 0.02 — harvest pollutes more
    wq_clean_amount: float = 0.025

    # Plaza waste dynamics
    wp_growth: float = 0.003                  # was 0.002 — faster baseline
    wp_growth_per_apple: float = 0.002        # was 0.001 — stronger global coupling
    wp_clean_amount: float = 0.025            # was 0.020 — matched cleaning power

    # Apple regrowth (orchard) — per-clan to break self-play symmetry.
    # Index q gives the orchard regrow_max for clan q (A=0, B=1, C=2, D=3).
    # The wealth gradient creates a persistent reason to engage other clans
    # via TRAVEL/RAID/GIFT rather than treating each clan as an independent
    # public-goods problem.
    apple_regrow_max_per_clan: Tuple[float, ...] = (0.20, 0.14, 0.10, 0.06)
    apple_regrow_slope: float = 2.0

    # Plaza bonus (de-emphasised so it doesn't dominate)
    bonus_threshold: float = 0.25             # was 0.35 — tighter shared-bonus window
    bonus_regrow_threshold: float = 0.35      # was 0.50 — regrowth disappears earlier
    bonus_regrow_prob: float = 0.06           # was 0.08
    bonus_value: float = 1.0                  # was 2.0 — half-strength shared payout
    # Plaza bonus is forfeited (collector + shared-bonus recipients) by any
    # agent whose held inventory at start-of-step is below this threshold.
    # Couples held-inventory to the public-goods payout, making RAID/GIFT
    # strategically meaningful: stealing held inventory denies the victim's
    # plaza eligibility, gifting confers it. Apple is still consumed even
    # when the gate fails, to avoid stalling plaza regrowth.
    plaza_bonus_held_min: int = 1

    # Action costs / probabilities
    clean_cost: float = 1.0
    raid_cost: float = 0.5
    raid_success_prob: float = 0.6
    gift_amount: int = 1
    travel_step_cost: float = 0.1
    travel_steps: int = 3

    # Initial conditions
    initial_wq: float = 0.1
    initial_wp: float = 0.1
    initial_orchard_fill: float = 0.4
    initial_plaza_fill: float = 0.5

    # Held-inventory mechanic (third nested dilemma — raid restraint).
    # Each held apple pays `held_apple_per_step_reward` per step, so inventory
    # is a future-reward stream worth stealing.  `initial_held` seeds every
    # agent so raids have something to target from t=0; auto-eat is decoupled
    # from held inventory (see Phase 5 in step()) so raising held above 2
    # actually pays out instead of consuming an auto-eat slot.
    held_apple_per_step_reward: float = 0.05
    initial_held: int = 2

    # ARIMD structural toggles (autoresearch/arimd/grammar.py).
    # All default False so the env's nominal behaviour is unchanged unless
    # Blue explicitly flips one.  Each toggle changes a *rule*, not a
    # numeric, and is the kind of structural lever H3 looks for.
    #
    # plaza_occupant_only_bonus: when a plaza apple's shared bonus fires
    #   (w_p ≤ bonus_threshold), pay the bonus only to other agents whose
    #   current cell is in the plaza.  Removes the global free-rider —
    #   non-plaza occupants no longer collect a passive payout.
    plaza_occupant_only_bonus: bool = False
    # plaza_local_clean_gate: gate the plaza bonus (both individual and
    #   shared) on the *collector's clan* river cleanliness:
    #   ``w_q[clan_of_collector] <= bonus_threshold``.  The plaza apple is
    #   still consumed if the gate fails, so polluting your own quadrant
    #   forfeits the plaza payout for your clan.
    plaza_local_clean_gate: bool = False
    # same_clan_retaliation: after a successful cross-clan raid by i on j,
    #   queue every clan-mate of j (other than j) for an automatic RAID(i)
    #   on the next step — applied at action-resolution time only if the
    #   would-be retaliator is adjacent to the raider.  One-step lifetime;
    #   the queue clears after each step regardless of execution.
    same_clan_retaliation: bool = False


DEFAULT_CONFIG = NestedCommonsConfig()


# ---------------------------------------------------------------------------
# Static layout
# ---------------------------------------------------------------------------
#
# Layout is deterministic and fixed (per spec §2.2).  Helpers below derive
# it from the config so a swap of plaza_lo/plaza_hi/grid_size stays
# consistent.
#
#   Quadrants:  A=NW (r<10, c<10), B=NE (r<10, c>=10),
#               C=SW (r>=10, c<10), D=SE (r>=10, c>=10).
#   Plaza:      rows/cols [plaza_lo, plaza_hi]  (default 8..11).
#   River:      8-cell strip on outer edge of each quadrant (row 0 / 19).
#   Orchard:    5x5 region on plaza-facing inner half, NOT overlapping plaza.
#   Spawns:     4 per quadrant at the outer-corner 2x2 patch.

def _build_layout(cfg: NestedCommonsConfig):
    """Return the static layout — river/orchard/spawn cells per quadrant.

    All returned coordinates are (row, col) ints.  Lists are ordered by
    quadrant index 0..3 (A, B, C, D).
    """
    G = cfg.grid_size
    half = G // 2  # = 10 for default

    # Rivers: 8 cells each along the outer row (top for A/B, bottom for C/D),
    # within the quadrant's column range, with a 2-cell inset from the inner
    # plaza-facing edge.  Inset choice keeps rivers at the outer corner side.
    river_cells: List[List[Tuple[int, int]]] = [
        [(0, c) for c in range(0, 8)],            # A: top, cols 0..7
        [(0, c) for c in range(G - 8, G)],        # B: top, cols 12..19
        [(G - 1, c) for c in range(0, 8)],        # C: bottom, cols 0..7
        [(G - 1, c) for c in range(G - 8, G)],    # D: bottom, cols 12..19
    ]

    # Orchards: 5x5 plaza-facing inner block of each quadrant.  Disjoint from
    # the 4x4 plaza (plaza_lo..plaza_hi) by construction.
    orchard_cells: List[List[Tuple[int, int]]] = [
        [(r, c) for r in range(3, 8) for c in range(3, 8)],          # A
        [(r, c) for r in range(3, 8) for c in range(G - 8, G - 3)],  # B  cols 12..16
        [(r, c) for r in range(G - 8, G - 3) for c in range(3, 8)],  # C  rows 12..16
        [(r, c) for r in range(G - 8, G - 3) for c in range(G - 8, G - 3)],  # D
    ]

    # Spawn points: 4 per quadrant in the outer-corner 2x2 patch (rows 1..2,
    # cols 1..2 for A; mirrored for B/C/D).
    spawn_cells: List[List[Tuple[int, int]]] = [
        [(r, c) for r in (1, 2) for c in (1, 2)],
        [(r, c) for r in (1, 2) for c in (G - 3, G - 2)],
        [(r, c) for r in (G - 3, G - 2) for c in (1, 2)],
        [(r, c) for r in (G - 3, G - 2) for c in (G - 3, G - 2)],
    ]

    # Plaza cells (4x4 by default).
    plaza_cells = [(r, c)
                   for r in range(cfg.plaza_lo, cfg.plaza_hi + 1)
                   for c in range(cfg.plaza_lo, cfg.plaza_hi + 1)]

    return river_cells, orchard_cells, spawn_cells, plaza_cells


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class NestedCommonsEnv:
    """Multi-agent Nested Commons gridworld.

    Drop-in replacement for Cleanup in the autoresearch framework: same
    `reset(seed)` / `step(actions)` / `compute_metrics` interface, full
    env-state access through the `env` handle inside policies, scalar
    `n_agents` / `max_steps` attributes.

    Parameters
    ----------
    config : NestedCommonsConfig
        Frozen dataclass holding all environment constants.
    seed : int or None
        Random seed.  Re-seeded on each `reset(seed=...)` call.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        config: NestedCommonsConfig = DEFAULT_CONFIG,
        seed: Optional[int] = None,
    ):
        self.config = config
        self.n_agents = config.n_agents
        self.max_steps = config.max_steps
        self.inventory_capacity = config.inventory_capacity
        self.height = config.grid_size
        self.width = config.grid_size
        # No walls in this environment; kept for API compatibility.
        self.walls = np.zeros((self.height, self.width), dtype=bool)

        # Static layout
        (self.river_cells_per_q,
         self.orchard_cells_per_q,
         self.spawn_cells_per_q,
         self.plaza_cells) = _build_layout(config)

        # Flat lists / quick-lookup sets.
        self.river_cells_list: List[Tuple[int, int]] = [
            cell for cells in self.river_cells_per_q for cell in cells
        ]
        self.orchard_cells_list: List[Tuple[int, int]] = [
            cell for cells in self.orchard_cells_per_q for cell in cells
        ]
        self.spawn_cells_list: List[Tuple[int, int]] = [
            cell for cells in self.spawn_cells_per_q for cell in cells
        ]
        self.river_cells_set = set(self.river_cells_list)
        self.orchard_cells_set = set(self.orchard_cells_list)
        self.plaza_cells_set = set(self.plaza_cells)

        # river_cell → quadrant_id (so CLEAN can credit the right w_q).
        self._river_to_q: Dict[Tuple[int, int], int] = {}
        for q, cells in enumerate(self.river_cells_per_q):
            for cell in cells:
                self._river_to_q[cell] = q

        # orchard_cell → quadrant_id (for collection bookkeeping).
        self._orchard_to_q: Dict[Tuple[int, int], int] = {}
        for q, cells in enumerate(self.orchard_cells_per_q):
            for cell in cells:
                self._orchard_to_q[cell] = q

        # Plaza cells lay across all four quadrants — track which quadrant
        # each plaza cell belongs to (for clan-home accounting per §2.3).
        self._plaza_to_q: Dict[Tuple[int, int], int] = {
            cell: self._quadrant_of(*cell) for cell in self.plaza_cells
        }

        # Quadrant of every spawn point.
        self.agent_clan = np.array(
            [i // (self.n_agents // NUM_CLANS) for i in range(self.n_agents)],
            dtype=np.int32,
        )

        # Counts used by gathering_policy.run_episode in verbose mode.
        self.n_apples = len(self.orchard_cells_list) + len(self.plaza_cells)
        # Approximate respawn time = 1 / mean regrow probability.
        _mean_regrow = float(np.mean(config.apple_regrow_max_per_clan))
        self.apple_respawn_time = int(round(1.0 / max(_mean_regrow, 1e-6)))

        # Random state
        self.rng = np.random.default_rng(seed)

        # --- Mutable state (initialised in reset) -----------------------
        self._step_count: int = 0
        self.agent_pos = np.zeros((self.n_agents, 2), dtype=np.int32)
        self.inventory = np.zeros(self.n_agents, dtype=np.int32)  # held apples
        # Apple presence on every cell (orchards + plaza bonus).
        self.orchard_apple = np.zeros((self.height, self.width), dtype=bool)
        self.bonus_apple = np.zeros((self.height, self.width), dtype=bool)
        # Waste levels.
        self.w_q = np.full(NUM_CLANS, config.initial_wq, dtype=np.float64)
        self.w_p = float(config.initial_wp)
        # TRAVEL queue per agent.
        self._travel_queue: List[deque] = [deque() for _ in range(self.n_agents)]
        # Compatibility shim — no agent ever times out in this game, but
        # gathering_policy.run_episode reads env.agent_timeout when present.
        self.agent_timeout = np.zeros(self.n_agents, dtype=np.int32)

        # Episode-level counters used in info / metrics.
        self._raid_attempts_total: int = 0
        self._raid_successes_total: int = 0
        self._gift_total: int = 0
        self._shared_bonus_total: int = 0

        # Same-clan retaliation queue: list of (retaliator_id, target_id)
        # tuples consumed at the start of the next step. Always present;
        # only populated when cfg.same_clan_retaliation is True.
        self._pending_retaliation: List[Tuple[int, int]] = []

        self.action_space_n = NUM_ACTIONS

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _quadrant_of(self, r: int, c: int) -> int:
        half = self.config.grid_size // 2
        if r < half and c < half:
            return CLAN_A
        if r < half and c >= half:
            return CLAN_B
        if r >= half and c < half:
            return CLAN_C
        return CLAN_D

    def quadrant_of(self, r: int, c: int) -> int:
        """Public helper: which clan-home quadrant does (r,c) live in?"""
        return self._quadrant_of(int(r), int(c))

    def _in_plaza(self, r: int, c: int) -> bool:
        lo, hi = self.config.plaza_lo, self.config.plaza_hi
        return lo <= r <= hi and lo <= c <= hi

    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.height and 0 <= c < self.width

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> Dict[int, object]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        cfg = self.config
        self._step_count = 0

        # Place agents at their clan's spawn points.  Convention: agents
        # 0..3 in clan A, 4..7 in clan B, etc.  Within a clan, deterministic
        # assignment to the 4 spawn cells.
        for i in range(self.n_agents):
            clan = int(self.agent_clan[i])
            slot = i % len(self.spawn_cells_per_q[clan])
            self.agent_pos[i] = self.spawn_cells_per_q[clan][slot]

        self.inventory[:] = cfg.initial_held
        self.agent_timeout[:] = 0

        # Pre-seed orchards: 40% of cells per quadrant.
        self.orchard_apple[:] = False
        for q, cells in enumerate(self.orchard_cells_per_q):
            n_seed = int(round(cfg.initial_orchard_fill * len(cells)))
            if n_seed > 0:
                idx = self.rng.choice(len(cells), size=n_seed, replace=False)
                for k in idx:
                    r, c = cells[int(k)]
                    self.orchard_apple[r, c] = True

        # Pre-seed plaza bonus: 50% of plaza cells.
        self.bonus_apple[:] = False
        n_bonus_seed = int(round(cfg.initial_plaza_fill * len(self.plaza_cells)))
        if n_bonus_seed > 0:
            idx = self.rng.choice(len(self.plaza_cells), size=n_bonus_seed, replace=False)
            for k in idx:
                r, c = self.plaza_cells[int(k)]
                self.bonus_apple[r, c] = True

        # Waste levels reset.
        self.w_q[:] = cfg.initial_wq
        self.w_p = float(cfg.initial_wp)

        # Travel queues.
        self._travel_queue = [deque() for _ in range(self.n_agents)]

        # Counters.
        self._raid_attempts_total = 0
        self._raid_successes_total = 0
        self._gift_total = 0
        self._shared_bonus_total = 0
        self._pending_retaliation = []

        return {i: None for i in range(self.n_agents)}

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self, actions: Dict[int, int],
    ) -> Tuple[Dict[int, object], Dict[int, float],
               Dict[int, bool], Dict[int, bool], Dict[int, dict]]:
        cfg = self.config
        self._step_count += 1
        rewards = {i: 0.0 for i in range(self.n_agents)}

        # Snapshot held inventory at step start for the auto-eat budget.
        held_at_start = self.inventory.copy()
        # Track per-agent fresh orchard collections this step.
        fresh_collected = np.zeros(self.n_agents, dtype=np.int32)
        # Per-quadrant orchard collection count for waste dynamics.
        apples_collected_per_q = np.zeros(NUM_CLANS, dtype=np.int32)
        # Per-step counters surfaced via info[i].
        raid_attempted_step = np.zeros(self.n_agents, dtype=np.int32)
        raid_succeeded_step = np.zeros(self.n_agents, dtype=np.int32)
        gift_given_step = np.zeros(self.n_agents, dtype=np.int32)
        bonus_collected_step = 0
        shared_bonus_step = 0

        # ------------------------------------------------------------------
        # Phase 0 — resolve the action each agent will execute this step.
        # If the agent has a queued travel move, that overrides their input.
        # If their input is TRAVEL, plan a path and execute its first step.
        #
        # Same-clan retaliation queue (if cfg.same_clan_retaliation): every
        # entry forces the listed retaliator to RAID the listed target this
        # step, provided they are adjacent at action-resolution time.  The
        # queue is drained unconditionally so retaliations have a one-step
        # lifetime regardless of execution.
        # ------------------------------------------------------------------
        forced_raid: Dict[int, int] = {}
        if cfg.same_clan_retaliation and self._pending_retaliation:
            for retaliator, target in self._pending_retaliation:
                if not (0 <= retaliator < self.n_agents):
                    continue
                if not (0 <= target < self.n_agents) or target == retaliator:
                    continue
                if self._travel_queue[retaliator]:
                    continue  # mid-travel agents can't retaliate this step
                if int(self.agent_clan[target]) == int(self.agent_clan[retaliator]):
                    continue  # same-clan raid would resolve as NOOP anyway
                rr, rc = int(self.agent_pos[retaliator, 0]), int(self.agent_pos[retaliator, 1])
                tr_, tc_ = int(self.agent_pos[target, 0]), int(self.agent_pos[target, 1])
                if abs(rr - tr_) + abs(rc - tc_) != 1:
                    continue  # adjacency required
                forced_raid[retaliator] = raid(target)
        self._pending_retaliation = []

        effective: List[int] = [int(Action.NOOP)] * self.n_agents
        for i in range(self.n_agents):
            if self._travel_queue[i]:
                # Mid-travel: pop next queued move.
                effective[i] = int(self._travel_queue[i].popleft())
                rewards[i] -= cfg.travel_step_cost
                continue

            if i in forced_raid:
                # Forced retaliation overrides the agent's submitted action.
                effective[i] = forced_raid[i]
                continue

            a = int(actions.get(i, int(Action.NOOP)))
            if is_travel_action(a):
                tgt_q = decode_travel_quadrant(a)
                plan = self._travel_plan(i, tgt_q)
                if plan:
                    take = plan[: cfg.travel_steps]
                    effective[i] = int(take[0])
                    for m in take[1:]:
                        self._travel_queue[i].append(int(m))
                    rewards[i] -= cfg.travel_step_cost
                # else: already in target quadrant → silent NOOP, no cost.
            else:
                effective[i] = a

        # ------------------------------------------------------------------
        # Phase 1 — atomic movement resolution.
        #
        # Same rule the rest of the framework uses: same-target conflicts
        # broken by lower agent_id; iterate to drop moves into cells held
        # by stationary agents until stable.
        # ------------------------------------------------------------------
        positions = [tuple(self.agent_pos[i]) for i in range(self.n_agents)]
        desired = list(positions)
        for i in range(self.n_agents):
            a = effective[i]
            if a not in _MOVE_OFFSETS:
                continue
            dr, dc = _MOVE_OFFSETS[a]
            nr, nc = positions[i][0] + dr, positions[i][1] + dc
            if not self._in_bounds(nr, nc):
                continue
            desired[i] = (nr, nc)

        claims: Dict[Tuple[int, int], List[int]] = {}
        for i in range(self.n_agents):
            if desired[i] != positions[i]:
                claims.setdefault(desired[i], []).append(i)
        for tgt, ids in claims.items():
            if len(ids) > 1:
                ids.sort()
                for loser in ids[1:]:
                    desired[loser] = positions[loser]

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

        # ------------------------------------------------------------------
        # Phase 2 — orchard apple collection (movement onto an apple cell).
        # Plaza-bonus collection (immediate consume, plus shared bonus).
        # ------------------------------------------------------------------
        for i in range(self.n_agents):
            r, c = int(self.agent_pos[i, 0]), int(self.agent_pos[i, 1])
            cell = (r, c)
            if cell in self.orchard_cells_set and self.orchard_apple[r, c]:
                self.orchard_apple[r, c] = False
                fresh_collected[i] += 1
                apples_collected_per_q[self._orchard_to_q[cell]] += 1

        # Plaza bonus: process in agent_id order so a deterministic single
        # collector picks each cell.  Shared-bonus payments use the
        # current w_p (snapshotted before any collection updates it).
        wp_before_collect = self.w_p
        for i in range(self.n_agents):
            r, c = int(self.agent_pos[i, 0]), int(self.agent_pos[i, 1])
            cell = (r, c)
            if cell not in self.plaza_cells_set:
                continue
            if not self.bonus_apple[r, c]:
                continue
            # Apple is consumed regardless of whether the gate fires —
            # otherwise an unclean clan stalls regrowth on the plaza.
            self.bonus_apple[r, c] = False
            collector_clan = int(self.agent_clan[i])
            local_gate_open = (
                not cfg.plaza_local_clean_gate
                or float(self.w_q[collector_clan]) <= cfg.bonus_threshold
            )
            if not local_gate_open:
                # Plaza payout denied: collector's home river is too dirty.
                continue
            # Held-inventory gate: collector forfeits payout if start-of-step
            # held inventory is below the threshold.  Couples plaza payouts
            # to the held-stream economics so RAID/GIFT have public-goods
            # consequences beyond redistribution.
            if int(held_at_start[i]) < cfg.plaza_bonus_held_min:
                continue
            rewards[i] += cfg.bonus_value
            bonus_collected_step += 1
            if wp_before_collect <= cfg.bonus_threshold:
                # Shared bonus: +bonus_value to every other agent (or only
                # to other agents currently in the plaza, if the
                # plaza_occupant_only_bonus toggle is on).  Each recipient
                # must also satisfy the held-inventory gate.
                for j in range(self.n_agents):
                    if j == i:
                        continue
                    if cfg.plaza_occupant_only_bonus:
                        jr, jc = int(self.agent_pos[j, 0]), int(self.agent_pos[j, 1])
                        if not self._in_plaza(jr, jc):
                            continue
                    if int(held_at_start[j]) < cfg.plaza_bonus_held_min:
                        continue
                    rewards[j] += cfg.bonus_value
                shared_bonus_step += 1

        # ------------------------------------------------------------------
        # Phase 3 — CLEAN actions.  Cost is paid whenever CLEAN is issued;
        # waste reduction only happens at a valid clean target (river-
        # adjacent or in plaza).
        # ------------------------------------------------------------------
        cleaned_q = np.zeros(NUM_CLANS, dtype=np.int32)
        cleaned_plaza = 0
        for i in range(self.n_agents):
            if effective[i] != int(Action.CLEAN):
                continue
            rewards[i] -= cfg.clean_cost
            r, c = int(self.agent_pos[i, 0]), int(self.agent_pos[i, 1])
            if self._in_plaza(r, c):
                self.w_p = max(0.0, self.w_p - cfg.wp_clean_amount)
                cleaned_plaza += 1
                continue
            # Otherwise, look for an adjacent river cell.
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nb = (r + dr, c + dc)
                if nb in self.river_cells_set:
                    q = self._river_to_q[nb]
                    self.w_q[q] = max(0.0, float(self.w_q[q]) - cfg.wq_clean_amount)
                    cleaned_q[q] += 1
                    break
            # Else: cost paid, no waste reduced.

        # ------------------------------------------------------------------
        # Phase 4 — RAID and GIFT (require adjacency at post-movement
        # positions).  Process in agent_id order; an agent can be both
        # raided and a raider in the same step.
        # ------------------------------------------------------------------
        for i in range(self.n_agents):
            a = effective[i]
            if not (is_raid_action(a) or is_gift_action(a)):
                continue
            r, c = int(self.agent_pos[i, 0]), int(self.agent_pos[i, 1])

            if is_raid_action(a):
                tgt = decode_raid_target(a)
                if tgt < 0 or tgt >= self.n_agents or tgt == i:
                    continue
                # Same-clan raids resolve as NOOP per spec; no cost.
                if self.agent_clan[tgt] == self.agent_clan[i]:
                    continue
                tr, tc = int(self.agent_pos[tgt, 0]), int(self.agent_pos[tgt, 1])
                if abs(tr - r) + abs(tc - c) != 1:
                    # Not adjacent → invalid → no cost, no attempt.
                    continue
                # Valid attempt: pay -0.5 regardless of outcome.
                rewards[i] -= cfg.raid_cost
                raid_attempted_step[i] += 1
                self._raid_attempts_total += 1
                if self.inventory[tgt] <= 0:
                    continue
                if self.rng.random() < cfg.raid_success_prob:
                    # Transfer 1 apple — capacity-cap on raider's inventory
                    # (excess is discarded; raid still empties from victim).
                    self.inventory[tgt] -= 1
                    if self.inventory[i] < cfg.inventory_capacity:
                        self.inventory[i] += 1
                    raid_succeeded_step[i] += 1
                    self._raid_successes_total += 1
                    if cfg.same_clan_retaliation:
                        # Queue every clan-mate of the victim (other than
                        # the victim itself) to auto-RAID the raider on
                        # the next step.  Adjacency is checked at execute
                        # time; non-adjacent retaliators silently drop.
                        victim_clan = int(self.agent_clan[tgt])
                        for k in range(self.n_agents):
                            if k == tgt or k == i:
                                continue
                            if int(self.agent_clan[k]) == victim_clan:
                                self._pending_retaliation.append((k, i))

            elif is_gift_action(a):
                tgt = decode_gift_target(a)
                if tgt < 0 or tgt >= self.n_agents or tgt == i:
                    continue
                if self.inventory[i] < cfg.gift_amount:
                    continue
                tr, tc = int(self.agent_pos[tgt, 0]), int(self.agent_pos[tgt, 1])
                if abs(tr - r) + abs(tc - c) != 1:
                    continue
                # Transfer (capacity-cap on recipient; donor always loses).
                self.inventory[i] -= cfg.gift_amount
                if self.inventory[tgt] < cfg.inventory_capacity:
                    self.inventory[tgt] = min(
                        cfg.inventory_capacity,
                        int(self.inventory[tgt]) + cfg.gift_amount,
                    )
                gift_given_step[i] += 1
                self._gift_total += 1

        # ------------------------------------------------------------------
        # Phase 5 — auto-eat fresh orchard collections.
        #
        # Held inventory does NOT reduce the eat budget any more — instead,
        # held apples earn an ongoing per-step holding reward (Phase 5b).
        # Decoupling these means raising held inventory (e.g., via raid or
        # gift) actually pays a net positive marginal stream, which is what
        # makes raid restraint a real strategic dilemma rather than a
        # structurally dead action.
        # ------------------------------------------------------------------
        cap = cfg.inventory_capacity
        for i in range(self.n_agents):
            if fresh_collected[i] > 0:
                rewards[i] += float(fresh_collected[i])

        # ------------------------------------------------------------------
        # Phase 5b — held-apple holding reward.
        #
        # Each apple in the agent's inventory at the start of the step pays
        # `held_apple_per_step_reward` reward this step.  Held inventory is a
        # future-reward stream worth stealing (raid) or transferring (gift).
        # ------------------------------------------------------------------
        if cfg.held_apple_per_step_reward > 0:
            for i in range(self.n_agents):
                held = int(held_at_start[i])
                if held > 0:
                    rewards[i] += cfg.held_apple_per_step_reward * float(held)

        # ------------------------------------------------------------------
        # Phase 6 — waste growth.
        # ------------------------------------------------------------------
        for q in range(NUM_CLANS):
            growth = cfg.wq_growth * (1.0 + cfg.wq_growth_per_apple * float(apples_collected_per_q[q]))
            self.w_q[q] = min(1.0, float(self.w_q[q]) + growth)
        total_collected = int(apples_collected_per_q.sum())
        self.w_p = min(1.0, self.w_p + cfg.wp_growth + cfg.wp_growth_per_apple * total_collected)

        # ------------------------------------------------------------------
        # Phase 7 — orchard regrowth (per-cell, gated by w_q).
        # ------------------------------------------------------------------
        agent_cells = set(
            (int(self.agent_pos[j, 0]), int(self.agent_pos[j, 1]))
            for j in range(self.n_agents)
        )
        for q, cells in enumerate(self.orchard_cells_per_q):
            p_regrow = max(0.0, cfg.apple_regrow_max_per_clan[q] * (1.0 - cfg.apple_regrow_slope * float(self.w_q[q])))
            if p_regrow <= 0:
                continue
            for r, c in cells:
                if self.orchard_apple[r, c]:
                    continue
                if (r, c) in agent_cells:
                    continue
                if self.rng.random() < p_regrow:
                    self.orchard_apple[r, c] = True

        # ------------------------------------------------------------------
        # Phase 8 — plaza bonus regrowth (gated by w_P ≤ bonus_regrow_threshold).
        # ------------------------------------------------------------------
        if self.w_p <= cfg.bonus_regrow_threshold:
            for r, c in self.plaza_cells:
                if self.bonus_apple[r, c]:
                    continue
                if (r, c) in agent_cells:
                    continue
                if self.rng.random() < cfg.bonus_regrow_prob:
                    self.bonus_apple[r, c] = True

        # ------------------------------------------------------------------
        # Phase 9 — termination, observations, info.
        # ------------------------------------------------------------------
        self._shared_bonus_total += shared_bonus_step
        done = self._step_count >= self.max_steps
        terminated = {i: done for i in range(self.n_agents)}
        truncated = {i: False for i in range(self.n_agents)}

        obs = {i: None for i in range(self.n_agents)}

        # Per-clan apple counts for feedback / profile signals.
        per_clan_inventory = np.zeros(NUM_CLANS, dtype=np.int32)
        for i in range(self.n_agents):
            per_clan_inventory[int(self.agent_clan[i])] += int(self.inventory[i])

        info: Dict[int, dict] = {}
        for i in range(self.n_agents):
            info[i] = {
                # Repurposed for the framework's run_episode loop, which
                # reads info[i]["timeout"] > 0 and forwards it as the per-
                # step bool used by compute_metrics.  Here it encodes
                # "agent attempted a raid this step" — see compute_metrics
                # for how it folds into the Peace metric.
                "timeout": int(raid_attempted_step[i]),
                "step": self._step_count,
                "clan": int(self.agent_clan[i]),
                "inventory": int(self.inventory[i]),
                "fresh_collected": int(fresh_collected[i]),
                "raid_attempted": int(raid_attempted_step[i]),
                "raid_succeeded": int(raid_succeeded_step[i]),
                "gift_given": int(gift_given_step[i]),
                "in_plaza": bool(self._in_plaza(int(self.agent_pos[i, 0]), int(self.agent_pos[i, 1]))),
                "current_quadrant": int(self._quadrant_of(int(self.agent_pos[i, 0]), int(self.agent_pos[i, 1]))),
                "w_q": tuple(float(x) for x in self.w_q),
                "w_p": float(self.w_p),
                "per_clan_inventory": tuple(int(x) for x in per_clan_inventory),
                "apples_collected_per_q": tuple(int(x) for x in apples_collected_per_q),
                "bonus_collected_step": int(bonus_collected_step),
                "shared_bonus_step": int(shared_bonus_step),
                "raid_attempts_total": int(self._raid_attempts_total),
                "raid_successes_total": int(self._raid_successes_total),
                "gift_total": int(self._gift_total),
                "shared_bonus_total": int(self._shared_bonus_total),
            }

        return obs, rewards, terminated, truncated, info

    # ------------------------------------------------------------------
    # Travel BFS helper
    # ------------------------------------------------------------------

    def _quadrant_bbox(self, q: int) -> Tuple[int, int, int, int]:
        """Return inclusive (r_lo, r_hi, c_lo, c_hi) of quadrant q."""
        half = self.config.grid_size // 2
        if q == CLAN_A:
            return (0, half - 1, 0, half - 1)
        if q == CLAN_B:
            return (0, half - 1, half, self.config.grid_size - 1)
        if q == CLAN_C:
            return (half, self.config.grid_size - 1, 0, half - 1)
        return (half, self.config.grid_size - 1, half, self.config.grid_size - 1)

    def _travel_plan(self, agent_id: int, target_q: int) -> List[int]:
        """Plan moves toward the nearest cell of `target_q`.

        Walls are absent in this env, so the shortest path is simply the
        Manhattan-step sequence to the closest in-quadrant cell (clipped
        coordinates).  We do NOT block on other agents — those may move,
        and at execution time a blocked move silently fails.

        Returns a list of MOVE_* action ints; empty if already in target.
        """
        if not (0 <= target_q < NUM_CLANS):
            return []
        r0, c0 = int(self.agent_pos[agent_id, 0]), int(self.agent_pos[agent_id, 1])
        if self._quadrant_of(r0, c0) == target_q:
            return []
        r_lo, r_hi, c_lo, c_hi = self._quadrant_bbox(target_q)
        tr = max(r_lo, min(r_hi, r0))
        tc = max(c_lo, min(c_hi, c0))
        moves: List[int] = []
        cr, cc = r0, c0
        # Vertical first, then horizontal — deterministic and grid-friendly.
        while cr != tr:
            if tr < cr:
                moves.append(int(Action.MOVE_N))
                cr -= 1
            else:
                moves.append(int(Action.MOVE_S))
                cr += 1
        while cc != tc:
            if tc < cc:
                moves.append(int(Action.MOVE_W))
                cc -= 1
            else:
                moves.append(int(Action.MOVE_E))
                cc += 1
        return moves

    # ------------------------------------------------------------------
    # Rendering (god's-eye debug view)
    # ------------------------------------------------------------------

    def render(self, cell_size: int = 12) -> np.ndarray:
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        COLOUR = {
            "background": np.array([28, 28, 32]),
            "river_clean": np.array([90, 160, 240]),
            "river_dirty": np.array([130, 80, 30]),
            "orchard_empty": np.array([40, 70, 30]),
            "orchard_apple": np.array([40, 200, 60]),
            "plaza_empty": np.array([90, 70, 130]),
            "plaza_bonus": np.array([240, 200, 80]),
            "agent_clan": [
                np.array([60, 110, 240]),     # A — blue
                np.array([220, 70, 70]),      # B — red
                np.array([70, 200, 130]),     # C — green
                np.array([240, 180, 70]),     # D — yellow
            ],
        }
        img[:] = COLOUR["background"]

        # Rivers.  Per-cell shade is binary by quadrant waste level.
        for cell, q in self._river_to_q.items():
            r, c = cell
            shade = COLOUR["river_dirty"] if self.w_q[q] >= 0.5 else COLOUR["river_clean"]
            img[r, c] = shade
        # Orchards.
        for cell in self.orchard_cells_list:
            r, c = cell
            img[r, c] = (COLOUR["orchard_apple"] if self.orchard_apple[r, c]
                         else COLOUR["orchard_empty"])
        # Plaza.
        for r, c in self.plaza_cells:
            img[r, c] = (COLOUR["plaza_bonus"] if self.bonus_apple[r, c]
                         else COLOUR["plaza_empty"])
        # Agents on top, coloured by clan; saturation = inventory level.
        for i in range(self.n_agents):
            r, c = int(self.agent_pos[i, 0]), int(self.agent_pos[i, 1])
            base = COLOUR["agent_clan"][int(self.agent_clan[i])]
            inv = int(self.inventory[i])
            scale = 0.6 + 0.4 * (inv / max(self.inventory_capacity, 1))
            img[r, c] = np.clip(base.astype(int) * scale, 0, 255).astype(np.uint8)

        if cell_size > 1:
            img = np.repeat(np.repeat(img, cell_size, axis=0), cell_size, axis=1)
        return img

    # ------------------------------------------------------------------
    # Social outcome metrics (Perolat et al., matched to existing envs)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_metrics(
        episode_rewards: Dict[int, List[float]],
        episode_timeouts: Dict[int, List[bool]],
    ) -> Dict[str, float]:
        """Per-episode social-outcome metrics.

        Note: in this environment, `episode_timeouts` carries per-step
        raid-attempt indicators (1 if agent attempted a RAID at step t,
        else 0) — there is no agent-tagout mechanism.  The Peace metric
        is therefore a raid-restraint analogue, computed in the same
        shape Cleanup uses for its tag-based version.
        """
        n = len(episode_rewards)
        T = len(next(iter(episode_rewards.values())))

        returns = {i: sum(episode_rewards[i]) for i in episode_rewards}
        R = np.array(list(returns.values()))

        # Efficiency — collective return per timestep (matches Cleanup).
        efficiency = float(R.sum() / T) if T > 0 else 0.0

        # Equality via Gini coefficient with negative-shift handling.
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

        # Peace — (n*T - total_raid_attempts) / T, same shape as Cleanup
        # so feedback templates that already display Peace work without
        # rescaling tweaks.
        total_raid_attempts = sum(
            sum(1 for v in vs if v) for vs in episode_timeouts.values()
        )
        peace = (
            float(n * T - total_raid_attempts) / T
            if T > 0 else float(n)
        )

        # Maximin (Rawlsian welfare).
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

def make_nested_commons(n_agents: int = 16, seed: Optional[int] = None,
                        **kwargs) -> NestedCommonsEnv:
    """Create a Nested Commons environment with default parameters.

    `n_agents` must be a multiple of 4 (one fourth per clan).  Any
    keyword args override the corresponding NestedCommonsConfig field;
    `seed` is forwarded to the env (not the config).
    """
    if n_agents % NUM_CLANS != 0:
        raise ValueError(
            f"n_agents must be a multiple of {NUM_CLANS} (one fourth per clan); "
            f"got {n_agents}"
        )
    cfg_kwargs = {"n_agents": n_agents}
    cfg_kwargs.update(kwargs)
    config = NestedCommonsConfig(**cfg_kwargs)
    return NestedCommonsEnv(config=config, seed=seed)


# ---------------------------------------------------------------------------
# Quick self-test / calibration demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Nested Commons smoke test ===")
    env = make_nested_commons(n_agents=16, seed=0)
    obs = env.reset(seed=42)
    print(f"Map size:             {env.height} x {env.width}")
    print(f"Agents:               {env.n_agents}  (clans {NUM_CLANS}, "
          f"{env.n_agents // NUM_CLANS} per clan)")
    print(f"Orchard cells:        {len(env.orchard_cells_list)}  "
          f"(seeded: {int(env.orchard_apple.sum())})")
    print(f"River cells:          {len(env.river_cells_list)}  "
          f"(per quadrant: {[len(c) for c in env.river_cells_per_q]})")
    print(f"Plaza cells:          {len(env.plaza_cells)}  "
          f"(seeded bonus: {int(env.bonus_apple.sum())})")
    print(f"Spawn cells:          {len(env.spawn_cells_list)}")
    print(f"Horizon:              {env.max_steps}")
    print(f"Num actions:          {env.action_space_n}")
    print()

    # ---- Random baseline ----
    ep_rewards = {i: [] for i in range(env.n_agents)}
    ep_timeouts = {i: [] for i in range(env.n_agents)}
    waste_q_log: List[float] = []
    waste_p_log: List[float] = []
    for step in range(env.max_steps):
        actions = {i: int(env.rng.integers(NUM_ACTIONS)) for i in range(env.n_agents)}
        obs, rewards, terminated, truncated, info = env.step(actions)
        for i in range(env.n_agents):
            ep_rewards[i].append(rewards[i])
            ep_timeouts[i].append(info[i]["timeout"] > 0)
        waste_q_log.append(float(np.mean(env.w_q)))
        waste_p_log.append(env.w_p)
    totals = {i: sum(ep_rewards[i]) for i in range(env.n_agents)}
    print("Random policy:")
    print(f"  Per-agent totals:   {[round(totals[i], 1) for i in range(env.n_agents)]}")
    print(f"  Mean per-agent:     {np.mean(list(totals.values())):.2f}")
    print(f"  Mean w_q (last):    {waste_q_log[-1]:.3f}")
    print(f"  w_P (last):         {waste_p_log[-1]:.3f}")
    print(f"  Raid attempts:      {env._raid_attempts_total} "
          f"(success {env._raid_successes_total})")
    print(f"  Gifts:              {env._gift_total}")
    print(f"  Shared bonuses:     {env._shared_bonus_total}")
    metrics = NestedCommonsEnv.compute_metrics(ep_rewards, ep_timeouts)
    print(f"  Metrics:            {metrics}")
    print()

    img = env.render(cell_size=18)
    try:
        from PIL import Image
        Image.fromarray(img).save("nested_commons_render.png")
        print("Saved render to nested_commons_render.png")
    except ImportError:
        print("PIL not installed; skipping image save.")
    print(f"Render shape:         {img.shape}")
    print("\nSmoke test complete.")
