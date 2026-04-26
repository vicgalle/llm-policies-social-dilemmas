# Nested Commons: Environment Specification

## 1. Overview

**Nested Commons** is a sequential social dilemma (SSD) designed to test *compositional mechanism design*. Unlike Cleanup or Gathering, which each pose a single dilemma, Nested Commons embeds two public-goods provision problems at different scales — a local one within each of four *clans*, and a global one across all clans at a shared *plaza* — together with an inter-agent predation dynamic. No single coordination mechanism (role rotation, zone partitioning, reciprocity) suffices; the environment admits high social welfare only when multiple mechanisms are composed correctly.

Design target: the environment should be solvable in principle (all-cooperation has strictly higher social welfare than all-defection) but require discovery of at least three interacting mechanisms by any policy-synthesis system. It is a drop-in replacement for Cleanup in the existing framework — same API shape, same homogeneous self-play, same metric contract.

## 2. World

### 2.1 Grid

- A single 20×20 gridworld, divided into four 10×10 **quadrants** plus a central 4×4 **plaza** superimposed on the inner region.
- Coordinates: `(x, y)` with `x, y ∈ [0, 19]`.
- Quadrants:
  - **A (NW):** `x ∈ [0,9], y ∈ [0,9]`
  - **B (NE):** `x ∈ [10,19], y ∈ [0,9]`
  - **C (SW):** `x ∈ [0,9], y ∈ [10,19]`
  - **D (SE):** `x ∈ [10,19], y ∈ [10,19]`
- **Plaza:** `x ∈ [8,11], y ∈ [8,11]` — a 4×4 region spanning all four quadrant corners. Cells inside the plaza still belong to a quadrant for clan-home purposes but have special resource rules (see §3).

### 2.2 Static terrain per quadrant

Each quadrant contains:

- A **river**: a 1-cell-wide strip of 8 cells along the outer edge of the quadrant (the edge furthest from the plaza). Waste accumulates here.
- An **orchard**: a 5×5 region on the quadrant's plaza-facing side where apples grow.
- **Open space**: the remainder.
- A fixed set of **spawn points**, one per agent (4 spawn points per quadrant).

Terrain is deterministic per episode; exact coordinates are fixed by a helper constructor and do not vary across episodes.

### 2.3 Plaza

The plaza hosts a **shared forest** of 16 cells (filling the entire 4×4). Each cell independently produces *bonus apples* when the plaza's waste level is below a threshold (see §3.3).

## 3. Resources and dynamics

### 3.1 Quadrant waste (local)

- Each quadrant has a waste level `w_q ∈ [0, 1]`.
- Per-step growth: `w_q ← min(1, w_q + 0.005)` (baseline accumulation rate).
- Growth is multiplied by `(1 + 0.02 × apples_collected_in_quadrant_this_step)`, capturing the pollution-via-activity link.
- Cleaning: an agent executing `CLEAN` on a cell adjacent to the river reduces `w_q` by 0.025 at a cost of `−1` reward to the acting agent (same shape as base Cleanup).

### 3.2 Quadrant apple regrowth (local)

- An apple in the quadrant's orchard regrows each step with probability `p_regrow(w_q) = max(0, 0.1 × (1 − 2 w_q))`.
- Apples are collected by moving onto an apple cell; yields `+1` to the collector.

### 3.3 Plaza waste (global)

- Plaza waste level `w_P ∈ [0, 1]`.
- Per-step growth: `w_P ← min(1, w_P + 0.002 + 0.001 × total_apples_collected_this_step_across_all_quadrants)`.
- Cleaning the plaza: an agent in any plaza cell executing `CLEAN` reduces `w_P` by 0.02 at cost `−1`. (Note: plaza cells do not border a river — cleaning is in-situ.)

### 3.4 Plaza bonus forest (global)

- When `w_P ≤ 0.35`, each plaza cell with a bonus apple yields `+2` to an agent standing on it *and* pays a simultaneous `+2` to every other agent in the episode (the shared bonus).
- Bonus apples regrow per-cell with probability `0.08` per step, conditional on `w_P ≤ 0.5` (they stop regrowing entirely above 0.5).
- When `w_P > 0.35`, existing bonus apples can still be collected by the standing agent (`+2`) but no shared-bonus payment is issued.
- This asymmetry (collection always pays the collector; shared bonus is threshold-gated) is the plaza's social dilemma.

## 4. Agents

### 4.1 Population

- `N = 16` agents, 4 per clan, clans labeled `A, B, C, D`.
- Each agent has an immutable `agent_id ∈ [0, 15]` and an immutable `clan_id ∈ {A, B, C, D}`. By convention: agents 0–3 are clan A, 4–7 clan B, 8–11 clan C, 12–15 clan D.
- Agents run a single shared policy (homogeneous self-play), consistent with the existing framework.

### 4.2 Action space

All actions are discrete; one action per agent per step.

| Action | Description |
|---|---|
| `NOOP` | Stay in place. |
| `MOVE_N`, `MOVE_S`, `MOVE_E`, `MOVE_W` | Attempt one-step movement. Blocked by other agents and out-of-bounds. |
| `CLEAN` | Clean the river cell adjacent to the agent (if any), or the plaza cell underneath (if in plaza). Costs `−1`. |
| `RAID(target_agent_id)` | Attempt to steal one apple from a target agent, valid only if the target is in an adjacent cell, is a member of a different clan, and holds at least one apple. Succeeds with probability `0.6`. On success, the target loses 1 apple and the raider gains 1 apple (apples here means unconsumed apples held in inventory — see §4.3). Costs `−0.5` to the raider regardless of outcome. If the target is the same clan, action resolves as `NOOP`. |
| `GIFT(target_agent_id)` | Transfer 1 apple from the agent's inventory to an adjacent agent of any clan. Free action. Requires holding ≥1 apple. |
| `TRAVEL(quadrant)` | A multi-step routing action. Commits the agent to a path toward the nearest cell of the named quadrant; during travel (lasting 3 steps), other actions are ignored. Cost: `−0.1` per travel step. Implemented as a queued sequence of `MOVE` commands computed by an internal BFS; policies can treat this as a high-level primitive. |

### 4.3 Inventory

- Each agent has an apple **inventory** with capacity 3.
- Apples collected from quadrant orchards go to inventory first; the inventory must be decremented by `EAT` for the reward to register.
- To keep the action space small, *apples auto-eat* at the end of each step: any apples exceeding capacity are discarded and do not pay reward. Agents receive reward `+1` per apple auto-eaten up to their capacity-minus-held-at-start, preventing free-hoarding exploits.
- Plaza bonus apples are *immediately consumed* on collection and pay `+2` directly to the collector (plus `+2` to all other agents if `w_P ≤ 0.35`); they never enter inventory.
- Apples held in inventory can be stolen via `RAID` or transferred via `GIFT`.

### 4.4 Observations

Each agent observes:

- An 11×11 egocentric view (zero-padded at boundaries) with per-cell channels for: `apple`, `bonus_apple`, `river_cell`, `plaza_cell`, `agent_present`, `clan_marker` (integer 0–3 for the clan of any agent in that cell, −1 otherwise), `waste_indicator` (binned).
- The agent's own `agent_id`, `clan_id`, inventory count, current quadrant.
- A public scalar per quadrant: `w_A, w_B, w_C, w_D, w_P`.
- The global step counter `t`.
- Full environment state is also accessible to policies via an `env` handle, following the existing framework's convention (same as Cleanup).

## 5. Episode structure

- Horizon `H = 1000` steps (matches Cleanup).
- Reset: all waste levels `w_q = 0.1`, `w_P = 0.1`; 40% of orchard cells pre-seeded with apples; 50% of plaza cells pre-seeded with bonus apples; inventories empty; agents at designated spawn points.
- Step order each tick: (1) resolve all agent actions simultaneously, with conflicts (two agents moving into the same cell) broken by lower `agent_id` priority; (2) apply raids and gifts; (3) auto-eat inventory to capacity; (4) update waste levels; (5) regrow apples and bonus apples; (6) emit observations and rewards.

## 6. Social welfare metrics

All four metrics from [Perolat et al. 2017] are computed per-episode, per-run, using the same formulas the existing framework uses for Cleanup and Gathering. Let `R_i` be agent `i`'s total undiscounted return.

- **Efficiency** `U = (1/N) Σ_i R_i`.
- **Equality** `E = 1 − (Σ_i Σ_j |R_i − R_j|) / (2 N Σ_i R_i)`, clipped to `[0, 1]` with the usual degenerate-case handling.
- **Sustainability** `S = (1/N) Σ_i t_i`, where `t_i` is the mean timestep at which agent `i` collects a reward-producing apple.
- **Peace** `P = (H − total_raid_attempts) / H` (replaces Cleanup's tag-based peace metric with a direct raid count; total raid attempts across all agents divided by horizon, then normalized).
- **Maximin** `min_i R_i`, for Rawlsian objectives `Φ_min`.

## 7. The three dilemmas (reference, not part of the API)

For clarity on why the env resists single-mechanism solutions:

1. **Intra-clan provision.** Within each quadrant, cleaning the river is a standard public-goods problem identical in structure to Cleanup. A free-riding clanmate can consume apples the cleaner helped grow. Solvable by rotation, role assignment, or zone partitioning within the clan.

2. **Inter-clan provision (plaza).** The plaza is a second public good, now across clans. An agent cleaning the plaza pays the cost alone but benefits all 16 agents. Free-riding is possible at the *clan* level (a whole clan ignoring the plaza while others maintain it) and at the *agent* level (individuals within a contributing clan skipping plaza duty). Note the two publics interact: cleaning the plaza requires travel, so an agent on plaza duty cannot simultaneously clean their home river.

3. **Raid restraint / deterrence.** Raid is individually rational against a lone high-inventory target in another clan; the equilibrium collapses if unchecked. Stable outcomes require either mutual restraint (norm), deterrence (retaliation), or structural avoidance (inventory-keeping protocols). Raid also affects equality: high-raid equilibria produce high-variance returns even at the same mean.

The three dilemmas trade off: an agent optimizing local cleaning cannot simultaneously contribute to the plaza; a clan optimizing the plaza weakens its raid-deterrence posture. No primitive discovered on Cleanup alone covers all three.

## 8. Implementation contract

To drop in as an inner-loop environment for the existing autoresearch framework:

- Expose a class `NestedCommonsEnv` with `reset(seed) → observations`, `step(actions) → (observations, rewards, done, info)`, and a `state` property exposing full grid state for policy access (matching Cleanup's interface).
- `info` dict per step includes: per-quadrant waste, plaza waste, per-clan apple counts, per-agent inventory, raid/gift counts, shared-bonus activations. These feed the feedback function `φ` the researcher modifies.
- Configurable constants (waste growth rate, raid success probability, plaza threshold) live in a `NestedCommonsConfig` dataclass. The researcher must not modify the config; only the pipeline surrounding it.
- Validation smoke test: 1 random-policy episode completes without raising; all metric signs match baseline expectation (random gets `U < 0`, `E ≈ 0.3`, `min_i R_i < 0`).

## 9. Calibration targets

The environment should be tuned so that:

- Random policy: `U ∈ [−3, −1]`, significantly worse than any coordinated policy.
- "Pure local" policy (each agent cleans its home river when waste is high, ignores plaza, never raids): `U ∈ [0.5, 1.5]`. Baseline competent behavior without inter-clan coordination.
- "Ideal" policy (rotate local cleaning, schedule plaza visits, no raids): target `U ≈ 4.0–5.0`. Ceiling should exceed Cleanup's (`U ≈ 3.2`) to give the autoresearch system room.
- The `U` gap between "pure local" and "ideal" should exceed 2.5, ensuring inter-clan coordination is the dominant design question rather than intra-clan tuning.

These targets should be verified with a hand-written reference policy before releasing the environment for autoresearch experiments, and the configurable constants tuned to hit them.

## 10. Connection to the autoresearch framework

The researcher agent `ℛ` running on this environment should expose characteristically new profile signals not present in Cleanup or Gathering:

- **Inter-clan divergence**: variance in returns across clans vs. within clans. Used in intra- vs. inter-clan mechanism diagnosis.
- **Plaza-occupancy time**: fraction of steps with at least one agent in the plaza; decomposed by clan.
- **Raid-retaliation autocorrelation**: conditional probability of a raid in step `t+k` given a raid in step `t`, by clan-pair. Signals cycle stability.
- **Cost-allocation asymmetry**: per-clan cleaning-cost totals; used to detect free-rider clans.

These are natural outputs of the environment-agnostic profile extractor and should be computable without environment-specific code by the researcher.
