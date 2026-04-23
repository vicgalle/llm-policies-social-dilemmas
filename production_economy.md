# Production Economy: Environment Specification

## 1. Overview

**Production Economy** is a sequential social dilemma with *temporal depth* and *capital formation*. Agents gather raw resources, craft intermediate goods at workshops, and produce final goods at forges. Two classes of final goods exist: *tools* (private, fast-payoff, renewable) and *shelter pieces* (public, delayed, threshold-triggered). The strategic arc — when and how to pivot from private tool production to public shelter contribution before a terminal deadline — introduces a temporal-coordination problem absent from Cleanup and Gathering, where strategies can be stationary.

Design target: the environment should admit multiple distinct viable strategies (tool-specialists, builder-specialists, hybrid pipelines) whose relative value depends on the distribution of other agents' strategies. It tests whether the autoresearch system can discover time-varying coordination — role assignments that change across the episode — rather than the stationary role assignments that sufficed in Cleanup.

## 2. World

### 2.1 Grid

- A single 15×15 gridworld. Coordinates `(x, y) ∈ [0, 14]²`.
- The grid is partitioned into four functional regions, fixed at episode reset:
  - **Forest** (6 cells, fixed positions scattered through upper half): sources of `WOOD`.
  - **Quarry** (6 cells, fixed positions scattered through lower half): sources of `STONE`.
  - **Workshops** (4 cells in the middle band): 2 sawmills (convert wood→plank), 2 masonries (convert stone→brick).
  - **Forges** (2 cells, center): convert planks + bricks into tools or shelter pieces.
  - Remaining cells are **open space** for movement and item drops.
- Exact coordinates are fixed by a helper constructor and do not vary across episodes. The layout is designed so that every workshop is within ~4 Manhattan steps of at least one raw-resource cluster and within ~3 steps of a forge.

### 2.2 Resource nodes

- Each forest cell independently yields 1 wood when an agent steps onto it; the cell is then depleted and respawns stochastically with probability `0.05` per step.
- Each quarry cell behaves identically with stone.
- At reset, 80% of resource nodes are stocked.

## 3. Items and production

### 3.1 Item types

| Item | Tier | Mobility |
|---|---|---|
| `WOOD` | 0 (raw) | Carried in inventory; droppable. |
| `STONE` | 0 (raw) | Carried in inventory; droppable. |
| `PLANK` | 1 (intermediate) | Carried in inventory; droppable. |
| `BRICK` | 1 (intermediate) | Carried in inventory; droppable. |
| `TOOL` | 2 (final, private) | Equipped (occupies equipment slot, not inventory). Spoils after `T_spoil = 80` steps equipped. |
| `SHELTER_PIECE` | 2 (final, public) | Deposited directly to a global counter on creation; not carried. |

### 3.2 Conversion rules

- **Sawmill** (workshop type 1): while an agent is on or adjacent to a sawmill cell and holds ≥1 wood, one wood in inventory is consumed and one plank is produced, one unit per step. Parallel production across agents is allowed (a sawmill is not a single-agent facility).
- **Masonry** (workshop type 2): same rule, stone→brick, one per step.
- **Forge (tool recipe)**: while an agent stands on a forge cell and holds ≥2 planks + ≥1 brick and has an empty equipment slot, the forge consumes those inputs and equips the agent with a tool. One recipe per step per forge per agent.
- **Forge (shelter recipe)**: while an agent stands on a forge cell and holds ≥3 planks + ≥3 bricks, the forge consumes those inputs and increments the global `shelter_count` by 1. One recipe per step per forge per agent.
- Conversions are **opt-in**: the agent must execute the `CRAFT` action (see §4.2) to trigger them. Standing on a workshop with inputs does not automatically convert.

### 3.3 Item dropping

- Agents may drop any carried item onto their current cell via the `DROP(item)` action. Dropped items persist on the ground (up to 5 items per cell) and can be picked up by any agent via `PICKUP(item)`.
- Drop/pickup is the *only* mechanism for inter-agent handoff; there is no direct gift action. This makes spatial coordination and route planning necessary for any pipeline-style strategy.

## 4. Agents

### 4.1 Population

- `N = 8` agents running a shared policy (homogeneous self-play).
- Each agent has `agent_id ∈ [0, 7]`, an inventory of capacity 3 (total slots, any item mix), and one equipment slot for a tool.

### 4.2 Action space

| Action | Description |
|---|---|
| `NOOP` | Stay in place. |
| `MOVE_N`, `MOVE_S`, `MOVE_E`, `MOVE_W` | One-step movement; collisions resolved by lower `agent_id` priority. |
| `GATHER` | Collect the resource at the current cell (if a stocked forest or quarry). Consumes the cell's stock. |
| `CRAFT` | Trigger conversion at the current or adjacent workshop/forge. Fails silently if preconditions unmet. At a forge, the agent must specify the recipe: `CRAFT_TOOL` or `CRAFT_SHELTER`. |
| `DROP(item)` | Drop one unit of the specified item onto the current cell. |
| `PICKUP(item)` | Pick up one unit of the specified item from the current cell. |

Movement costs nothing; `GATHER`, `CRAFT`, `DROP`, `PICKUP` are free actions (no step cost beyond opportunity cost).

### 4.3 Equipped tool effect

- While an agent has a tool equipped, every `GATHER` action yields +2 units instead of +1, and the `GATHER` completes in 1 step rather than the normal 1 step (i.e., effectively doubles raw throughput). This is the private payoff of tool production.
- The tool pays `+2` reward per step it is equipped up to a cap of `T_spoil = 80` steps per tool. This is the agent's private reward from tool ownership.
- A tool spoils (disappears from the equipment slot) exactly `T_spoil` steps after being equipped. The agent may craft a new tool immediately afterward if resources permit.

### 4.4 Observations

Each agent observes:

- An 11×11 egocentric view with per-cell channels for: `is_forest`, `is_quarry`, `is_sawmill`, `is_masonry`, `is_forge`, `resource_present` (0 for depleted / 1 for stocked on resource cells), `dropped_items` (one channel per item type), `agent_present`, `agent_has_tool`.
- The agent's own `agent_id`, inventory contents (4-dim: wood/stone/plank/brick counts), `has_tool`, `tool_age` (steps since equipping, if applicable).
- Global scalars: `shelter_count`, `steps_until_winter`, total population `N`.
- Full environment state is accessible via an `env` handle (matching the existing framework's convention).

## 5. Episode structure

- Horizon `H = 300` steps.
- **Winter** occurs at step 200. At that tick, each agent receives:
  - `+50` if `shelter_count ≥ 6` (the public-goods threshold).
  - `−50` otherwise.
- The remaining 100 steps after winter are *endgame*: agents can still gather, craft, and accumulate tool-step rewards, but no further winter triggers. The endgame exists so that post-winter strategy (including whether to finish shelter-building for a future winter-like event) is non-trivial.
- **Reset state**: all resource nodes 80% stocked, no inventory, no equipped tools, `shelter_count = 0`, all agents at fixed spawn points near the edges of the grid.
- **Step order**: (1) apply all movements with collision resolution; (2) apply `GATHER` / `CRAFT` / `DROP` / `PICKUP` in `agent_id` order; (3) age all equipped tools, unequip any that have reached `T_spoil`; (4) pay `+2` per surviving tool; (5) trigger the winter event if `t == 200`; (6) regrow resource nodes; (7) emit observations.

## 6. Reward streams

Total agent reward `R_i` consists of four components, all summed into a single scalar per step as seen by the policy:

- **Tool payoff**: +2 per step per surviving equipped tool (up to its spoil limit).
- **Winter payoff**: ±50 once per episode at step 200, based on `shelter_count`.
- **No direct reward** for gathering, crafting intermediates, or contributing shelter pieces. All reward flows through the two channels above.
- **No action penalties** for gathering or crafting. Opportunity cost is the only pressure.

This reward sparsity is deliberate: it forces agents to plan over a 200-step horizon and to coordinate on the shelter threshold, which a greedy policy cannot achieve.

## 7. Social welfare metrics

Computed identically to the existing framework for consistency.

- **Efficiency** `U = (1/N) Σ_i R_i`.
- **Equality** `E` as defined in the framework (Gini-like, 1 − normalized pairwise difference).
- **Sustainability** `S = (1/N) Σ_i t_i` where `t_i` is the mean timestep of a reward-producing event (tool-step or winter tick) for agent `i`.
- **Peace**: this environment has no aggressive action; `P` is fixed at 1.0. It remains in the output vector for API compatibility but carries no signal. (An alternative is to drop `P` from the feedback template for this env; the researcher may choose to do so.)
- **Maximin** `min_i R_i`.

## 8. The dilemmas (reference, not part of the API)

For clarity on why the env tests temporal coordination:

1. **Pipeline-balance dilemma.** If too many agents gather, workshops idle and raw items pile up to the inventory cap and are wasted. If too many craft, workshops starve. Steady-state throughput requires a stable allocation across tiers, but with no central coordinator and only emergent norms the allocation must either be self-enforcing or learned by feedback.

2. **Private-vs-public investment.** Tools pay `+2` per step for up to 80 steps (max `+160` per tool, renewable). Shelter pieces pay `+50` each, once, only if the aggregate clears 6 units — effectively `+50/6 ≈ +8.3` per shelter piece divided among contributors plus every free-rider. A myopic agent always prefers tools. A cooperative welfare-maximizer pivots to shelter at some point. The pivot timing is the core strategic variable and has no stationary optimum.

3. **Handoff-logistics dilemma.** Inventory caps (3 items) force drops when agents accumulate mixed stockpiles or when the pipeline is unbalanced. *Where* to drop matters: drops near a workshop enable downstream specialists, drops mid-grid fragment the pipeline. Discovery of drop-zone conventions is a spatial-coordination mechanism with no counterpart in Cleanup.

## 9. Implementation contract

Drop-in contract for the autoresearch framework:

- Class `ProductionEconomyEnv` with `reset(seed) → observations`, `step(actions) → (observations, rewards, done, info)`, `state` property for full policy access.
- `info` dict per step includes: per-agent inventory, per-agent `has_tool` and `tool_age`, per-cell item counts for each item type, per-workshop production counts, `shelter_count`, `steps_until_winter`, `winter_triggered` flag on the relevant tick. These feed the feedback function.
- Configurable constants (`T_spoil`, winter threshold, winter reward, regrow probabilities, forge recipe ratios) live in `ProductionEconomyConfig` and are frozen for the researcher.
- Validation smoke test: a random policy completes without error; `shelter_count` at winter is typically 0, producing `R_i ≈ −50` for all agents; post-winter reward accrual is non-negative.

## 10. Calibration targets

Tuning should target the following hand-policy benchmarks:

- **Random policy**: `U ≈ −45` to `−50` per agent (winter failure dominates; minimal tool production).
- **All-tools policy** (every agent specializes in self-sufficient tool production, ignores shelter): `U ≈ 40` to `60`. Winter still fails (−50 each), but tool payoffs compensate partially.
- **All-shelter policy** (every agent contributes to shelter from step 0, no tools): `U ≈ 30` to `45`. Winter succeeds, but no tool payoff; low per-agent totals.
- **Mixed-pipeline policy** (a hand-designed mix: e.g., 2 dedicated gatherers, 2 dedicated crafters, 2 hybrid tool users, 2 dedicated builders pivoting at step ~150): `U ≈ 100` to `130`. Clear ceiling above either pure strategy.
- Variance between mixed-pipeline seeds should remain modest (≤15% of mean), so that observed improvement under autoresearch is attributable to mechanism discovery rather than seed luck.

These targets should be verified with hand-written reference policies before releasing the environment.

## 11. Connection to the autoresearch framework

Characteristic profile signals this environment exposes that are not present in Cleanup or Gathering:

- **Pipeline occupancy by tier**: fraction of agent-steps spent at gathering / workshops / forges, and the time-series of those fractions. Imbalance patterns diagnose missing coordination.
- **Temporal phase structure**: the autocorrelation of per-agent action type at various lags. A well-coordinated policy should exhibit a clear pivot signature (a regime change around step 150–200) that uncoordinated policies lack.
- **Inventory-cap loss rate**: count of items auto-discarded or items sitting in full inventories for multiple steps. Signals handoff-logistics failures.
- **Shelter-contribution Gini**: equality specifically over shelter contributions (orthogonal to overall reward equality).
- **Tool-age distribution**: tool renewal cadence across the population; stable cadence indicates a working steady-state.

These are natural outputs of the environment-agnostic profile extractor. None require the researcher to read environment source code.

## 12. Extension points (for later work)

- **Staggered winters**: multiple winter ticks at steps 200, 400, 600 with escalating shelter thresholds. Extends horizon to `H = 800` and tests recursive temporal planning.
- **Resource depletion**: forest/quarry regrowth rate decreases with total cumulative extraction, introducing genuine scarcity.
- **Heterogeneous forges**: different forges specialize in tool vs. shelter recipes, forcing spatial choice.
- **Weather shocks**: random perturbations to regrowth rates, testing robustness of discovered pipelines.

These are deliberately left out of the core spec to keep the first experimental round focused; they are natural follow-ups once the baseline env is saturated.
