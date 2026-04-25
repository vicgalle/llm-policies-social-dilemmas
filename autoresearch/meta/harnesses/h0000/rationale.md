# h0000 — weak greedy seed (production_economy)

Deliberately incompetent baseline:

- No phase routing (no awareness of the winter step at t=200).
- No shelter coordination (every agent treats CRAFT_TOOL as the goal).
- No tool maintenance (no pre-spoil kit-building).
- No role assignment (agents don't specialise wood vs stone).
- No pool sharing (no cap-aware drops, no pickup logic for tool inputs).

Expected efficiency: low single digits, possibly negative if the
winter penalty fires. This seed exists so the meta-harness search
has somewhere to climb from.

The headline experiment of PLAN.md Phase 2 is to verify that a
fully autonomous proposer (Claude Code + meta-harness tools) can
discover the strategies that h0001 uses, climbing from this weak
seed to ≥10 efficiency without human intervention.
