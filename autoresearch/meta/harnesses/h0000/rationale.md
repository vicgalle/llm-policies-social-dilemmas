# h0000 — weak greedy seed (production_economy)

Deliberately incompetent baseline:

- No phase routing (no awareness of any time-dependent structure in the env).
- No specialisation between agents.
- No coordinated handoff of intermediate goods.
- Greedy local choices only: walk → gather → craft → repeat.

Expected efficiency: low. Possibly negative if the env has time-gated
penalties that this baseline doesn't satisfy.

This seed exists so the meta-harness search has somewhere to climb from.
Read the env source (`production_economy_env.py`) to understand the
mechanics; propose harnesses that diagnose specific failure modes
of this baseline; iterate.
