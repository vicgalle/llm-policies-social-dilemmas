# h0001 — hand-crafted seed (production_economy)

Seed harness. The policy is a verbatim copy of the repo's hand-crafted
`production_economy_policy.py` (Claude Opus + ~50 interactive iterations).
On 5 seeds (the default eval block) it averages **efficiency ≈ 10.27**.

This is the upper-anchor seed for the meta-harness search on
production_economy: any proposer-authored M3 harness should match or
exceed it. Together with the weak seed `h0000`, it bookends the
search space.

## Strategy summary

Phase-routed self-play with three mechanisms doing the heavy lifting:

1. **Pre-winter shelter rush**, anchored by agent 7 — seven of eight
   agents do a tool-first opening, agent 7 skips that and feeds the brick
   pool from step 0 so the rush can start immediately.
2. **Cap-aware pool drops** during the steady-state tool cycle —
   tooled agents drop intermediates only when the per-type pool slot is
   below 3 *and* the cell still has drop room (cap 5). Without these
   guards the cell-cap silently fails drops and tooled agents jam on
   full forge cells, blocking tool-less peers.
3. **Pre-spoil personal kit** — when tool age ≥ 60 (20 steps before
   spoil), the tooled agent stops dropping into the pool and instead
   accumulates 2p+1b in inv via cross-role gathering, then parks
   adjacent to a forge for an instant re-craft post-spoil.

See the policy docstring for full details.

## Why this seeds the search

The handcraft demonstrates the *upper bound* the harness-as-policy mode
(M3) can reach with full proposer access to env source + diagnostic
profile. The meta-harness experiment is then: can a fully autonomous
proposer recover 10+ from a weaker seed without human help?
