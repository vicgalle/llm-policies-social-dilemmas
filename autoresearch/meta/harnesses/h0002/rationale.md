# h0002 — M2 (scaffold-supplier) seed

A pure-code scaffold with documented hooks where the proposer can
introduce inference-time LLM calls. The seed itself runs without any
LLM at inference (`base_model=null`), which makes it a degenerate M2
that's structurally similar to an M3 — but the *shape* is different:
the scaffold separates "decision" hooks from "mechanics" code, so a
proposer can wire an LLM into a single decision point without
restructuring everything.

Mode: **M2**. The harness defines `harness/scaffold.py` exposing
`make_policy(base_model: str | None) -> Callable[[env, int], int]`.
The factory captures the base_model and any per-episode state, then
returns a closure that the meta-eval loop calls per agent per step.

Search axes the proposer can vary:
- **Where to put LLM calls** — turn one of the decision hooks into a
  `base_model`-gated callout. Track inference_token_cost in the
  manifest so the cost-Pareto axis becomes meaningful.
- **Phase / role logic** — the scaffold hardcodes a simple step-based
  phase plus agent-id parity for role assignment; the proposer can
  swap in env-state-conditional logic.
- **Mechanics** — the navigation, gathering, crafting micro-policies
  inside `_dispatch_action` are all replaceable.
- **Base model** — null (pure code) → `claude-haiku-4-5-20251001` →
  `gemini-3.1-pro-preview`, etc.

The seed's expected efficiency is low (it has no shelter coordination
and no tool-aging logic). It exists as a *clean canvas*, not as a
contender. The proposer should iterate.
