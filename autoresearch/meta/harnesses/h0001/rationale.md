# h0001 — M1 (feedback-shaper) seed

A minimal feedback-shaper harness. The proposer can fork this to vary
the system prompt, the per-iteration feedback construction, the helper
namespace exposed to the synthesizer, and the iteration budget /
eval-seed count.

Mode: **M1**. The harness defines a four-file pipeline
(`harness/pipeline/{prompts,feedback,helpers,config}.py`). At spawn
time, the framework runs the inner-loop synthesis (K iterations of
LLM policy generation + self-play eval) using `base_model` and caches
the final-iteration policy code at `harness/synthesized_policy.py`.
The cached policy is what gets evaluated in the meta-harness loop.

This seed's pipeline is deliberately bare — it lets the synthesizer
LLM see only the env description and the previous iteration's reward.
The proposer should diagnose what the synthesizer needs (better
worked examples? a profiling channel? per-agent stats? a different
iteration cadence?) and spawn variants that test those hypotheses.

Search axes the proposer can vary:
- **Prompt content** in `prompts.py`
- **Feedback richness** in `feedback.py`
- **Helper namespace** in `helpers.py` (e.g. add a BFS implementation
  the synthesizer can call)
- **Iteration budget K** in `config.py`
- **Base model** via the manifest (`claude-sonnet-4-6` →
  `gemini-3.1-pro-preview` is a single edit)
