# Improvement Plan: Profile-Guided Autoresearch for Production Economy

## Context and Diagnosis

The two-level autoresearch framework saturates at ~3.25 on Cleanup (≈65–80% of ceiling) but only reaches ~8.44 on Production Economy (≈50% of a ~16 ceiling). The gap is not a capability ceiling of the Opus 4.6 researcher. It is an **information-theoretic bottleneck** in the outer loop.

At each outer iteration the researcher observes roughly one scalar (Φ plus the social-metrics vector m). On Cleanup this was sufficient because the environment admits essentially one dominant coordination mechanism. On Production Economy at least three mechanisms must be discovered and composed simultaneously — pipeline balance, pivot timing, handoff logistics — and they all alias into a single Φ value. The researcher is reduced to guessing which mechanism is currently bottlenecking.

This is structurally the same problem compiler optimization faced before PGO: scalar runtime tells you the program is slow but not where. The fix is the same fix — enrich the observation.

## Primary Contribution: Profile-Guided Autoresearch (PGA)

### Change in one sentence

Replace the researcher's per-outer-iteration observation from `(diff, Φ, m)` with `(diff, Φ, m, P)`, where `P` is a structured execution profile computed by a fixed, environment-agnostic extractor.

### Profile contents

The extractor runs over the inner-loop trajectories and emits at minimum:

- **Action histograms over time-buckets.** Split the horizon into thirds (or configurable bins). For PE this surfaces "no agent ever invoked `CRAFT_SHELTER`" or "`CRAFT_TOOL` dominates throughout" — information that collapses into U but is not recoverable from it.
- **Per-agent reward timelines with change-point detection.** Standard CPD (e.g. binary segmentation over cumulative reward) applied to each agent's trajectory. Reveals whether the reward stream is stationary or has a detectable phase transition near step 150–200. A pivot strategy produces change points; a myopic strategy does not.
- **Inter-agent divergence by role.** Under homogeneous self-play, agents should diverge in *role* on PE (some gather, some craft) while sharing a policy. Measure: pairwise KL or variation distance between per-agent action distributions, plus the correlation of role with `agent_id`. Zero divergence → no specialization. High divergence correlated with `agent_id` rather than state → brittle static assignment.
- **AST-branch coverage of the synthesized policy.** Trace which policy branches fired and how often during inner-loop evaluation. Dead branches are silent bugs; hot branches that never produce reward are targeted feedback opportunities. Use `sys.settrace` or AST instrumentation; no environment knowledge required.
- **Precondition-failure and cap-saturation rates.** Count `PICKUP` attempts on empty cells, `CRAFT` attempts missing inputs, steps with full inventory, workshop-idle frames. These are PE-characteristic but the measurement is generic: "action-precondition-unmet rate" and "resource-cap hit rate" apply to any gridworld env.

All of these are computable without environment-specific code. The extractor is a fixed piece of infrastructure that would equally apply to Cleanup, Gathering, Nested Commons, or a future unrelated LLM pipeline.

### Consumption by the researcher

Prompt the researcher to reason **diagnostically** rather than optimistically: "given Φ went from 6.8 to 8.4 and the profile shows X, which mutation class should I try?" This reframes the outer loop from a one-dimensional Bayesian optimization into multi-signal attribution. With ~15 dimensions of structured profile, the researcher can actually attribute effects to causes.

### Falsifiable prediction

PGA closes at least half the PE gap (U from 8.4 to ≥12), and most of the closure comes from iterations where the profile revealed a specific mechanism the scalar signal had not flagged. A run-level ablation — scalar Φ vs. scalar + one profile channel vs. full profile — tells us which channels are load-bearing.

### Why this is the right headline for the paper

- Environment-agnostic: the extractor is fixed Python, not domain-specific.
- Compiler-grounded: PGO is a well-understood reference framework.
- Clean ablations available (channel-by-channel).
- Generality argument transfers: the same extractor drives autoresearch for any LLM pipeline whose inner loop produces traces, not just SSDs.

## Secondary Contribution: MH-Accepted Inner Loop (Variance Containment)

### Motivation

PGA raises the signal; the noise floor must also fall or the richer signal gets washed out. The original paper documents the mechanism explicitly: "identical configurations can yield U=3.2 then U=0.09." Full-policy regeneration per inner step is the culprit — one bad regeneration discards hard-won structure.

### Change

Replace each K inner-loop iteration with a STOKE-style local step:

1. Start inner loop from the current best π.
2. LLM proposes a **local mutation** m(π_{k−1}) targeted by the profile P (e.g. "the `CRAFT_SHELTER` branch is cold — propose mutations there").
3. Evaluate Φ(π_k) over a small seed set.
4. Accept with probability `min(1, exp(β (Φ_k − Φ_{k−1})))`, annealing β across the loop.

### Why this composes with PGA

Orthogonal effects: PGA makes each outer observation richer; MH makes each observation less noisy. The composition should also *shorten* the outer loop — the original paper's mean of 8.3 outer iterations reflects significant probe-and-discard. Diagnostic profiles plus variance reduction should need fewer probes. Worth tracking this as a secondary result.

## Experimental Plan

Three contrasts carry the paper, in this order:

### Experiment 1 — Matched-compute on PE

Scalar autoresearch (existing baseline) vs. PGA vs. PGA + MH. Same outer iteration count, same compute budget. Two policy LLMs (Gemini 3.1 Pro, Sonnet 4.6), two replications each.

Primary metric: final U and maximin. Secondary: fraction of outer iterations that produced a kept mutation (expected to rise under PGA). Target: PGA closes ≥50% of the 16 − 8.4 = 7.6 gap on PE.

### Experiment 2 — Profile-channel ablation

PGA with a progressively larger profile:

- Action histograms only.
- + change-point detection.
- + inter-agent divergence.
- + AST-branch coverage.
- Full profile.

One replication per condition, one policy LLM (Gemini) to keep cost manageable. Output: a bar chart of marginal contribution per channel. This is the scientific backbone of the paper — it tells us which observational dimensions actually drive the improvement, giving a principled guide for profile design in future environments.

### Experiment 3 — Transfer check

Run PGA-researcher on Cleanup, where scalar autoresearch already reached ~80% of ceiling. Two outcomes, both publishable:

- **If Cleanup performance improves or holds constant**: PGA is strictly better, dominates scalar autoresearch.
- **If Cleanup performance is unchanged**: the profile is specifically load-bearing for *temporally deep* dilemmas, which is the more interesting scientific finding. This separates "PGA is a general upgrade" from "PGA is the right tool for temporal depth specifically."

Either way the experiment produces a clean claim.

## Pre-Implementation Pilot (Strongly Recommended)

Before building the profile-extractor infrastructure, spend ~2 hours on a cheap pilot:

1. Pick one PE run that stalled at U ≈ 8.
2. Manually compute a profile over its inner-loop trajectories (action histogram, change-point detection, inventory saturation, a hand-read branch coverage).
3. Feed that profile into a one-off researcher prompt alongside the existing scalar signal.
4. Compare: does the researcher suggest qualitatively different modifications than scalar-only? If yes, the full PGA hypothesis is confirmed with almost no engineering. If no, reconsider the profile contents before building infrastructure.

This pilot de-risks the larger effort and takes an afternoon.

## Paper Narrative (Suggested Framing)

> On simple dilemmas (Cleanup, Gathering) scalar autoresearch suffices because a single coordination mechanism is sufficient for social welfare. On temporally deep dilemmas (Production Economy) scalar autoresearch hits an information-theoretic ceiling unrelated to researcher capability: multiple coordination mechanisms must be discovered and composed, and they alias into a single Φ value. Compiler-inspired observation enrichment (profile-guided autoresearch) closes the gap; MH-based variance reduction in the inner loop is a complementary technique that further reduces the outer-loop sample complexity.

This framing positions PGA as an architectural advance in LLM-driven automated research, not an SSD-specific trick.

## Deferred: Phase-Structured Policies (Future Work)

An additional contribution worth exploring but **not** front-loading into this paper: have the researcher synthesize not a single π but a tuple `(π_pre, π_pivot, π_post)` with learned transition points. This is the compiler analogue of loop unrolling or function extraction and is the natural representation for PE specifically.

It entangles policy representation with the autoresearch claim, which muddles the paper's central argument. Phase-structured policies are a stronger follow-up paper than a co-contribution to this one. Flag in future-work section only.

## Implementation Roadmap

1. **Profile extractor module** (`pipeline/profile.py`): fixed infrastructure, environment-agnostic. Functions: `action_histogram_over_buckets`, `change_point_detection`, `inter_agent_divergence`, `ast_branch_coverage`, `precondition_failure_rates`. Returns a structured `Profile` dataclass.
2. **Researcher prompt update**: add profile serialization to the outer-loop prompt, with natural-language descriptions of each channel. Update history format to include `P_j` alongside `(Δ_j, Φ_j, m_j)`.
3. **MH inner loop** (`pipeline/inner_loop_mh.py`): alternative inner-loop implementation with local mutation proposal and MH acceptance. Keep the original inner loop available as a baseline.
4. **Mutation-targeting hook**: a small component that reads the profile and includes targeted hints in the synthesizer's prompt. E.g. "the following branches fired 0 times in the last evaluation: [...]".
5. **Pilot run** as described above before steps 1–4.
6. Experiment 1, then 2, then 3, with the outputs going into the paper.

## Open Questions Worth Resolving Early

- **Profile size vs. context budget.** Serializing the full profile may consume 2–5k tokens. If this competes with the diff history or source code in the researcher's context, truncate the oldest profiles first and keep recent ones intact.
- **Change-point detection choice.** Binary segmentation is simple but sensitive to noise. Consider PELT or a simple windowed-variance heuristic; the exact choice probably matters less than having *some* detector, but this should be confirmed in the pilot.
- **MH temperature schedule.** Start with linear annealing β_k = β_0 · k/K; revisit if acceptance rates are pathological (too permissive → variance returns; too strict → no exploration).
- **Interaction with GEPA comparison.** The original paper includes a GEPA baseline. PGA should also be compared against GEPA on PE; expected result is that the gap widens further, because GEPA's scalar-only signal has the same information-theoretic limitation as baseline autoresearch.
