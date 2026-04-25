# HARN_RELATED — comparison with Meta-Harness (Lee 2026) and AutoHarness (Lou 2026)

**Date**: 2026-04-25
**Purpose**: positioning notes for the workshop paper. Maps our framework
onto the two prior works it adapts, identifies what's borrowed vs novel,
and flags the spots reviewers will probe.

## What each prior paper does (concretely)

### Meta-Harness — Lee, Nair, Zhang, Lee, Khattab, Finn (Stanford / MIT / KRAFTON, 2026)

- **Domains**: online text classification (LawBench, Symptom2Disease, USPTO-50k), RAG math reasoning (200 IMO problems), agentic coding (TerminalBench-2).
- **Harness structure**: each harness *always* wraps an LLM at runtime — the search optimizes prompts, retrieval, memory, orchestration logic. There is no "no-LLM-at-runtime" mode.
- **Proposer**: Claude Code with Opus 4.6.
- **Outer loop**: population $\mathcal{H}$ + Pareto frontier + filesystem feedback channel; **proposer picks parent freely (no parent-selection rule)**, sequential.
- **Scale**: ~60 harnesses over 20 iterations; 2 candidates per iteration; ~82 files read per proposer iteration; up to ~10M tokens of feedback per evaluation (3 OOM beyond OPRO/TextGrad/GEPA/TTT-Discover).
- **Headline results**: 7.7-point improvement on text classification with 4× fewer context tokens; #1 among Haiku-4.5 agents on TerminalBench-2; matches OpenEvolve / TTT-Discover at 0.1× evaluations.
- **"Money plot"**: information ablation (their Table 3) showing scores-only < scores+summary < full-trace access.
- **Base model M is always frozen** per domain.

### AutoHarness — Lou, Lázaro-Gredilla, Dedieu, Wendelken, Lehrach, Murphy (DeepMind, ICLR RSI Workshop 2026)

- **Domain**: TextArena (145 games; eval on 16 1-player + 16 2-player turn-based games). **Never simultaneous-move; never symmetric self-play.**
- **Three explicit modes**:
  - *harness-as-action-filter*: LLM proposes a set of legal moves, harness ranks them with chain-of-thought.
  - *harness-as-action-verifier*: LLM proposes an action; harness verifies via `is_legal_action()`; refines if invalid.
  - *harness-as-policy*: pure code; no LLM at runtime.
- **Search algorithm**: tree search with Thompson sampling (Tang 2024). Critic → Refiner → Evaluator loop. Heuristic = average legal-move accuracy + reward.
- **One harness per game**, trained with Gemini-2.5-Flash; eval against Gemini-2.5-Pro and GPT-5.2-High.
- **Scale**: average 89 iterations for harness-as-policy training (max 256).
- **Headline results**:
  - Action-verifier with smaller LLM beats Gemini-2.5-Pro alone (56% win vs 38%).
  - Pure-code harness-as-policy beats GPT-5.2-High average reward across 16 1P games (0.870 vs 0.844).

## Direct comparison — eleven axes

| Axis | Lee (Meta-Harness) | Lou (AutoHarness) | Ours |
|---|---|---|---|
| **Setting** | Single-agent benchmarks | 1P / turn-based 2P games | **Simultaneous-move multi-agent SSDs, symmetric self-play (N agents share policy)** |
| **Harness modes** | One: always LLM-at-runtime | Three: filter / verifier / policy | Three (M1/M2/M3) — only M3 implemented in run1 |
| **LLM at decision-time** | Always | Yes (filter, verifier) or No (policy) | M3=no, M2=optional, M1=no (synthesised once) |
| **Search algorithm** | Population + Pareto, sequential, no parent rule | Tree + Thompson sampling | Population + Pareto, sequential, no parent rule |
| **Diagnostic channel** | Filesystem, ~82 files/iter, ~10M tokens/eval | Critic-fed failure traces (smaller) | Filesystem, per-step JSONL; smaller scale than Lee |
| **Information ablation** | **Yes — their headline figure** | No | **Not yet — gap** |
| **Reward signal** | Scalar (accuracy / pass-rate) | Scalar (legal moves + reward) | **Multi-objective: 5 SSD metrics (Perolat 2017)** |
| **Pareto axes** | {accuracy, context tokens} | Implicit {accuracy, model size} | {efficiency, maximin, equality, synthesis cost, inference cost} |
| **Base-model variation** | Frozen per domain | Varied (Flash train, Pro/GPT-5.2 eval) | Not varied (Opus proposer + M3-only policy) |
| **Information hygiene** | Implicit | Implicit | **Explicit off-limits list** (PLAN.md, prior solutions, prior pipeline files) |
| **Headline result** | Beats hand-engineered TerminalBench agents | Smaller-LLM + harness ≥ larger-LLM alone | M3 search exceeds hand-crafted human reference; reveals env miscalibration |

## Where we genuinely innovate (defensible against either paper)

1. **Symmetric self-play setting.** Neither prior paper has it. AutoHarness explicitly excludes simultaneous-move games and notes that strategic 2P would need world-model-MCTS techniques. The N-agent-sharing-one-policy constraint creates a distinct optimisation landscape that is unique to our setting.
2. **Multi-objective Pareto with social-dilemma metrics.** Lee has Pareto over {accuracy, context cost}; Lou is single-objective per game. Ours brings the five Perolat (2017) social-dilemma metrics into the meta-harness frame for the first time.
3. **The env-calibration-probe finding.** Neither prior paper produces a finding *about the environment*. Both produce findings about the search method. Our h0048 result that "production_economy's social-dilemma calibration rewards defection" is qualitatively different from "our method beats the baseline" — and is uniquely available because the multi-agent setting *has* a calibration question to discover.
4. **Information-hygiene formalism.** Neither paper writes down "the proposer cannot read the human reference solution". We do, both as a design rule (program.md) and via path-whitelist guards (tools.py). Reviewers in cooperative AI / safety venues will care.
5. **Sharper M1/M2/M3 axis than Lou's three modes.** Lou's three modes mix two orthogonal axes (LLM at runtime? code-or-text artifact?). Our taxonomy factorises them cleanly: M1 = LLM-at-synthesis only, M2 = LLM-at-runtime for high-level decisions, M3 = no LLM at all. This is a contribution to the harness-mode taxonomy.

## Where we directly follow Lee (cite explicitly)

- **Filesystem-as-context.** Our `autoresearch/meta/` tree mirrors Lee's setup; same architecture.
- **Coding-agent proposer (Claude Code + Opus).** Identical proposer choice.
- **Population + Pareto frontier with no parent-selection rule.** Same.
- **Per-iteration scale.** ~50 harnesses in 48 iterations vs Lee's ~60 in 20. Comparable.

## Where we follow Lou (cite explicitly)

- **The mode taxonomy** (filter / verifier / policy → M1 / M2 / M3). Borrowed and renamed.
- **The "no LLM at runtime" extreme.** Lou's harness-as-policy = our M3.
- **The cost-Pareto framing.** Lou's "smaller model + harness beats larger model" claim depends on the cost axis; ours has the same axis (`neg_inference_token_cost`).

## Where we *diverge* — things reviewers will probe

1. **No tree search / Thompson sampling.** Lou's search algorithm is a contribution; ours is plain sequential. *Defense*: Lee is also sequential and beats baselines. *Fix*: not needed for workshop.
2. **Our M1 ≠ Lou's action-filter.** Lou's filter has an LLM at every decision step; our M1 has an LLM only at policy synthesis time, then runs as cached code. Distinct point on the {when does LLM run?} × {what does it produce?} grid. *Fix*: be explicit in the paper to avoid conflation.
3. **Information channel scale.** Lee writes ~10M tokens / eval; ours is much smaller (single env, ~5–20 seeds, traces ~3 MB). *Defense*: scale is set by env complexity; the channel is *used* (the proposer cited specific seed/cell evidence). Show this with quotes from `proposer_log.md`.
4. **No information ablation yet.** This is Lee's headline justification. Without it, our claim "the trace channel matters" is asserted, not measured. **High-priority fix**: re-run the same env with scores-only feedback and show the gap (HARN_RUN4 in our roadmap).
5. **Single base-model variation.** Lou's contribution depends on varying the base model. We've only run Opus. **Easy fix**: one HARN_RUN with Sonnet proposer (HARN_RUN5).
6. **No formal qualitative evaluation of discovered policies.** Lee has appendix-level qualitative analysis. Ours is `proposer_log.md`. Fine for workshop; marginal for main conference.

## Implications for the paper

- **Lead with what's novel** (multi-agent self-play, social-dilemma multi-objective Pareto, calibration finding) rather than method-level novelty (where we don't have much).
- **Cite Lee as the methodological backbone** explicitly. Position the contribution as: *Meta-Harness applied to a setting with [symmetric self-play] reveals [calibration phenomenon not observable in single-agent settings]*. This rewards the multi-agent novelty without overclaiming on method.
- **Match Lee's information ablation** in HARN_RUN4. It's the strongest single experiment justifying the trace-rich design — and is directly comparable to their Table 3.
- **Borrow Lou's "smaller-model + harness beats larger-model"** experiment design with HARN_RUN5 (Sonnet vs Opus proposer, or a Sonnet base model in M2 mode). Even a partial result is informative.
- **Don't claim a new search algorithm.** It's basically Lee's. Claim what we own: the setting, the multi-objective metric set, the calibration finding, the information-hygiene rule.

## Cleanest one-paragraph positioning (paper-ready draft)

> Meta-Harness (Lee et al. 2026) and AutoHarness (Lou et al. 2026)
> demonstrate that automated search over harness code outperforms
> hand-crafted alternatives in single-agent benchmarks and turn-based
> games respectively. We adapt the framework to multi-agent symmetric
> self-play in social-dilemma environments, where the harness must
> produce identical behaviour for all $N$ agents simultaneously. The
> setting unlocks a class of finding inaccessible in prior work: not
> whether the search beats a baseline, but **whether the environment's
> intended cooperation tension actually presents a tension under
> self-play optimization**. On a published-style production-economy
> game, the autonomous proposer climbs to a strategy that exceeds the
> human-crafted reference (efficiency 10.84 vs 10.33) by **defecting
> on the public good** — a finding the env's authors had not
> anticipated. We frame meta-harness search as a calibration probe
> for SSD designers, not just a cooperation generator.

## Pointer table for the paper's related-work section

| Claim | Cite |
|---|---|
| Filesystem-as-context for code-search proposers | Lee et al. 2026 (Meta-Harness) |
| Information-rich feedback > scalar feedback | Lee et al. 2026, §4.1 + Table 3 |
| Population + Pareto frontier + no parent rule | Lee et al. 2026, §3 |
| Code-as-policy; no LLM at runtime | Lou et al. 2026; Liang et al. 2023 (code-as-policies) |
| Smaller-LLM + harness ≥ larger-LLM alone | Lou et al. 2026, §4.2 |
| Three harness modes (filter / verifier / policy) | Lou et al. 2026, §3 |
| Tree search + Thompson sampling for code refinement | Tang et al. 2024 (cited by Lou); not used by us |
| Five SSD metrics (efficiency, equality, sustainability, peace, maximin) | Perolat et al. 2017 |
| Hand-crafted SSD policy synthesis baseline | Gallego 2026 (current PGA paper) |

This file lives at the repo root alongside `HARN_RUN1.md`. Update it as
the paper draft evolves and as new experiments add to the comparison.
