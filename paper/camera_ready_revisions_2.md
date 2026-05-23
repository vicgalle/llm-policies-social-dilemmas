# Camera-Ready Revision Plan — NeurIPS 2026 Workshop

**Paper:** Discovering Cooperative Pipelines: Autoresearch for Sequential Social Dilemmas
**Constraints:** 3 days, no new experiments, 9 pages main text (before references).

Good news: both of the reviewer's substantive concerns are addressable *with framing changes only*. The reviewer explicitly says the centralized-control issue "doesn't invalidate the paper" and frames the inner-LLM-ablation as a way to "strengthen" the results — meaning honest acknowledgment + a tight theoretical argument is acceptable, even if the full ablation has to wait for a later revision.

## What needs to be addressed

1. **Centralized-policy reframing.** The reviewer is right that with one program controlling all N agents, the classical SSD "dilemma" structure is largely absent — what remains is a joint-optimization/coordination problem. The MARL allusion in the intro is the most exposed sentence.
2. **Why the outer loop, vs. just a stronger inner LLM?** The reviewer suspects the gains might come from Opus 4.6 doing the work, not from the two-level architecture per se. The matched ablation (Opus as both R and M, single-level) is for the next revision, but you can make the *theoretical* argument now from existing data.
3. **Extrapolation discussion** in limitations — small, easy.

---

## Proposed changes

### Change 1: Add a "framing note" in Section 2.1 (~8 lines)

After the social-metrics block, before §2.2, add:

> **A note on the dilemma's status under symmetric programmatic policies.** We adopt the SSD environments of [1, 5, 6] as benchmarks, but in our synthesis setup a single Python function π controls all N agents (§2.2). This reframes the strategic problem: the individual-rationality constraint that makes classical SSDs a *dilemma* is replaced by a joint coordination/scheduling problem with the welfare objective Φ as the explicit target. Locally myopic per-agent code can still recreate dilemma-shaped behavior (and the baseline pipeline in fact does), but cooperation here is a joint-optimization outcome, not an equilibrium under individual rationality. We interpret discovered mechanisms (duty rotation, role assignment) accordingly: they are coordination solutions in algorithm space that resemble the fairness mechanisms one would want a decentralized MARL system to converge to, not equilibria induced by self-interested agents.

This is the single most important addition — it directly answers Concern 1 and lets you keep the SSD benchmark naming honestly.

### Change 2: Light edit in Section 1 (intro)

The sentence "Standard multi-agent reinforcement learning (MARL) struggles in this regime due to credit assignment, non-stationarity, and large joint action spaces [2]" pre-commits you to a MARL framing that the reviewer flagged as misleading. Change to:

> Standard multi-agent reinforcement learning (MARL) struggles in this regime due to credit assignment, non-stationarity, and large joint action spaces [2]. A complementary approach, recently introduced by Gallego [3], **sidesteps this by replacing decentralized parameter-space optimization with centralized algorithm-space synthesis**: a frozen LLM writes a Python policy function …

Adding "decentralized" / "centralized" makes the contrast explicit instead of leaving it for the reader to discover later.

### Change 3: New paragraph in Section 6 — "Why the outer loop?" (~12 lines)

Insert after "Mechanism design in action" and before "Human oversight":

> **Why the outer loop, not just a stronger inner LLM?** A natural concern is whether the gains in Table 2 are attributable specifically to the two-level architecture, or simply to a stronger model (Opus 4.6) doing the work. The cleanest test — Opus 4.6 as *both* R and M in a single-level inner loop with all other components fixed — is the single most important follow-up and we flag it explicitly in Appendix A. Three observations from the present results, however, bound how much of the effect that ablation could absorb. First, the outer loop modifies components the inner loop cannot touch within a single generation: the helper library H, the feedback function ϕ, and the iteration logic ι are *inputs* to every inner call, and no single synthesis call rewrites the harness it runs under. Second, GEPA, run at matched environment-evaluation budget and with the same access to a strong LLM, optimizes the prompt only and still trails our method by 2–3× on Cleanup (Table 2) — capability alone with a single-component search does not close the gap. Third, the explicit fairness mechanism appears in 4/4 maximin runs and 0/4 efficiency runs despite an objective-agnostic researcher prompt pR (§4, Finding 4; Listing 1): this is an information-design move conditioned on Φ, with no within-inner-loop analogue. Taken together, these point to the architecture, not raw capability, as the primary driver — though we agree that empirically pinning down the decomposition is the obvious next step.

This (a) directly answers the reviewer's question, (b) reuses existing results (GEPA, Finding 4) rather than asking the reader to take it on faith, and (c) honestly concedes the ablation. The reviewer signaled this concession alone is sufficient to "strengthen the results."

### Change 4: Expand Appendix A (Limitations) — no length cost

Add two new items (keep them in the appendix; doesn't touch main text):

> **4. Centralized programmatic-policy framing.** We use SSDs as benchmarks, but our synthesis setup is symmetric in code: a single program controls all N agents (§2.1, framing note). The individual-rationality constraint that defines classical SSDs as strategic dilemmas is therefore absent, and the problem we are solving is closer to joint coordination/scheduling under a welfare objective Φ. Allusions to MARL, duty rotation as "fairness," etc. should be read in this lens: the discovered mechanisms are coordination solutions in algorithm space, not equilibria under self-interest. Whether the same framework can be productively applied to genuinely decentralized SSDs (asymmetric per-agent programs, partial observation, individual reward functions) is left for future work, and ties to the "asymmetric programs" direction in §6.
>
> **5. No matched-LLM inner-loop ablation.** All synthesizer LLMs in the main experiments (Gemini 3.1 Pro, Sonnet 4.6) are at or below the researcher (Opus 4.6) in coding capability, so the contribution of the two-level architecture and the contribution of researcher-LLM capability are partially entangled. The matched ablation — Opus 4.6 as both R and M in a single-level inner loop with all other components fixed — is the cleanest direct test of the architectural claim and the single most important empirical follow-up alongside (1). The arguments in §6 ("Why the outer loop?") bound but do not eliminate this concern.

Limitation (1) (R-LLM sweep) is already there — these slot in cleanly after it.

### Change 5: Extrapolation note (small)

The reviewer's limitations comment asks for a few sentences on extrapolation. Existing Appendix A.3 (gridworld scope) already does most of this — just add one closing sentence:

> Encouragingly, the outer-loop *operating conditions* (noisy multi-seed eval, code-level edits, bounded Φ-query budget) are not gridworld-specific, so we expect the framework to transfer wherever an inner-loop LLM pipeline produces evaluable artifacts under a scalar Φ; what is bounded by the gridworld benchmark is the *content* of the discovered mechanisms, not the discovery process itself.

---

## Length management

The two additions to main text are roughly 8 + 12 = 20 lines. To make room without losing substance, the cleanest trims are:

- **Section 4.1, "Common failure modes" paragraph** (~7 lines): the full taxonomy lives in App. B.1, so trim to 2–3 lines and point to the appendix.
- **Section 4.1, "Spec-gaming on the modifiable surface" paragraph** (~12 lines): the meat of this is reassurance for a careful reader; you can compress it to 4–5 lines stating the conclusion ("we inspected all final pipelines and found no reward-hacking patterns; ϕ-edits are thresholded interventions on the optimized metric, H-edits are spatial heuristics and state queries") and point to a new appendix subsection holding the current full text.

That covers the new prose with a small surplus.

---

## What to defer to the later revision

The reviewer's "really strong result, especially if you can figure out why" is the Opus-as-both-R-and-M experiment. That's a single new run per condition — feasible but not in 3 days alongside the rewrite. The current text changes acknowledge it explicitly as the priority follow-up, which is the standard and accepted move for camera-ready under this kind of review.

---

## Minor polish

The abstract currently opens with "two-level autoresearch for cooperation" — given the centralized framing now made explicit, consider "two-level autoresearch for **coordination**" or "for **cooperative-policy synthesis**." Tiny change, but it pre-empts a casual reader making exactly the assumption the reviewer made.

---

## Open questions / next steps

- [ ] Draft the exact LaTeX diffs against the source.
- [ ] Sketch the Opus-vs-Opus ablation design for the post-camera-ready arXiv revision.
- [ ] Decide on abstract wording: "cooperation" vs. "coordination" vs. "cooperative-policy synthesis".
- [ ] Confirm which paragraphs in §4.1 to trim to make room (failure modes / spec-gaming).