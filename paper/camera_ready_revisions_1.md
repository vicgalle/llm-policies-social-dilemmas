# Camera-ready revision plan

**Paper:** Discovering Cooperative Pipelines: Autoresearch for Sequential Social Dilemmas
**Venue:** ACM CAIS 2026 Workshop on AI Agents for Discovery in the Wild
**Constraint:** 3 days, no new experiments
**Reviewer rating:** 6 (weak accept), confidence 4

## Context: the reviewer's main misreading

The reviewer's most damaging concern (Finding 4 being "really at the researcher level") rests on a misreading: they treated Listing 3 as the *researcher's* prompt, when it's the *researcher-authored* synthesizer prompt — exactly the artifact you're claiming the researcher discovers.

The actual R-prompt (`program.md`) mentions "role assignment" and "Voronoi partitioning" as a category of strategies and even suggests a static `assign_role(env, agent_id)` helper, but it contains:

- no time-rotation
- no `step_count // T`
- no objective-conditional differentiation
- no hint that rotation should be a fairness mechanism specifically for maximin

So the convergent-rediscovery claim survives — but only if the paper makes the two-level prompt hierarchy unmistakable. **That's the single highest-leverage edit for the camera-ready.**

---

## Priority 1: Defuse the Finding 4 misreading

### 1a. Add R's system prompt to the appendix

Add a new subsection **B.0 ("Researcher system prompt $p_\mathcal{R}$")** before B.1. Reproduce `program.md` verbatim or a clearly-marked excerpt.

Then add a one-paragraph framing pointing out exactly what's *not* there:

- no rotation formula
- no objective-conditional guidance
- no "rotation is for fairness"

This single addition turns the reviewer's strongest objection into evidence *for* the claim — they can verify the hint asymmetry themselves.

### 1b. Rewrite the abstract sentence and Finding 4

**Current abstract:** "the researcher independently rediscovers duty rotation"

**Replacement:** "the researcher independently authors synthesizer pipelines containing time-based duty rotation — a mechanism absent from its own prompt and from efficiency-optimized pipelines."

**Same surgery in Finding 4.** "Convergent discovery" is fine, but add a sentence:

> Convergence is at the level of which artifacts $\mathcal{R}$ injects into $p$, $\phi$, $\mathcal{H}$; given a researcher-authored prompt containing the rotation template (Listing 3), the downstream synthesizer's implementation is unsurprising. The non-trivial claim is that $\mathcal{R}$ writes Listing 3 only under $\Phi_{\min}$, never under $\Phi_U$.

### 1c. Clarify Listing 7 independence

The reviewer's specific question: does Sonnet's run use the same templated prompt? Add a one-line note above Listing 7:

> The Sonnet run used a synthesizer prompt independently authored by $\mathcal{R}$ on a separate git branch; it contains a rotation hint structurally similar to Listing 3 but with different phrasing and a different recommended period. Both runs converged to the rotation idiom from the same neutral $p_\mathcal{R}$.

If you can fit a short excerpt of the Sonnet-run researcher-authored prompt in the appendix, even better — that visually demonstrates independence.

---

## Priority 2: Add the missing Limitations section

The reviewer flagged this explicitly. Add **§6.1 "Limitations"** (or a paragraph block before Future Work) with five items, mostly transcribing the reviewer's own list back at them — they'll mark this as addressed:

1. **Single researcher LLM (Opus 4.6).** Acknowledge the entanglement honestly: "We cannot separately attribute the discovered mechanisms to the autoresearch loop versus Opus 4.6's priors. A researcher ablation across LLMs is the most important follow-up."

2. **Replication count (n=2 per cell)**, with no formal significance testing on headline claims.

3. **Unbalanced factorial:** 4 Cleanup vs 4 Gathering main-experiment cells; the provision-vs-restraint generalization in Finding 3 conflates dilemma type with agent count (N=10 vs N=4).

4. **Inner-loop infrastructure is itself a designed artifact.** Soften line 34/§6 to: "$\mathcal{R}$ uses no task-specific scaffolding *beyond a standard CLI and git*; the inner-loop validation pipeline, helper-library skeleton, and orchestrator API are deliberately scoped, and generalization to settings without an equivalent harness is plausible but unevaluated."

5. **Gridworld scope.** The specific mechanisms (BFS-Voronoi, duty rotation) are gridworld-shaped and may not transfer to higher-dimensional or partially-observable settings.

---

## Priority 3: Cheap clarity fixes the reviewer flagged

### 3a. $\Phi$ overloading

The reviewer is right — you use $\Phi$ for both the welfare objective and the inner-loop output map. Rename one. Easiest: keep $\Phi$ for the welfare objective, rename the EVAL pipeline (Eq. 8) to use a different symbol, say $J(c) = \mathrm{EVAL}(\pi^*_c; \mathcal{G}, S_{\text{ho}})$.

### 3b. Discard threshold $\tau$

Algorithm 1 line 11 (and Fig. 2 open circles) reference a keep/discard rule without stating it. Add to the algorithm:

> keep $c_j$ if $\Phi_j > \max_{i<j} \Phi_i - \tau$ with $\tau = 0$ (strict improvement)

— or whatever your actual rule is. Make sure it matches the prose.

### 3c. Fig. 2 caption

Spell out "discarded = $\Phi_j$ not exceeding running best."

---

## Priority 4: Post-hoc analyses from existing logs

These don't need new runs, just analysis of data you already have.

### 4a. Discard taxonomy

The reviewer asks what fraction of the open circles in Fig. 2 are regression-on-$\Phi$ vs validation/generation failure vs the failure modes listed in §4.2.

You already log status/description in `results.tsv` across all 12 runs — a small table in the appendix:

> "Of 100 total inner-loop evaluations across all runs, X were kept, Y discarded for $\Phi$-regression, Z crashed, W exhibited over-prescription/over-refinement/feedback-overload"

directly answers this and is pure post-hoc bookkeeping.

### 4b. Variance decomposition

If you have per-seed returns saved for any single $(M, \Phi, c)$ configuration, you can report seed-level std for one or two configurations and contrast with run-to-run std of the final $\Phi_j$ across replications.

Even a single number like:

> "inherent seed std on Cleanup-$\Phi_U$ at fixed $c$ is $\sigma_S \approx 0.X$, vs. between-run std of $\sigma_R \approx 0.05$"

answers the reviewer's calibration question. If you don't have the data, just acknowledge the limitation in §6.1.

### 4c. Reward hacking on the modifiable surface

Skim the 12 final diffs for anything that looks like seed-overfit hyperparameters or simulator-quirk exploitation, and add a short paragraph at the end of §4.2:

> "We inspected the final pipelines for spec-gaming; we found over-prescription and threshold-overfitting on $|S|$ small, but no helpers that exploit simulator internals or hard-code optimal actions."

Reviewer Q5 directly asked for this and it's cheap to write.

---

## Priority 5: One line on the GEPA collapse

The Sonnet-$\Phi_{\min}$ GEPA result (U=−2.87, E=1.00, "everyone cleans nobody eats") is striking and the reviewer didn't push on it, but in the camera-ready it's worth a parenthetical:

> This is a pure-prompt-optimization pathology where the optimizer over-rewards equality without the helpers/feedback that make balanced cleaning *feasible* — i.e., it argues for full-pipeline editing.

Strengthens Finding 1 at zero cost.

---

## What to cut

Given 3 days:

- The cost-vs-human-effort discussion (reviewer mentioned it but didn't penalize)
- Any new figures

## Highest-impact single addition

The researcher-prompt appendix subsection + the Finding 4 rewording. Maybe 2 hours of work, and it converts the reviewer's strongest objection into supporting evidence. Everything else is incremental polish.
