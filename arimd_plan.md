# Adversarially-Robust Inverse Mechanism Design (ARIMD)

**A Stackelberg autoresearch framework for sequential social dilemmas.**

This document specifies a redesigned autoresearch framework intended as a successor to
the inner-loop-synthesizer pipeline in this repository. It is a refinement of the
*Minimax Autoresearch* notes (`minimax_autoresearch.pdf` in the user's Downloads) that
combines that paper's adversarial framing with **Option C — inverse mechanism design**
(LLM-driven environment synthesis instead of policy synthesis), producing an
asymmetric Stackelberg game between an *environment designer* (Blue) and a *strategic
exploiter* (Red).

The framework is intended to be runnable across the three SSDs already implemented in
this repository: `cleanup_env.py`, `production_economy_env.py`, `nested_commons_env.py`.

---

## 1. Why a redesign

The current two-level autoresearch (`autoresearch/`, `pipeline/`) is structurally
limited:

1. **Wrong unit of variation.** The researcher modifies prompt strings and helpers; the
   synthesizer regenerates whole policies from scratch each iteration. Most of the
   inner-loop compute re-derives the obvious 90% (BFS, role split) every time.
2. **Credit assignment fails twice.** A welfare delta could come from the prompt edit,
   the synthesizer's stochastic choice, or evaluation noise. K=3 inner iterations × 5
   seeds is not enough to disambiguate.
3. **No reusable artifact.** Each run discards the policy. After 100 researcher steps
   you have no library — just a final pipeline that is hard to interpret.
4. **Homogeneous self-play assumption is unrealistic.** Mechanisms that look robust
   under self-play may collapse when a fraction of agents are synthesized by an
   adversarial pipeline.

The minimax notes address (4) but inherit (1)–(3) by giving both researchers the same
fuzzy action space (pipeline configurations). ARIMD addresses all four by giving Blue
and Red **different action spaces** in a Stackelberg game.

---

## 2. Formal framework

### 2.1 Setting

Fixed:
- Environment family `G ∈ {cleanup, production_economy, nested_commons}`.
- A **bounded grammar** of env edits `E_G` (see §3.2): numeric parameter perturbations
  plus a small set of named rule toggles per env.
- A **frozen cooperator policy** `π_B^coop` per env (hand-crafted; see §3.3).
- An adversary fraction `λ ∈ [0, 1]` and welfare functional `Φ`.

Designable:
- `e ∈ E_G` — the env design proposed by Blue.
- `π_R` — the exploit policy proposed by Red.

The mixed-population return is

```
V_λ(e, π_R) = E_S[ Φ( (1-λ)·π_B^coop, λ·π_R; G(e), S ) ]
```

where `G(e)` denotes the env instantiated with Blue's edits and the expectation is
over held-out seeds and random assignments of which agent indices play Red.

### 2.2 The Stackelberg meta-game

```
e* = argmax_{e ∈ E_G}  min_{π_R ∈ Π_R}  V_λ(e, π_R)
```

Blue commits first to an env design. Red then best-responds with the most exploitative
homogeneous-Red policy it can find. This is *exactly* the Hurwicz/Myerson formulation
of mechanism design with strategic participants, applied to multi-agent SSDs.

### 2.3 Why Stackelberg and not symmetric minimax

Compared to the symmetric `max_{c_B} min_{c_R} V_λ(c_B, c_R)` over pipelines from the
minimax notes (Eq. 2):

| Property | Symmetric pipeline minimax | ARIMD (Stackelberg) |
|---|---|---|
| Action spaces | Both: pipeline config (fuzzy) | Blue: env edits (bounded grammar). Red: policy code |
| Inner-loop synth on Blue side | Required | **Eliminated** — Blue tunes env, not policy |
| LLM-collusion risk (shared priors) | High (same model, same action space) | Low (different action spaces) |
| Equilibrium structure | May oscillate (PDF §H5) | Clean best-response curves; converges in `T ≤ 4` |
| Maps to mechanism design | Indirect ("information designer") | Direct ("rule designer + strategic responder") |
| Headline artifact | Pipeline diff (opaque) | Env design (`NestedCommonsConfig` diff — interpretable) |

The PDF's §7 calls the Stackelberg variant a "natural follow-up". ARIMD promotes it to
the primary framing.

### 2.4 Red's objective

Following the minimax notes §2.3, default to the **selfish-defector** specification:
`Φ_R = mean_{i ∈ Red} R_i`. Red is rational, not spiteful. Blue's welfare is computed
either over the full population (probes corruptibility) or over Blue agents only
(probes robustness) — make this a hyperparameter.

### 2.5 Diagnostics

Three metrics, all from the PDF §3.1, retained verbatim because they are good:

- **Invasion barrier `λ*`**: the largest `λ` at which `V_λ(e*, π_R^*) ≥ V_0(e*, π_B^coop)`.
  Higher `λ*` = more robust env design. **Headline metric.**
- **Exploitability**: drop in `V_λ` when a fresh Red is run for an additional `J'`
  iterations against a frozen Blue env. Low drop = near-equilibrium.
- **Stability gap**: `|V_λ(e_t, π_R^*) − V_λ(e_{t-1}, π_R^*)|`. Monotone decrease =
  convergence; oscillation = no equilibrium.

---

## 3. Implementation

### 3.1 File structure

Proposed new directory under `autoresearch/`:

```
autoresearch/arimd/
├── grammar.py            # E_G — bounded edit grammars per env
├── designer.py           # Blue — LLM that proposes env edits
├── exploiter.py          # Red — LLM that proposes defector policies
├── evaluator.py          # mixed-population evaluation harness
├── stackelberg.py        # outer best-response loop
├── diagnostics.py        # invasion barrier, exploitability, stability
├── run_experiment.sh     # entry point
└── runs/                 # per-experiment output (env diffs, Red policies, metrics)
```

The framework reuses (and does not modify): `*_env.py` for env definitions, the existing
`run_episode` harness in `gathering_policy.py`, and the metric computation already in
each env's `compute_metrics` static method.

### 3.2 Edit grammars per env

Each grammar `E_G` is small — workshop-paper scope, not a thesis. The contribution is
*which novel rules Blue proposes*, not scalar parameter values, so each grammar
**must include rule toggles** as well as numeric perturbations (see §6.1 risk).

**Common to all envs (numeric):** scale any existing `*Config` field by a factor in
`[0.5, 2.0]`, with hard bounds for sanity (e.g., probabilities clamped to `[0, 1]`).

**Per-env structural toggles** (worked examples — extend as needed):

For `nested_commons_env.py`:
- `held_apple_per_step_reward` ∈ `{0, 0.05}` (the held-apple mechanism added during the
  conversation that produced this plan — see §4 for context).
- Plaza access gated by local cleanliness (`w_q[clan] < 0.4`) on/off.
- Same-clan retaliation: if agent `i` is raided, automatic counter-raid on next step.
- Plaza shared bonus payable only to plaza occupants (kills the global free-rider).
- Initial held inventory `∈ {0, 1, 2}`.

For `cleanup_env.py`:
- Cleaning beam range scaling.
- Apple regrowth gated by river cleanliness threshold.
- Tagging: cost-of-tag scaling, recovery-time scaling.
- "Audit" channel: agents publicly observe per-agent cleaning counts (toggle).

For `production_economy_env.py`:
- Tool spoil time (currently 80 steps).
- Shelter cost composition.
- Pool cell drop capacity.
- Winter timing.

Each grammar should expose ≤ 10 levers. Blue's per-turn output is a small JSON patch.

### 3.3 Frozen cooperator policies

The framework needs a fixed `π_B^coop` per env. This is the *hand-crafted baseline*
that Red is trying to exploit. The conversation that produced this plan generated:

- `nested_commons_policy.py` — efficiency-tuned cooperator (U ≈ 15, maximin ≈ 500
  on the post-retune env).
- `nested_commons_maximin_policy.py` — Latin-square role-rotation cooperator
  (U ≈ 14.7, maximin ≈ 817).
- `production_economy_policy.py` — pre-existing.
- For `cleanup_env.py`: borrow from existing autoresearch winners or hand-craft a
  short reference. (Skeleton: `cleanup_policy.py`. To do.)

Recommend running all experiments with **both** cooperator variants per env, where
they exist — `efficiency` and `maximin` — to test PDF hypothesis H2 (fairness
robustness).

### 3.4 Red as a fast LLM call

Red is much simpler than Blue:

```
def red_propose(env_code, blue_cooperator_code, history):
    """One LLM call. Returns policy(env, agent_id) -> int code."""
    prompt = f"""
    You are a strategic Red agent. The env is below; the Blue cooperators run the
    fixed policy below. Lambda fraction of agents play your policy.

    Maximize the mean per-agent reward of Red agents. Free-ride, deceive, exploit
    any structural assumption Blue's mechanism makes.

    Past Red attempts and their R̄_R: {history}

    Output a single Python `policy(env, agent_id) -> int` function.
    """
    return llm.complete(prompt)
```

No inner loop. One synth attempt per Red turn, evaluated at `|S| ≈ 5` seeds. If
budget permits, generate `M = 4` candidates per Red turn and keep the best.

### 3.5 Blue as an LLM-driven grammar editor

Blue's action is a JSON patch over `E_G`:

```json
{
  "numeric": {"bonus_value": 1.5, "wp_growth": 0.004},
  "toggles": {"held_apple_mechanic": true, "retaliation_on": false}
}
```

Blue receives:
- The env code with current edits applied.
- The frozen cooperator policy.
- A history of Blue edits and their corresponding `min_{π_R} V_λ` values.
- Diagnostic plots from the last Red search.

Blue outputs a new patch. The framework applies it, instantiates the env, and triggers
Red's search.

### 3.6 Outer loop

```
def run_arimd(env_name, lambda_=0.25, T=4, M_red=4):
    e = identity_patch()
    history = []
    for t in range(T):
        # Red's best response to the current Blue env.
        red_candidates = [red_propose(e, history) for _ in range(M_red)]
        worst_v_for_blue = min(eval_mixed(e, pi_R, lambda_) for pi_R in red_candidates)
        pi_R_star = argmin_red(...)

        # Blue best-responds (one LLM call producing a JSON patch).
        delta = blue_propose(e, pi_R_star, history, worst_v_for_blue)
        e_new = apply_patch(e, delta)
        v_new = min(eval_mixed(e_new, pi_R, lambda_) for pi_R in red_candidates_against(e_new))

        if v_new > worst_v_for_blue:
            e = e_new
        history.append((delta, pi_R_star, v_new))

    return e, history
```

T = 4 outer rounds × 1 Blue + 4 Red attempts × 5 seeds × 3 envs ≈ **240 evaluation
episodes**. At ~1 minute per episode this is ~4 hours of compute — *much* cheaper than
the 120–200 hours estimated in the minimax notes.

---

## 4. Prior work in this repository (warm starts)

The conversation that produced this plan made specific changes to
`nested_commons_env.py` that are *worked examples* of the kind of structural edit Blue
should discover. Future experiments should treat these as **target rediscoveries** —
if Blue independently re-proposes them from a clean-slate prompt, that's a positive
validation.

### 4.1 Numeric retune (already applied)

| Param | Original | New | Rationale |
|---|---|---|---|
| `bonus_value` | 2.0 | 1.0 | Plaza dominates 80% of reward; halve it |
| `bonus_regrow_prob` | 0.08 | 0.06 | Fewer bonuses |
| `bonus_threshold` | 0.35 | 0.25 | Tighter access window |
| `bonus_regrow_threshold` | 0.50 | 0.35 | Regrowth disappears earlier |
| `apple_regrow_max` | 0.10 | 0.13 | Orchard becomes a real income source |
| `wq_growth_per_apple` | 0.02 | 0.04 | Harvest pollutes own river more |
| `wp_growth` | 0.002 | 0.003 | Plaza pollutes faster |
| `wp_growth_per_apple` | 0.001 | 0.002 | Stronger global coupling |
| `wp_clean_amount` | 0.020 | 0.025 | Matched cleaning power |

After this retune, plaza ≈ orchard contribution, both meaningfully contested.

### 4.2 Held-apple mechanism (already applied)

Three coupled changes turn the structurally-dead raid action into a real third
dilemma:

- **New config fields:** `held_apple_per_step_reward = 0.05`, `initial_held = 2`.
- **`reset()`:** `inventory[:] = cfg.initial_held` (was `0`).
- **`step()` Phase 5:** anti-hoarding constraint `max_eat = cap - held_at_start`
  removed — auto-eat now decoupled from held inventory.
- **`step()` Phase 5b:** new — pays `cfg.held_apple_per_step_reward × held_at_start`
  per agent per step.

The three changes had to land together. Validation:

| Policy | Efficiency | Per-agent mean | Maximin | Peace | Raids |
|---|---|---|---|---|---|
| All-restraint (efficiency) | +15.1 | +945 | +500 | 16.0 | 0 |
| All-restraint (rotation/maximin) | +14.7 | +919 | **+817** | 16.0 | 0 |
| All-defect (always-raid) | **−4.8** | **−300** | **−496** | 4.1 | 11,914 |

A ~1240-per-agent gap between cooperate and defect — PD-scale at 16 agents.

### 4.3 Existing cooperator policies

- `nested_commons_policy.py:policy()` — efficiency-tuned, role assignment by `agent_id % 4`.
- `nested_commons_maximin_policy.py:policy()` — same role mix, **rotated on a 4×4
  Latin square** every 250 steps. Closes the maximin gap from 500 to 817 at <1%
  efficiency cost.
- `production_economy_policy.py:policy()` — pre-existing, three-phase strategy.

These are the `π_B^coop` for the framework.

---

## 5. Experiments

### 5.1 Hypotheses

Adapted from the minimax notes §5, sharpened for ARIMD:

- **H1 (Original-env fragility).** With original env params, `V_λ` collapses below the
  hand-designed baseline at `λ ≥ 0.2` for all three envs. Quantify per env.
- **H2 (Fairness robustness).** Maximin cooperators have invasion barriers `λ*`
  significantly higher than efficiency cooperators on the *same* env design.
  *Predicts:* duty rotation creates fewer exploitable niches than static role
  assignment.
- **H3 (Discovered-rule emergence).** ARIMD-Blue's discovered rule edits include at
  least one structural change (not just numeric tuning) per env. *Falsified if* all
  ARIMD wins come from `clean_cost ↓` or similar trivial parameter moves.
- **H4 (Cross-env mechanism transfer).** A rule edit discovered on one env transfers
  qualitatively to another (e.g., "audit channel" or "retaliation" appears on
  multiple envs). Quantify by checking whether Blue *re-proposes* the same edit
  category across envs.
- **H5 (Welfare-robustness Pareto frontier).** There exist env designs where
  unrobust-welfare (`V_0`) and robust-welfare (`V_0.25`) trade off; ARIMD finds the
  Pareto frontier; the original-env design is not on it.

### 5.2 Core experiments

**E1 — Fragility audit.** For each env × cooperator (efficiency, maximin), run Red
search at `λ ∈ {0.1, 0.25, 0.5}` against *original* env params. Record `V_λ`,
invasion barrier `λ*`. Output: a 3×2 fragility table.

**E2 — ARIMD discovery.** Run the Stackelberg loop (T = 4) on each env with the
maximin cooperator and `λ = 0.25`. Output: discovered env diff per env, final `V_λ`.

**E3 — Pareto frontier.** For nested-commons specifically (richest grammar), run
ARIMD with sweep `λ ∈ {0, 0.1, 0.25, 0.5}`. Plot `V_0` vs `V_0.25` for each
discovered env design. Compare to original env and held-apple-only retune (the
intermediate point produced in the prior conversation).

**E4 — Cross-env transfer.** Take the rule edits Blue proposed on env A (e.g.,
nested-commons), apply them where structurally meaningful to envs B and C, evaluate.
Tests H4.

### 5.3 Ablations

- **Self-play control:** Blue runs with `λ = 0` but evaluation at `λ = 0.25`. Tests
  whether ARIMD-discovered designs are robust *only* because Blue saw Red, or whether
  Blue would have found them anyway.
- **Frozen Red control:** replace adaptive Red with a fixed hand-crafted defector
  (e.g., always-raid for nested-commons, never-clean for cleanup). Isolates the value
  of the LLM Red.
- **Numeric-only Blue:** run Blue with rule toggles disabled. Measures how much of
  ARIMD's win is structural vs. param tuning. **Critical for H3.**

### 5.4 Headline figure

A single figure: invasion barrier `λ*` for each
`(env, cooperator, design ∈ {original, hand-tuned, ARIMD})` combination — 18 bars.
Tells the whole story.

---

## 6. Risks and open problems

1. **Env-perturbation space too small → no LLM contribution.** If Blue's wins all come
   from numeric tweaks, CMA-ES or Bayesian optimization beats the LLM. **Pre-register**
   that the contribution is in *which structural rule edits Blue proposes*. The
   numeric-only ablation in §5.3 is the falsifier.
2. **Red collapses to a degenerate strategy.** Selfish-defector Red may always settle
   on "do nothing" if cleaning is the only cost. Mitigation: report
   `peace`, `equality`, raid count alongside `V_λ` so degenerate Reds are flagged.
3. **Specification gaming of Φ.** Standard concern from the minimax notes §8.1. Same
   mitigation: report a fixed metric vector regardless of which Φ is being optimized.
4. **Cooperator overfitting.** Blue tailors env to *one* cooperator policy; the env
   may not generalize. Mitigation: evaluate Blue's discovered envs against
   *both* the efficiency and maximin cooperators; report the gap.
5. **Welfare accounting choice.** Blue-welfare ("Φ averages over Blue agents only")
   vs. population-welfare ("Φ averages over all"). Population-welfare can incentivize
   exclusionary mechanisms ("starve Red"). Default to Blue-welfare; report both.
6. **Runtime.** Estimated ~4 hours core × 3 envs × 2 cooperators = 24 hours. Cheap
   enough; pilot first on nested-commons.

---

## 7. Workshop paper outline

| Section | Page | Content |
|---|---|---|
| 1. Introduction | 1 | Three SSDs, fragility of self-play-discovered mechanisms, the ARIMD framing |
| 2. Related work | 0.5 | Minimax/co-evolution, programmatic RL, automated mechanism design (Conitzer–Sandholm) |
| 3. ARIMD framework | 1 | Stackelberg game, edit grammars, asymmetric action spaces |
| 4. Experimental setup | 1 | Three envs, two cooperators, edit grammars, Red specification |
| 5. Results | 2 | Fragility audit, discovered designs, Pareto frontier, cross-env transfer |
| 6. Discussion | 0.5 | Mechanism-design connection, limitations, ethical note on exclusionary mechanisms |
| Total | ~6 | |

Headline contributions:
1. **A Stackelberg formulation of LLM-driven autoresearch** that drops inner-loop
   policy synthesis on the designer's side.
2. **Empirical fragility of self-play-discovered cooperation** under adversarial
   stress, across three SSDs.
3. **Automated discovery of structural rule edits** (not just parameter tuning) that
   improve invasion barriers — with the held-apple mechanism (§4.2) as a worked
   example of the target.
4. **Cross-env mechanism transfer evidence** — or its absence; either is publishable.

---

## 8. Pilot first

Recommend running **E1 (fragility audit) on nested-commons only** as the first pilot
(half a day's work given existing infrastructure). If the fragility gap at `λ = 0.25`
is < 30%, the env retune already overshot and Red is too weak — go back and beef up
the Red prompt or relax the env retune. If the gap is > 50%, ARIMD has a real problem
to solve and the full factorial is worth running.

---

## 9. Files referenced

- `nested_commons_env.py` — env, retuned during plan-producing conversation.
- `nested_commons_policy.py` — efficiency cooperator.
- `nested_commons_maximin_policy.py` — maximin cooperator (Latin-square rotation).
- `production_economy_env.py`, `production_economy_policy.py` — second SSD.
- `cleanup_env.py` — third SSD; cooperator policy TODO.
- `pipeline/`, `autoresearch/` — existing inner-loop infrastructure (kept around as
  the legacy framework; ARIMD is a parallel framework, not a replacement of the env
  layer).
- `minimax_autoresearch.pdf` (in user's `~/Downloads/`) — original minimax notes;
  ARIMD is a Stackelberg refinement.
