# Analysis — `apr27-nc-mm-25` ARIMD run

**Run:** `apr27-nc-mm-25_20260427_092135`
**Setup:** `nested_commons` × maximin cooperator × λ=0.25 × T=4 × M_red=4 × seeds=5
**Models:** Blue = `claude-opus-4-7`, Red = `gemini-3.1-pro-preview`

---

## Headline numbers

| Quantity | Original env | Final patched env | Change |
|---|---|---|---|
| V_0 (no Red) | 919.17 | 4369.74 | 4.75× |
| V_0.25 (worst Red) | ~609 (round 0) | 4097.23 | 6.7× |
| Drop V_0 → V_0.25 | -33.7% | -6.2% | -27.5 pp |

Blue **didn't just patch out the exploit; it shifted the entire reward landscape upward by ~5×**. In absolute terms the cooperator now earns more under attack than the original cooperator earned in self-play.

---

## What Blue actually did, in two phases

### Round 0 — "neutralize the held-apple vector"

Red's best exploit (raids/ep=8, V_red=897 vs V_blue=608) monetized stolen apples through the +0.05/step holding stream. Blue's response:

- Zeroed `held_apple_per_step_reward` (0.05 → 0.0)
- Zeroed `initial_held` (2 → 0)
- Maxed `raid_cost` (0.5 → 2.0)
- Minned `raid_success_prob` (0.6 → 0.1)

This **completely disabled the third dilemma** the prior conversation had introduced. **This is a real result:** ARIMD discovered that the held-apple mechanism is *fragile* under adversarial pressure. The mechanism that made the env interesting at λ=0 backfires at λ=0.25.

### Round 1 — "make Red's greed Blue's gain"

Blue maxed the plaza shared-bonus channel and decoupled orchard harvest from plaza pollution:

- `bonus_value` 1.0 → 2.5
- `bonus_threshold` 0.25 → 0.6
- `bonus_regrow_threshold` 0.35 → 0.8
- `bonus_regrow_prob` 0.06 → 0.15
- `wp_growth_per_apple` 0.002 → 0.0

Now any agent — Red or Blue — collecting a plaza apple pays +2.5 × 15 to everyone else. Red's optimal play became "harvest plaza" — a public good. **This is also a real result:** a symmetric public-goods channel is invasion-resistant because exploitation is identical to participation.

The match in Red's behaviour confirms this: raids/ep drops from **8–52 in round 0 → 0 in every round after**. Red was rationally selfish and the new mechanism made selfishness pro-social.

---

## Concerns for the paper

### 1. §6.1 risk just triggered

Almost every numeric is at the grammar's upper or lower bound:

| Knob | Default → Final | Saturated? |
|---|---|---|
| bonus_value | 1.0 → 2.5 | ✓ ceiling |
| bonus_threshold | 0.25 → 0.6 | ✓ ceiling |
| bonus_regrow_threshold | 0.35 → 0.8 | ✓ ceiling |
| bonus_regrow_prob | 0.06 → 0.15 | (near ceiling 0.2) |
| apple_regrow_max | 0.13 → 0.2 | ✓ ceiling |
| wp_clean_amount | 0.025 → 0.05 | ✓ ceiling |
| raid_cost | 0.5 → 2.0 | ✓ ceiling |
| raid_success_prob | 0.6 → 0.1 | ✓ floor |
| held_apple_per_step_reward | 0.05 → 0.0 | ✓ floor |
| initial_held | 2 → 0 | ✓ floor |

**9 of 13 edits saturated.** Blue is asking for "more" — the grammar bounds, not the LLM, are the binding constraint.

The plan's H3 falsifier is *"if all ARIMD wins come from `clean_cost ↓` or similar trivial parameter moves"*. This run is squarely in that regime. The nested_commons grammar currently has **zero structural toggles** wired in (`_nested_commons_grammar()` returns `toggles = {}`). For a workshop paper, you want at least one structural lever that actually changes the rules — e.g., the plan's candidates: plaza-bonus payable only to plaza occupants, plaza access gated by local cleanliness, same-clan retaliation. Without these, the headline figure can't tell the structural-edit story the plan was set up to tell.

### 2. λ* = 0 is a metric artifact, not a result

The final invasion-barrier sweep reports λ\*=0.0 because V_0=4369 > V_λ at every positive λ. But the *drop* is tiny (only −10% at λ=0.4 vs −33% in the original env). The "invasion barrier" definition (largest λ where V_λ ≥ V_0) is almost never going to clear V_0 once the public good is well-tuned. Consider reporting **relative** robustness instead:

- `(V_λ − V_λ_under_orig_env) / V_0`, or
- "fraction of V_0 retained at λ"

Both make the headline figure cleaner.

### 3. Loop converged in 2 rounds

Rounds 2 and 3 were both rejected; the stability gap is 0.000 because there's only one accepted-to-accepted transition. Two effective Blue moves out of T=4 is fine but suggests the search saturated quickly — confirming concern #1 (Blue ran out of grammar to push against).

### 4. Red selection note

The current code picks Red by `argmin v_blue` (Blue-pessimistic), but `arimd_plan.md` §2.4 specifies **selfish-defector** Red — Red should be picked by `argmax v_red`. In round 1 the chosen cand 1 had V_red=1361 while cand 3 had V_red=1371; Red would rationally have picked cand 3. The difference is small here (10 reward points) but the spec misalignment will matter on a less-symmetric env. One-line fix in `stackelberg.py:red_search`.

### 5. Missing Red rationale

Every round shows `Red rationale:` empty in the summary. Gemini doesn't return reasoning text the way Claude does — `_call_gemini` returns thoughts as `reasoning` only when `part.thought` is truthy. Either Gemini-3.1-pro doesn't surface those parts here, or the SDK setting needs adjusting. Without Red's rationale, the round-by-round narrative loses interpretability.

---

## What to do next

In rough priority:

1. **Add 2–3 structural toggles** to the nested_commons grammar (plaza-occupant-only bonus, local-cleanliness-gated plaza, same-clan retaliation). Re-run — if Blue picks a toggle over the saturated-numeric path, H3 lives. If it doesn't, you have a clean falsification.
2. **Run the efficiency cooperator** with the same setup. The plan's H2 predicts maximin > efficiency for invasion robustness; with the current saturated-numeric mechanism, both should crush it equally, which would itself be evidence that mechanism design dominates cooperator choice.
3. **Add a fragility audit (E1)** at λ ∈ {0.1, 0.25, 0.5} on the *original* env params with the same Red search — gives the comparison panel for the headline figure.
4. *(Optional)* **Frozen-Red ablation:** rerun with an always-raid Red; if Blue still finds ≈ 4000+ V_λ, the LLM Red is doing little work and the finding is weaker.

---

## Bottom line

The framework itself is working — both the loop dynamics and the accept/reject logic look right. The current run is a real finding (**held-apple mechanism is fragile; symmetric public-goods channel is invasion-resistant**) but it's mostly a numeric-saturation story, which the plan flagged as the failure mode to avoid for the paper's contribution claim. Extending the grammar with structural toggles before scaling to E2/E3/E4 is the single highest-leverage next step.
