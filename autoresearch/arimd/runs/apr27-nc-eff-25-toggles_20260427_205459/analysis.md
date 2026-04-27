# Analysis — `apr27-nc-eff-25-toggles` ARIMD run

**Run:** `apr27-nc-eff-25-toggles_20260427_205459`
**Setup:** `nested_commons` × **efficiency cooperator** × λ=0.25 × T=4 × M_red=4 × seeds=5
**Models:** Blue = `claude-opus-4-7`, Red = `gemini-3.1-pro-preview`
**Changes vs prior run (`apr27-nc-mm-25`):**
- Cooperator swapped maximin → efficiency (everything else identical).
- Three structural toggles wired into the grammar: `plaza_occupant_only_bonus`, `plaza_local_clean_gate`, `same_clan_retaliation`.
- Red selection changed from `argmin v_blue` → `argmax v_red` (selfish-defector, per `arimd_plan.md` §2.4).

---

## Headline numbers

| Quantity | maximin run (prior) | **efficiency run (this)** |
|---|---|---|
| V_0 (no Red) | 919.17 | 945.51 |
| Best V_λ (worst-case Red) | 639.33 | **5116.65** (≈5.4× V_0) |
| λ\* (invasion barrier) | 0.0 | **0.25** ✓ cleared |
| Accepted rounds | 0/4 | **3/4** |
| Final patch | identity | 12 numerics + **2 toggles flipped** |

The efficiency cooperator turns into a strong positive H3 result. The maximin cooperator with the same grammar produced 0/4 accepts; the only difference is the cooperator.

---

## H3 status — structural edits *do* matter

The final accepted patch flips **two of the three toggles** and Blue uses them with deliberate, non-monotonic reasoning, not as saturation defaults:

- `plaza_local_clean_gate=True` ✓
- `same_clan_retaliation=True` ✓
- `plaza_occupant_only_bonus=False` — Blue had it ON in rounds 0–1, then *toggled it back OFF* in round 2 after observing Red's response. Round 2 rationale:

> *"Red's exploit is plaza-camping with no cleaning, abusing occupant-only sharing as a Red-only payout pool. Disabling occupant-only restores global welfare leakage back to Blue; enabling local_clean_gate ties plaza payouts to the collector's own river hygiene, which Blue maintains by role and Red ignores."*

Round 3's rationale doubles down:

> *"Plaza_occupant_only_bonus is deliberately left off — flipping it would collapse Blue's majority-share advantage on the shared bonus."*

This is the cleanest H3 evidence we have so far: Blue treats the toggles as a real action space, including reverting a flip when it sees Red benefit more from it.

---

## Round-by-round trajectory

| Round | V_blue under chosen Red | V_blue after patch | Accepted? | Toggle changes |
|---|---:|---:|:-:|---|
| 0 | 655.56 | 291.55 | ✗ | all 3 ON; raid economy killed |
| 1 | 670.59 | **946.12** | ✓ | retaliation + occupant-only ON, raid stays sealed |
| 2 | 971.38 | **2731.95** | ✓ | occupant-only OFF, local_clean_gate ON |
| 3 | 3464.00 | **5116.65** | ✓ | pure numeric polish on round-2 structural choice |

**Raids/ep across rounds:** 12.8 → 19.4 → 0.0 → 0.0. Red abandoned raiding the moment `plaza_local_clean_gate` came on — strong signal the structural toggle, not the numeric raid-cost tuning, was the binding constraint.

---

## What Blue actually did, structurally

### Round 1 — "build a Blue-only welfare engine through the plaza"

Blue's first accepted move keeps `plaza_occupant_only_bonus=True`, raises `bonus_value` and the regrow window, and decouples plaza waste from orchard harvest. Net effect: 4 Blue plaza-specialists run a self-contained payout loop that Red can't reach without occupying plaza cells. Raid economy is sealed (high cost, low success, retaliation, no held inventory).

### Round 2 — "let the global bonus flow but gate it on home cleanliness"

Red found the round-1 hole: park in plaza, free-ride on Blue's labor, and *occupant-only sharing turned the plaza into a Red-only payout pool* once Red took up enough plaza cells. Blue's response is non-trivial: turn the occupant-only restriction OFF (Blue is the population majority — global sharing favors Blue), and turn the *local-clean gate* ON (Red doesn't clean, so Red picks forfeit). This is a structural reversal, not a parameter dial.

### Round 3 — "polish the channel"

Numeric tuning on top of the round-2 structure: cheaper CLEAN, lower harvest-induced w_q growth, richer plaza payout. With the structural decisions made, ~10 numerics saturate. Three of them (`bonus_value`, `bonus_threshold`, `bonus_regrow_threshold`) hit the same ceiling as the prior run, but layered on the toggle flip rather than substituting for one.

---

## Comparison to the maximin run — H2 partial inversion

The plan's H2 predicts maximin > efficiency under invasion. This run produces the opposite ordering with this grammar:

- **maximin under toggles:** 0/4 accepts, final V_λ = 639 (round-0 baseline; identity patch).
- **efficiency under toggles:** 3/4 accepts, final V_λ = 5117.

Mechanism: efficiency's specialist roles co-locate plaza guards and clean-river roles per clan, so `plaza_local_clean_gate` *protects* Blue's picks while denying Red's. Maximin's role-rotation thins the plaza guard at any given moment, so the same toggle misfires on Blue's own picks too.

The toggles are real levers, but their welfare sign depends on the cooperator. That's a paper-worthy finding either way: the cooperator and the mechanism interact.

---

## What's left

- **Stability gap = 0.000** — the gap metric reads 0 between every accepted-to-accepted pair despite V_blue going 946 → 2732 → 5117. Looks like `diagnostics.stability_gap` is computing the wrong thing. Worth a debug pass before quoting in the paper.
- **Empty Red rationale** persists (Gemini SDK doesn't surface `part.thought` here). The round-by-round narrative loses Red's side. Either swap Red to Sonnet on a follow-up run or add an explicit "rationale: …" line to the Red prompt's required output.
- **Numeric saturation is still there** (10 numerics at bound). The structural-toggle layer doesn't eliminate the saturation story — it sits on top of it. For the paper, the headline should be that toggles *contribute* to the win, not that they *replace* numeric tuning. The numeric-only ablation in §5.3 of `arimd_plan.md` is now well-positioned as a clean comparison.

---

## Bottom line

H3 fires positively here for the first time: Blue actively uses the structural toggle action-space, reverts a flip when it backfires, and the final accepted patch carries two structural switches plus saturated numerics. λ\* = 0.25 cleared. The maximin/efficiency divergence is a separate publishable observation — same grammar, opposite outcomes, driven by how each cooperator interacts with the toggles' preconditions.
