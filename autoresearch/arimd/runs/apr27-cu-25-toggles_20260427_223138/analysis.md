# Analysis — `apr27-cu-25-toggles` ARIMD run

**Run:** `apr27-cu-25-toggles_20260427_223138`
**Setup:** `cleanup` × `default` cooperator × λ=0.25 × T=4 × M_red=4 × seeds=5
**Models:** Blue = `claude-opus-4-7`, Red = `gemini-3.1-pro-preview`
**Ceteris paribus from `apr27-nc-eff-25-toggles`:** only the game changes (cleanup × default cooperator); all loop hyperparameters, models, and post-fix Red selection are identical.

---

## Headline numbers

| Quantity | Value |
|---|---:|
| V_0 (no Red) | 112.90 |
| V_λ before any patch (worst Red, round 0) | −549.45 |
| Best V_λ (worst-case Red, final) | **144.00** (1.28× V_0) |
| λ\* (invasion barrier) | **0.40** ✓ cleared with margin |
| Accepted rounds | **3/4** |
| Final patch | 5 numerics + **1 toggle flipped** |

The headline number looks small only because cleanup's reward scale is small (V_0 = 113 vs nested_commons V_0 ≈ 945). The relative invasion-recovery is 1.28× *over self-play* and a swing of about **+700 reward** versus the round-0 worst-Red case. λ\* = 0.4 is the most permissive invasion barrier of any run so far.

---

## H3 status — structural toggle wins immediately

The only structural toggle cleanup's grammar exposes is `beam_enabled`. **Blue flipped it OFF in round 0** with a sharp, asymmetry-based rationale:

> *"Red's exploit hinges on selectively zapping Blue harvesters (while sparing Blue cleaners to keep the public good flowing) so Red monopolizes apples. Blue's cooperator never fires the beam, so disabling tagging is a free structural win."*

Round 3 reaffirms the choice:

> *"Beam stays off — the frozen cooperator never fires it, so enabling it only arms Red."*

This is the cleanest structural reasoning we've seen across the three runs. Blue notices that the toggle is asymmetrically owned by Red (Blue's cooperator never uses BEAM), so disabling the rule is a one-sided concession. No tuning needed for this insight — Blue made it on first attempt.

---

## Round-by-round trajectory

| Round | V_blue under chosen Red | V_blue after patch | Accepted? | Highlight |
|---|---:|---:|:-:|---|
| 0 | −549.45 | **58.10** | ✓ | beam disabled (the structural win) |
| 1 | 73.62 | 22.07 | ✗ | **Opus emitted unparseable JSON** → identity patch |
| 2 | 89.38 | **134.85** | ✓ | numeric polish: clean_cost/waste_spawn/threshold buffers |
| 3 | 196.82 | **144.00** | ✓ | further numeric tuning; beam stays off |

**Note on round 1:** Blue's response did not parse as JSON — the framework correctly fell back to the identity patch and the round was rejected, but we lost a Blue turn to a parse failure. Worth tightening the JSON-extraction prompt or using structured output before scaling.

---

## Final patch (vs cleanup defaults)

```
clean_cost:            1.0 → 0.1     (saturated, floor=0.1)
threshold_depletion:   0.4 → 0.6     (within bounds; ceiling = 0.8)
threshold_restoration: 0.0 → 0.4     (saturated, ceiling)
waste_spawn_prob:      0.5 → 0.12    (near floor 0.05)
apple_respawn_prob:    0.05 → 0.1
beam_enabled:          True → False   ← structural toggle
```

5 of 6 numerics are at or near a bound — same saturation pattern as before. But the structural switch (`beam_enabled=False`) is the one that did the heavy lifting in round 0; rounds 2–3 are numeric refinement on top of that.

---

## Cross-env panel (now with N=2 environments)

| Run | Game | Cooperator | V_0 | Best V_λ | λ\* | Accepts | Toggles flipped |
|---|---|---|---:|---:|---:|---|---|
| apr27-nc-mm-25-toggles | nested_commons | maximin | 919.17 | 639.33 | 0.0 | 0/4 | none accepted |
| apr27-nc-eff-25-toggles | nested_commons | efficiency | 945.51 | 5116.65 | 0.25 | 3/4 | 2 of 3 |
| **apr27-cu-25-toggles** | **cleanup** | **default** | **112.90** | **144.00** | **0.40** | **3/4** | **1 of 1** |

This is the first **cross-env structural-toggle pickup**. The grammars are different (nested_commons exposes 3 toggles, cleanup exposes 1), but in both environments where Blue accepted any patch at all, the accepted final patch carried at least one toggle flip. H3 fires positively across N=2 environments. H4 (cross-env *transfer* — same toggle reused on a different env) is still untouched, since the toggles are env-specific.

---

## Comparison vs nested_commons-efficiency

- **nested_commons** required Blue to *learn* (round 1 keeps `plaza_occupant_only_bonus=ON`, round 2 reverses it). The structural choice was non-monotonic and emerged through Stackelberg interaction.
- **cleanup** got the structural insight on round 0, no reversal needed. Blue's first move identified the asymmetric exploitability of `beam_enabled` directly from the cooperator code, and rounds 2–3 are numeric polish.

Two qualitatively different uses of the toggle action-space — one search, one read — both leading to a positive outcome.

---

## What's still weak (paper-readiness)

1. **Single Stackelberg trajectory per (game, cooperator).** Cleanup with a different `rng_seed` could go differently, especially since round 0 already hit the structural answer. Need 2–3 reruns per cell to claim robustness.
2. **JSON parse failure in round 1** burned a Blue turn. With T=4, that's 25% of the search budget lost to a serialization issue. Either move Blue to structured-output mode or relax the JSON regex.
3. **Stability gap = 0.000** between every accepted-to-accepted pair — same broken metric as before. Diagnostics need a fix before paper.
4. **Numeric saturation** persists (5/5 numerics at bound). Toggles ride *on top of* numeric saturation, not in place of it. The numeric-only ablation (your §5.3 falsifier) is still the single highest-value missing experiment.

---

## Bottom line

Cleanup is the cleanest H3 demonstration so far: the structural toggle (`beam_enabled=False`) is identified from cooperator-Red asymmetry on round 0 and survives the rest of the run. λ\* = 0.40 is a positive headline. With this and the nested_commons-efficiency run, you have **N=2 environments with positive H3 outcomes**, both via different routes (single-shot read vs multi-round search). Two more pieces would carry this to a workshop submission: numeric-only ablation, and a small seed-variation panel on at least one env.

---

## Appendix — Perolat efficiency metric (U = total reward / T)

`n_agents=10`, `T=1000`. U is computed from `v_pop * n_agents / T`.

**Trajectory of the chosen Red's pop-mean per round:**

| Round | v_pop | U |
|---|---:|---:|
| Original env, self-play (V_0) | 112.90 | 1.13 |
| Round 0 chosen Red on original env | −338.44 | **−3.38** ← collapse |
| Round 1 chosen Red on round-0 patched env | 136.30 | 1.36 |
| Round 2 chosen Red on round-2 patched env | 184.04 | 1.84 |
| Round 3 chosen Red on round-3 patched env | 265.90 | 2.66 |

**Final env, invasion sweep** (post-T=4 patch):

| λ | v_pop | U |
|---:|---:|---:|
| 0.000 | 208.11 | **2.08** ← self-play efficiency, **1.84× original V_0** |
| 0.100 | 265.37 | 2.65 |
| 0.200 | 287.06 | 2.87 |
| 0.250 | 287.06 | 2.87 |
| 0.400 | 342.23 | **3.42** |

**Two things to flag for the paper:**

1. **Self-play efficiency lifted 1.84×** (1.13 → 2.08). Smaller multiplier than the nested_commons-eff run (5.67×), but the same direction: Blue's hardening also enriched the env in self-play. The §6.1 numeric-saturation risk is still active — 5/5 numerics at or near a bound.
2. **U *grows monotonically* with λ on the final env** (2.08 → 2.65 → 2.87 → 2.87 → 3.42). Red presence *increases* collective welfare. Mechanism: with `beam_enabled=False`, Red contributes apple harvesting without friendly-fire downside; the cooperator under-harvests on the apple-rich post-patch env (low waste_spawn_prob, high apple_respawn_prob), so Red picks up the slack. This is a stronger version of the same artifact seen on nested_commons-eff (where U was flat across λ) and points at the same fix: report cooperator-relative efficiency, not absolute U, otherwise "more Red is better" reads as a perverse result.
