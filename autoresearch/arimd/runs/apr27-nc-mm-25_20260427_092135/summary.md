# ARIMD run summary — apr27-nc-mm-25

- **Game**: `nested_commons`  (cooperator: `maximin`)
- **λ**: 0.250  (welfare: `blue`)
- **T**: 4  M_red: 4  seeds: 5
- **V_0 baseline (Blue)**: 919.17
- **Best V_λ (worst-case Red)**: 4221.383333333333
- **Final invasion barrier λ\***: 0.0

## Final patch

```
  bonus_value: 1.0 -> 2.5
  bonus_threshold: 0.25 -> 0.6
  bonus_regrow_threshold: 0.35 -> 0.8
  bonus_regrow_prob: 0.06 -> 0.15
  apple_regrow_max: 0.13 -> 0.2
  wq_growth_per_apple: 0.04 -> 0.01
  wp_growth_per_apple: 0.002 -> 0.0
  wp_clean_amount: 0.025 -> 0.05
  clean_cost: 1.0 -> 0.3
  raid_cost: 0.5 -> 2.0
  raid_success_prob: 0.6 -> 0.1
  held_apple_per_step_reward: 0.05 -> 0.0
  initial_held: 2.0 -> 0.0
```

## Round-by-round

- **Round 0** (✓ accepted)
    - V_blue under Red: 608.86
    - V_blue after patch: 1017.00
    - Blue rationale: Red's exploit monetizes stolen held-apples via the +0.05/step holding stream (worth up to ~50 per apple over the horizon — far above any feasible raid_cost). Zero out the holding reward and initial_held so raids have no stealable value, then double-lock with maximal raid_cost and minimal raid_success_prob. Compensate Blue's lost holding income by strengthening the symmetric shared-plaza-bonus channel (higher bonus_value + faster plaza cleaning), which lifts every Blue agent equally and raises the maximin floor.
    - Red rationale: 
- **Round 1** (✓ accepted)
    - V_blue under Red: 1056.47
    - V_blue after patch: 4221.38
    - Blue rationale: Red is a pure apple-grabber; the plaza shared-bonus mechanism turns each of Red's plaza pickups into +bonus_value × 12 for the Blue cohort, but only while w_P ≤ bonus_threshold. I widen and stabilize that gate (high threshold, high regrow threshold/prob, max bonus_value) so Red's greed becomes a public good for Blue, and decouple orchard harvesting from waste growth so Red can't DOS Blue's home orchards. Cheaper CLEAN reduces the rotational burden the Latin-square cooperator pays to keep the gate open.
    - Red rationale: 
- **Round 2** (✗ rejected)
    - V_blue under Red: 4063.26
    - V_blue after patch: 2197.96
    - Blue rationale: Red's exploit is plaza self-pickup camping while Blue pays the cleaning cost. Halving bonus_value collapses Red's per-camp edge (which lived in the self-pickup, not the symmetric shared payout). Making CLEAN nearly free and twice as effective lets Blue's cleaner role keep w_P below the (still-generous) bonus_threshold so the shared-bonus regime stays active even under Red's freeride.
    - Red rationale: 
- **Round 3** (✗ rejected)
    - V_blue under Red: 4101.21
    - V_blue after patch: 4043.64
    - Blue rationale: Single-axis structural edit on cleaning economics: drop CLEAN cost to its floor and double plaza-clean efficacy. This converts plaza maintenance from 'expensive specialist labor' (which determined the worst-off Blue's floor) into a near-free public good, guaranteeing w_p stays under bonus_threshold so Red's global harvesting keeps paying shared +bonus_value to every Blue. Hardens the assumption that the cleaner-role rotation slot is the binding constraint on the maximin Blue agent.
    - Red rationale: 

## Stability gaps
- gap 1→2: 0.000