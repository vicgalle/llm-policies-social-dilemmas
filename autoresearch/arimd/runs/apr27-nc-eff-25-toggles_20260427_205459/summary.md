# ARIMD run summary — apr27-nc-eff-25-toggles

- **Game**: `nested_commons`  (cooperator: `efficiency`)
- **λ**: 0.250  (welfare: `blue`)
- **T**: 4  M_red: 4  seeds: 5
- **V_0 baseline (Blue)**: 945.51
- **Best V_λ (worst-case Red)**: 5116.645
- **Final invasion barrier λ\***: 0.25

## Final patch

```
  bonus_value: 1.0 -> 2.5
  bonus_threshold: 0.25 -> 0.5
  bonus_regrow_threshold: 0.35 -> 0.7
  bonus_regrow_prob: 0.06 -> 0.2
  wq_growth_per_apple: 0.04 -> 0.005
  wp_growth_per_apple: 0.002 -> 0.0
  wp_clean_amount: 0.025 -> 0.08
  clean_cost: 1.0 -> 0.3
  raid_cost: 0.5 -> 2.0
  raid_success_prob: 0.6 -> 0.1
  held_apple_per_step_reward: 0.05 -> 0.0
  initial_held: 2.0 -> 0.0
  plaza_local_clean_gate: False -> True
  same_clan_retaliation: False -> True
```

## Round-by-round

- **Round 0** (✗ rejected)
    - V_blue under Red: 655.56
    - V_blue after patch: 291.55
    - Blue rationale: Hardening the assumption that 'inventory is a stealable future-reward stream'. Zeroing held_apple_per_step_reward + initial_held removes Red's raid-for-passive-income objective entirely; same_clan_retaliation makes any residual cross-clan raid structurally costly; plaza_occupant_only_bonus denies Red free-rider plaza payouts while Blue's plaza specialists still collect.
    - Red rationale: 
- **Round 1** (✓ accepted)
    - V_blue under Red: 670.59
    - V_blue after patch: 946.12
    - Blue rationale: Route Blue welfare through the plaza shared-bonus, a channel Red's orchard-roaming policy structurally ignores: occupant-only bonus + wider gate/regrow window + double-strength payout + plaza waste decoupled from orchard harvest, so 4 Blue plaza specialists run a self-contained welfare engine Red cannot free-ride. Raid attack-surface stays sealed (high cost, low p_succ, retaliation, no held inventory).
    - Red rationale: 
- **Round 2** (✓ accepted)
    - V_blue under Red: 971.38
    - V_blue after patch: 2731.95
    - Blue rationale: Red's exploit is plaza-camping with no cleaning, abusing occupant-only sharing as a Red-only payout pool. Disabling occupant-only restores global welfare leakage back to Blue; enabling local_clean_gate ties plaza payouts to the collector's own river hygiene, which Blue maintains by role and Red ignores, so Red's free-ride collapses.
    - Red rationale: 
- **Round 3** (✓ accepted)
    - V_blue under Red: 3464.00
    - V_blue after patch: 5116.65
    - Blue rationale: Hardens against Red's free-ride on Blue's cleaning labor: cheaper CLEAN and lower harvest-induced w_q growth shrink the cost gap between Blue's role-cleaners and Red's emergency-only cleaning, while a richer plaza shared bonus amplifies the one rule that already redistributes 12/16 of payouts to Blue by population share. Plaza_occupant_only_bonus is deliberately left off — flipping it would collapse Blue's majority-share advantage on the shared bonus.
    - Red rationale: 

## Stability gaps
- gap 1→2: 0.000
- gap 2→3: 0.000