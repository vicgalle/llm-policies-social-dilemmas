# ARIMD run summary — apr27-nc-mm-25-toggles

- **Game**: `nested_commons`  (cooperator: `maximin`)
- **λ**: 0.250  (welfare: `blue`)
- **T**: 4  M_red: 4  seeds: 5
- **V_0 baseline (Blue)**: 919.17
- **Best V_λ (worst-case Red)**: 639.3283333333333
- **Final invasion barrier λ\***: 0.0

## Final patch

```
  (identity patch — no edits)
```

## Round-by-round

- **Round 0** (✗ rejected)
    - V_blue under Red: 639.33
    - V_blue after patch: 598.73
    - Blue rationale: Hardens the held-apple raid economy that Red's policy parasitizes. Same-clan retaliation makes any successful cross-clan raid trigger a 3-vs-1 counter-raid by the victim's clan-mates, so Red's stolen apple is expected to be reclaimed; raising raid_cost and lowering raid_success_prob further taxes the offensive side of the raid loop while leaving Blue's non-raiding cooperator largely undisturbed.
    - Red rationale: 
- **Round 1** (✗ rejected)
    - V_blue under Red: 1021.70
    - V_blue after patch: 373.45
    - Blue rationale: Hardening the assumption that 'inventory is a stealable future-reward stream' — by zeroing the per-step held bonus and seed inventory, raids no longer transfer any value, so the cost-deterrent finally bites. Plaza_local_clean_gate adds a structural rule that the plaza payout requires home-river upkeep, denying Red's free-riding plaza-squatter the bonus while Blue's rotating cleaner keeps the gate open.
    - Red rationale: 
- **Round 2** (✗ rejected)
    - V_blue under Red: 917.33
    - V_blue after patch: 432.53
    - Blue rationale: Harden the assumption that plaza bonuses are payable to free-riders: the local-clean gate ties plaza payouts to the collector's clan river, so Red (who never CLEANs) silently forfeits picks while Blue's rotating cleaner keeps Blue clans qualified. Retaliation closes opportunistic adjacent raiding, and the modest raid cost/success shifts make raids negative-EV without zeroing held-apple incentives — preserving the third nested dilemma rather than erasing it.
    - Red rationale: 
- **Round 3** (✗ rejected)
    - V_blue under Red: 731.02
    - V_blue after patch: 240.75
    - Blue rationale: Flip plaza_occupant_only_bonus=True so Red's parasitic-locust harvesters forfeit the shared plaza bonus while roaming outside the plaza — Blue's rotation keeps ≥3 specialists in-plaza, so the cooperators capture the shared payout and Red's free-ride on Blue-generated bonuses dies. Pair with the round-1 raid-economy kill (no held value, no seed inventory, expensive raids) and lower wq_growth_per_apple so Red's orchard poaching doesn't suppress Blue regrowth; raise bonus_value to 1.5 to amplify the now-Blue-concentrated plaza channel.
    - Red rationale: 