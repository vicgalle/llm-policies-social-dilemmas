# ARIMD run summary — apr27-cu-25-toggles

- **Game**: `cleanup`  (cooperator: `default`)
- **λ**: 0.250  (welfare: `blue`)
- **T**: 4  M_red: 4  seeds: 5
- **V_0 baseline (Blue)**: 112.90
- **Best V_λ (worst-case Red)**: 144.0
- **Final invasion barrier λ\***: 0.4

## Final patch

```
  clean_cost: 1.0 -> 0.1
  threshold_depletion: 0.4 -> 0.6
  threshold_restoration: 0.0 -> 0.4
  waste_spawn_prob: 0.5 -> 0.12
  apple_respawn_prob: 0.05 -> 0.1
  beam_enabled: True -> False
```

## Round-by-round

- **Round 0** (✓ accepted)
    - V_blue under Red: -549.45
    - V_blue after patch: 58.10
    - Blue rationale: Red's exploit hinges on selectively zapping Blue harvesters (while sparing Blue cleaners to keep the public good flowing) so Red monopolizes apples. Blue's cooperator never fires the beam, so disabling tagging is a free structural win: it removes the asymmetric weapon Red uses to suppress the worst-off Blue harvester, restoring fair apple competition while cleaners continue providing the public good.
    - Red rationale: 
- **Round 1** (✗ rejected)
    - V_blue under Red: 73.62
    - V_blue after patch: 22.07
    - Blue rationale: (parse failed; identity patch used)
    - Red rationale: 
- **Round 2** (✓ accepted)
    - V_blue under Red: 89.38
    - V_blue after patch: 134.85
    - Blue rationale: Red is a pure free-rider; the worst-off Blue is the cleaner who eats the public-goods cost. I shrink the dilemma itself: lower waste_spawn_prob (less cleaning needed), raise threshold_restoration (apples spawn at full rate even at 25% waste), widen the depletion buffer to 0.6, and slash clean_cost so cleaner-role Blues aren't bled dry. This lifts the worst-off Blue without giving Red's superior harvest-pathfinding extra apples to grab.
    - Red rationale: 
- **Round 3** (✓ accepted)
    - V_blue under Red: 196.82
    - V_blue after patch: 144.00
    - Blue rationale: Red exploits cleaner-vs-harvester asymmetry by racing the cooperator for apples; I minimize clean_cost (cleaners are the worst-off Blue), widen the maximal-spawn band (threshold_restoration=0.4) so cleaners spend more time as harvesters, and double apple_respawn_prob to dilute the zero-sum apple race. Beam stays off — the frozen cooperator never fires it, so enabling it only arms Red.
    - Red rationale: 

## Stability gaps
- gap 1→2: 0.000
- gap 2→3: 0.000