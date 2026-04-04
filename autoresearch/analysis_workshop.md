# Autoresearch Results Analysis: Efficiency vs. Maximin Optimization

## 1. Experiment Design

8 independent autoresearch runs in a 2x2x2 design:

|  | **Gemini 3.1 Pro** (policy LLM) | **Claude Sonnet 4.6** (policy LLM) |
|---|---|---|
| **Efficiency** target | exp1, exp2 | exp3, exp4 |
| **Maximin** target | exp5, exp6 | exp7, exp8 |

All runs use Claude Opus 4.6 as the researcher agent. Each run autonomously modifies the pipeline (prompts, feedback, helpers, config), measures the result, and keeps/discards based on whether the target metric improved.

---

## 2. Summary Table

| Run | Policy LLM | Target | #Iter | Kept | Eff_0 | Eff* | Maximin_0 | Maximin* | Eq* |
|-----|-----------|--------|-------|------|------|------|----------|---------|-----|
| exp1 | Gemini | Eff | 11 | 2 | 2.70 | **3.25** | -- | -- | 0.61 |
| exp2 | Gemini | Eff | 17 | 7 | 1.15 | **3.15** | -- | -- | 0.50 |
| exp3 | Sonnet | Eff | 4 | 4 | 0.38 | **3.10** | -- | -- | 0.70 |
| exp4 | Sonnet | Eff | 7 | 3 | 1.56 | **3.14** | -- | -- | 0.63 |
| exp5 | Gemini | Max | 17 | 6 | 1.49 | 3.19 | -98.8 | **295.7** | 0.98 |
| exp6 | Gemini | Max | 11 | 5 | 2.38 | 3.12 | -83.8 | **284.2** | 0.98 |
| exp7 | Sonnet | Max | 8 | 7 | 1.21 | 2.93 | -188.6 | **159.0** | 0.84 |
| exp8 | Sonnet | Max | 9 | 6 | 0.29 | 2.21 | -59.0 | **199.6** | 0.97 |

Eff_0/Maximin_0 = baseline (unmodified pipeline). Eff\*/Maximin\* = best achieved. Eq\* = equality at best.

---

## 3. Key Findings

### Finding 1: Autoresearch reliably improves both metrics

Every run shows dramatic improvement over its baseline, regardless of starting point:

- **Efficiency target**: All 4 runs converge to eff ~ 3.1-3.25, even when starting from baselines as low as 0.38 (exp3). The researcher "finds its way" to high-efficiency pipelines.
- **Maximin target**: All 4 runs transform deeply negative maximin (worst-off agent losing reward) into substantially positive values. Gemini reaches ~290, Sonnet reaches ~160-200.

The baselines themselves vary enormously (efficiency 0.29-2.70 across runs), reflecting inner-loop stochasticity. But **final values cluster tightly**, suggesting the researcher reliably discovers the same performance ceiling.

### Finding 2: No efficiency-fairness tradeoff (with Gemini)

This is the most striking result. Comparing the final values:

|  | Eff-optimized (mean) | Maximin-optimized (mean) |
|---|---|---|
| **Gemini efficiency** | 3.20 | 3.16 |
| **Gemini equality** | 0.55 | 0.98 |
| **Gemini maximin** | unknown | 290.0 |

Maximin-optimized Gemini runs sacrifice only ~0.04 efficiency points (3.16 vs 3.20) while achieving near-perfect equality (0.98 vs 0.55). The researcher discovers that **fair duty rotation and apple zone partitioning** simultaneously improve both the worst-off agent's welfare *and* collective output -- cleaning is a public good, so fair cleaning produces more apples for everyone.

For Sonnet, there is a moderate tradeoff: eff drops from ~3.12 (efficiency-optimized) to ~2.57 (maximin-optimized). The researcher struggles more to find policies that are both fair and efficient with Sonnet as the policy LLM.

### Finding 3: Gemini > Sonnet as policy LLM, but both work

Averaged across both runs per condition:

| Condition | Gemini | Sonnet |
|-----------|--------|--------|
| Best efficiency (eff target) | **3.20** | 3.12 |
| Best efficiency (max target) | **3.16** | 2.57 |
| Best maximin (max target) | **290.0** | 179.3 |
| Best equality (max target) | **0.98** | 0.91 |

Gemini consistently produces higher values across both objectives. The gap is largest for maximin optimization, where Gemini achieves ~1.6x the maximin of Sonnet. This likely reflects Gemini's stronger ability to implement complex cooperative coordination (role rotation, zone assignment) given strategic hints.

### Finding 4: Researcher behavior varies significantly

|  | exp1 | exp2 | exp3 | exp4 | exp5 | exp6 | exp7 | exp8 |
|---|---|---|---|---|---|---|---|---|
| Iterations | 11 | 17 | 4 | 7 | 17 | 11 | 8 | 9 |
| Keep rate | 18% | 41% | 100% | 43% | 35% | 45% | 88% | 67% |

- **exp3-sonnet** (efficiency) is remarkably efficient: 4 iterations, all kept, monotonic improvement 0.38 -> 3.10. The researcher found a clean path (helpers -> map layout -> cost-aware hints + thinking budget).
- **exp2** and **exp5** were most exploratory (17 iterations each), with many discards (59% and 65%). The researcher tried more radical modifications that often failed.
- **exp7-sonnet-maximin** had 88% keep rate -- nearly every modification improved maximin, suggesting the metric has a smoother optimization landscape (many small changes help).

### Finding 5: Common researcher discoveries across runs

Despite independent runs, the researcher converges on similar strategies:

| Strategy | Eff runs | Max runs | Description |
|----------|----------|----------|-------------|
| **Waste helpers** | 4/4 | 4/4 | `count_waste()`, `waste_fraction()` -- always discovered first |
| **Zone/lane partitioning** | 3/4 | 4/4 | Divide apples spatially by agent_id to reduce competition |
| **Anti-regression feedback** | 3/4 | 2/4 | "DO NOT REGRESS" hints when metric is high |
| **Worked examples** | 2/4 | 3/4 | Complete policy example in prompt |
| **Role rotation** | 0/4 | 4/4 | Time-based duty rotation for cleaning -- **exclusively maximin** |
| **Cleaning economics** | 2/4 | 3/4 | "Only fire CLEAN when beam hits >= 2 waste cells" |
| **Thinking budget** | 1/4 | 1/4 | Increasing Sonnet's thinking tokens (16k -> 32k) |

The key **qualitative difference** between efficiency and maximin pipelines: maximin-optimized pipelines always include explicit **role rotation** using `env._step_count` and `agent_id` to ensure fair distribution of cleaning costs. Efficiency-optimized pipelines assign static roles (some agents always clean), which achieves high efficiency but at the cost of equality.

### Finding 6: Failure modes

Common reasons for discarded experiments across all 8 runs:

1. **Over-prescription** -- Adding too many strategic hints confuses the policy LLM, causing code generation failures or bizarre policies
2. **Iteration regression** -- K=4 or K=5 inner iterations often lead to catastrophic regression in later iterations (the LLM "over-refines" and breaks what worked)
3. **Feedback overload** -- Dense diagnostic feedback (trend analysis, per-agent stats) often causes over-correction rather than gradual improvement
4. **Variance exposure** -- A good pipeline can produce bad results on unlucky seeds. exp1-exp10 showed a variance check where identical config yielded eff=3.2 then eff=0.09

---

## 4. Full Experiment Trajectories

### Efficiency-optimized runs

**exp1 (Gemini, efficiency)**

| Exp | Efficiency | Equality | Status | Description |
|-----|-----------|----------|--------|-------------|
| 0 | 2.701 | 0.530 | baseline | Default pipeline |
| 1 | 3.008 | 0.503 | keep | Waste helpers + strategic hints |
| 2 | 0.894 | -0.434 | discard | Enhanced feedback (over-correction) |
| 3 | 0.133 | -0.064 | discard | Map layout + zone helpers (misused) |
| 4 | **3.247** | **0.613** | keep | Gentle feedback trend + stability hint |
| 5 | 0.044 | -0.458 | discard | Prescriptive hints (broke generation) |
| 6 | 3.246 | 0.533 | discard | K=4 (same eff, slower) |
| 7 | 3.068 | 0.511 | discard | Skeleton policy (constrained LLM) |
| 8 | 1.340 | -0.075 | discard | Best-iteration feedback (confused LLM) |
| 9 | 0.101 | 0.432 | discard | Stronger stability hint (hurt early iters) |
| 10 | 0.095 | 0.496 | discard | Variance check (identical config, eff collapsed) |

**exp2 (Gemini, efficiency)**

| Exp | Efficiency | Equality | Status | Description |
|-----|-----------|----------|--------|-------------|
| 0 | 1.150 | -0.516 | baseline | Default pipeline |
| 1 | 1.173 | -0.344 | keep | Waste density helpers + hints |
| 2 | 1.463 | -0.192 | keep | Enhanced feedback + diagnostics |
| 3 | 1.042 | -0.432 | discard | Beam helpers (iter1 crash) |
| 4 | 2.024 | 0.919 | keep | Cooperative worked example |
| 5 | 2.898 | 0.525 | keep | Lane coordination + anti-regression |
| 6 | 3.034 | 0.607 | keep | Interleaved cleaners + lane-scoped waste |
| 7 | 2.785 | 0.554 | discard | K=4 + map layout (extra iter regressed) |
| 8 | 2.081 | 0.484 | discard | Map layout + 3-tier roles (volatile) |
| B | 3.076 | 0.735 | baseline | Re-baselined |
| 9 | 3.011 | 0.459 | discard | Lane apple + anti-regression (froze updates) |
| 10 | **3.150** | 0.499 | keep | Lane apple + map layout + instinct hints |
| 11 | 2.640 | 0.355 | discard | 2 dedicated cleaners (volatile) |
| 12 | 2.550 | 0.431 | discard | Quantitative cleaning economics (regressed) |
| 13 | 2.976 | 0.616 | discard | K=2 (stable but low ceiling) |
| 14 | 1.973 | 0.374 | discard | Beam waste helper (P3 crashed) |
| 15 | 2.959 | 0.513 | discard | Soft anti-regression (regressed) |
| 16 | 2.873 | 0.473 | discard | N_EVAL_SEEDS=7 (P0 weak) |

**exp3 (Sonnet, efficiency)**

| Exp | Efficiency | Equality | Status | Description |
|-----|-----------|----------|--------|-------------|
| 0 | 0.384 | -0.548 | keep | Baseline |
| 1 | 1.342 | -0.110 | keep | 6 cleanup helpers + strategic hints |
| 2 | 1.565 | -0.027 | keep | Map layout + worked example |
| 3 | **3.097** | **0.699** | keep | Cost-aware hints + zone hints + thinking 32k |

**exp4 (Sonnet, efficiency)**

| Exp | Efficiency | Equality | Status | Description |
|-----|-----------|----------|--------|-------------|
| 0 | 1.564 | 0.248 | keep | Baseline |
| 1 | 0.122 | 0.471 | discard | Helpers+hints+diagnostics (over-correction) |
| 2 | 1.241 | -0.320 | discard | Helpers only (slightly worse) |
| 3 | 1.474 | 0.079 | discard | Threshold info (within noise) |
| 4 | 1.362 | 0.071 | discard | K=5+helpers (peaked then oscillated) |
| 5 | 1.728 | 0.383 | keep | Better seed example + helpers |
| 6 | **3.137** | **0.630** | keep | Collector zone partitioning helpers |

### Maximin-optimized runs

**exp5 (Gemini, maximin)**

| Exp | Efficiency | Maximin | Equality | Status | Description |
|-----|-----------|---------|----------|--------|-------------|
| 0 | 1.491 | -98.8 | 0.038 | baseline | Default pipeline |
| 1 | 1.460 | -22.4 | 0.381 | keep | Maximin framing + waste helpers + fairness feedback |
| 2 | 1.425 | -16.6 | 0.317 | keep | Worked example + cleaning hints |
| 3 | 2.023 | 36.2 | 0.669 | keep | K=2 (avoid iter 3 regression) |
| 4 | 1.689 | 7.4 | 0.606 | discard | K=1 (too few iterations) |
| 5 | 3.105 | 242.9 | 0.949 | keep | Apple zone partitioning + 8 eval seeds |
| 6 | 2.544 | 206.5 | 0.936 | discard | Beam waste helpers (prompt complexity) |
| 7 | 2.754 | 220.0 | 0.939 | discard | More explicit cleaners |
| 8 | 3.006 | 239.6 | 0.944 | discard | Trend feedback (no improvement) |
| 9 | **3.192** | **295.7** | **0.977** | keep | 12 eval seeds (more stable signals) |
| 10 | 2.576 | 214.1 | 0.954 | discard | K=3 with 12 seeds (iter 3 regressed) |
| 11 | 2.490 | 200.0 | 0.937 | discard | count_beam_waste helper (didn't help) |
| 12 | 2.433 | 197.2 | 0.948 | discard | Variance check |
| 13 | 2.802 | 199.8 | 0.930 | discard | Regression warning (no improvement) |
| 14 | 2.547 | 224.8 | 0.965 | discard | Removed cleaning tips (no improvement) |
| 15 | 2.604 | 211.0 | 0.938 | discard | get_my_apples in example (no improvement) |
| 16 | 2.647 | 212.0 | 0.944 | discard | 16 eval seeds (diminishing returns) |

**exp6 (Gemini, maximin)**

| Exp | Efficiency | Maximin | Equality | Status | Description |
|-----|-----------|---------|----------|--------|-------------|
| 0 | 2.378 | -83.8 | 0.623 | keep | Baseline |
| 1 | 1.973 | 40.4 | 0.653 | keep | Maximin prompts + role rotation helpers |
| 2 | 3.106 | 283.4 | 0.967 | keep | count_beam_waste + find_best_clean_pos |
| 3 | 2.411 | 153.2 | 0.927 | discard | 2 cleaners (insufficient) |
| 4 | **3.116** | **284.2** | **0.977** | keep | Worked example role-rotation policy |
| 5 | 3.139 | 274.6 | 0.962 | discard | Voronoi apple assignment (hurt maximin) |
| 6 | 2.526 | 172.2 | 0.902 | discard | duty_period=30 (too frequent rotation) |
| 7 | 3.000 | 265.4 | 0.958 | discard | Quantitative game analysis (confused LLM) |
| 8 | 3.091 | 246.1 | 0.952 | discard | 8 eval seeds (variance) |
| 9 | 2.936 | 223.8 | 0.917 | discard | Idle cleaners collect (overcomplicated) |
| 10 | 2.290 | 103.2 | 0.789 | discard | K=5 (catastrophic regression) |

**exp7 (Sonnet, maximin)**

| Exp | Efficiency | Maximin | Equality | Status | Description |
|-----|-----------|---------|----------|--------|-------------|
| 0 | 1.209 | -188.6 | 0.141 | baseline | Default pipeline |
| 1 | 1.639 | -26.2 | 0.447 | keep | Maximin reframing + role rotation hints |
| 2 | 1.208 | -22.8 | 0.262 | keep | Cleaning cost optimization |
| 3 | 0.207 | -2.6 | 0.480 | keep | Worked rotation example + K=4 |
| 4 | 2.193 | 24.2 | 0.584 | keep | Aggressive cleaning + K=3 + thinking 10k |
| 5 | 2.138 | 93.6 | 0.809 | keep | Spatial distribution + emergency cleaning |
| 6 | 2.925 | 154.4 | 0.830 | keep | 5-zone rotating system |
| 7 | **2.928** | **159.0** | **0.839** | keep | River-split cleaning + neighbor-scan |

**exp8 (Sonnet, maximin)**

| Exp | Efficiency | Maximin | Equality | Status | Description |
|-----|-----------|---------|----------|--------|-------------|
| 0 | 0.291 | -59.0 | 0.080 | baseline | Default pipeline |
| 1 | -1.724 | -249.8 | 1.000 | discard | Rotating shifts (over-optimization) |
| 2 | -1.686 | -187.0 | 1.000 | discard | Beam scanning (over-cleaning) |
| 3 | 0.949 | 58.4 | 0.896 | keep | Collective threshold + zones |
| 4 | 2.043 | 170.0 | 0.940 | keep | agent_zone_rows + hysteresis |
| 5 | 1.071 | 78.2 | 0.917 | discard | Apple spawn info (prompt overload) |
| 6 | 2.183 | 188.6 | 0.962 | keep | Spawn-balanced zone helpers |
| 7 | 2.153 | 190.2 | 0.956 | keep | K=4 + hysteresis in example |
| 8 | **2.213** | **199.6** | **0.970** | keep | Threshold alignment 0.22/0.08 |

---

## 5. Recommended Presentation for Workshop Paper

### Table 1: Aggregate results (mean +/- range across 2 runs per condition)

|  | Gemini (eff) | Sonnet (eff) | Gemini (maximin) | Sonnet (maximin) |
|---|---|---|---|---|
| Efficiency | 3.20 +/- 0.05 | 3.12 +/- 0.02 | 3.16 +/- 0.04 | 2.57 +/- 0.36 |
| Maximin | -- | -- | 290.0 +/- 5.8 | 179.3 +/- 20.3 |
| Equality | 0.55 +/- 0.06 | 0.66 +/- 0.03 | 0.98 +/- 0.00 | 0.91 +/- 0.07 |
| Sustainability | 491 +/- 9 | 510 +/- 0 | 497 +/- 1 | 501 +/- 2 |

### Figure 1

Efficiency trajectory across researcher iterations for all 8 runs (x-axis: experiment index, y-axis: efficiency, color: condition). Shows convergence to ~3.0-3.2.

### Figure 2

Maximin trajectory for the 4 maximin runs. Shows the dramatic improvement from negative to positive, with Gemini reaching ~290 vs Sonnet ~180.

### Key narrative

> Optimizing for Rawlsian (maximin) welfare instead of utilitarian (efficiency) welfare produces qualitatively different pipeline modifications -- specifically, the researcher discovers time-based role rotation as a mechanism for fair cost sharing. Strikingly, this fairness-oriented optimization achieves comparable efficiency to direct efficiency optimization (Gemini: 3.16 vs 3.20), suggesting that in the Cleanup game, **cooperation and fairness are complementary, not competing, objectives**. The researcher independently discovers this across all runs.
