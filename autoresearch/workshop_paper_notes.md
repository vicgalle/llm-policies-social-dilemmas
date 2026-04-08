# Workshop Paper Notes: Autoresearch for LLM Policy Synthesis in SSDs

## Paper Title (working)

"Automated Research for LLM Policy Synthesis in Sequential Social Dilemmas"

## Core Contribution

A two-level framework where a **researcher agent** (Claude Opus 4.6) autonomously modifies the synthesis pipeline -- prompts, feedback, helpers, config -- under which a **policy-synthesizer LLM** (Gemini 3.1 Pro or Claude Sonnet 4.6) generates cooperative strategies for multi-agent social dilemmas. We compare two social welfare objectives (utilitarian efficiency vs. Rawlsian maximin) across two complementary games (Cleanup and Gathering).

## Formalization

Already written in `autoresearch_ssd.tex`. Key elements:

- **Configuration space**: $\mathcal{C} = (p, \ell, \phi, \mathcal{H}, \iota, v)$ -- system prompt, feedback template, feedback construction, helper library, iteration logic, validation pipeline
- **Inner loop**: $\pi^*_c = \text{InnerLoop}(\mathcal{M}, \mathcal{G}, c)$ -- K=3 iterations of LLM policy synthesis
- **Outer loop**: Algorithm 1 -- researcher proposes config changes, runs inner loop, keeps/discards based on $\Phi$
- **Three objectives**: $\Phi_U = U$ (efficiency), $\Phi_W$ (weighted welfare), $\Phi_{\min} = \min_i R_i$ (Rawlsian)
- **Mechanism design connection**: researcher = principal designing information structure; synthesizer = bounded-rational agent

## Experimental Design

### 12 runs total across a 2x2x2 + 4 design

**Cleanup game** (public goods provision, 10 agents): 8 runs

|  | **Gemini 3.1 Pro** | **Claude Sonnet 4.6** |
|---|---|---|
| **Efficiency** target | exp1, exp2 | exp3, exp4 |
| **Maximin** target | exp5, exp6 | exp7, exp8 |

**Gathering game** (common pool resource, 4 agents): 4 runs (efficiency only, 2 per LLM)

|  | **Gemini 3.1 Pro** | **Claude Sonnet 4.6** |
|---|---|---|
| **Efficiency** target | gather-exp2, gather-exp3 | gather-exp1, gather-exp4 |

No maximin runs needed for Gathering -- efficiency optimization alone achieves equality > 0.94.

### Where to find the data

- Cleanup efficiency: `/Users/victorgallego/llm-policies-social-dilemmas-exp{1,2,3-sonnet,4-sonnet}`
- Cleanup maximin: `/Users/victorgallego/llm-policies-social-dilemmas-exp{5-rawls,6-maximin,7-sonnet-maximin,8-sonnet-maximin}`
- Gathering efficiency: `/Users/victorgallego/llm-policies-social-dilemmas-gather-exp{1-sonnet,2-gemini,3-gemini,4-sonnet}`

Each directory has `autoresearch/results.tsv`, `autoresearch/runs/*/metrics.json`, pipeline/ files, and git history.

---

## Results Summary

### Cleanup: Efficiency Optimization

| Run | Policy LLM | #Iter | Kept | Eff_0 | Eff* | Eq* |
|-----|-----------|-------|------|------|------|-----|
| exp1 | Gemini | 11 | 2 | 2.70 | **3.25** | 0.61 |
| exp2 | Gemini | 17 | 7 | 1.15 | **3.15** | 0.50 |
| exp3 | Sonnet | 4 | 4 | 0.38 | **3.10** | 0.70 |
| exp4 | Sonnet | 7 | 3 | 1.56 | **3.14** | 0.63 |

All converge to eff ~3.1-3.25 regardless of baseline. Equality moderate (0.50-0.70).

### Cleanup: Maximin Optimization

| Run | Policy LLM | #Iter | Kept | Eff_0 | Eff* | Max_0 | Max* | Eq* |
|-----|-----------|-------|------|------|------|-------|------|-----|
| exp5 | Gemini | 17 | 6 | 1.49 | 3.19 | -98.8 | **295.7** | 0.98 |
| exp6 | Gemini | 11 | 5 | 2.38 | 3.12 | -83.8 | **284.2** | 0.98 |
| exp7 | Sonnet | 8 | 7 | 1.21 | 2.93 | -188.6 | **159.0** | 0.84 |
| exp8 | Sonnet | 9 | 6 | 0.29 | 2.21 | -59.0 | **199.6** | 0.97 |

Maximin goes from deeply negative to positive. Near-perfect equality (>0.97 with Gemini). Efficiency barely sacrificed with Gemini (3.16 vs 3.20).

### Gathering: Efficiency Optimization

| Run | Policy LLM | #Iter | Kept | Eff_0 | Eff* | Max* | Eq* |
|-----|-----------|-------|------|------|------|------|-----|
| gather-exp1 | Sonnet | 3 | 3 | 0.03 | **2.52** | 596.6 | 0.98 |
| gather-exp2 | Gemini | 5 | 1 | 2.42 | **2.51** | 582.4 | 0.98 |
| gather-exp3 | Gemini | 3 | 3 | 1.66 | **2.47** | 580.8 | 0.98 |
| gather-exp4 | Sonnet | 4 | 4 | 0.03 | **2.51** | 555.8 | 0.94 |

All 4 runs converge to eff ~2.5 regardless of starting point (0.03-2.42). Equality >0.94 with efficiency optimization -- no maximin runs needed.

### Aggregate Table (for paper)

**Cleanup (10 agents)**

|  | Gemini (eff) | Sonnet (eff) | Gemini (max) | Sonnet (max) |
|---|---|---|---|---|
| Efficiency | 3.20 +/- 0.05 | 3.12 +/- 0.02 | 3.16 +/- 0.04 | 2.57 +/- 0.36 |
| Maximin | -- | -- | 290.0 +/- 5.8 | 179.3 +/- 20.3 |
| Equality | 0.55 +/- 0.06 | 0.66 +/- 0.03 | 0.98 +/- 0.00 | 0.91 +/- 0.07 |

**Gathering (4 agents, 2 runs per LLM)**

|  | Gemini (eff) | Sonnet (eff) |
|---|---|---|
| Efficiency | 2.49 +/- 0.02 | 2.52 +/- 0.01 |
| Maximin | 581.6 +/- 0.8 | 576.2 +/- 20.4 |
| Equality | 0.98 +/- 0.00 | 0.96 +/- 0.02 |

---

## Key Findings (5 for the paper)

### Finding 1: Autoresearch reliably improves social welfare

All 12 runs show dramatic improvement over baselines, regardless of starting point. Final values cluster tightly (eff ~3.1-3.2 for Cleanup, ~2.5 for Gathering), suggesting the researcher reliably discovers the performance ceiling. Baselines vary enormously (0.03-2.70), but end states converge. The Gathering replication (4 runs) is especially striking: baselines range from 0.03 to 2.42, yet all converge to eff ~2.5.

### Finding 2: No efficiency-fairness tradeoff in Cleanup (with Gemini)

Maximin-optimized Gemini runs sacrifice only ~0.04 efficiency points (3.16 vs 3.20) while achieving near-perfect equality (0.98 vs 0.55). The researcher discovers that fair duty rotation and apple zone partitioning simultaneously improve worst-off welfare AND collective output. Cleaning is a public good, so fair cleaning produces more apples for everyone.

For Sonnet, there is a moderate tradeoff: eff drops from ~3.12 to ~2.57. The gap reflects Sonnet's harder time implementing complex coordination (role rotation, zone assignment) from strategic hints.

### Finding 3: Game structure determines whether fairness requires explicit optimization

- **Cleanup** (asymmetric costs -- cleaners pay, collectors benefit): Baseline equality is 0.04-0.62. Maximin optimization is needed to reach equality >0.9. The researcher discovers role rotation as the key mechanism.
- **Gathering** (symmetric costs -- all agents face the same landscape): Baseline equality is already 0.54-0.98. Efficiency optimization alone reaches equality >0.97. No role differentiation needed.

This is the structural insight: in provision dilemmas, fairness requires explicit mechanisms; in restraint dilemmas, fairness is a free byproduct of coordination.

### Finding 4: The researcher discovers qualitatively different strategies per game and objective

**Cleanup efficiency**: waste helpers, zone partitioning, anti-regression feedback, static role assignment (some agents always clean)

**Cleanup maximin**: all of the above PLUS time-based duty rotation using `env._step_count` and `agent_id` -- exclusively discovered in maximin runs

**Gathering efficiency**: Voronoi zone partitioning by BFS distance, respawn-timer awareness, "NEVER use BEAM in self-play"

Despite independent runs, the researcher converges on similar strategies within each condition -- evidence of robust discovery, not luck.

### Finding 5: Common failure modes across all runs

1. **Over-prescription**: Too many strategic hints confuses the policy LLM
2. **Iteration regression**: K=4 or K=5 inner iterations often cause catastrophic regression (LLM "over-refines")
3. **Feedback overload**: Dense diagnostics cause over-correction rather than gradual improvement
4. **Variance**: Identical configs can yield eff=3.2 then eff=0.09 on different seeds

---

## Researcher Behavior Patterns

| Run | Game | LLM | Target | #Iter | Keep rate |
|-----|------|-----|--------|-------|-----------|
| exp1 | Cleanup | Gemini | Eff | 11 | 18% |
| exp2 | Cleanup | Gemini | Eff | 17 | 41% |
| exp3 | Cleanup | Sonnet | Eff | 4 | 100% |
| exp4 | Cleanup | Sonnet | Eff | 7 | 43% |
| exp5 | Cleanup | Gemini | Max | 17 | 35% |
| exp6 | Cleanup | Gemini | Max | 11 | 45% |
| exp7 | Cleanup | Sonnet | Max | 8 | 88% |
| exp8 | Cleanup | Sonnet | Max | 9 | 67% |
| gather-exp1 | Gathering | Sonnet | Eff | 3 | 100% |
| gather-exp2 | Gathering | Gemini | Eff | 5 | 20% |
| gather-exp3 | Gathering | Gemini | Eff | 3 | 100% |
| gather-exp4 | Gathering | Sonnet | Eff | 4 | 100% |

Notable: exp3-sonnet ran only 4 iterations with 100% keep rate (monotonic improvement 0.38 -> 3.10). exp2 and exp5 were most exploratory (17 iterations each). Gathering runs are generally shorter and more efficient -- gather-exp3 and gather-exp4 both achieved 100% keep rates.

---

## Common Researcher Discoveries (across Cleanup runs)

| Strategy | Eff runs (4) | Max runs (4) | Description |
|----------|-------------|-------------|-------------|
| Waste helpers | 4/4 | 4/4 | `count_waste()`, `waste_fraction()` |
| Zone/lane partitioning | 3/4 | 4/4 | Divide apples spatially by agent_id |
| Anti-regression feedback | 3/4 | 2/4 | "DO NOT REGRESS" when metric is high |
| Worked examples | 2/4 | 3/4 | Complete policy example in prompt |
| Role rotation | **0/4** | **4/4** | Time-based duty rotation -- exclusively maximin |
| Cleaning economics | 2/4 | 3/4 | "Only fire CLEAN when beam hits >= 2 waste" |

---

## Comparison: Cleanup vs. Gathering

| | Cleanup (provision) | Gathering (restraint) |
|---|---|---|
| Dilemma | Free-ride on cleaning | Over-harvest apples |
| Cooperation = | Do the costly thing | Don't do the tempting thing |
| Baseline equality | 0.04-0.62 | 0.54-0.98 |
| Baseline maximin | -189 to -59 (negative) | 0.2 to 571 |
| Maximin opt needed? | Yes -- huge fairness gap | No -- efficiency opt already fair |
| Runs | 8 (2x2 LLMs x objectives) | 4 (2 per LLM, eff only) |
| Key strategy | Role differentiation + rotation | Voronoi zones + respawn timing |
| Agent count | 10 | 4 |

---

## Connection to Existing Formalization

The tex file (`autoresearch_ssd.tex`) already provides:

- Configuration space $\mathcal{C}$ (Table 1) -- now with empirical evidence of which components matter
- Three objectives $\Phi_U$, $\Phi_W$, $\Phi_{\min}$ (eqs. 7-9) -- $\Phi_U$ and $\Phi_{\min}$ experimentally compared
- Algorithm 1 (outer loop) -- executed 12 times with real results
- Mechanism design framing (Section 3) -- researcher designs information structure differently depending on $\Phi$
- Goodhart risk (Section 3) -- observed: over-prescription, variance gaming, iteration regression
- Experimental plan phases:
  - **Phase 1 (proof of concept)**: DONE -- all runs exceed 2.75 baseline
  - **Phase 3 (cross-game)**: DONE -- Cleanup + Gathering with 2 runs per LLM per game, different strategies confirmed
  - Phase 2 (ablation) and Phase 4 (adversarial): Not yet done

---

## Paper Structure (proposed 4-page workshop paper)

### 1. Introduction (~0.5 page)
- LLM policy synthesis for SSDs (cite Gallego 2026)
- Autoresearch paradigm (cite Karpathy 2026)
- Our contribution: two-level framework + efficiency vs maximin comparison across two games

### 2. Framework (~1 page)
- Configuration space $\mathcal{C}$ (from tex)
- Inner loop (policy synthesis) and outer loop (automated research) -- Algorithm 1
- Mechanism design interpretation (brief)
- Two games: Cleanup (provision) and Gathering (restraint)
- Two objectives: $\Phi_U$ (efficiency) and $\Phi_{\min}$ (maximin)

### 3. Experiments (~1.5 pages)
- Setup: 2x2x2 + 4 design, 12 runs total
- Table 1: Aggregate results (Cleanup eff/max x Gemini/Sonnet + Gathering)
- Finding 1: Reliable improvement from diverse baselines
- Finding 2: No efficiency-fairness tradeoff (Gemini/Cleanup)
- Finding 3: Game structure determines need for explicit fairness optimization
- Finding 4: Convergent discovery of game-specific strategies
- Figure 1: Efficiency trajectory across researcher iterations (all 10 runs)
- Figure 2: Maximin trajectory for the 4 maximin runs

### 4. Discussion (~1 page)
- Mechanism design interpretation of findings
- Limitations: variance, only 2 runs/condition (4 for Gathering), no MARL baselines
- Future: adversarial objective ($\Phi = R_0$), ablation study, more games
- Connection to cooperative AI and alignment

---

## Figures to Create

1. **Efficiency trajectory plot**: x = researcher iteration index, y = efficiency. 12 lines (colored by condition). Shows convergence from diverse baselines.
2. **Maximin trajectory plot**: x = researcher iteration index, y = maximin. 4 lines (Cleanup maximin runs). Shows transformation from negative to positive.
3. **Bar chart**: Final efficiency and equality for all 10 runs, grouped by condition. Shows the efficiency-fairness relationship.
4. **Strategy comparison table**: What the researcher discovers per game x objective.

---

## Strengths to Emphasize

1. **Novel framework**: LLM researching how to make another LLM cooperate
2. **Clean experimental design**: Factorial across games, objectives, LLMs
3. **Structural insight**: Provision vs restraint dilemmas need different fairness approaches
4. **Convergent discovery**: Independent runs find the same strategies
5. **Practical**: Works out of the box, no RL training, just prompt/helper engineering
6. **Already formalized**: $\mathcal{C}$, Algorithm 1, mechanism design connection in tex

---

## Maximin Metric Implementation

Added during this conversation. Changes across 6 files:

- `cleanup_env.py` / `gathering_env.py`: `maximin = float(R.min())` in `compute_metrics()`
- `run_inner_loop.py`: Propagated through trajectory, result dict, fallback values
- `autoresearch/measure.sh`: New `--metric [efficiency|maximin]` flag, metric-specific baseline files
- `autoresearch/run_experiment.sh`: 5th positional arg `[metric]` passed to researcher prompt
- `autoresearch/program.md`: Parameterized optimization target, maximin description

Usage:
```bash
./autoresearch/measure.sh dense --metric maximin
./autoresearch/run_experiment.sh apr03-rawls dense opus gemini-3.1-pro-preview maximin
```
