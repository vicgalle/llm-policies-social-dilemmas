# LLM Policy Synthesis for Sequential Social Dilemmas — Autoresearch

## What this repo does

This repo implements **iterative LLM policy synthesis** for multi-agent Sequential Social Dilemmas (SSDs). An LLM generates Python policy functions, evaluates them in self-play, and refines them using performance feedback. The paper: [Gallego 2026, "Cooperation and Exploitation in LLM Policy Synthesis for Sequential Social Dilemmas"](https://arxiv.org/abs/2603.19453).

On top of this, we built an **autoresearch layer** (inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch)) where a *researcher agent* autonomously modifies the synthesis pipeline to improve the resulting policies.

## Architecture: Two-Level Framework

```
OUTER LOOP — Researcher agent (Claude Opus)
  Modifies: pipeline/ (prompts, feedback, helpers, config)
  Observes: efficiency metric after each inner loop run
  Loop: modify pipeline → run inner loop → keep/discard → repeat

INNER LOOP — Policy synthesizer LLM (Gemini 3.1 Pro or Claude Sonnet)
  Reads: system prompt, feedback from previous iteration
  Generates: Python policy function
  Evaluated: in self-play (all agents run same policy)
  Loop: generate → validate → evaluate → feedback → refine (K=3 iterations)

FROZEN — Not modifiable by the researcher
  Environments: cleanup_env.py, gathering_env.py
  Evaluation: metric computation (efficiency, equality, sustainability, peace)
  Orchestrator: run_inner_loop.py
  Infrastructure: llm_self_play.py (LLM calls, validation, policy loading)
```

## File structure

### Original codebase (frozen infrastructure)
- `llm_self_play.py` — Main self-play framework (LLM calls, validation, evaluation, prompt construction). Supports both Claude (via Agent SDK) and Gemini (via google-genai).
- `cleanup_env.py` — Cleanup game environment (public goods dilemma: clean pollution vs collect apples)
- `gathering_env.py` — Gathering game environment (common pool resource dilemma)
- `gathering_policy.py` — Base policy utilities, BFS pathfinding, helper functions

### Pipeline (modifiable by researcher)
- `pipeline/prompts.py` — System prompts for the policy LLM ($p$ in the formalism)
- `pipeline/feedback.py` — Feedback construction: what metrics/info the policy LLM sees between iterations ($\ell$, $\phi$)
- `pipeline/helpers.py` — Extra helper functions injected into the policy namespace ($\mathcal{H}$)
- `pipeline/config.py` — Iteration parameters: K, eval seeds, retries ($\iota$)

### Orchestrator
- `run_inner_loop.py` — Composes pipeline/ with frozen infrastructure, runs the inner loop, outputs metrics as JSON. Args: `--game`, `--model`, `--map`, `--n-agents`, `--output-dir`.

### Autoresearch infrastructure
- `autoresearch/program.md` — Full instructions for the researcher agent (the "research program")
- `autoresearch/measure.sh` — Runs inner loop and reports metrics. Usage: `./autoresearch/measure.sh [sparse|dense] [--metric efficiency|maximin] [--model MODEL ...]`
- `autoresearch/run_experiment.sh` — Launches an autonomous researcher run. Usage: `./autoresearch/run_experiment.sh <tag> [feedback_mode] [researcher_model] [policy_model] [metric]`
- `autoresearch/analyze.py` — Results analysis and convergence plots
- `autoresearch/results.tsv` — Experiment log (tab-separated)
- `autoresearch/runs/` — Per-run output directories with policies, metrics, history

### Paper notes
- `autoresearch_ssd.tex` — Formal description of the two-level framework, connection to automated mechanism design, experimental plan

## Key concepts

### Games
- **Cleanup** (primary): 2D gridworld with river (pollution) and orchard (apples). CLEAN action removes waste (-1 cost) but enables apple growth for everyone. Public goods dilemma. 9 actions.
- **Gathering**: 2D gridworld with apples and tagging beams. Common pool resource. 8 actions.

### Metrics (Perolat et al. 2017)
- **Efficiency (U)**: collective reward per timestep (higher = more apples collected). Default optimization target.
- **Maximin**: minimum per-agent total return (Rawlsian welfare). Maximizes worst-off agent's reward. Alternative optimization target.
- **Equality (E)**: reward fairness via Gini coefficient (1.0 = perfect equality).
- **Sustainability (S)**: average time rewards are collected (higher = resources preserved later).
- **Peace (P)**: absence of beam aggression (higher = less conflict).

### Feedback modes
- **Sparse** (reward-only): policy LLM sees only average reward per iteration
- **Dense** (reward+social): policy LLM sees reward + all 4 social metrics with definitions

### Models
- Policy LLM: `gemini-3.1-pro-preview` (default) or `claude-sonnet-4-6` (pass `--model`)
- Researcher agent: Claude Opus (via Claude Code CLI)

## Running experiments

```bash
# Single inner loop run (no researcher, just evaluate current pipeline)
uv run run_inner_loop.py --game cleanup --model gemini-3.1-pro-preview --map large --n-agents 10

# Same but with Sonnet as the policy LLM
uv run run_inner_loop.py --game cleanup --model claude-sonnet-4-6 --map large --n-agents 10

# Measure current pipeline (used by researcher agent)
./autoresearch/measure.sh dense                              # Gemini (default), optimize efficiency
./autoresearch/measure.sh dense --model claude-sonnet-4-6    # Sonnet
./autoresearch/measure.sh dense --metric maximin             # Optimize maximin (Rawlsian welfare)

# Launch autonomous researcher (Opus researcher, Gemini policy LLM)
./autoresearch/run_experiment.sh mar30-test dense

# Launch with Sonnet as the policy LLM
./autoresearch/run_experiment.sh mar30-sonnet dense opus claude-sonnet-4-6

# Launch optimizing maximin instead of efficiency
./autoresearch/run_experiment.sh apr03-rawls dense opus gemini-3.1-pro-preview maximin

# Analyze results
python3 autoresearch/analyze.py
```

## Design principles

1. **Separation of concerns**: Researcher $\mathcal{R}$ and synthesizer $\mathcal{M}$ are distinct LLM calls. $\mathcal{R}$ reasons about *what information helps $\mathcal{M}$ write better policies*; $\mathcal{M}$ reasons about *what policy to write*.
2. **Read-only environment**: The researcher can read env source code but cannot modify it. Prevents trivially making the game easier.
3. **Fixed compute budget**: Each inner loop runs exactly K iterations with fixed seeds. The researcher changes *what happens within* those iterations, not how many there are.
4. **Diff-based history**: The researcher sees code diffs + outcomes from previous experiments, enabling causal reasoning about which modifications drove improvements.
