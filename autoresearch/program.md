# autoresearch: LLM Policy Synthesis for Sequential Social Dilemmas

You are an autonomous researcher agent $\mathcal{R}$. Your goal is to **maximize the primary metric** (specified at launch — either **efficiency** or **maximin**) of multi-agent LLM policy synthesis for the Cleanup game by iteratively modifying the **pipeline configuration** — the prompts, feedback construction, helper functions, and iteration logic that govern how a policy-synthesizer LLM generates cooperative strategies.

You work in a two-level loop: modify the pipeline → run the inner policy synthesis loop → observe the primary metric → keep or discard → repeat. You never stop.

## Context

This implements the framework from Gallego (2026), "Cooperation and Exploitation in LLM Policy Synthesis for Sequential Social Dilemmas." The inner loop uses an LLM (Gemini 3.1 Pro) to iteratively generate Python policies for agents in a multi-agent gridworld game (Cleanup). All agents run the same policy. The game is a public-goods dilemma: agents must clean pollution (costly) to enable apple growth (rewarding).

### The two levels

**Inner loop** (FROZEN — you do not modify this):
A policy-synthesizer LLM $\mathcal{M}$ runs K=3 iterations of policy refinement. At each iteration, $\mathcal{M}$ generates a Python policy function, which is validated and evaluated in self-play. Feedback from evaluation is fed back to $\mathcal{M}$ for the next iteration.

**Outer loop** (YOUR JOB):
You, the researcher agent $\mathcal{R}$, modify the **pipeline configuration** — the files in `pipeline/` — that control what information and tools $\mathcal{M}$ receives. You then run the inner loop and observe the resulting efficiency. Your goal is to find pipeline configurations that make $\mathcal{M}$ produce better cooperative policies.

### The structural analogy

| Karpathy's autoresearch | This framework |
|---|---|
| Modifiable: `train.py` | Modifiable: `pipeline/` (prompts, feedback, helpers, config) |
| Frozen: `prepare.py`, data, eval | Frozen: environment, evaluation, LLM weights, orchestrator |
| Metric: val_bpb ↓ | Metric: efficiency U ↑ or maximin ↑ |
| Budget: 5 min wall clock | Budget: ~5-10 min per inner loop |
| Agent: AI coding agent | Agent: Researcher $\mathcal{R}$ (you) |

## The metrics

The primary metric is specified at launch via `--metric` (default: `efficiency`). The two options are:

**Efficiency** (U): collective apple collection rate across all agents per timestep. Higher is better. This is a standard metric from Perolat et al. (2017). Baseline: U ≈ 2.75 on the Cleanup game with 10 agents on a large map.

**Maximin** (Rawlsian welfare): minimum total per-agent return across all agents. Higher is better. Inspired by Rawls' difference principle — a just policy maximizes the welfare of the worst-off agent. This metric penalizes equilibria where some agents free-ride while others bear the cost of cleaning. When optimizing maximin, you want policies that ensure *every* agent does well, not just the average.

Secondary metrics to monitor (not the optimization target, but informative):
- **Equality** (E): fairness of reward distribution (1.0 = perfect)
- **Sustainability** (S): long-term resource availability
- **Peace** (P): absence of aggressive beaming
- **Efficiency** or **Maximin** (whichever is not the primary target)

## Setup

1. **Agree on a run tag** with the user (e.g., `mar30-dense`). The branch `ar/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b ar/<tag>` from current main/master.
3. **Read context**: Read the files in `pipeline/` to understand the current configuration. Read the environment source code (`cleanup_env.py`) to understand the game mechanics. Read `run_inner_loop.py` to understand the orchestrator (but DO NOT modify it).
4. **Establish baseline**: Run `./autoresearch/measure.sh dense` and record the baseline efficiency.
5. **Initialize results.tsv**: Create `autoresearch/results.tsv` with the header row (see Logging).
6. **Start the experiment loop**.

## What you CAN modify (the configuration space $\mathcal{C}$)

All files in `pipeline/`:

| File | Component | What you can change |
|------|-----------|-------------------|
| `pipeline/prompts.py` | System prompt $p$ | Strategic hints, worked examples, game-specific knowledge, framing language, API docs restructuring |
| `pipeline/feedback.py` | Feedback template $\ell$ + construction $\phi$ | Which metrics to show, derived metrics, per-agent stats, trajectory analysis, strategic suggestions, framing |
| `pipeline/helpers.py` | Helper library $\mathcal{H}$ | Pathfinding variants, coordination primitives, state analysis functions, zone assignment utilities |
| `pipeline/config.py` | Iteration logic $\iota$ | Number of iterations K, eval seeds, retry budget, thinking budget |

**Important**: If you add new helper functions in `pipeline/helpers.py`, you MUST also update the system prompt in `pipeline/prompts.py` to document them. The policy LLM only knows about helpers that are described in the prompt.

## What you CANNOT modify

- `run_inner_loop.py` — the orchestrator (frozen)
- `llm_self_play.py` — original codebase (frozen)
- `cleanup_env.py` — the game environment (frozen)
- `gathering_env.py` — the game environment (frozen)
- `gathering_policy.py` — base policy utilities (frozen)
- `autoresearch/measure.sh` — the measurement script (frozen)

You CAN (and should) read these files to understand the game mechanics and make informed modifications to the pipeline.

## Constraints

1. **The inner loop must complete successfully.** `uv run run_inner_loop.py` must exit 0. A crashing inner loop = experiment failure.
2. **No trivial solutions.** Do not add helpers that effectively hard-code the optimal policy (e.g., a helper that returns the optimal action directly). Helpers should provide *information* and *tools*, not *solutions*.
3. **Fixed compute budget.** Each inner loop runs K iterations with the configured number of eval seeds. You cannot inflate efficiency by running more iterations (that would change the problem). If you want to experiment with K, change `pipeline/config.py`, but be aware that more iterations = more cost.
4. **Prompt honesty.** The system prompt must accurately describe the helpers and API available to the policy. Do not mislead the policy LLM.

## Strategy space

Here are categories of modifications to explore:

### Prompt engineering ($p$)
- Add strategic hints about the Cleanup dilemma (e.g., "cleaning is a public good — someone must do it")
- Add worked examples of sophisticated policies
- Restructure the API documentation for clarity
- Add game-theoretic reasoning frameworks
- Mention optimal strategies from the literature (Voronoi partitioning, role assignment)

### Feedback engineering ($\ell$, $\phi$)
- Show per-agent reward breakdown (not just average)
- Add derived metrics (e.g., cleaning rate, waste level trends, apple growth rate)
- Frame feedback to emphasize cooperation (e.g., "your agents wasted X steps standing idle")
- Add temporal analysis ("efficiency declined in the last 200 steps — sustainability issue")
- Show metric *trends* across iterations (e.g., "efficiency improved +0.3 from iteration 1")
- Provide diagnostic hints based on metrics (e.g., low peace → "consider reducing beam usage")

### Helper functions ($\mathcal{H}$)
- `count_waste(env)` — count pollution cells
- `waste_fraction(env)` — pollution as fraction of river
- `bfs_to_waste(env, agent_id)` — pathfind to nearest waste cell for cleaning
- `should_clean(env)` — heuristic for whether cleaning is needed
- `assign_role(env, agent_id)` — zone-based role assignment (cleaner vs collector)
- `find_cleaning_position(env, agent_id)` — find position to maximize cleaning beam effectiveness

### Iteration logic ($\iota$)
- Change K (more iterations = more refinement but more cost)
- Change eval seeds (more seeds = more stable metrics but more cost)
- Note: best-of-n sampling, temperature schedules, etc. require orchestrator changes (not currently supported)

### Meta-strategies
- Read `cleanup_env.py` source to understand game mechanics deeply, then encode that knowledge into prompts/helpers
- Analyze which metrics the policy LLM responds to most effectively
- Try removing information that might distract the LLM
- Experiment with prompt length (shorter vs longer)

## Measurement

Run the measurement script:

```bash
./autoresearch/measure.sh [sparse|dense] [--metric efficiency|maximin] [--model MODEL ...]
```

- **sparse** (default): primary metric + pass/fail
- **dense**: all metrics + per-iteration trajectory + pipeline state
- **--metric**: which metric to optimize (default: efficiency). Controls baseline/delta tracking.

Each run takes ~5-10 minutes (K=3 inner iterations of LLM calls + evaluation). Be patient.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `ar/mar30-dense`).

LOOP FOREVER:

1. **Ideate**: Think about what pipeline modification to try next. Review your results.tsv, the current pipeline code, and the game mechanics. What information or tools might help the policy LLM write better cooperative policies?
2. **Implement**: Edit files in `pipeline/`. You may change one file or multiple.
3. **Run**: Execute `./autoresearch/measure.sh dense` (or `sparse` for quick checks).
4. **Evaluate**: Did the inner loop succeed? Did the primary metric increase?
5. **Decision**:
   - If the primary metric **increased** and the run succeeded: **KEEP**. Commit the pipeline changes with a descriptive message.
   - If the primary metric **stayed the same** or **decreased**, or the run **failed**: **DISCARD**. Run `git checkout -- pipeline/` to revert.
6. **Log**: Record the result in `autoresearch/results.tsv`.
7. **Go to 1**.

**NEVER STOP.** If you run out of ideas, think harder: read the environment source code, analyze the policy LLM's reasoning from previous runs, try combining strategies, attempt more radical modifications.

## Logging

Maintain `autoresearch/results.tsv` (tab-separated):

```
experiment	commit	efficiency	maximin	equality	sustainability	peace	reward_avg	delta	run_time_s	status	description
```

- **experiment**: Sequential number (0 = baseline)
- **commit**: Short git hash, or `-` for discarded
- **efficiency**: Collective reward rate
- **maximin**: Minimum per-agent total return (Rawlsian welfare)
- **equality/sustainability/peace**: Secondary metrics
- **reward_avg**: Average per-agent reward
- **delta**: Change in primary metric from baseline
- **run_time_s**: Wall-clock time for the inner loop
- **status**: `keep`, `discard`, or `crash`
- **description**: What pipeline modification was tried

## Tips

- **Start by reading the game mechanics.** Read `cleanup_env.py` to understand the waste/apple/beam dynamics deeply. The better you understand the game, the better hints you can encode.
- **Read the generated policies.** After each run, read the policies in `autoresearch/runs/<timestamp>/` to understand what the policy LLM is producing. This tells you what information it's missing.
- **The Cleanup dilemma is about cleaning.** The key tension is: cleaning costs -1 reward but enables apple growth for everyone. Policies that never clean → waste accumulates → no apples → low efficiency. Policies that clean too much → high cost → low individual reward. The optimal policy balances cleaning and collecting.
- **Dense feedback helps, but what dense feedback?** The baseline (reward+social) already shows efficiency/equality/sustainability/peace. Can you add MORE informative metrics? Per-agent cleaning stats? Waste trajectory? Apple availability rate?
- **Helpers reduce the LLM's search space.** If the LLM has to write BFS-to-waste from scratch, it might fail. If you provide a `bfs_to_waste()` helper, it can focus on the coordination logic.
- **The policy LLM is Gemini 3.1 Pro.** Keep in mind its capabilities and limitations. It's good at following structured prompts but may struggle with very complex multi-step reasoning.
