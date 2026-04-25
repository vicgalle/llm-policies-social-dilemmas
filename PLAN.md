# Meta-Harness for Sequential Social Dilemmas — Implementation Plan

## Motivation

The current two-level autoresearch caps around **efficiency ≈ 8** on
production_economy. A hand-crafted policy authored interactively (Claude Code
+ Opus, ~50 iterations, with diagnostic probes) reached **10.27** with the
same env, same metric. The gap is not the proposer model — it's the
**information channel** between proposer and outcome:

- Researcher sees diffs of `pipeline/*.py` + scalar efficiency.
- Researcher cannot see *why* a policy underperformed (no traces, no per-agent
  state, no event timeline).
- Researcher cannot write probes, run targeted simulations, or accumulate
  scratch artifacts across iterations.
- Researcher debugs through the synthesizer LLM (which writes the policy)
  via prompt edits — a noisy, lossy channel.

Two 2026 papers map directly onto this diagnosis:

- **Meta-Harness** (Lee et al., Stanford/MIT/KRAFTON) — outer-loop search over
  harness code with **filesystem access to all prior code, scores, and
  execution traces**. Reads ~82 files per iter, ~10M tokens of diagnostic
  context per evaluation (3 OOM more than prior text optimizers like
  OPRO/TextGrad/GEPA/OpenEvolve/TTT-Discover). Beats hand-engineered TerminalBench-2
  agents.
- **AutoHarness** (Lou et al., DeepMind) — three harness flavors
  (action-filter, action-verifier, **harness-as-policy** = pure code, no LLM
  at runtime). Gemini-2.5-Flash + harness-as-policy beats GPT-5.2-High on
  TextArena.

Both findings reproduce in our setting (the hand-crafted policy is exactly
"harness-as-policy"). We should formalize this as the framework.

## Goals

1. **Match or beat 10.3 efficiency** on production_economy with a fully
   autonomous proposer (no human in the loop).
2. **Discover transferable harnesses** that work across env variants
   (cleanup ↔ gathering ↔ coop_mining ↔ production_economy, different maps,
   different `n_agents`).
3. **Multi-objective optimization**: explicit Pareto over
   {efficiency, maximin, equality, inference cost, context cost}.
4. **Plottable convergence story** comparable to Meta-Harness Fig. 1 + Fig. 3:
   harness-search progress curve + accuracy/cost Pareto frontier.

## Architecture

### Three harness modes (Pareto endpoints, not exclusive)

| Mode | What's optimized | LLM at inference? | Closest analog |
|---|---|---|---|
| **M1: Feedback-shaper** | `pipeline/{prompts,feedback,helpers,config}.py` — synthesizer still writes the policy each inner-loop iteration | Yes, every step | Current autoresearch; AutoHarness "action-filter" |
| **M2: Scaffold-supplier** | Structural Python scaffold (action validators, navigation helpers, phase routers, kit-builders) + a small LLM-decided strategic loop | Yes, but only for high-level decisions | AutoHarness "action-verifier" |
| **M3: Policy-as-code** | Pure Python `policy(env, agent_id) -> int`, no inference-time LLM | No | AutoHarness "harness-as-policy"; what the hand-crafted policy is |

The proposer chooses the mode for each candidate. Different modes occupy
different points on the cost ↔ performance ↔ transferability frontier.

### Model layering

Two distinct model slots — easy to conflate, important to keep separate
(Meta-Harness does the same split):

| Slot | Role | Configuration |
|---|---|---|
| **Proposer** $P$ | Outer-loop coding agent that searches over harnesses (reads filesystem, edits code, writes probes, decides what to eval). Same model across the whole search. | Claude Code with Opus 4.6+ (fixed). |
| **Base model** $M$ | The model the harness calls *at inference time* (per-step or per-decision). Varies per harness as a search axis. | `gemini-3.1-pro-preview`, `claude-sonnet-4-6`, or `none` (M3 mode). |

By mode:

- **M1 (feedback-shaper)**: $M$ is called every step / every synthesis
  iteration. Search optimizes prompts/feedback/helpers *for that specific $M$*.
- **M2 (scaffold-supplier)**: $M$ is called only for strategic decisions
  inside a code scaffold; lower call rate.
- **M3 (policy-as-code)**: $M$ is `none`. No inference-time model. The harness
  *is* the policy. (Note: $P$ is still Claude Code while *writing* the
  harness; M3 just means the resulting artifact has no LLM dependency.)

Each harness folder declares its base model in `manifest.json` (see
filesystem layout below). The eval pipeline already accepts `--model`
(see `CLAUDE.md`); `tools.py:eval_harness` reads the manifest and
forwards the right value to `run_inner_loop.py`. The proposer can branch
a harness with a different base model as a single edit:

```python
spawn_branch(parent="h0042", edits={"base_model": "claude-sonnet-4-6"})
```

This makes "swap Gemini for Sonnet" a first-class search action, and lets
the Pareto frontier span {efficiency × inference-cost × base-model
dependency} — directly reproducing the AutoHarness "smaller base + good
harness ≥ larger base alone" claim if it holds in the SSD setting.

### Filesystem layout (Meta-Harness style)

```
autoresearch/
  meta/
    harnesses/
      h0001/
        manifest.json         # {mode, base_model, env, n_agents, parent}
        harness/              # The harness source files (mode-dependent)
        rationale.md          # Proposer's diagnosis + hypothesis
        metrics.json          # {efficiency, maximin, equality, ctx_tokens, infer_cost}
        traces/
          seed_00.jsonl       # Per-step state for one rollout
          seed_01.jsonl
          ...
          summary.json        # Aggregated event log, action distribution, deadlocks
      h0002/...
    pareto.json               # Current frontier IDs per objective
    queue.jsonl               # Pending evaluations
    proposer_log.md           # Append-only narrative of the search
    scratch/                  # Persistent probes the proposer writes
      diag_*.py
```

The proposer reads/writes this entire tree via `grep`/`cat`/`Read`/`Edit`.
**No structured prompt** — the filesystem is the context.

### Execution traces (the "10M-token diagnostic footprint")

Per-seed per-rollout, store one JSONL line per step:

```jsonl
{"t": 0, "agents": [{"id": 0, "pos": [4,0], "inv": [0,0,0,0], "tool": false, "tool_age": 0, "action": "MOVE_E", "reward": 0}, ...], "shelter": 0, "forge_pools": [[0,0],[0,0]], "events": []}
{"t": 30, ..., "events": [{"agent": 0, "type": "CRAFT_TOOL"}]}
```

Plus an aggregated `summary.json`:

```json
{
  "agent_totals": [414, 466, ..., 236],
  "tool_history": {"0": [[42, 121], [197, 276]], "7": [[244, "end"]]},
  "deadlock_events": [{"t": 256, "agent": 7, "blocked_by": [0, 1, 3], "duration": 22, "cell": [6, 8]}],
  "action_counts": {"0": {"NOOP": 101, "GATHER": 9, ...}},
  "forge_contention_steps": 47
}
```

This is what makes the "agent 7 stuck for 22 steps because forge cap=5"
finding *recoverable* by an agent reading prior runs.

## Components to build

### Trace recording (`pipeline/trace.py`)

```python
class TraceRecorder:
    def __init__(self, path: str): ...
    def record_step(self, env, actions, rewards): ...
    def record_event(self, t: int, kind: str, payload: dict): ...
    def finalize(self) -> dict: ...  # writes summary.json
```

Hook into `run_inner_loop.py` behind `--record-traces` flag. Recording adds
~10% step time, acceptable for evaluation runs.

Aggregation post-processor extracts:

- per-agent tool history (start/end of each tool cycle)
- contention events (agent attempted move blocked by another stationary agent)
- pool saturation events (DROP/CRAFT silent failures due to cap)
- per-agent action distribution
- per-cell visit counts

### Harness abstraction (`pipeline/harness.py`)

```python
class Harness(Protocol):
    mode: Literal["M1", "M2", "M3"]
    def run_episode(self, env, seed: int, recorder: TraceRecorder | None) -> EpisodeResult: ...

class M1FeedbackShaper(Harness): ...   # wraps pipeline/* + llm_self_play
class M2Scaffold(Harness): ...         # mixed code + LLM inner decisions
class M3PolicyAsCode(Harness): ...     # imports a pure Python policy()
```

Migrate the existing `run_inner_loop.py` to instantiate one of these. The
hand-crafted `production_economy_policy.py` becomes a seed M3 harness.

### Proposer tools (`autoresearch/meta/tools.py`)

A small Python module the proposer imports. Each function is also exposed
as a CLI script for Claude Code Bash use.

```python
def list_harnesses(filter: dict = None) -> list[str]: ...
def read_harness(hid: str) -> dict: ...                  # source + metrics + summary
def read_trace(hid: str, seed: int, agents: list[int] | None = None) -> list[dict]: ...
def eval_harness(hid: str, n_seeds: int = 2, record: bool = False) -> dict: ...
def diff_harnesses(hid_a: str, hid_b: str) -> str: ...
def pareto_frontier(objectives: list[str]) -> list[str]: ...
def spawn_branch(parent_hid: str, edits: dict) -> str: ...   # returns new hid
def write_probe(name: str, code: str) -> None: ...           # persists in scratch/
def run_probe(name: str, args: list[str] = []) -> str: ...   # captured stdout
```

These are the tools I missed in the current framework. The proposer should
also have raw `Bash`/`Read`/`Edit`/`Write` over the filesystem (Claude Code
defaults).

### Adaptive evaluator (`autoresearch/meta/evaluator.py`)

Sequential testing — exploration is cheap, commitment is expensive:

```python
def adaptive_eval(hid: str, target_pareto: list[str]) -> EvalResult:
    # Stage 1: 2 seeds. If clearly dominated by a frontier point, stop.
    # Stage 2: 5 seeds. If on/near frontier, continue.
    # Stage 3: 20 seeds + record traces. Used for committing to frontier.
```

Saves compute on dead-end candidates while sharpening the signal on
contenders.

### Population manager (`autoresearch/meta/population.py`)

```python
def update_frontier(new_hid: str, objectives: list[str]) -> dict:
    # Returns {dominates: [...], dominated_by: [...], on_frontier: bool}

def cull_dominated(keep_n_per_objective: int = 5) -> list[str]:
    # Move strictly dominated harnesses to archive/ to keep main listing tractable
```

Frontier is multi-objective; we don't collapse to scalar score.

### Outer loop (`autoresearch/meta/loop.py`)

Replaces `run_experiment.sh`. Pseudocode:

```python
def meta_harness_loop(env: str, n_iters: int, proposer: ClaudeCodeAgent):
    seed_population(env)  # zero-shot, few-shot, hand-crafted, current LLM-only
    for t in range(n_iters):
        # Proposer reads filesystem, decides what to do this iteration
        action = proposer.act(read_only_filesystem=PATH)
        # action is one of:
        #   propose_harness(parent, edits)
        #   write_probe(name, code) + run_probe(name)
        #   eval_harness(hid, n_seeds)
        #   spawn_subagent(hypothesis)  # parallel branch
        execute(action)
    return pareto_frontier(["efficiency", "maximin", "ctx_tokens"])
```

## Migration path

### Phase 1 — Tracing + filesystem (smallest viable change)
1. Add `pipeline/trace.py` and `--record-traces` flag to `run_inner_loop.py`.
2. Add `autoresearch/meta/` directory layout. Existing `autoresearch/runs/`
   stays untouched.
3. Verify trace files are readable + diff'able with current proposer.

**Deliverable**: existing autoresearch run produces traces; old framework still
works.

### Phase 2 — Tool layer + M3 mode
1. Build `tools.py` proposer toolset.
2. Add `M3PolicyAsCode` harness class. Wrap `production_economy_policy.py`
   as a seed M3 harness.
3. Run a single Claude Code session with the new tools on production_economy
   from a weak M3 seed — *should* recover ~10.3 efficiency unsupervised.
   This is the headline experiment; if it doesn't reproduce, the framework is
   broken.

**Deliverable**: M3-mode autoresearch on production_economy reaches ≥10.

### Phase 3 — Population + Pareto + adaptive eval
1. Multi-objective frontier tracking.
2. Adaptive seed allocation.
3. Outer loop driver.

**Deliverable**: Pareto plot (efficiency vs maximin) with ≥3 distinct
non-dominated harnesses.

### Phase 4 — Cross-env + ablations
1. Run on cleanup, gathering, coop_mining, production_economy.
2. OOD transfer: train on cleanup default map, eval on cleanup_large.
3. Information ablation (Meta-Harness Table 3 analog): scores-only vs
   scores+summary vs scores+traces.

**Deliverable**: paper-ready figures.

## Experiments

### E1 — Mode comparison (within-env)
Run all three modes on production_economy with matched eval budget.
**Hypothesis**: M3 ≫ M2 ≳ M1 on this env. M1 caps near 8, M3 reaches 10+.

### E2 — Pareto over social objectives
Multi-objective search optimizing simultaneously
{efficiency, maximin, equality} on coop_mining (the stag-hunt env where
these objectives genuinely trade off).
**Hypothesis**: discovered harnesses span the frontier. Some lean
efficiency-max (defection-tolerant), others maximin (egalitarian).

### E3 — Cross-env transfer
Train M3 harness on cleanup default. Evaluate on:
- cleanup with 6 vs 10 agents
- cleanup_large
- gathering (different mechanic — does the agent-aware BFS helper still help?)

**Hypothesis**: structural helpers (BFS variants, role-assignment heuristics)
transfer; phase logic does not.

### E4 — Information ablation (the Meta-Harness money plot)
Same proposer, three info conditions:
1. **Scores only** — scalar efficiency per harness, no code, no traces
2. **Scores + summary** — + LLM-generated summary of failures
3. **Full** — code + scores + traces (ours)

**Hypothesis** (per Meta-Harness Table 3): full interface ≫ summaries
> scores-only. Quantify the lift attributable to traces specifically.

### E5 — Inference-cost frontier
Plot performance vs LLM-tokens-per-episode across modes.
M3 is the zero-cost endpoint (no inference). M1 is the high-cost endpoint.
**Hypothesis**: M3 is on the Pareto frontier, validating the AutoHarness
"smaller model + good harness > larger model alone" claim.

## Open questions / design decisions to make

1. **Mixed-mode harnesses**: Can a harness be partially M3 (e.g.,
   navigation/kit-builder in code) and partially M1 (LLM picks high-level
   intent)? How does the search space handle this? *Probably yes — let
   the proposer decide; the API is `run_episode`.*
2. **Trace privacy across seeds**: Do we want the proposer to see all 5+
   seed traces, or sample? Storage is cheap; let it grep what it wants.
3. **Probe persistence**: When should probes be deleted? Probably never
   automatically — they're proposer-authored cognition. Keep them.
4. **Curriculum**: Should easy variants (small map, fewer agents) seed
   harder ones? Worth testing — might be a free win.
5. **Sub-agent spawning**: Meta-Harness uses a single proposer; AutoHarness
   uses Thompson sampling over a tree. We could let the proposer
   `spawn_branch()` to explore divergent hypotheses in parallel. Decide:
   single sequential vs population-based proposer.
6. **Variance budget**: Eval std is ~20 on production_economy with 5 seeds.
   For Pareto-frontier commitment we probably need 20+ seeds. How much
   compute can we budget per harness? Affects how aggressive Phase 3 is.
7. **Env modification ban**: Meta-Harness lets the proposer rewrite anything
   not whitelisted as frozen. Our envs are explicitly frozen for game-design
   integrity. Make this a hard guard via path whitelist in `tools.py`.

## Reference: papers

- `~/Downloads/2603.28052v1-2.pdf` — Lee, Nair, Zhang, Lee, Khattab, Finn (2026).
  *Meta-Harness: End-to-End Optimization of Model Harnesses.*
- `~/Downloads/68_AUTOHARNESS_IMPROVING_LLM_A-2.pdf` — Lou, Lázaro-Gredilla,
  Dedieu, Wendelken, Lehrach, Murphy (DeepMind, ICLR Recursive Self-Improvement
  Workshop 2026). *AutoHarness: Improving LLM Agents by Automatically
  Synthesizing a Code Harness.*

Both are directly applicable. Cite as the methodological foundation; the
SSD setting is the novel contribution (multi-agent self-play, social-dilemma
multi-objective, transferable across env variants).

## Story for the paper

> **Meta-Harness for Multi-Agent Sequential Social Dilemmas: When Coding
> Agents Outperform Synthesized LLM Policies.** We adapt Meta-Harness
> (Lee et al., 2026) to the multi-agent self-play SSD setting and show
> that pure-code harnesses (AutoHarness's harness-as-policy mode, Lou
> et al., 2026) Pareto-dominate LLM-in-the-loop policies on three of
> four environments. The discovered harnesses transfer across maps and
> agent counts. The social-dilemma multi-objective frontier
> (efficiency / maximin / equality) reveals harnesses with qualitatively
> different cooperation strategies that no single scalar reward would
> have surfaced.
