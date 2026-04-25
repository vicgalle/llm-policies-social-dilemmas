#!/usr/bin/env bash
#
# autoresearch/meta/run_experiment.sh — launch a meta-harness search.
#
# Usage:
#   ./autoresearch/meta/run_experiment.sh <tag> [game] [proposer_model] [metric]
#
# Examples:
#   ./autoresearch/meta/run_experiment.sh apr25-pe                                # PE, Opus, efficiency
#   ./autoresearch/meta/run_experiment.sh apr25-cleanup cleanup                   # cleanup, Opus, efficiency
#   ./autoresearch/meta/run_experiment.sh apr25-pe production_economy opus        # explicit
#   ./autoresearch/meta/run_experiment.sh apr27-pe-mm production_economy opus maximin
#                                                                                  # PE, Opus, maximin-primary
#
# [metric] selects the *primary* axis the proposer optimizes, by reordering
# autoresearch/meta/pareto.json so that axis is index 0. The adaptive
# evaluator and population manager read objectives[0] as primary; the
# launch prompt tells the proposer which metric to lead with.
#
# Valid metrics: efficiency, maximin, equality, sustainability, peace.
#
# Side effects:
#   1. Creates a git branch ar-meta/<tag>.
#   2. Reorders pareto.json's objectives so [metric] is primary.
#   3. Launches Claude Code on autoresearch/meta/program.md.
#   4. The agent runs autonomously until interrupted (Ctrl-C).
#

set -euo pipefail

TAG="${1:?Usage: $0 <tag> [game] [proposer_model] [metric]}"
GAME="${2:-production_economy}"
PROPOSER_MODEL="${3:-opus}"
METRIC="${4:-efficiency}"

case "$METRIC" in
    efficiency|maximin|equality|sustainability|peace) ;;
    *) echo "ERROR: metric must be one of efficiency|maximin|equality|sustainability|peace (got '$METRIC')"; exit 1 ;;
esac

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

BRANCH="ar-meta/${TAG}"

echo "=== Meta-Harness search ==="
echo "Tag:             ${TAG}"
echo "Game:            ${GAME}"
echo "Proposer model:  ${PROPOSER_MODEL}"
echo "Primary metric:  ${METRIC}"
echo "Branch:          ${BRANCH}"
echo ""

# Branch setup
if git show-ref --verify --quiet "refs/heads/${BRANCH}" 2>/dev/null; then
    echo "Branch ${BRANCH} already exists. Checking it out..."
    git checkout "$BRANCH"
else
    echo "Creating branch ${BRANCH}..."
    git checkout -b "$BRANCH"
fi

# Sanity checks
test -f autoresearch/meta/program.md || { echo "missing program.md"; exit 1; }
test -d autoresearch/meta/harnesses || { echo "missing harnesses/"; exit 1; }
test -f autoresearch/meta/pareto.json || { echo "missing pareto.json"; exit 1; }

# Reorder pareto.json so $METRIC is the primary axis (index 0). The
# evaluator and population manager use objectives[0] as the early-stop
# reference; the proposer reads pareto.json to learn what's primary.
python3 - <<PY
import json, sys
p = "autoresearch/meta/pareto.json"
metric = "$METRIC"
d = json.load(open(p))
existing = d.get("objectives", [])
others = [m for m in existing if m != metric]
d["objectives"] = [metric] + others
# Wipe any previously cached scores: their primary-axis ordering would
# now be wrong. The proposer re-evals on first iteration anyway.
d["scores"] = {}
d["frontier"] = []
d["updated"] = None
json.dump(d, open(p, "w"), indent=2)
print(f"  pareto.json: objectives reordered, primary={metric}")
PY

# Per-game default seeds. The proposer can override these in spawn manifests.
case "$GAME" in
    cleanup)             N_AGENTS=10 ;;
    coop_mining)         N_AGENTS=6  ;;
    gathering)           N_AGENTS=4  ;;
    production_economy)  N_AGENTS=8  ;;
    *) echo "Unknown game: $GAME"; exit 1 ;;
esac

PROMPT="Read autoresearch/meta/program.md carefully. This is your research program.

You are the proposer P in a meta-harness search for the ${GAME} env
with ${N_AGENTS} agents.

PRIMARY OBJECTIVE: ${METRIC}.
The first axis in pareto.json is the metric you should optimize. The
adaptive evaluator early-stops candidates that are clearly dominated on
that axis; secondary axes only matter when a candidate is competitive on
the primary. If the proposer log from a prior run quotes a different
primary metric (e.g. efficiency in run1), do not anchor on those numbers
— the search is for the best ${METRIC} under self-play, which may
yield a qualitatively different strategy.

Workflow per iteration:
  1. tools.py list                 # see harness population
  2. tools.py pareto               # see current frontier
  3. (read traces / probes for diagnosis)
  4. tools.py spawn <parent> ...   # author a new harness
  5. tools.py validate <new_id>    # 20-step smoke
  6. tools.py eval <new_id>        # adaptive-eval with auto Pareto update
  7. tools.py log \"...\"          # narrative entry

Run tools.py via:
  uv run python -m autoresearch.meta.tools <subcommand> [args]

Hard constraints:
  * Never modify frozen files (envs, pipeline/{trace,harness,...}.py,
    llm_self_play.py, run_inner_loop.py).
  * Never modify an existing harness folder.
  * Always validate before evaluating.
  * Read the OFF-LIMITS FILES section in program.md and respect it.
    The headline experiment is invalid if the search uses a known
    reference solution.

Discover the search space yourself: tools.py list to see what's seeded,
read the env source to understand mechanics, propose harnesses, evaluate.
Goal: find M3/M2/M1 harnesses that span the Pareto frontier
(efficiency / maximin / equality / cost).

NEVER STOP. Run experiments continuously until interrupted."

echo "Launching Claude Code proposer agent..."
echo "Press Ctrl-C to stop."
echo "---"

claude --model "$PROPOSER_MODEL" --dangerously-skip-permissions -p "$PROMPT"
