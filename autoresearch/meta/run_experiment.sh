#!/usr/bin/env bash
#
# autoresearch/meta/run_experiment.sh — launch a meta-harness search.
#
# Usage:
#   ./autoresearch/meta/run_experiment.sh <tag> [game] [proposer_model]
#
# Examples:
#   ./autoresearch/meta/run_experiment.sh apr25-pe                     # production_economy, Opus proposer
#   ./autoresearch/meta/run_experiment.sh apr25-cleanup cleanup
#   ./autoresearch/meta/run_experiment.sh apr25-pe production_economy opus
#
# Side effects:
#   1. Creates a git branch ar-meta/<tag>.
#   2. Launches Claude Code on autoresearch/meta/program.md.
#   3. The agent runs autonomously until interrupted (Ctrl-C).
#

set -euo pipefail

TAG="${1:?Usage: $0 <tag> [game] [proposer_model]}"
GAME="${2:-production_economy}"
PROPOSER_MODEL="${3:-opus}"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

BRANCH="ar-meta/${TAG}"

echo "=== Meta-Harness search ==="
echo "Tag:             ${TAG}"
echo "Game:            ${GAME}"
echo "Proposer model:  ${PROPOSER_MODEL}"
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

The framework's seed harnesses for production_economy are:
  - h0000 (weak greedy baseline; ~few efficiency)
  - h0001 (hand-crafted upper anchor; ~10.27 efficiency)

Goal: discover M3/M2/M1 harnesses that match or beat h0001 on the
primary metric, ideally finding distinct points on the Pareto frontier
(different cost / equality / maximin tradeoffs).

NEVER STOP. Run experiments continuously until interrupted."

echo "Launching Claude Code proposer agent..."
echo "Press Ctrl-C to stop."
echo "---"

claude --model "$PROPOSER_MODEL" --dangerously-skip-permissions -p "$PROMPT"
