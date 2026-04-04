#!/usr/bin/env bash
#
# run_experiment.sh — Launch an autonomous autoresearch run for SSD policy synthesis.
#
# Usage:
#   ./autoresearch/run_experiment.sh <tag> [feedback_mode] [model]
#
# Examples:
#   ./autoresearch/run_experiment.sh mar30-test dense                                    # Opus researcher, Gemini policy LLM
#   ./autoresearch/run_experiment.sh mar30-sonnet dense opus claude-sonnet-4-6           # Opus researcher, Sonnet policy LLM
#   ./autoresearch/run_experiment.sh mar30-gemini sparse sonnet gemini-3.1-pro-preview   # Sonnet researcher, Gemini policy LLM
#
# This script:
#   1. Creates a git branch for the experiment
#   2. Launches Claude Code with the research program
#   3. The agent runs autonomously until interrupted (Ctrl-C)
#

set -euo pipefail

TAG="${1:?Usage: $0 <tag> [sparse|dense] [researcher_model] [policy_model] [metric] [game]}"
FEEDBACK="${2:-dense}"
RESEARCHER_MODEL="${3:-opus}"  # Model for the *researcher* agent (not the policy LLM)
POLICY_MODEL="${4:-gemini-3.1-pro-preview}"  # Model for the policy synthesizer LLM
METRIC="${5:-efficiency}"  # Primary metric to optimize: efficiency or maximin
GAME="${6:-cleanup}"  # Game: cleanup, gathering, or coop_mining

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

BRANCH="ar/${TAG}"

echo "=== Autoresearch: SSD Policy Synthesis ==="
echo "Tag:              ${TAG}"
echo "Feedback:         ${FEEDBACK}"
echo "Researcher model: ${RESEARCHER_MODEL}"
echo "Policy model:     ${POLICY_MODEL}"
echo "Metric:           ${METRIC}"
echo "Game:             ${GAME}"
echo "Branch:           ${BRANCH}"
echo ""

# --- Setup branch ---
if git show-ref --verify --quiet "refs/heads/${BRANCH}" 2>/dev/null; then
    echo "Branch ${BRANCH} already exists. Checking it out..."
    git checkout "$BRANCH"
else
    echo "Creating branch ${BRANCH}..."
    git checkout -b "$BRANCH"
fi

echo "Working on: $(git branch --show-current)"
echo ""

# --- Verify infrastructure ---
echo "Verifying run_inner_loop.py exists..."
if [[ ! -f "run_inner_loop.py" ]]; then
    echo "ERROR: run_inner_loop.py not found."
    exit 1
fi
echo "OK."

echo "Verifying pipeline/ exists..."
if [[ ! -d "pipeline" ]]; then
    echo "ERROR: pipeline/ directory not found."
    exit 1
fi
echo "OK."
echo ""

# --- Construct the prompt ---
GAME_ARGS="--game ${GAME}"
if [[ "$GAME" == "cleanup" ]]; then
    GAME_ARGS="--game cleanup --n-agents 10"
    GAME_DESC="Cleanup with 10 agents on large map (public goods dilemma)"
    ENV_FILE="cleanup_env.py"
elif [[ "$GAME" == "coop_mining" ]]; then
    GAME_ARGS="--game coop_mining --n-agents 6"
    GAME_DESC="Coop Mining with 6 agents on large map (Stag Hunt coordination)"
    ENV_FILE="coop_mining_env.py"
elif [[ "$GAME" == "gathering" ]]; then
    GAME_ARGS="--game gathering --n-agents 4"
    GAME_DESC="Gathering with 4 agents on large map (common pool resource)"
    ENV_FILE="gathering_env.py"
fi

PROMPT="Read autoresearch/program.md carefully. This is your research program.

Set up a new autoresearch run:
- Tag: ${TAG}
- Branch: ${BRANCH} (already created and checked out)
- Feedback mode: ${FEEDBACK} (use ./autoresearch/measure.sh ${FEEDBACK} --metric ${METRIC} --model ${POLICY_MODEL} ${GAME_ARGS})
- Policy LLM: ${POLICY_MODEL}
- Primary metric: ${METRIC} (higher = better)
- Game: ${GAME_DESC}

Important context:
- You are the RESEARCHER agent. You modify files in pipeline/ to improve how the POLICY LLM generates cooperative strategies.
- The primary metric to optimize is ${METRIC} (higher = better).
- Each inner loop run takes ~5-10 minutes. Be patient.
- Read ${ENV_FILE} to understand the game mechanics.
- Read the current pipeline/ files to understand the starting configuration.

The branch is ready. Establish the baseline by running ./autoresearch/measure.sh ${FEEDBACK} --metric ${METRIC} --model ${POLICY_MODEL} ${GAME_ARGS}, then begin the experiment loop.
When running measure.sh, ALWAYS pass --metric ${METRIC} --model ${POLICY_MODEL} ${GAME_ARGS} as extra arguments after the feedback mode.

Remember: NEVER STOP. Run experiments continuously until I interrupt you."

echo "Launching Claude Code researcher agent..."
echo "Press Ctrl-C to stop."
echo "---"

# --- Launch ---
claude --model "$RESEARCHER_MODEL" --dangerously-skip-permissions -p "$PROMPT"
