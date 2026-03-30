#!/usr/bin/env bash
#
# run_experiment.sh — Launch an autonomous autoresearch run for SSD policy synthesis.
#
# Usage:
#   ./autoresearch/run_experiment.sh <tag> [feedback_mode] [model]
#
# Examples:
#   ./autoresearch/run_experiment.sh mar30-test dense
#   ./autoresearch/run_experiment.sh mar30-opus dense opus
#   ./autoresearch/run_experiment.sh mar30-sonnet sparse sonnet
#
# This script:
#   1. Creates a git branch for the experiment
#   2. Launches Claude Code with the research program
#   3. The agent runs autonomously until interrupted (Ctrl-C)
#

set -euo pipefail

TAG="${1:?Usage: $0 <tag> [sparse|dense] [researcher_model]}"
FEEDBACK="${2:-dense}"
RESEARCHER_MODEL="${3:-opus}"  # Model for the *researcher* agent (not the policy LLM)

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

BRANCH="ar/${TAG}"

echo "=== Autoresearch: SSD Policy Synthesis ==="
echo "Tag:              ${TAG}"
echo "Feedback:         ${FEEDBACK}"
echo "Researcher model: ${RESEARCHER_MODEL}"
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
PROMPT="Read autoresearch/program.md carefully. This is your research program.

Set up a new autoresearch run:
- Tag: ${TAG}
- Branch: ${BRANCH} (already created and checked out)
- Feedback mode: ${FEEDBACK} (use ./autoresearch/measure.sh ${FEEDBACK})
- Policy LLM: gemini-3.1-pro-preview (default, configured in run_inner_loop.py)
- Game: Cleanup with 10 agents on large map (default)

Important context:
- You are the RESEARCHER agent. You modify files in pipeline/ to improve how the POLICY LLM generates cooperative strategies.
- The metric is EFFICIENCY (higher = better). Baseline from Gallego 2026: ~2.75.
- Each inner loop run takes ~5-10 minutes. Be patient.
- Read cleanup_env.py to understand the game mechanics.
- Read the current pipeline/ files to understand the starting configuration.

The branch is ready. Establish the baseline by running ./autoresearch/measure.sh ${FEEDBACK}, then begin the experiment loop.

Remember: NEVER STOP. Run experiments continuously until I interrupt you."

echo "Launching Claude Code researcher agent..."
echo "Press Ctrl-C to stop."
echo "---"

# --- Launch ---
claude --model "$RESEARCHER_MODEL" --dangerously-skip-permissions -p "$PROMPT"
