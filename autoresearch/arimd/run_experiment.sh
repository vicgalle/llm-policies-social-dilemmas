#!/usr/bin/env bash
#
# run_experiment.sh — Launch an ARIMD Stackelberg search.
#
# ARIMD = Adversarially-Robust Inverse Mechanism Design (arimd_plan.md).
# Blue (LLM) tunes a bounded grammar of env edits; Red (LLM) writes an
# exploit policy.  Per round: Red search → Blue patch → accept/reject.
#
# Usage:
#   ./autoresearch/arimd/run_experiment.sh <tag> <game> [cooperator] [lambda] [T] [M_red] [seeds]
#
# Examples:
#   ./autoresearch/arimd/run_experiment.sh apr26-nc-eff       nested_commons efficiency 0.25
#   ./autoresearch/arimd/run_experiment.sh apr26-nc-mm        nested_commons maximin    0.25
#   ./autoresearch/arimd/run_experiment.sh apr26-cu           cleanup        default    0.25
#   ./autoresearch/arimd/run_experiment.sh apr26-pe           production_economy default 0.25
#
# Smoke test (T=1, M_red=1, 2 seeds):
#   ./autoresearch/arimd/run_experiment.sh smoke nested_commons efficiency 0.25 1 1 2
#
# Use environment variables to switch models:
#   ARIMD_BLUE_MODEL=claude-opus-4-7  ARIMD_RED_MODEL=claude-sonnet-4-6 \
#     ./autoresearch/arimd/run_experiment.sh ...

set -euo pipefail

TAG="${1:?Usage: $0 <tag> <game> [cooperator] [lambda] [T] [M_red] [seeds]}"
GAME="${2:?Usage: $0 <tag> <game> [cooperator] [lambda] [T] [M_red] [seeds]}"
COOPERATOR="${3:-default}"
LAMBDA="${4:-0.25}"
T_VALUE="${5:-4}"
M_RED="${6:-4}"
SEEDS="${7:-5}"

BLUE_MODEL="${ARIMD_BLUE_MODEL:-claude-opus-4-7}"
RED_MODEL="${ARIMD_RED_MODEL:-claude-sonnet-4-6}"
WELFARE="${ARIMD_WELFARE:-blue}"
INVASION_LAMBDAS="${ARIMD_INVASION_LAMBDAS:-0.0,0.1,0.2,0.25,0.4}"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

OUTPUT_DIR="autoresearch/arimd/runs/${TAG}_$(date +%Y%m%d_%H%M%S)"

echo "=== ARIMD Stackelberg run ==="
echo "Tag:           ${TAG}"
echo "Game:          ${GAME}"
echo "Cooperator:    ${COOPERATOR}"
echo "Lambda:        ${LAMBDA}"
echo "T:             ${T_VALUE}"
echo "M_red:         ${M_RED}"
echo "Seeds:         0..$((SEEDS-1))"
echo "Welfare:       ${WELFARE}"
echo "Blue model:    ${BLUE_MODEL}"
echo "Red model:     ${RED_MODEL}"
echo "Output:        ${OUTPUT_DIR}"
echo ""

uv run python -m autoresearch.arimd.stackelberg \
  --game "${GAME}" \
  --cooperator "${COOPERATOR}" \
  --lambda "${LAMBDA}" \
  --welfare "${WELFARE}" \
  --T "${T_VALUE}" \
  --M-red "${M_RED}" \
  --seeds "${SEEDS}" \
  --blue-model "${BLUE_MODEL}" \
  --red-model "${RED_MODEL}" \
  --invasion-lambdas "${INVASION_LAMBDAS}" \
  --tag "${TAG}" \
  --output-dir "${OUTPUT_DIR}"
