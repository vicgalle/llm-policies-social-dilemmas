#!/usr/bin/env bash
#
# measure.sh — Run the inner policy synthesis loop and report metrics.
#
# Usage:
#   ./autoresearch/measure.sh [sparse|dense] [--game cleanup] [--model gemini-3.1-pro-preview] ...
#
# Modes:
#   sparse  (default)  — efficiency score + pass/fail only
#   dense              — all 4 metrics + per-iteration trajectory + timing
#
# All extra arguments are passed to run_inner_loop.py.
# Default: --game cleanup --model gemini-3.1-pro-preview --map large --n-agents 10
#
# Exit codes:
#   0 — inner loop succeeded
#   1 — inner loop failed
#

set -euo pipefail

FEEDBACK_MODE="${1:-sparse}"
shift 2>/dev/null || true  # Remove feedback mode from args

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

BASELINE_FILE="autoresearch/.baseline_efficiency"

# Extra args from caller (e.g., --model claude-sonnet-4-6)
EXTRA_ARGS="$*"

# Build run args: start with defaults, then overlay extra args.
# Extra args like --model X will override the default --model.
DEFAULT_ARGS="--game cleanup --map large --n-agents 10"

if [[ -z "$EXTRA_ARGS" ]]; then
    # No extra args: use defaults with default model
    RUN_ARGS="$DEFAULT_ARGS --model gemini-3.1-pro-preview"
else
    # Extra args provided: add defaults for anything not specified
    RUN_ARGS="$DEFAULT_ARGS $EXTRA_ARGS"
fi

# Create unique output dir for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="autoresearch/runs/${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

# --- Run inner loop ---
echo "=== Running inner loop ==="
echo "Args: $RUN_ARGS --output-dir $OUTPUT_DIR"
RUN_START=$(date +%s)

RUN_OUTPUT=$(uv run run_inner_loop.py $RUN_ARGS --output-dir "$OUTPUT_DIR" 2>&1) || {
    RUN_END=$(date +%s)
    RUN_TIME=$((RUN_END - RUN_START))
    echo ""
    echo "--- RESULT ---"
    echo "status:     FAIL"
    echo "run_time:   ${RUN_TIME}s"
    echo ""
    if [[ "$FEEDBACK_MODE" == "dense" ]]; then
        echo "--- ERRORS ---"
        echo "$RUN_OUTPUT" | tail -40
    else
        echo "Inner loop failed. Run with 'dense' for error details."
    fi
    exit 1
}

RUN_END=$(date +%s)
RUN_TIME=$((RUN_END - RUN_START))

# --- Parse metrics ---
# Extract the JSON block after "=== INNER LOOP COMPLETE ==="
METRICS_JSON=$(echo "$RUN_OUTPUT" | sed -n '/=== INNER LOOP COMPLETE ===/,$ p' | tail -n +2)

if [[ -z "$METRICS_JSON" ]]; then
    echo "ERROR: Could not find metrics in output."
    echo "$RUN_OUTPUT" | tail -20
    exit 1
fi

# Parse with python
EFFICIENCY=$(echo "$METRICS_JSON" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['efficiency'])")
EQUALITY=$(echo "$METRICS_JSON" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('equality', 0))")
SUSTAINABILITY=$(echo "$METRICS_JSON" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('sustainability', 0))")
PEACE=$(echo "$METRICS_JSON" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('peace', 0))")
REWARD_AVG=$(echo "$METRICS_JSON" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['reward_avg'])")
WALL_TIME=$(echo "$METRICS_JSON" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['wall_time_s'])")

# Compute delta from baseline
DELTA="n/a"
if [[ -f "$BASELINE_FILE" ]]; then
    BASELINE=$(cat "$BASELINE_FILE")
    # Use python for float arithmetic
    DELTA=$(python3 -c "b=$BASELINE; e=$EFFICIENCY; d=e-b; print(f'+{d:.4f}' if d>=0 else f'{d:.4f}')")
else
    echo "$EFFICIENCY" > "$BASELINE_FILE"
    DELTA="+0.0000"
fi

# --- Report ---
echo ""
echo "--- RESULT ---"
echo "status:          OK"
echo "efficiency:      ${EFFICIENCY}"
echo "delta:           ${DELTA}"
echo "reward_avg:      ${REWARD_AVG}"
echo "run_time:        ${RUN_TIME}s"
echo "output_dir:      ${OUTPUT_DIR}"

if [[ "$FEEDBACK_MODE" == "dense" ]]; then
    echo ""
    echo "--- ALL METRICS ---"
    echo "efficiency:      ${EFFICIENCY}"
    echo "equality:        ${EQUALITY}"
    echo "sustainability:  ${SUSTAINABILITY}"
    echo "peace:           ${PEACE}"
    echo "reward_avg:      ${REWARD_AVG}"
    echo "wall_time:       ${WALL_TIME}s"
    echo ""
    echo "--- PER-ITERATION TRAJECTORY ---"
    echo "$METRICS_JSON" | python3 -c "
import json, sys
d = json.load(sys.stdin)
for t in d.get('trajectory', []):
    print(f\"  iter {t['iteration']}: reward={t['reward_avg']:.1f}  eff={t['efficiency']:.4f}  eq={t.get('equality',0):.3f}  sus={t.get('sustainability',0):.1f}  peace={t.get('peace',0):.2f}\")
"
    echo ""
    echo "--- PIPELINE STATE ---"
    echo "Files in pipeline/:"
    for f in pipeline/*.py; do
        echo "  $f ($(wc -l < "$f") lines, $(wc -c < "$f") bytes)"
    done
fi

# Save metrics JSON to output dir
echo "$METRICS_JSON" > "${OUTPUT_DIR}/metrics.json"
