#!/bin/bash
# Checkpoint Evaluation Script - compare to baseline
# Usage: ./scripts/evaluate_checkpoint.sh <checkpoint_dir> <baseline_metrics>

set -e

CHECKPOINT_DIR="${1:-models/socrates-qwen-500step-validation/checkpoint-500}"
BASELINE_METRICS="${2:-evaluation_results/baseline/metrics.json}"
VAL_DATA="data/socsci210_full/val.jsonl"
OUTPUT_DIR="${CHECKPOINT_DIR}/evaluation"
MAX_EXAMPLES=10000

echo "========================================"
echo "Checkpoint Evaluation & GO/NO-GO Decision"
echo "========================================"
echo "Checkpoint: ${CHECKPOINT_DIR}"
echo "Baseline: ${BASELINE_METRICS}"
echo "Output: ${OUTPUT_DIR}"
echo ""

# Check baseline exists
if [ ! -f "$BASELINE_METRICS" ]; then
    echo "ERROR: Baseline metrics not found: $BASELINE_METRICS"
    echo "Run ./scripts/evaluate_baseline.sh first!"
    exit 1
fi

# Check checkpoint exists
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT_DIR"
    exit 1
fi

# Run evaluation
echo "Running checkpoint evaluation..."
python scripts/evaluate.py \
    --model "$CHECKPOINT_DIR" \
    --test-data "$VAL_DATA" \
    --output "$OUTPUT_DIR" \
    --max-examples "$MAX_EXAMPLES"

echo ""
echo "========================================"
echo "Comparison to Baseline"
echo "========================================"

# Extract metrics
BASELINE_WD=$(cat "$BASELINE_METRICS" | python -c "import json, sys; print(json.load(sys.stdin)['wasserstein_distance'])")
CHECKPOINT_WD=$(cat "${OUTPUT_DIR}/metrics.json" | python -c "import json, sys; print(json.load(sys.stdin)['wasserstein_distance'])")

IMPROVEMENT=$(python -c "print(f'{(1 - $CHECKPOINT_WD/$BASELINE_WD) * 100:.1f}')")

echo "Wasserstein Distance:"
echo "  Baseline:    ${BASELINE_WD}"
echo "  Checkpoint:  ${CHECKPOINT_WD}"
echo "  Improvement: ${IMPROVEMENT}%"
echo ""

# Decision logic
if (( $(echo "$IMPROVEMENT >= 5" | bc -l) )); then
    echo "✅ GO - Improvement ≥5%"
    echo "Proceed to full training!"
    exit 0
else
    echo "❌ NO-GO - Improvement <5%"
    echo "Debug hyperparameters before full training"
    exit 1
fi
