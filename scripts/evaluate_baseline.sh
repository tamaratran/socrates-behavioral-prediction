#!/bin/bash
# Baseline Evaluation Script
# Evaluates untrained base model (Qwen2.5-14B-Instruct) on validation set
# This establishes the "zero learning" baseline for comparison

set -e

# Configuration
BASE_MODEL="Qwen/Qwen2.5-14B-Instruct"
VAL_DATA="data/socsci210_full/val.jsonl"
OUTPUT_DIR="evaluation_results/baseline"
MAX_EXAMPLES=10000  # Sample for speed

echo "================================"
echo "Baseline Model Evaluation"
echo "================================"
echo "Model: ${BASE_MODEL}"
echo "Validation data: ${VAL_DATA}"
echo "Sampling: ${MAX_EXAMPLES} examples"
echo "Output: ${OUTPUT_DIR}"
echo ""

# Check if validation data exists
if [ ! -f "$VAL_DATA" ]; then
    echo "ERROR: Validation data not found: $VAL_DATA"
    echo "Please ensure data/socsci210_full/val.jsonl exists"
    exit 1
fi

# Run evaluation
echo "Running baseline evaluation..."
echo "This will take 2-3 hours on a single GPU"
echo ""

python scripts/evaluate.py \
    --model "$BASE_MODEL" \
    --test-data "$VAL_DATA" \
    --output "$OUTPUT_DIR" \
    --max-examples "$MAX_EXAMPLES"

echo ""
echo "================================"
echo "Baseline Evaluation Complete"
echo "================================"
echo "Results saved to: ${OUTPUT_DIR}"
echo ""
echo "Key baseline metrics:"
cat "${OUTPUT_DIR}/metrics.json" | python -m json.tool | grep -E "wasserstein_distance|mae|correlation"

echo ""
echo "Next steps:"
echo "1. Note the baseline Wasserstein distance"
echo "2. Train to 500 steps: Use config_500step_validation.json"
echo "3. Evaluate 500-step checkpoint and compare to this baseline"
echo "4. GO if 5-10% improvement, NO-GO if no improvement"
echo ""
