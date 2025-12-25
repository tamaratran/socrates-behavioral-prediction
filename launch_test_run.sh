#!/bin/bash
# Launch 50-step test run on Thunder Compute
# Purpose: Verify all systems before full multi-day training
# Expected duration: 10-15 minutes
# Expected checkpoints: 5 (steps 10, 20, 30, 40, 50)

cd ~/socrates-training
source venv/bin/activate

# Set environment variables
export HF_TOKEN="${HF_TOKEN:-YOUR_HF_TOKEN_HERE}"
export CUDA_VISIBLE_DEVICES=0,1  # 2 GPUs for test

echo "================================"
echo "SOCRATES Test Run (50 steps)"
echo "================================"
echo "Started: $(date)"
echo "Config: config_test_run.json"
echo "GPUs: 2x A100"
echo "Expected duration: 10-15 minutes"
echo "================================"
echo ""

# Launch training with DeepSpeed (2 GPUs)
nohup deepspeed --num_gpus=2 scripts/train_full_dataset.py \
    --config config_test_run.json \
    --deepspeed deepspeed_config_test.json \
    > training_test.log 2>&1 &

TEST_PID=$!

echo "✓ Test run launched!"
echo "PID: ${TEST_PID}"
echo "Log file: ~/socrates-training/training_test.log"
echo ""
echo "Monitor progress:"
echo "  tail -f ~/socrates-training/training_test.log"
echo ""
echo "Check GPU usage:"
echo "  nvidia-smi"
echo ""
echo "Expected checkpoints will be saved to:"
echo "  ~/socrates-training/models/socrates-qwen-test-run/"
echo ""
