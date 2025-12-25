#!/bin/bash
# Launch full training on Thunder Compute
# Usage: ./scripts/launch_full_training.sh

set -e

echo "========================================"
echo "Launch Full Training"
echo "========================================"
echo ""

# Check HF token
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN environment variable not set"
    echo "Set it with: export HF_TOKEN=\"your_token\""
    exit 1
fi

# Configuration
INSTANCE_ID="${1:-0}"  # Default to instance 0
CONFIG="config_full_dataset.json"
DEEPSPEED="deepspeed_config.json"
NUM_GPUS=4

echo "Instance ID: $INSTANCE_ID"
echo "Config: $CONFIG"
echo "DeepSpeed: $DEEPSPEED"
echo "GPUs: $NUM_GPUS"
echo ""

# Launch training via SSH
echo "Launching training..."
tnr ssh $INSTANCE_ID "cd ~/socrates-training && \
  source venv/bin/activate && \
  export HF_TOKEN=\"$HF_TOKEN\" && \
  export CUDA_VISIBLE_DEVICES=0,1,2,3 && \
  nohup deepspeed --num_gpus=$NUM_GPUS scripts/train_full_dataset.py \
    --config $CONFIG \
    --deepspeed $DEEPSPEED \
    > training.log 2>&1 &"

echo ""
echo "✓ Training launched!"
echo ""
echo "Monitor progress:"
echo "  tnr ssh $INSTANCE_ID 'tail -f ~/socrates-training/training.log'"
echo ""
echo "Check GPU utilization:"
echo "  tnr ssh $INSTANCE_ID 'nvidia-smi'"
echo ""
echo "First checkpoint expected at step 500 (~2 hours)"
echo ""
