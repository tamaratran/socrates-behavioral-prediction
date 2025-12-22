#!/bin/bash
cd ~/socrates-training
source venv/bin/activate

# Set environment variables (set HF_TOKEN before running)
export HF_TOKEN="${HF_TOKEN:-YOUR_HF_TOKEN_HERE}"
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Launch training directly with deepspeed
nohup deepspeed --num_gpus=4 scripts/train_full_dataset.py \
    --config config_full_dataset.json \
    --deepspeed deepspeed_config.json \
    --use-wandb \
    > training_direct.log 2>&1 &

echo "Training launched directly with DeepSpeed"
echo "PID: $!"
echo "Log: ~/socrates-training/training_direct.log"
