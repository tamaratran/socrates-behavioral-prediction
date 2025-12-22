#!/bin/bash
cd ~/socrates-training
source venv/bin/activate

# Set HuggingFace token (set this before running)
export HF_TOKEN="${HF_TOKEN:-YOUR_HF_TOKEN_HERE}"

# Launch training with nohup
nohup bash scripts/run_full_training.sh --wandb > training.log 2>&1 &

echo "Training launched in background (PID: $!)"
echo "Log file: ~/socrates-training/training.log"
echo "To monitor: tail -f ~/socrates-training/training.log"
