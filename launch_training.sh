#!/bin/bash
cd ~/socrates-training
source venv/bin/activate

# Set HuggingFace token (set this before running)
export HF_TOKEN="${HF_TOKEN:-YOUR_HF_TOKEN_HERE}"

# Launch training in tmux session
tmux new-session -d -s training "bash scripts/run_full_training.sh --wandb 2>&1 | tee training.log"

echo "Training launched in tmux session 'training'"
echo "To monitor: tmux attach -t training"
echo "To detach: Ctrl+B then D"
