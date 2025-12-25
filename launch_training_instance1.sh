#!/bin/bash
# Launch DeepSpeed Training on Instance 1
# Usage: ./launch_training_instance1.sh

export TNR_API_TOKEN="af22ef23aca94b95661ed2a3f5d8dbe94e78bc39d8c2a38c4c34ac25a77f3759"

echo "========================================="
echo "Launching DeepSpeed Training on Instance 1"
echo "========================================="
echo "Configuration:"
echo "  - GPUs: 4x A100 80GB"
echo "  - Global Batch Size: 256"
echo "  - Per-device Batch: 8"
echo "  - Gradient Accumulation: 8"
echo "  - DeepSpeed ZeRO-2"
echo "  - LoRA r=32, alpha=64"
echo ""

/usr/local/bin/python3.11 -m thunder.thunder connect 1 << 'EOF'
cd ~/socrates-training
source venv/bin/activate
# HF_TOKEN should be set in the environment (e.g., in ~/.bashrc or passed from calling environment)
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "=== Launching DeepSpeed Training ==="
nohup deepspeed --num_gpus=4 scripts/train_full_dataset.py \
  --config config_full_dataset.json \
  --deepspeed deepspeed_config.json \
  > training.log 2>&1 &

echo "Training launched! PID: $!"
sleep 3

echo ""
echo "Process count:"
ps aux | grep train_full_dataset | grep -v grep | wc -l

echo ""
echo "Initial log output:"
tail -30 training.log
EOF

echo ""
echo "========================================="
echo "Training launch complete!"
echo "========================================="
echo ""
echo "Monitor with:"
echo "  tnr ssh 1 'tail -f ~/socrates-training/training.log'"
echo ""
echo "Check GPU utilization:"
echo "  tnr ssh 1 'nvidia-smi'"
