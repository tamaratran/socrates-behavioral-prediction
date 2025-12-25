#!/bin/bash
# Automated Setup and Training Launch for Thunder Compute
# Usage: ./scripts/setup_and_launch_training.sh [INSTANCE_ID]
# This script handles the complete setup from local machine to running training

set -e

INSTANCE_ID="${1:-0}"
TNR_API_TOKEN="af22ef23aca94b95661ed2a3f5d8dbe94e78bc39d8c2a38c4c34ac25a77f3759"
HF_TOKEN="${HF_TOKEN:-}"  # Set this in your environment

echo "========================================"
echo "SOCRATES Training Setup & Launch"
echo "========================================"
echo "Instance ID: $INSTANCE_ID"
echo ""

# Check HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN environment variable not set"
    echo "Set it with: export HF_TOKEN=\"your_token\""
    exit 1
fi

# Export API token
export TNR_API_TOKEN="$TNR_API_TOKEN"

# Check instance exists
echo "Checking instance status..."
if ! /usr/local/bin/python3.11 -m thunder.thunder status | grep -q "$INSTANCE_ID"; then
    echo "ERROR: Instance $INSTANCE_ID not found"
    echo "Create instance first via Thunder web UI with:"
    echo "  - GPU: A100 80GB"
    echo "  - Num GPUs: 4"
    echo "  - Disk: 500GB"
    exit 1
fi
echo "✓ Instance $INSTANCE_ID found"
echo ""

# Create remote directory structure
echo "Setting up remote directories..."
/usr/local/bin/python3.11 -m thunder.thunder connect $INSTANCE_ID << 'EOF'
mkdir -p ~/socrates-training/{scripts,data,models,config}
EOF
echo "✓ Directories created"
echo ""

# Upload fixed training script
echo "Uploading scripts..."
/usr/local/bin/python3.11 -m thunder.thunder upload $INSTANCE_ID scripts/train_full_dataset.py socrates-training/scripts/
/usr/local/bin/python3.11 -m thunder.thunder upload $INSTANCE_ID scripts/monitor_gpu_startup.sh socrates-training/scripts/
/usr/local/bin/python3.11 -m thunder.thunder upload $INSTANCE_ID scripts/prepare_full_data.py socrates-training/scripts/
echo "✓ Scripts uploaded"
echo ""

# Upload configs
echo "Uploading configuration files..."
/usr/local/bin/python3.11 -m thunder.thunder upload $INSTANCE_ID config_full_dataset.json socrates-training/
/usr/local/bin/python3.11 -m thunder.thunder upload $INSTANCE_ID deepspeed_config.json socrates-training/
/usr/local/bin/python3.11 -m thunder.thunder upload $INSTANCE_ID requirements_full_training.txt socrates-training/
echo "✓ Configs uploaded"
echo ""

# Setup Python environment
echo "Setting up Python environment..."
/usr/local/bin/python3.11 -m thunder.thunder connect $INSTANCE_ID << 'EOF'
cd ~/socrates-training
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements_full_training.txt
echo "✓ Python environment ready"
EOF
echo ""

# Download and prepare data
echo "Downloading and preparing dataset..."
echo "This may take 10-15 minutes for 2.9M examples..."
/usr/local/bin/python3.11 -m thunder.thunder connect $INSTANCE_ID << EOF
cd ~/socrates-training
source venv/bin/activate
export HF_TOKEN="$HF_TOKEN"
python scripts/prepare_full_data.py --output data/socsci210_full --seed 42
echo "✓ Dataset ready"
EOF
echo ""

# Launch training
echo "Launching training with DeepSpeed..."
/usr/local/bin/python3.11 -m thunder.thunder connect $INSTANCE_ID << EOF
cd ~/socrates-training
source venv/bin/activate
export HF_TOKEN="$HF_TOKEN"
export CUDA_VISIBLE_DEVICES=0,1,2,3
nohup deepspeed --num_gpus=4 scripts/train_full_dataset.py \
  --config config_full_dataset.json \
  --deepspeed deepspeed_config.json \
  > training.log 2>&1 &
echo "✓ Training launched! PID: \$!"
sleep 3
echo "Process count:"
ps aux | grep train_full_dataset | grep -v grep | wc -l
EOF
echo ""

# Start GPU monitoring
echo "Starting GPU startup monitor..."
echo "Will check GPU utilization for 5 minutes..."
echo "Auto-shutdown if GPUs never reach >80%"
echo ""

chmod +x scripts/monitor_gpu_startup.sh
./scripts/monitor_gpu_startup.sh $INSTANCE_ID

# If we get here, monitoring passed
echo ""
echo "========================================"
echo "Training Started Successfully!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Monitor progress:"
echo "   tnr ssh $INSTANCE_ID 'tail -f ~/socrates-training/training.log'"
echo ""
echo "2. Check GPU utilization:"
echo "   tnr ssh $INSTANCE_ID 'nvidia-smi'"
echo ""
echo "3. First checkpoint expected at step 500 (~2 hours)"
echo "   Download with:"
echo "   tnr download $INSTANCE_ID socrates-training/models/socrates-qwen-full-dataset/checkpoint-500 models/"
echo ""
echo "4. Full training: ~95 hours (4 days)"
echo "   Total cost: ~\$360"
echo ""
echo "Instance will continue running until:"
echo "- Training completes"
echo "- You manually delete it"
echo "- It crashes (ephemeral storage - data lost!)"
echo ""
echo "IMPORTANT: Download checkpoints regularly to avoid data loss!"
echo "========================================"
