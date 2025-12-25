#!/bin/bash
# Master Deployment Script for SOCRATES Full Training
# Single command to: create instance → setup environment → launch training → monitor GPUs
#
# Usage: ./deploy_full_training.sh
#
# Prerequisites:
#   export HF_TOKEN="your_huggingface_token"

set -e

echo "========================================"
echo "SOCRATES FULL TRAINING DEPLOYMENT"
echo "========================================"
echo "This will:"
echo "  1. Create Thunder instance (4x A100 80GB)"
echo "  2. Upload all scripts and configs"
echo "  3. Setup Python environment"
echo "  4. Download 2.9M training examples"
echo "  5. Launch training with DeepSpeed"
echo "  6. Monitor GPU startup (auto-shutdown if hung)"
echo ""
echo "Estimated time: ~20 minutes setup"
echo "Expected cost: ~\$360 for full 95-hour training"
echo ""

# Check HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN environment variable not set"
    echo "Set it with: export HF_TOKEN=\"your_token\""
    exit 1
fi

# Step 1: Create instance
echo "========================================"
echo "STEP 1/2: Creating Thunder Instance"
echo "========================================"
echo ""

python3 scripts/create_thunder_instance.py

# Get instance ID from file created by creation script
if [ ! -f ".thunder_instance_id" ]; then
    echo "ERROR: Instance creation failed - no instance ID file found"
    exit 1
fi

INSTANCE_ID=$(cat .thunder_instance_id)
echo ""
echo "Instance ID: $INSTANCE_ID"
echo ""

# Step 2: Setup and launch training
echo "========================================"
echo "STEP 2/2: Setup & Launch Training"
echo "========================================"
echo ""

./scripts/setup_and_launch_training.sh $INSTANCE_ID

# Done!
echo ""
echo "========================================"
echo "DEPLOYMENT COMPLETE!"
echo "========================================"
echo "Instance $INSTANCE_ID is running training"
echo ""
echo "Monitor commands:"
echo "  tnr ssh $INSTANCE_ID 'tail -f ~/socrates-training/training.log'"
echo "  tnr ssh $INSTANCE_ID 'nvidia-smi'"
echo ""
echo "Download checkpoints (every 500 steps):"
echo "  tnr download $INSTANCE_ID socrates-training/models/socrates-qwen-full-dataset/ models/"
echo ""
echo "Stop training and delete instance:"
echo "  tnr delete $INSTANCE_ID"
echo ""
echo "Instance will run until:"
echo "  - Training completes (~95 hours)"
echo "  - You manually delete it"
echo "  - It crashes (data lost! download checkpoints regularly)"
echo "========================================"
