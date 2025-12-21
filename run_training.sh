#!/bin/bash
#
# SOCRATES Paper Replication - One-Command Training Launcher
# Based on: Finetuning LLMs for Human Behavior Prediction (arxiv:2509.05830)
#
# Usage:
#   ./run_training.sh
#

set -e  # Exit on error

echo "================================================================================"
echo "SOCRATES PAPER REPLICATION - Training Launcher"
echo "Paper: Finetuning LLMs for Human Behavior Prediction (arxiv:2509.05830)"
echo "================================================================================"
echo ""

# Check if running on GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️  WARNING: nvidia-smi not found. Are you running on a GPU machine?"
    echo "   This training requires an NVIDIA GPU (preferably A100 40GB+)"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✓ GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install/update dependencies
echo "Installing dependencies..."
pip install -q torch transformers datasets accelerate peft trl huggingface-hub \
    python-dotenv tqdm pandas "numpy<2" bitsandbytes scipy scikit-learn \
    matplotlib seaborn

# Check if data exists
if [ ! -d "data/socsci210_1pct" ]; then
    echo ""
    echo "⚠️  ERROR: Training data not found at data/socsci210_1pct/"
    echo "   Please run: python scripts/prepare_data.py --subset 0.01 --output data/socsci210_1pct"
    exit 1
fi

echo ""
echo "================================================================================"
echo "Starting training with paper-exact configuration..."
echo "================================================================================"
echo ""
echo "Configuration:"
echo "  - Model: Qwen/Qwen2.5-14B-Instruct"
echo "  - Method: Supervised Fine-Tuning (SFT) with QLoRA"
echo "  - Dataset: data/socsci210_1pct/"
echo "  - Epochs: 1"
echo "  - Batch Size: 256 (effective, via gradient accumulation)"
echo "  - Learning Rate: 1e-05"
echo ""
echo "Estimated time: 2-4 hours on A100 40GB"
echo "================================================================================"
echo ""

# Run training
python scripts/train_paper_replication.py --config config_paper.json

echo ""
echo "================================================================================"
echo "Training Complete!"
echo "================================================================================"
echo ""
echo "Next steps:"
echo "  1. Evaluate the model:"
echo "     python scripts/evaluate.py --model models/socrates-qwen-paper-replication"
echo ""
echo "  2. Test interactively:"
echo "     python scripts/test_inference.py --model models/socrates-qwen-paper-replication"
echo ""
echo "================================================================================"
