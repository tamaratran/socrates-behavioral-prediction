#!/bin/bash
###############################################################################
# Full Dataset Training Launcher for SOCRATES
# Optimized for Thunder Compute with 8x A100 80GB GPUs
#
# Usage:
#   bash scripts/run_full_training.sh [--prepare-data] [--resume CHECKPOINT]
###############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NUM_GPUS=8
CONFIG_FILE="config_full_dataset.json"
DEEPSPEED_CONFIG="deepspeed_config.json"
DATA_DIR="data/socsci210_full"
OUTPUT_DIR="models/socrates-qwen-full-dataset"

# Parse arguments
PREPARE_DATA=false
RESUME_CHECKPOINT=""
USE_WANDB=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --prepare-data)
      PREPARE_DATA=true
      shift
      ;;
    --resume)
      RESUME_CHECKPOINT="$2"
      shift 2
      ;;
    --wandb)
      USE_WANDB=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--prepare-data] [--resume CHECKPOINT] [--wandb]"
      exit 1
      ;;
  esac
done

###############################################################################
# Header
###############################################################################

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                  SOCRATES FULL DATASET TRAINING                            ║${NC}"
echo -e "${BLUE}║                  2.9M Examples on 8x A100 80GB                             ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

###############################################################################
# Environment Check
###############################################################################

echo -e "${YELLOW}[1/6] Checking environment...${NC}"

# Check CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}ERROR: nvidia-smi not found. CUDA may not be installed.${NC}"
    exit 1
fi

# Check GPU count
AVAILABLE_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "✓ Found $AVAILABLE_GPUS GPUs"

if [ "$AVAILABLE_GPUS" -lt "$NUM_GPUS" ]; then
    echo -e "${YELLOW}WARNING: Requested $NUM_GPUS GPUs but only $AVAILABLE_GPUS available${NC}"
    echo "Continuing with $AVAILABLE_GPUS GPUs..."
    NUM_GPUS=$AVAILABLE_GPUS
fi

# Check Python
if ! command -v python &> /dev/null; then
    echo -e "${RED}ERROR: Python not found${NC}"
    exit 1
fi

PYTHON_VERSION=$(python --version)
echo "✓ $PYTHON_VERSION"

# Check required packages
echo "Checking required packages..."
python -c "import torch; import transformers; import peft; import datasets; import trl" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Missing required packages${NC}"
    echo "Please install: pip install torch transformers peft datasets trl accelerate deepspeed bitsandbytes"
    exit 1
fi

echo "✓ All required packages installed"

# Check config files
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}ERROR: Config file not found: $CONFIG_FILE${NC}"
    exit 1
fi

if [ ! -f "$DEEPSPEED_CONFIG" ]; then
    echo -e "${RED}ERROR: DeepSpeed config not found: $DEEPSPEED_CONFIG${NC}"
    exit 1
fi

echo "✓ Configuration files found"
echo ""

###############################################################################
# Data Preparation
###############################################################################

echo -e "${YELLOW}[2/6] Checking dataset...${NC}"

if [ ! -d "$DATA_DIR" ] || [ "$PREPARE_DATA" = true ]; then
    echo "Preparing full dataset (2.9M examples)..."
    echo "This will take 30-60 minutes..."
    python scripts/prepare_full_data.py --output "$DATA_DIR"

    if [ $? -ne 0 ]; then
        echo -e "${RED}ERROR: Data preparation failed${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Data preparation complete${NC}"
else
    echo "✓ Dataset found at $DATA_DIR"

    # Check for required files
    if [ ! -f "$DATA_DIR/train.jsonl" ] || [ ! -f "$DATA_DIR/val.jsonl" ]; then
        echo -e "${RED}ERROR: Missing train.jsonl or val.jsonl in $DATA_DIR${NC}"
        exit 1
    fi

    # Count examples
    TRAIN_COUNT=$(wc -l < "$DATA_DIR/train.jsonl")
    VAL_COUNT=$(wc -l < "$DATA_DIR/val.jsonl")
    echo "  Train: $(printf '%d' $TRAIN_COUNT | sed ':a;s/\B[0-9]\{3\}\>/,&/;ta') examples"
    echo "  Val:   $(printf '%d' $VAL_COUNT | sed ':a;s/\B[0-9]\{3\}\>/,&/;ta') examples"
fi
echo ""

###############################################################################
# GPU Information
###############################################################################

echo -e "${YELLOW}[3/6] GPU Configuration${NC}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader | \
    awk -F', ' '{printf "  GPU %s: %s (%s total, %s free)\n", $1, $2, $3, $4}'
echo ""

###############################################################################
# Training Configuration Summary
###############################################################################

echo -e "${YELLOW}[4/6] Training Configuration${NC}"
echo "  Config: $CONFIG_FILE"
echo "  DeepSpeed: $DEEPSPEED_CONFIG"
echo "  GPUs: $NUM_GPUS"
echo "  Output: $OUTPUT_DIR"

if [ -n "$RESUME_CHECKPOINT" ]; then
    echo "  Resume from: $RESUME_CHECKPOINT"
fi

if [ "$USE_WANDB" = true ]; then
    echo "  Logging: Weights & Biases"
fi

echo ""

# Parse key config values
MODEL_NAME=$(python -c "import json; print(json.load(open('$CONFIG_FILE'))['model_name'])")
EPOCHS=$(python -c "import json; print(json.load(open('$CONFIG_FILE'))['training']['num_epochs'])")
BATCH_SIZE=$(python -c "import json; print(json.load(open('$CONFIG_FILE'))['training']['global_batch_size'])")
LR=$(python -c "import json; print(json.load(open('$CONFIG_FILE'))['training']['learning_rate'])")
LORA_R=$(python -c "import json; print(json.load(open('$CONFIG_FILE'))['lora']['r'])")

echo "  Model: $MODEL_NAME"
echo "  Epochs: $EPOCHS"
echo "  Global Batch Size: $BATCH_SIZE"
echo "  Learning Rate: $LR"
echo "  LoRA Rank: $LORA_R"
echo ""

###############################################################################
# Estimated Time and Cost
###############################################################################

echo -e "${YELLOW}[5/6] Estimates${NC}"

# Estimate training time
# Rough estimate: 2.9M examples / batch_size * 3 seconds per step
TRAIN_COUNT=$(wc -l < "$DATA_DIR/train.jsonl" 2>/dev/null || echo "2900000")
STEPS=$((TRAIN_COUNT / BATCH_SIZE * EPOCHS))
TIME_SECONDS=$((STEPS * 3))
TIME_HOURS=$((TIME_SECONDS / 3600))

echo "  Training steps: ~$(printf '%d' $STEPS | sed ':a;s/\B[0-9]\{3\}\>/,&/;ta')"
echo "  Estimated time: ~$TIME_HOURS hours (assuming ~3s/step)"

# Cost estimate (Thunder Compute: $1.10/GPU/hour)
COST_PER_GPU_HOUR=1.10
TOTAL_COST=$(echo "$TIME_HOURS * $NUM_GPUS * $COST_PER_GPU_HOUR" | bc)
echo "  Estimated cost: ~\$$TOTAL_COST (Thunder Compute @ \$1.10/GPU/hr)"
echo ""

###############################################################################
# Launch Training
###############################################################################

echo -e "${YELLOW}[6/6] Launching training...${NC}"
echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  Training started at $(date +'%Y-%m-%d %H:%M:%S')                                       ║${NC}"
echo -e "${GREEN}║  Press Ctrl+C to interrupt (model will be saved at last checkpoint)       ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Build command
CMD="deepspeed --num_gpus=$NUM_GPUS scripts/train_full_dataset.py \
    --config $CONFIG_FILE \
    --deepspeed $DEEPSPEED_CONFIG"

if [ "$USE_WANDB" = true ]; then
    CMD="$CMD --use-wandb"
fi

if [ -n "$RESUME_CHECKPOINT" ]; then
    CMD="$CMD --resume-from-checkpoint $RESUME_CHECKPOINT"
fi

# Log command
echo "Command: $CMD"
echo ""

# Execute
eval $CMD

EXIT_CODE=$?

###############################################################################
# Completion
###############################################################################

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                       TRAINING COMPLETED SUCCESSFULLY                      ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${GREEN}Model saved to: $OUTPUT_DIR${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Evaluate model:"
    echo "     python scripts/evaluate.py --model $OUTPUT_DIR"
    echo ""
    echo "  2. Test inference:"
    echo "     python scripts/test_inference.py --model $OUTPUT_DIR"
else
    echo -e "${RED}╔════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║                         TRAINING FAILED                                    ║${NC}"
    echo -e "${RED}╚════════════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${RED}Training exited with code $EXIT_CODE${NC}"
    echo ""
    echo "Check logs above for error details."

    if [ -d "$OUTPUT_DIR" ]; then
        echo ""
        echo "Resume from last checkpoint with:"
        echo "  bash scripts/run_full_training.sh --resume $OUTPUT_DIR/checkpoint-*"
    fi
fi

exit $EXIT_CODE
