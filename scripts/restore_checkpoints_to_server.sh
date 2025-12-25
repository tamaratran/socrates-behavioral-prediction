#!/bin/bash
# Restore SOCRATES Checkpoints from Local Laptop to Thunder GPU Server
# Uses Thunder CLI to upload checkpoints for resuming training

set -e  # Exit on error

# Default parameters
INSTANCE=0
CHECKPOINT=""
LOCAL_DIR="./backups/socrates-checkpoints"
DRY_RUN=false
TNR_API_TOKEN="${TNR_API_TOKEN:-af22ef23aca94b95661ed2a3f5d8dbe94e78bc39d8c2a38c4c34ac25a77f3759}"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --instance)
      INSTANCE="$2"
      shift 2
      ;;
    --checkpoint)
      CHECKPOINT="$2"
      shift 2
      ;;
    --local-dir)
      LOCAL_DIR="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Restore SOCRATES checkpoints from local laptop to Thunder GPU server"
      echo ""
      echo "Options:"
      echo "  --instance N        Thunder instance number (default: 0)"
      echo "  --checkpoint NAME   Specific checkpoint to restore (e.g., checkpoint-500)"
      echo "                      If not specified, the latest checkpoint will be used"
      echo "  --local-dir PATH    Local backup directory (default: ./backups/socrates-checkpoints)"
      echo "  --dry-run           Show what would be transferred without actually uploading"
      echo "  -h, --help          Show this help message"
      echo ""
      echo "Examples:"
      echo "  $0                                   # Restore latest checkpoint to instance 0"
      echo "  $0 --instance 1                      # Restore latest checkpoint to instance 1"
      echo "  $0 --checkpoint checkpoint-1500      # Restore specific checkpoint"
      echo "  $0 --dry-run                         # Preview what would be uploaded"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use -h or --help for usage information"
      exit 1
      ;;
  esac
done

# Remote paths
REMOTE_DIR="~/socrates-training/models/socrates-qwen-full-dataset"

echo "========================================="
echo "SOCRATES Checkpoint Restore"
echo "========================================="

# If no checkpoint specified, find the latest one
if [ -z "$CHECKPOINT" ]; then
  echo "No checkpoint specified, searching for latest..."
  if [ ! -d "$LOCAL_DIR" ]; then
    echo -e "${RED}✗${NC} Local backup directory not found: $LOCAL_DIR"
    exit 1
  fi

  LATEST_CHECKPOINT=$(ls -d "$LOCAL_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -1 | xargs basename 2>/dev/null)

  if [ -z "$LATEST_CHECKPOINT" ]; then
    echo -e "${RED}✗${NC} No checkpoints found in: $LOCAL_DIR"
    echo "Please run backup_checkpoints_from_laptop.sh first or specify --checkpoint"
    exit 1
  fi

  CHECKPOINT="$LATEST_CHECKPOINT"
  echo -e "${GREEN}✓${NC} Found latest checkpoint: $CHECKPOINT"
fi

# Verify checkpoint exists locally
CHECKPOINT_PATH="$LOCAL_DIR/$CHECKPOINT"
if [ ! -d "$CHECKPOINT_PATH" ]; then
  echo -e "${RED}✗${NC} Checkpoint not found: $CHECKPOINT_PATH"
  exit 1
fi

echo ""
echo "Instance:           $INSTANCE"
echo "Checkpoint:         $CHECKPOINT"
echo "Local path:         $CHECKPOINT_PATH"
echo "Remote directory:   $REMOTE_DIR"
echo "Dry run:            $DRY_RUN"
echo ""

# Verify checkpoint structure
echo "Verifying checkpoint structure..."
ADAPTER_COUNT=$(find "$CHECKPOINT_PATH" -name "adapter_*.safetensors" -o -name "adapter_*.bin" | wc -l | tr -d ' ')
OPTIMIZER_COUNT=$(find "$CHECKPOINT_PATH" -name "optimizer.pt" -o -name "optimizer.bin" | wc -l | tr -d ' ')

if [ $ADAPTER_COUNT -eq 0 ]; then
  echo -e "${RED}✗${NC} Checkpoint missing LoRA adapter files"
  exit 1
fi

if [ $OPTIMIZER_COUNT -eq 0 ]; then
  echo -e "${YELLOW}⚠${NC} Warning: Checkpoint missing optimizer state (not a full DeepSpeed checkpoint)"
  echo "Training will restart from scratch with these weights, not resume from exact state."
  read -p "Continue anyway? (y/N): " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
  fi
else
  echo -e "${GREEN}✓${NC} Checkpoint appears complete (adapter + optimizer)"
fi

# Calculate checkpoint size
CHECKPOINT_SIZE=$(du -sh "$CHECKPOINT_PATH" | cut -f1)
echo ""
echo "Checkpoint size: $CHECKPOINT_SIZE"
echo ""

# Upload checkpoint
export TNR_API_TOKEN="$TNR_API_TOKEN"

if [ "$DRY_RUN" = true ]; then
  echo "[DRY RUN] Would upload $CHECKPOINT to instance $INSTANCE"
  echo "[DRY RUN] Files that would be uploaded:"
  find "$CHECKPOINT_PATH" -type f | head -20 | sed 's/^/  /'
  FILE_COUNT=$(find "$CHECKPOINT_PATH" -type f | wc -l | tr -d ' ')
  if [ $FILE_COUNT -gt 20 ]; then
    echo "  ... and $((FILE_COUNT - 20)) more files"
  fi
else
  echo "Connecting to Thunder instance $INSTANCE..."

  # Create remote directory structure
  echo "Creating remote directory..."
  /usr/local/bin/python3.11 -m thunder.thunder connect $INSTANCE << EOF
mkdir -p $REMOTE_DIR
ls -la $REMOTE_DIR
EOF

  if [ $? -ne 0 ]; then
    echo -e "${RED}✗${NC} Failed to create remote directory"
    exit 1
  fi

  # Create tarball for efficient transfer
  echo ""
  echo "Creating tarball for transfer..."
  TARBALL="/tmp/${CHECKPOINT}.tar.gz"
  tar -czf "$TARBALL" -C "$LOCAL_DIR" "$CHECKPOINT"

  TARBALL_SIZE=$(du -sh "$TARBALL" | cut -f1)
  echo -e "${GREEN}✓${NC} Tarball created: $TARBALL ($TARBALL_SIZE)"

  # Upload tarball
  echo ""
  echo "Uploading checkpoint to instance $INSTANCE..."
  echo "This may take several minutes depending on checkpoint size..."

  # Upload via Thunder connect (streaming base64)
  cat "$TARBALL" | /usr/local/bin/python3.11 -m thunder.thunder connect $INSTANCE << 'EOF' > /dev/null
cat > /tmp/checkpoint_upload.tar.gz
EOF

  if [ $? -ne 0 ]; then
    echo -e "${RED}✗${NC} Failed to upload tarball"
    rm "$TARBALL"
    exit 1
  fi

  echo -e "${GREEN}✓${NC} Upload complete"

  # Extract on remote server
  echo ""
  echo "Extracting checkpoint on remote server..."
  /usr/local/bin/python3.11 -m thunder.thunder connect $INSTANCE << EOF
cd $REMOTE_DIR
tar -xzf /tmp/checkpoint_upload.tar.gz
rm /tmp/checkpoint_upload.tar.gz
ls -lh $CHECKPOINT/
EOF

  if [ $? -ne 0 ]; then
    echo -e "${RED}✗${NC} Failed to extract checkpoint"
    rm "$TARBALL"
    exit 1
  fi

  echo -e "${GREEN}✓${NC} Checkpoint extracted successfully"
  rm "$TARBALL"

  # Upload config files if they exist
  echo ""
  echo "Uploading config files..."
  for CONFIG in config_full_dataset.json deepspeed_config.json; do
    if [ -f "$LOCAL_DIR/$CONFIG" ]; then
      cat "$LOCAL_DIR/$CONFIG" | /usr/local/bin/python3.11 -m thunder.thunder connect $INSTANCE << EOF > /dev/null
cat > ~/socrates-training/$CONFIG
EOF
      if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} Uploaded: $CONFIG"
      else
        echo -e "${YELLOW}⚠${NC} Failed to upload: $CONFIG"
      fi
    fi
  done

  # Verify remote checkpoint
  echo ""
  echo "Verifying remote checkpoint..."
  REMOTE_CHECK=$(/usr/local/bin/python3.11 -m thunder.thunder connect $INSTANCE << EOF
cd $REMOTE_DIR/$CHECKPOINT
ls -1 | wc -l
EOF
)

  if [ "$REMOTE_CHECK" -gt 0 ]; then
    echo -e "${GREEN}✓${NC} Remote checkpoint verified ($REMOTE_CHECK files)"
  else
    echo -e "${RED}✗${NC} Remote checkpoint verification failed"
    exit 1
  fi
fi

# Summary and resume command
echo ""
echo "========================================="
echo "Restore Complete!"
echo "========================================="

if [ "$DRY_RUN" = false ]; then
  echo -e "${GREEN}✓${NC} Checkpoint restored to instance $INSTANCE:"
  echo "  $REMOTE_DIR/$CHECKPOINT"
  echo ""
  echo "To resume training from this checkpoint, run:"
  echo ""
  echo "  tnr ssh $INSTANCE"
  echo "  cd ~/socrates-training"
  echo "  source venv/bin/activate"
  echo "  export HF_TOKEN=\"your_token_here\""
  echo "  export CUDA_VISIBLE_DEVICES=0,1,2,3"
  echo ""
  echo "  deepspeed --num_gpus=4 scripts/train_full_dataset.py \\"
  echo "    --config config_full_dataset.json \\"
  echo "    --deepspeed deepspeed_config.json \\"
  echo "    --resume-from-checkpoint models/socrates-qwen-full-dataset/$CHECKPOINT"
  echo ""
  echo "Note: The training will resume from the exact state saved in this checkpoint,"
  echo "      including optimizer state, learning rate schedule, and global step counter."
else
  echo "[DRY RUN] No files were actually uploaded."
  echo "Run without --dry-run to perform the restore."
fi

echo "========================================="
