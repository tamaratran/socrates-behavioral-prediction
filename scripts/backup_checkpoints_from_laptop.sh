#!/bin/bash
# Backup SOCRATES Checkpoints from Thunder GPU Server to Local Laptop
# Uses rsync over SSH via Thunder CLI for resumable, efficient transfers

set -e  # Exit on error

# Default parameters
INSTANCE=0
MAX_CHECKPOINTS=2
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
    --max-checkpoints)
      MAX_CHECKPOINTS="$2"
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
      echo "Backup SOCRATES checkpoints from Thunder GPU server to local laptop"
      echo ""
      echo "Options:"
      echo "  --instance N          Thunder instance number (default: 0)"
      echo "  --max-checkpoints N   Number of most recent full checkpoints to download (default: 2)"
      echo "  --local-dir PATH      Local backup directory (default: ./backups/socrates-checkpoints)"
      echo "  --dry-run             Show what would be transferred without actually downloading"
      echo "  -h, --help            Show this help message"
      echo ""
      echo "Examples:"
      echo "  $0                                    # Backup 2 most recent checkpoints from instance 0"
      echo "  $0 --instance 1 --max-checkpoints 3  # Backup 3 checkpoints from instance 1"
      echo "  $0 --dry-run                          # Preview what would be downloaded"
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
echo "SOCRATES Checkpoint Backup"
echo "========================================="
echo "Instance:           $INSTANCE"
echo "Remote directory:   $REMOTE_DIR"
echo "Local directory:    $LOCAL_DIR"
echo "Max checkpoints:    $MAX_CHECKPOINTS"
echo "Dry run:            $DRY_RUN"
echo ""

# Get Thunder SSH connection info
echo "Connecting to Thunder instance $INSTANCE..."
export TNR_API_TOKEN="$TNR_API_TOKEN"

# Create local backup directory
if [ "$DRY_RUN" = false ]; then
  mkdir -p "$LOCAL_DIR"
  echo -e "${GREEN}✓${NC} Local backup directory ready: $LOCAL_DIR"
fi

# List available checkpoints on remote server
echo ""
echo "Fetching checkpoint list from server..."
CHECKPOINT_LIST=$(/usr/local/bin/python3.11 -m thunder.thunder connect $INSTANCE << 'EOF'
cd ~/socrates-training/models/socrates-qwen-full-dataset 2>/dev/null || exit 1
ls -d checkpoint-* 2>/dev/null | sort -V | tail -n 20 || echo "NO_CHECKPOINTS"
EOF
)

if [ "$CHECKPOINT_LIST" = "NO_CHECKPOINTS" ] || [ -z "$CHECKPOINT_LIST" ]; then
  echo -e "${RED}✗${NC} No checkpoints found on server"
  echo "Training may not have reached the first checkpoint (step 500) yet."
  exit 1
fi

# Get the N most recent checkpoints
CHECKPOINTS_TO_BACKUP=$(echo "$CHECKPOINT_LIST" | tail -n $MAX_CHECKPOINTS)
CHECKPOINT_COUNT=$(echo "$CHECKPOINTS_TO_BACKUP" | wc -l | tr -d ' ')

echo -e "${GREEN}✓${NC} Found $CHECKPOINT_COUNT checkpoint(s) to backup:"
echo "$CHECKPOINTS_TO_BACKUP" | sed 's/^/  - /'
echo ""

# Download each checkpoint using rsync
for CHECKPOINT in $CHECKPOINTS_TO_BACKUP; do
  echo "----------------------------------------"
  echo "Downloading: $CHECKPOINT"
  echo "----------------------------------------"

  RSYNC_OPTS="-avz --progress --partial"
  if [ "$DRY_RUN" = true ]; then
    RSYNC_OPTS="$RSYNC_OPTS --dry-run"
  fi

  # Use Thunder CLI to establish SSH connection and rsync
  # We need to get the SSH connection details from Thunder
  SSH_INFO=$(/usr/local/bin/python3.11 -m thunder.thunder ssh $INSTANCE --print-command 2>&1 | grep -o "ssh.*" || echo "")

  if [ -z "$SSH_INFO" ]; then
    echo -e "${YELLOW}⚠${NC} Using Thunder connect method (direct SSH not available)"

    # Alternative: Download via Thunder connect with tar+base64
    if [ "$DRY_RUN" = false ]; then
      echo "Creating tarball on remote server..."
      /usr/local/bin/python3.11 -m thunder.thunder connect $INSTANCE << EOF
cd ~/socrates-training/models/socrates-qwen-full-dataset
tar -czf /tmp/${CHECKPOINT}.tar.gz $CHECKPOINT/
ls -lh /tmp/${CHECKPOINT}.tar.gz
EOF

      echo "Downloading tarball..."
      /usr/local/bin/python3.11 -m thunder.thunder connect $INSTANCE << EOF > "/tmp/${CHECKPOINT}.tar.gz"
cat /tmp/${CHECKPOINT}.tar.gz
EOF

      echo "Extracting to local directory..."
      mkdir -p "$LOCAL_DIR"
      tar -xzf "/tmp/${CHECKPOINT}.tar.gz" -C "$LOCAL_DIR/"
      rm "/tmp/${CHECKPOINT}.tar.gz"

      echo -e "${GREEN}✓${NC} Downloaded: $CHECKPOINT"
    else
      echo "[DRY RUN] Would download: $CHECKPOINT"
    fi
  else
    # Use rsync over SSH (preferred method)
    REMOTE_HOST=$(echo "$SSH_INFO" | grep -o "[^@]*@[^ ]*" | head -1)

    rsync $RSYNC_OPTS \
      -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
      "$REMOTE_HOST:$REMOTE_DIR/$CHECKPOINT/" \
      "$LOCAL_DIR/$CHECKPOINT/"

    if [ "$DRY_RUN" = false ]; then
      echo -e "${GREEN}✓${NC} Downloaded: $CHECKPOINT"
    fi
  fi
done

# Download config files and metadata (always download these, they're small)
echo ""
echo "----------------------------------------"
echo "Downloading config files and metadata"
echo "----------------------------------------"

CONFIG_FILES="config_full_dataset.json deepspeed_config.json"

for CONFIG in $CONFIG_FILES; do
  if [ "$DRY_RUN" = false ]; then
    /usr/local/bin/python3.11 -m thunder.thunder connect $INSTANCE << EOF > "$LOCAL_DIR/$CONFIG"
cat ~/socrates-training/$CONFIG
EOF
    if [ $? -eq 0 ]; then
      echo -e "${GREEN}✓${NC} Downloaded: $CONFIG"
    else
      echo -e "${YELLOW}⚠${NC} Could not download: $CONFIG (may not exist)"
    fi
  else
    echo "[DRY RUN] Would download: $CONFIG"
  fi
done

# Summary
echo ""
echo "========================================="
echo "Backup Complete!"
echo "========================================="

if [ "$DRY_RUN" = false ]; then
  echo -e "${GREEN}✓${NC} Backed up $CHECKPOINT_COUNT checkpoint(s) to:"
  echo "  $LOCAL_DIR"
  echo ""
  echo "Backup contents:"
  du -sh "$LOCAL_DIR"/* 2>/dev/null | sed 's/^/  /'
  echo ""
  echo "Total backup size:"
  du -sh "$LOCAL_DIR"

  # Verify checkpoint structure
  echo ""
  echo "Verifying checkpoint structure..."
  for CHECKPOINT in $CHECKPOINTS_TO_BACKUP; do
    if [ -d "$LOCAL_DIR/$CHECKPOINT" ]; then
      ADAPTER_COUNT=$(find "$LOCAL_DIR/$CHECKPOINT" -name "adapter_*.safetensors" -o -name "adapter_*.bin" | wc -l | tr -d ' ')
      OPTIMIZER_COUNT=$(find "$LOCAL_DIR/$CHECKPOINT" -name "optimizer.pt" -o -name "optimizer.bin" | wc -l | tr -d ' ')

      if [ $ADAPTER_COUNT -gt 0 ] && [ $OPTIMIZER_COUNT -gt 0 ]; then
        echo -e "  ${GREEN}✓${NC} $CHECKPOINT: Complete (adapter + optimizer)"
      elif [ $ADAPTER_COUNT -gt 0 ]; then
        echo -e "  ${YELLOW}⚠${NC} $CHECKPOINT: Adapter only (missing optimizer state)"
      else
        echo -e "  ${RED}✗${NC} $CHECKPOINT: Incomplete"
      fi
    fi
  done
else
  echo "[DRY RUN] No files were actually downloaded."
  echo "Run without --dry-run to perform the backup."
fi

echo ""
echo "Next steps:"
echo "  1. Verify backup integrity in: $LOCAL_DIR"
echo "  2. To restore to a new server: ./scripts/restore_checkpoints_to_server.sh"
echo "========================================="
