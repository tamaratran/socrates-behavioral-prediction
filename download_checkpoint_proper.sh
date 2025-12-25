#!/bin/bash
#
# Proper Checkpoint Download Using Thunder Native SCP
# This is the standard method - no weird binary trimming needed!
#

set -e  # Exit on error

# Configuration
INSTANCE="${INSTANCE:-1}"
REMOTE_CHECKPOINT_DIR="${REMOTE_CHECKPOINT_DIR:-~/socrates-training/models/socrates-qwen-full-dataset/checkpoint-10}"
LOCAL_BACKUP_DIR="${LOCAL_BACKUP_DIR:-./backups/socrates-checkpoints}"
TNR_API_TOKEN="${TNR_API_TOKEN:-af22ef23aca94b95661ed2a3f5d8dbe94e78bc39d8c2a38c4c34ac25a77f3759}"

export TNR_API_TOKEN

mkdir -p "$LOCAL_BACKUP_DIR"

echo "=========================================="
echo "SOCRATES Checkpoint Download (Proper SCP)"
echo "=========================================="
echo ""
echo "Instance: $INSTANCE"
echo "Remote:   $REMOTE_CHECKPOINT_DIR"
echo "Local:    $LOCAL_BACKUP_DIR"
echo ""

# Download entire checkpoint directory using Thunder's native SCP
echo "[1/2] Downloading checkpoint directory..."
/usr/local/bin/python3.11 -m thunder.thunder scp \
  "${INSTANCE}:${REMOTE_CHECKPOINT_DIR}" \
  "$LOCAL_BACKUP_DIR/"

if [ $? -ne 0 ]; then
    echo "ERROR: SCP download failed"
    exit 1
fi

echo ""
echo "[2/2] Verifying download..."
CHECKPOINT_NAME=$(basename "$REMOTE_CHECKPOINT_DIR")
if [ ! -d "$LOCAL_BACKUP_DIR/$CHECKPOINT_NAME" ]; then
    echo "ERROR: Checkpoint directory not found at $LOCAL_BACKUP_DIR/$CHECKPOINT_NAME"
    exit 1
fi

# Show what we downloaded
echo ""
echo "Downloaded checkpoint contents:"
ls -lh "$LOCAL_BACKUP_DIR/$CHECKPOINT_NAME/" | head -15

echo ""
echo "=========================================="
echo "SUCCESS: Checkpoint downloaded!"
echo "=========================================="
echo ""
echo "Location: $LOCAL_BACKUP_DIR/$CHECKPOINT_NAME/"
echo ""
echo "Next steps:"
echo "  1. Verify contents: ls -lh $LOCAL_BACKUP_DIR/$CHECKPOINT_NAME/"
echo "  2. Use for evaluation or resume training"
echo ""
