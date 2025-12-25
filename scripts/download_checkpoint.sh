#!/bin/bash
# Download a specific checkpoint from Thunder Compute to local machine
# Usage: ./scripts/download_checkpoint.sh <checkpoint_number>
# Example: ./scripts/download_checkpoint.sh 500

set -e

if [ -z "$1" ]; then
    echo "Error: Checkpoint number required"
    echo "Usage: $0 <checkpoint_number>"
    echo "Example: $0 500"
    exit 1
fi

CHECKPOINT_NUM=$1
REMOTE_PATH="~/socrates-training/models/socrates-qwen-full-dataset/checkpoint-${CHECKPOINT_NUM}"
LOCAL_DIR="models/socrates-qwen-full-dataset"
LOCAL_PATH="${LOCAL_DIR}/checkpoint-${CHECKPOINT_NUM}"

# Create local directory if it doesn't exist
mkdir -p "$LOCAL_DIR"

echo "================================"
echo "Checkpoint Download"
echo "================================"
echo "Checkpoint: ${CHECKPOINT_NUM}"
echo "Remote path: ${REMOTE_PATH}"
echo "Local path: ${LOCAL_PATH}"
echo ""

# Check if checkpoint already exists locally
if [ -d "$LOCAL_PATH" ]; then
    echo "⚠️  Checkpoint already exists locally: ${LOCAL_PATH}"
    read -p "Overwrite? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Download cancelled"
        exit 0
    fi
    echo "Removing existing checkpoint..."
    rm -rf "$LOCAL_PATH"
fi

echo "Downloading checkpoint from Thunder Compute..."
echo ""

# Download using tnr scp (recursive for directory)
# Syntax: tnr scp <instance_id> <remote_path> <local_path>
if tnr scp 0 "${REMOTE_PATH}" "${LOCAL_PATH}"; then
    echo ""
    echo "✅ Checkpoint downloaded successfully!"
    echo ""
    echo "Local path: ${LOCAL_PATH}"

    # Show checkpoint contents
    echo ""
    echo "Checkpoint contents:"
    ls -lh "$LOCAL_PATH"

    # Calculate size
    SIZE=$(du -sh "$LOCAL_PATH" | cut -f1)
    echo ""
    echo "Total size: ${SIZE}"
else
    echo ""
    echo "❌ Download failed!"
    echo ""
    echo "Troubleshooting:"
    echo "1. Check if Thunder instance is running: tnr status"
    echo "2. Check if checkpoint exists on remote: tnr ssh 0 'ls ~/socrates-training/models/socrates-qwen-full-dataset/'"
    echo "3. Verify TNR_API_TOKEN is set correctly"
    exit 1
fi

echo ""
echo "================================"
echo "Download Complete"
echo "================================"
