#!/bin/bash
#
# Production-Ready Checkpoint Download Script
# Downloads checkpoint tarball from Thunder Compute using proven streaming method
#
# Based on best practices from scripts/backup_checkpoints_from_laptop.sh
# Uses tarball + streaming with proper shell redirection to avoid binary corruption
#

set -e  # Exit on error

# Configuration
INSTANCE="${INSTANCE:-1}"
REMOTE_FILE="${REMOTE_FILE:-/tmp/checkpoint-10.tar.gz}"
LOCAL_DIR="${LOCAL_DIR:-./backups/test-checkpoints}"
LOCAL_FILE="${LOCAL_FILE:-${LOCAL_DIR}/checkpoint-10.tar.gz}"
TNR_API_TOKEN="${TNR_API_TOKEN:-af22ef23aca94b95661ed2a3f5d8dbe94e78bc39d8c2a38c4c34ac25a77f3759}"

# Export API token
export TNR_API_TOKEN

# Create backup directory
mkdir -p "$LOCAL_DIR"

echo "=========================================="
echo "SOCRATES Checkpoint Download (Production)"
echo "=========================================="
echo ""
echo "Instance:     $INSTANCE"
echo "Remote file:  $REMOTE_FILE"
echo "Local dir:    $LOCAL_DIR"
echo "Local file:   $LOCAL_FILE"
echo ""
echo "Expected time: 2-5 minutes for 1.8GB file"
echo ""

# Step 1: Verify remote file exists and get size
echo "[1/4] Verifying remote file..."
REMOTE_SIZE=$(/usr/local/bin/python3.11 -m thunder.thunder connect $INSTANCE << 'EOF'
if [ -f /tmp/checkpoint-10.tar.gz ]; then
    stat -f%z /tmp/checkpoint-10.tar.gz 2>/dev/null || stat -c%s /tmp/checkpoint-10.tar.gz
else
    echo "0"
fi
EOF
)

# Clean output (remove Thunder banner)
REMOTE_SIZE=$(echo "$REMOTE_SIZE" | grep -E '^[0-9]+$' | tail -1)

if [ -z "$REMOTE_SIZE" ] || [ "$REMOTE_SIZE" -eq 0 ]; then
    echo "ERROR: Remote file not found or empty"
    echo "Please ensure checkpoint tarball exists at: $REMOTE_FILE"
    exit 1
fi

REMOTE_SIZE_MB=$((REMOTE_SIZE / 1024 / 1024))
echo "Remote file size: ${REMOTE_SIZE_MB} MB"
echo ""

# Step 2: Download via streaming
echo "[2/4] Downloading checkpoint (this may take 2-5 minutes)..."
echo "Streaming $REMOTE_FILE to $LOCAL_FILE..."

# Critical: Shell redirection OUTSIDE Thunder CLI to avoid capturing banner
/usr/local/bin/python3.11 -m thunder.thunder connect $INSTANCE << 'THUNDER_EOF' > "$LOCAL_FILE"
cat /tmp/checkpoint-10.tar.gz
THUNDER_EOF

if [ $? -ne 0 ]; then
    echo "ERROR: Download failed"
    rm -f "$LOCAL_FILE"  # Clean up partial file
    exit 1
fi

echo "Download complete!"
echo ""

# Step 3: Verify local file size
echo "[3/4] Verifying download..."
if [ ! -f "$LOCAL_FILE" ]; then
    echo "ERROR: Local file not found after download"
    exit 1
fi

LOCAL_SIZE=$(stat -f%z "$LOCAL_FILE" 2>/dev/null || stat -c%s "$LOCAL_FILE")
LOCAL_SIZE_MB=$((LOCAL_SIZE / 1024 / 1024))

echo "Downloaded: ${LOCAL_SIZE_MB} MB"

# Verify size is reasonable (should be >100MB for valid checkpoint)
if [ $LOCAL_SIZE_MB -lt 100 ]; then
    echo "ERROR: File size too small (${LOCAL_SIZE_MB} MB), download may have failed"
    echo "Expected: ~${REMOTE_SIZE_MB} MB"
    exit 1
fi

# Check if sizes match (allowing small difference due to compression)
SIZE_DIFF=$((REMOTE_SIZE - LOCAL_SIZE))
SIZE_DIFF=${SIZE_DIFF#-}  # Absolute value

if [ $SIZE_DIFF -gt 1048576 ]; then  # 1MB tolerance
    echo "WARNING: Size mismatch detected"
    echo "  Remote: ${REMOTE_SIZE_MB} MB"
    echo "  Local:  ${LOCAL_SIZE_MB} MB"
    echo "  Difference: $((SIZE_DIFF / 1024 / 1024)) MB"
fi

echo ""

# Step 4: Verify tarball integrity
echo "[4/4] Verifying tarball integrity..."
if ! tar -tzf "$LOCAL_FILE" > /dev/null 2>&1; then
    echo "ERROR: Tarball is corrupted or incomplete"
    echo "You may need to delete $LOCAL_FILE and try again"
    exit 1
fi

# Count files in tarball
FILE_COUNT=$(tar -tzf "$LOCAL_FILE" | wc -l | tr -d ' ')
echo "Tarball contains $FILE_COUNT files"

# List first few files
echo ""
echo "Sample contents (first 10 files):"
tar -tzf "$LOCAL_FILE" | head -10

echo ""
echo "=========================================="
echo "SUCCESS: Checkpoint downloaded!"
echo "=========================================="
echo ""
echo "File saved to: $LOCAL_FILE"
echo "Size: ${LOCAL_SIZE_MB} MB"
echo "Status: Verified and ready to use"
echo ""
echo "Next steps:"
echo "  1. Extract: tar -xzf $LOCAL_FILE -C $LOCAL_DIR/"
echo "  2. Verify checkpoint contents in: ${LOCAL_DIR}/checkpoint-10/"
echo "  3. Use for training resume or model evaluation"
echo ""
