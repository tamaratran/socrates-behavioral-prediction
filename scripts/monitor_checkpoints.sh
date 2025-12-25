#!/bin/bash
# Automatic checkpoint monitoring and download script
# Monitors Thunder Compute instance for new checkpoints and downloads them automatically
# Usage: ./scripts/monitor_checkpoints.sh [instance_id] [check_interval_seconds]
# Example: ./scripts/monitor_checkpoints.sh 0 300  (check every 5 minutes)

set -e

INSTANCE_ID=${1:-0}
CHECK_INTERVAL=${2:-300}  # Default: check every 5 minutes
REMOTE_DIR="~/socrates-training/models/socrates-qwen-full-dataset"
LOCAL_DIR="models/socrates-qwen-full-dataset"
STATE_FILE=".checkpoint_monitor_state"

# Create local directory if it doesn't exist
mkdir -p "$LOCAL_DIR"

# Initialize state file (tracks which checkpoints we've already downloaded)
if [ ! -f "$STATE_FILE" ]; then
    echo "# Checkpoint download state" > "$STATE_FILE"
    echo "# Format: checkpoint-NNN" >> "$STATE_FILE"
fi

echo "================================"
echo "Checkpoint Monitor"
echo "================================"
echo "Instance ID: ${INSTANCE_ID}"
echo "Check interval: ${CHECK_INTERVAL}s"
echo "Remote directory: ${REMOTE_DIR}"
echo "Local directory: ${LOCAL_DIR}"
echo "State file: ${STATE_FILE}"
echo ""
echo "Monitoring started at: $(date)"
echo "Press Ctrl+C to stop"
echo "================================"
echo ""

# Function to get list of checkpoints on remote
get_remote_checkpoints() {
    tnr ssh "$INSTANCE_ID" "ls -1 ${REMOTE_DIR} 2>/dev/null | grep '^checkpoint-' | sort -V" 2>/dev/null || echo ""
}

# Function to check if checkpoint already downloaded
is_downloaded() {
    local checkpoint=$1
    grep -q "^${checkpoint}$" "$STATE_FILE" 2>/dev/null
}

# Function to mark checkpoint as downloaded
mark_downloaded() {
    local checkpoint=$1
    echo "$checkpoint" >> "$STATE_FILE"
}

# Function to download a checkpoint
download_checkpoint() {
    local checkpoint=$1
    local checkpoint_num=$(echo "$checkpoint" | sed 's/checkpoint-//')

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Downloading ${checkpoint}..."

    if bash scripts/download_checkpoint.sh "$checkpoint_num"; then
        mark_downloaded "$checkpoint"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ ${checkpoint} downloaded successfully"
        return 0
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ Failed to download ${checkpoint}"
        return 1
    fi
}

# Main monitoring loop
ITERATION=0
while true; do
    ITERATION=$((ITERATION + 1))

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Check #${ITERATION}: Scanning for new checkpoints..."

    # Get list of remote checkpoints
    REMOTE_CHECKPOINTS=$(get_remote_checkpoints)

    if [ -z "$REMOTE_CHECKPOINTS" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] No checkpoints found on remote (training may not have started yet)"
    else
        # Count total checkpoints
        TOTAL_COUNT=$(echo "$REMOTE_CHECKPOINTS" | wc -l | tr -d ' ')
        NEW_COUNT=0

        # Check each checkpoint
        while IFS= read -r checkpoint; do
            if ! is_downloaded "$checkpoint"; then
                NEW_COUNT=$((NEW_COUNT + 1))
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] Found new checkpoint: ${checkpoint}"
                download_checkpoint "$checkpoint" &
                # Wait a bit between downloads to avoid overwhelming the connection
                sleep 10
            fi
        done <<< "$REMOTE_CHECKPOINTS"

        # Wait for all downloads to complete
        wait

        if [ "$NEW_COUNT" -eq 0 ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] No new checkpoints (${TOTAL_COUNT} already downloaded)"
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Downloaded ${NEW_COUNT} new checkpoint(s)"
        fi
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting ${CHECK_INTERVAL}s until next check..."
    echo ""
    sleep "$CHECK_INTERVAL"
done
