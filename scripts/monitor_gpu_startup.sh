#!/bin/bash
# GPU Startup Validator for Thunder Compute
# Monitors GPU utilization for 5 minutes after training launch
# Shuts down instance if GPUs never reach >80% utilization
#
# Usage: ./scripts/monitor_gpu_startup.sh [instance_id]
# Example: ./scripts/monitor_gpu_startup.sh 0

set -e

INSTANCE_ID=${1:-0}
CHECK_INTERVAL=30  # seconds
MAX_CHECKS=10      # 5 minutes total (10 checks × 30 seconds)
THRESHOLD=80       # GPU utilization threshold (%)

echo "========================================="
echo "GPU Startup Validator"
echo "========================================="
echo "Instance ID: $INSTANCE_ID"
echo "Monitoring for 5 minutes (10 checks × 30s)"
echo "Auto-shutdown if GPUs never reach >$THRESHOLD%"
echo ""

for i in $(seq 1 $MAX_CHECKS); do
    echo "[Check $i/$MAX_CHECKS] $(date '+%H:%M:%S')"

    # Get GPU utilization from nvidia-smi via Thunder CLI
    GPU_UTIL=$(export TNR_API_TOKEN="af22ef23aca94b95661ed2a3f5d8dbe94e78bc39d8c2a38c4c34ac25a77f3759" && /usr/local/bin/python3.11 -m thunder.thunder connect $INSTANCE_ID 2>/dev/null <<'EOF'
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits
EOF
)

    if [ -z "$GPU_UTIL" ]; then
        echo "  ERROR: Cannot reach instance or nvidia-smi failed"
        echo "  Instance may have crashed or been deleted"
        exit 1
    fi

    # Get maximum GPU utilization across all GPUs
    MAX_GPU_UTIL=$(echo "$GPU_UTIL" | awk '{if($1>max)max=$1}END{print max}')

    if [ -z "$MAX_GPU_UTIL" ]; then
        echo "  WARNING: Could not parse GPU utilization"
        sleep $CHECK_INTERVAL
        continue
    fi

    echo "  Max GPU utilization: $MAX_GPU_UTIL%"

    # Check if any GPU exceeds threshold
    if [ "$MAX_GPU_UTIL" -gt "$THRESHOLD" ]; then
        echo ""
        echo "========================================="
        echo "✅ SUCCESS: GPUs engaged!"
        echo "========================================="
        echo "Max GPU utilization: $MAX_GPU_UTIL% (threshold: >$THRESHOLD%)"
        echo "Training started successfully."
        echo "Safe to exit monitor - training will continue."
        echo ""
        exit 0
    fi

    # Wait before next check
    if [ $i -lt $MAX_CHECKS ]; then
        sleep $CHECK_INTERVAL
    fi
done

echo ""
echo "========================================="
echo "❌ FAILURE: GPUs never reached >$THRESHOLD%"
echo "========================================="
echo "Training likely hung during startup."
echo "Shutting down instance to prevent costs..."
echo ""

# Shutdown the instance to stop charges
export TNR_API_TOKEN="af22ef23aca94b95661ed2a3f5d8dbe94e78bc39d8c2a38c4c34ac25a77f3759"
/usr/local/bin/python3.11 -m thunder.thunder delete $INSTANCE_ID

echo "Instance $INSTANCE_ID deleted."
echo "Saved approximately \$7.16/hour in GPU charges."
echo ""
exit 1
