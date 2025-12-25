#!/bin/bash
# Download training log from Thunder Compute instance 0

export TNR_API_TOKEN="af22ef23aca94b95661ed2a3f5d8dbe94e78bc39d8c2a38c4c34ac25a77f3759"

echo "Downloading training log from instance 0..."

/usr/local/bin/python3.11 -m thunder.thunder connect 0 << 'EOF' > training_instance0.log 2>&1
cat ~/socrates-training/training.log
EOF

if [ -f "training_instance0.log" ]; then
    SIZE=$(wc -c < training_instance0.log)
    echo "✓ Log downloaded: training_instance0.log ($SIZE bytes)"
    echo ""
    echo "Latest training progress:"
    grep "{'loss':" training_instance0.log | tail -5
else
    echo "✗ Failed to download log"
    exit 1
fi
