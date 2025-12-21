#!/bin/bash
# Run proper evaluation on Thunder Compute GPU
# This evaluates the model on 20 truly unseen studies (3,289 examples)
# Following SOCRATES paper Section 5.3 methodology

set -e

echo "====================================="
echo "SOCRATES Proper Evaluation - Study-Level Split"
echo "====================================="
echo ""
echo "This will evaluate on 3,289 examples from 20 unseen studies"
echo "Expected runtime: ~4 hours on T4 16GB GPU (~$2.64)"
echo ""

# Create package with all necessary files
echo "Creating evaluation package..."
tar -czf evaluation-package.tar.gz \
    scripts/evaluate.py \
    config_paper.json \
    data/socsci210_1pct_proper_split/ \
    requirements.txt

PACKAGE_SIZE=$(ls -lh evaluation-package.tar.gz | awk '{print $5}')
echo "Package created: $PACKAGE_SIZE"
echo ""

# Encode the package as base64 for transfer
echo "Encoding package for transfer..."
base64 evaluation-package.tar.gz > evaluation-package.tar.gz.b64

echo "Starting evaluation on GPU instance..."
echo ""

# Run the evaluation via Python script
python3.11 << 'PYTHON_EOF'
import subprocess
import sys

# Connect to Thunder instance and set up evaluation
cmd = ['python3.11', '-m', 'thunder.thunder', 'connect', '0']
proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# Read the base64-encoded package
with open('evaluation-package.tar.gz.b64', 'r') as f:
    package_b64 = f.read()

commands = f"""
cd /home/ubuntu

# Receive and decode the package
cat > evaluation-package.tar.gz.b64 << 'PACKAGE_EOF'
{package_b64}
PACKAGE_EOF

# Decode and extract
echo "Decoding package..."
base64 -d evaluation-package.tar.gz.b64 > evaluation-package.tar.gz
tar -xzf evaluation-package.tar.gz

# Install dependencies
echo "Installing dependencies..."
pip install -q torch transformers peft accelerate datasets scikit-learn scipy matplotlib pandas numpy

# Verify test data
echo ""
echo "Test data info:"
wc -l data/socsci210_1pct_proper_split/test.jsonl
echo ""

# Run evaluation
echo ""
echo "Starting evaluation on 3,289 test examples..."
echo "This will take approximately 4 hours..."
echo ""

nohup python scripts/evaluate.py \
  --model models/socrates-qwen-paper-replication \
  --test-data data/socsci210_1pct_proper_split/test.jsonl \
  --output results_proper_split \
  > evaluation_proper.log 2>&1 &

echo "Evaluation started in background!"
echo "PID: \$(ps aux | grep 'evaluate.py' | grep -v grep | awk '{{print \$2}}')"

exit
"""

try:
    stdout, stderr = proc.communicate(input=commands, timeout=180)
    print(stdout)
    if stderr:
        print(stderr, file=sys.stderr)
except subprocess.TimeoutExpired:
    proc.kill()
    print("Connection timed out")
    sys.exit(1)

PYTHON_EOF

# Cleanup
rm -f evaluation-package.tar.gz.b64

echo ""
echo "====================================="
echo "Evaluation launched successfully!"
echo "====================================="
echo ""
echo "Next steps:"
echo "  1. Monitor: python3.11 -m thunder.thunder connect 0"
echo "     Then: tail -f evaluation_proper.log"
echo ""
echo "  2. Check progress periodically"
echo ""
echo "Expected results:"
echo "  - Wasserstein distance: ~0.20-0.25"
echo "  - This is the PROPER evaluation on truly unseen studies"
