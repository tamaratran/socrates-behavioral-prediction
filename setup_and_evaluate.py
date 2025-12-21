#!/usr/bin/env python3
"""
Setup and run proper evaluation on Thunder Compute GPU
Evaluates on 3,289 test examples from 20 truly unseen studies
"""

import subprocess
import sys
import time
import json
from pathlib import Path

def run_command(cmd, timeout=60):
    """Run a command and return output"""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return result.stdout, result.stderr, result.returncode

def create_instance():
    """Create a new Thunder Compute GPU instance"""
    print("\n" + "="*50)
    print("Creating Thunder Compute GPU instance (T4 16GB)")
    print("="*50)

    # Create T4 instance for evaluation
    stdout, stderr, code = run_command([
        'python3.11', '-m', 'thunder.thunder', 'create',
        '--gpu', 't4',
        '--disk', '100',
        '--mode', 'prototyping'
    ], timeout=120)

    print(stdout)
    if code != 0:
        print(f"Error creating instance: {stderr}", file=sys.stderr)
        return False

    print("✓ Instance created successfully!")
    print("Waiting 30 seconds for instance to fully initialize...")
    time.sleep(30)
    return True

def upload_files_to_gpu():
    """Upload model, script, and data to GPU via SSH connection"""
    print("\n" + "="*50)
    print("Uploading files to GPU")
    print("="*50)

    # Read files
    print("Reading evaluate.py...")
    with open('scripts/evaluate.py', 'r') as f:
        evaluate_script = f.read()

    print(f"Reading test data (3,289 lines)...")
    with open('data/socsci210_1pct_proper_split/test.jsonl', 'r') as f:
        test_data = f.read()

    print(f"Reading metadata...")
    with open('data/socsci210_1pct_proper_split/metadata.json', 'r') as f:
        metadata = f.read()

    # Create tar package
    print("\nCreating upload package...")
    subprocess.run([
        'tar', '-czf', 'evaluation-upload.tar.gz',
        'models/socrates-qwen-paper-replication/',
        'scripts/evaluate.py',
        'data/socsci210_1pct_proper_split/',
        'requirements.txt'
    ], check=True)

    # Get size
    size_mb = Path('evaluation-upload.tar.gz').stat().st_size / 1024 / 1024
    print(f"✓ Package created: {size_mb:.1f} MB")

    return True

def run_evaluation_on_gpu():
    """Connect to GPU and run evaluation"""
    print("\n" + "="*50)
    print("Running evaluation on GPU")
    print("="*50)

    cmd = ['python3.11', '-m', 'thunder.thunder', 'connect', '0']
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, text=True)

    # Commands to run on GPU
    commands = """
cd /home/ubuntu

# Check if package needs to be uploaded
if [ ! -f "evaluation-upload.tar.gz" ]; then
    echo "ERROR: evaluation-upload.tar.gz not found!"
    echo "Please upload it manually"
    exit 1
fi

# Extract package
echo "Extracting files..."
tar -xzf evaluation-upload.tar.gz

# Install dependencies
echo "Installing dependencies..."
pip install -q torch transformers peft accelerate datasets scikit-learn scipy matplotlib pandas numpy

# Verify files
echo ""
echo "Verification:"
echo "Model files:"
ls -lh models/socrates-qwen-paper-replication/ | head -5
echo ""
echo "Test data:"
wc -l data/socsci210_1pct_proper_split/test.jsonl
echo ""

# Run evaluation
echo "========================================="
echo "Starting evaluation on 3,289 test examples"
echo "Expected runtime: ~4 hours"
echo "========================================="
echo ""

nohup python scripts/evaluate.py \
  --model models/socrates-qwen-paper-replication \
  --test-data data/socsci210_1pct_proper_split/test.jsonl \
  --output results_proper_split \
  > evaluation_proper.log 2>&1 &

EVAL_PID=$!
echo "Evaluation started!"
echo "PID: $EVAL_PID"
echo ""
echo "To monitor progress:"
echo "  tail -f evaluation_proper.log"
echo ""
echo "To check status:"
echo "  ps aux | grep evaluate.py"

exit
"""

    try:
        stdout, stderr = proc.communicate(input=commands, timeout=120)
        print(stdout)
        if stderr and 'exit' not in stderr.lower():
            print(f"STDERR: {stderr}", file=sys.stderr)
        return True
    except subprocess.TimeoutExpired:
        proc.kill()
        print("Connection timed out")
        return False

def main():
    print("="*60)
    print("SOCRATES Proper Evaluation Setup")
    print("="*60)
    print("")
    print("This will:")
    print("  1. Create a T4 16GB GPU instance (~$0.66/hr)")
    print("  2. Upload model and test data")
    print("  3. Run evaluation on 3,289 examples from 20 unseen studies")
    print("  4. Expected runtime: ~4 hours (~$2.64 total)")
    print("")

    input("Press Enter to continue or Ctrl+C to cancel...")

    # Step 1: Create instance
    if not create_instance():
        print("Failed to create instance")
        return 1

    # Step 2: Upload files
    if not upload_files_to_gpu():
        print("Failed to prepare files")
        return 1

    print("\n" + "="*60)
    print("IMPORTANT: Manual upload required")
    print("="*60)
    print("")
    print("The package 'evaluation-upload.tar.gz' has been created.")
    print("You need to manually copy it to the GPU instance:")
    print("")
    print("Option 1 - Use scp (if Thunder provides SSH access):")
    print("  scp evaluation-upload.tar.gz user@instance:/home/ubuntu/")
    print("")
    print("Option 2 - Copy via your method of choice")
    print("")
    input("Press Enter after you've uploaded evaluation-upload.tar.gz...")

    # Step 3: Run evaluation
    if not run_evaluation_on_gpu():
        print("Failed to start evaluation")
        return 1

    print("\n" + "="*60)
    print("✓ Evaluation Started Successfully!")
    print("="*60)
    print("")
    print("Next steps:")
    print("  1. Monitor: python3.11 -m thunder.thunder connect 0")
    print("     Then: tail -f evaluation_proper.log")
    print("")
    print("  2. Check after ~4 hours for results")
    print("")
    print("  3. Download results when complete")
    print("")

    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nCancelled by user")
        sys.exit(1)
