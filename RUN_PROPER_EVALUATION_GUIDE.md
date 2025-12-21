# Guide: Run Proper Evaluation on GPU

This guide will help you run the proper evaluation on 3,289 test examples from 20 truly unseen studies, following the SOCRATES paper methodology.

## Overview

- **Test Set**: 3,289 examples from 20 unseen studies (proper study-level split)
- **Metrics**: Wasserstein distance, MAE, RMSE, Correlation
- **Hardware**: T4 16GB GPU (~$0.66/hr)
- **Expected Time**: ~4 hours
- **Expected Cost**: ~$2.64

## Step 1: Create GPU Instance

```bash
cd "/Users/tamaratran/Desktop/Desktop - MacBook Pro (8)/prediction-agents-copy"

# Create T4 instance
python3.11 -m thunder.thunder create \
  --gpu t4 \
  --disk 100 \
  --mode prototyping
```

Wait 30 seconds for the instance to initialize.

## Step 2: Prepare Upload Package

```bash
# Create a compressed package with all necessary files
tar -czf evaluation-package.tar.gz \
  models/socrates-qwen-paper-replication/ \
  scripts/evaluate.py \
  data/socsci210_1pct_proper_split/ \
  requirements.txt

# Check package size
ls -lh evaluation-package.tar.gz
# Should be ~140-150 MB
```

## Step 3: Transfer Files to GPU

Since Thunder Compute doesn't have a direct upload command, we'll use a workaround:

### Option A: Via GitHub (Recommended)

1. First, commit the new files to git:
```bash
git add data/socsci210_1pct_proper_split/ scripts/evaluate.py
git commit -m "Add proper study-level split data and Wasserstein metric"
git push
```

2. On the GPU, clone/pull the repo:
```bash
python3.11 -m thunder.thunder connect 0
```

Then in the GPU terminal:
```bash
cd /home/ubuntu
git clone https://github.com/tamaratran/socrates-behavioral-prediction prediction-agents-copy
cd prediction-agents-copy
```

### Option B: Transfer via Base64 (For Small Files)

For the evaluate.py script only (if model is already on GPU):

```bash
# On local machine
cd "/Users/tamaratran/Desktop/Desktop - MacBook Pro (8)/prediction-agents-copy"

# Encode evaluate.py
base64 scripts/evaluate.py > evaluate_py_b64.txt

# Similarly for test data
tar -czf test_data.tar.gz data/socsci210_1pct_proper_split/
base64 test_data.tar.gz > test_data_b64.txt
```

Then connect to GPU and paste the base64 content to decode.

### Option C: Recreate Files Directly on GPU (Simplest)

Copy-paste the file contents directly via the SSH session.

## Step 4: Set Up Environment on GPU

```bash
python3.11 -m thunder.thunder connect 0
```

In the GPU terminal:

```bash
cd /home/ubuntu/prediction-agents-copy

# Install dependencies
pip install torch transformers peft accelerate datasets scikit-learn scipy matplotlib pandas numpy

# Verify test data
wc -l data/socsci210_1pct_proper_split/test.jsonl
# Should show: 3289

# Verify model exists
ls -lh models/socrates-qwen-paper-replication/
# Should show adapter_model.safetensors (131M) and other files
```

## Step 5: Run Evaluation

```bash
# Start evaluation in background
nohup python scripts/evaluate.py \
  --model models/socrates-qwen-paper-replication \
  --test-data data/socsci210_1pct_proper_split/test.jsonl \
  --output results_proper_split \
  > evaluation_proper.log 2>&1 &

# Get PID
echo $!

# Check it's running
ps aux | grep evaluate.py
```

## Step 6: Monitor Progress

```bash
# Check log file
tail -f evaluation_proper.log

# Check progress (Ctrl+C to exit tail)
# Every ~12 seconds per example = ~11 hours for 3,289 examples
# With batch optimizations, expect ~4 hours
```

## Step 7: Check Results (After ~4 Hours)

```bash
# Check if evaluation completed
ps aux | grep evaluate.py

# View results
cat results_proper_split/metrics.json | python -m json.tool

# Expected output:
# {
#   "wasserstein_distance": 0.20-0.25,  (paper: 0.151 on full data)
#   "mae": ~5-7,
#   "correlation": ~0.6-0.7  (lower than our inflated 0.815)
# }
```

## Step 8: Download Results

From your local machine:

```bash
# Connect and cat the results
python3.11 -m thunder.thunder connect 0
```

Then in the GPU terminal:

```bash
cd /home/ubuntu/prediction-agents-copy

# View metrics
cat results_proper_split/metrics.json

# Sample predictions
head -20 results_proper_split/predictions.json
```

Copy the JSON output and save locally.

## Step 9: Clean Up

```bash
# Destroy the GPU instance to stop billing
python3.11 -m thunder.thunder destroy 0
```

## Quick Start (If Model Already on GPU)

If you still have the model on a running GPU instance:

```bash
# Connect
python3.11 -m thunder.thunder connect 0

# Copy updated evaluate.py (paste content directly or use git pull)

# Run evaluation
cd /home/ubuntu/prediction-agents-copy
python scripts/evaluate.py \
  --model models/socrates-qwen-paper-replication \
  --test-data data/socsci210_1pct_proper_split/test.jsonl \
  --output results_proper_split

# Wait ~4 hours and check results
```

## Expected Results

Based on the SOCRATES paper and our 1% training subset:

- **Wasserstein Distance**: 0.20-0.25
  - Paper (100% data): 0.151
  - Ours (1% data): Expected to be higher (worse)

- **Correlation**: 0.6-0.7
  - Our previous (data leak): 0.815
  - Proper evaluation should be lower

- **MAE**: 5-7
  - Harder task = higher error

This represents the model's TRUE performance on genuinely unseen experimental studies.
