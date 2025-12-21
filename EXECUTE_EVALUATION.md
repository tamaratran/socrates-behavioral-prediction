# Execute Proper Evaluation - Quick Start Guide

This guide provides copy-paste commands to run the proper evaluation on 3,289 test examples from 20 unseen studies.

---

## Step 1: Create GPU Instance (Your Terminal)

Open a new terminal and run:

```bash
cd "/Users/tamaratran/Desktop/Desktop - MacBook Pro (8)/prediction-agents-copy"

# Create T4 GPU instance
python3.11 -m thunder.thunder create
```

**Interactive prompts** - Select:
- GPU: **t4** (or **a10g** for faster evaluation)
- Disk: **100 GB**
- Mode: **prototyping**
- vCPUs: Accept default
- RAM: Accept default

Wait 30 seconds for instance to initialize.

---

## Step 2: Setup Environment on GPU

```bash
# Connect to instance
python3.11 -m thunder.thunder connect 0
```

Once connected, **copy-paste this entire block**:

```bash
cd /home/ubuntu

# Clone repository (contains all updated code and data)
git clone https://github.com/tamaratran/socrates-behavioral-prediction prediction-agents-copy
cd prediction-agents-copy

# Install dependencies
echo "Installing dependencies..."
pip install -q torch transformers peft accelerate datasets scikit-learn scipy matplotlib pandas numpy

# Verify files
echo ""
echo "=== Verification ==="
echo "Test data:"
wc -l data/socsci210_1pct_proper_split/test.jsonl

echo ""
echo "Model check:"
if [ -d "models/socrates-qwen-paper-replication" ]; then
    ls -lh models/socrates-qwen-paper-replication/ | head -5
    echo "✓ Model found locally"
else
    echo "⚠️  Model NOT found - need to upload or re-train"
    echo ""
    echo "Option 1: If model is on another GPU instance, copy it"
    echo "Option 2: Re-train using: python scripts/train_paper_replication.py --config config_paper.json"
fi

echo ""
echo "Ready to evaluate!"
```

---

## Step 3: Run Evaluation

### Option A: Quick Test First (Recommended)

Test on 100 examples to verify everything works (~20 minutes, $0.22):

```bash
python scripts/evaluate.py \
  --model models/socrates-qwen-paper-replication \
  --test-data data/socsci210_1pct_proper_split/test.jsonl \
  --output results_test_100 \
  --max-examples 100
```

**Watch for**:
- Wasserstein distance should compute successfully
- No errors in loading model or data
- Results saved to `results_test_100/metrics.json`

**View results**:
```bash
cat results_test_100/metrics.json | python -m json.tool
```

### Option B: Full Evaluation

Once verified, run full evaluation on all 3,289 examples (~4 hours, $2.64):

```bash
nohup python scripts/evaluate.py \
  --model models/socrates-qwen-paper-replication \
  --test-data data/socsci210_1pct_proper_split/test.jsonl \
  --output results_proper_split \
  > evaluation_proper.log 2>&1 &

# Get process ID
echo "Evaluation PID: $(ps aux | grep 'evaluate.py' | grep -v grep | awk '{print $2}')"
```

**Monitor progress**:
```bash
# Watch log file (Ctrl+C to exit)
tail -f evaluation_proper.log

# Check progress
grep "Generating predictions" evaluation_proper.log | tail -1

# Check if still running
ps aux | grep evaluate.py
```

---

## Step 4: View Results

After evaluation completes (~4 hours for full, ~20 min for test):

```bash
# View metrics
echo "=== METRICS ==="
cat results_proper_split/metrics.json | python -m json.tool

echo ""
echo "=== SAMPLE PREDICTIONS ==="
cat results_proper_split/predictions.json | python -m json.tool | head -100
```

**Copy the output** from metrics.json and save it locally.

---

## Step 5: Download Results to Local Machine

From the GPU terminal, display the results:

```bash
# Full metrics
cat results_proper_split/metrics.json

# First 20 predictions
head -20 results_proper_split/predictions.json
```

Copy the JSON output and save locally as:
- `evaluation_metrics_proper.json`
- `evaluation_predictions_proper_sample.json`

---

## Step 6: Disconnect and Clean Up

```bash
# Exit GPU instance
exit

# From your local terminal, destroy instance to stop billing
python3.11 -m thunder.thunder destroy 0
```

---

## Expected Results

Based on SOCRATES paper methodology and 1% training data:

```json
{
  "wasserstein_distance": 0.20-0.25,  // Paper: 0.151 (on 100% data)
  "correlation": 0.60-0.70,           // Previous (data leak): 0.815
  "mae": 5-7,                         // Previous: 3.72
  "num_predictions": 3289,
  "wasserstein_num_studies": 20
}
```

**Lower performance is CORRECT** - this represents true generalization to unseen experimental studies!

---

## Troubleshooting

### Model Not Found
If `models/socrates-qwen-paper-replication/` doesn't exist:

**Option 1: Upload from local**
- The model exists locally at: `/Users/tamaratran/Desktop/Desktop - MacBook Pro (8)/prediction-agents-copy/models/socrates-qwen-paper-replication/`
- Size: ~140MB (adapter + tokenizer)
- You'll need to manually copy it to the GPU

**Option 2: Re-train** (~2 hours, $1.32 on A100):
```bash
python scripts/train_paper_replication.py --config config_paper.json
```

### Evaluation Runs Out of Memory
If you get OOM errors:
- Use A10G GPU instead of T4 (more memory)
- Or reduce batch size in the evaluation loop

### Connection Lost
If your connection drops, the evaluation continues running in background:
```bash
# Reconnect
python3.11 -m thunder.thunder connect 0

# Check if still running
ps aux | grep evaluate.py

# View progress
tail -f evaluation_proper.log
```

---

## Quick Command Reference

```bash
# Create instance
python3.11 -m thunder.thunder create

# Connect
python3.11 -m thunder.thunder connect 0

# Check instance status
python3.11 -m thunder.thunder status

# Destroy instance
python3.11 -m thunder.thunder destroy 0
```

---

## Next Steps After Evaluation

Once you have the results:
1. Share the `metrics.json` output with me
2. I'll analyze the results and compare to the paper
3. We'll write the final `PROPER_EVALUATION_RESULTS.md` report
4. Commit and push final results to GitHub

**Ready to start? Begin with Step 1!**
