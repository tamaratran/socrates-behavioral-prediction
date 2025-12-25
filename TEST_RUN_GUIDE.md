# Full Training Guide - Continuous Run with Auto-Checkpoint Downloads

## Overview

We've already validated our hyperparameters with 1% baseline training (29K examples, Wasserstein 0.0811, 81.5% correlation). These excellent results prove the configuration works.

**Workflow:**
1. **Launch full training** - Runs continuously to completion (~50 hours)
2. **Auto-download checkpoints** - Monitor script downloads checkpoints as they save
3. **Manual review** - You review checkpoints on your own schedule
4. **No interruptions** - Training runs uninterrupted

**Rationale:** Hyperparameters already validated. Training runs continuously with automatic checkpoint downloads for safety. You review checkpoints manually whenever you want.

**Training Details:**
- **Duration:** ~50 hours (~2 days)
- **Cost:** ~$360
- **Hardware:** 4x A100 80GB GPUs on Thunder Compute
- **Checkpoints:** Every 500 steps (~2 hours), ~23 total
- **Auto-download:** Checkpoints download locally as they save

---

## Quick Start: Full Training Workflow

### Step 1: Create Thunder Instance & Upload Files

```bash
export TNR_API_TOKEN="your_api_key"
export HF_TOKEN="your_hf_token"

# Create instance
tnr create \
  --name "socrates-full-training" \
  --gpu A100-80GB:4 \
  --disk 500 \
  --region us-east

# Upload files (see detailed setup below if needed)
```

### Step 2: Launch Full Training

```bash
./scripts/launch_full_training.sh 0
```

This starts training immediately. Training runs continuously to completion.

### Step 3: Start Checkpoint Auto-Download (Separate Terminal)

```bash
# In a separate terminal on your local machine
./scripts/monitor_checkpoints.sh 0 300  # Check every 5 minutes
```

This will automatically download checkpoints as they're created:
- Checks remote every 5 minutes for new checkpoints
- Downloads any new checkpoints found
- Tracks what's already downloaded
- Runs until you stop it (Ctrl+C)

**Checkpoints download to:** `models/socrates-qwen-full-dataset/checkpoint-NNN/`

### Step 4: Monitor Training Progress

```bash
# Watch training log
tnr ssh 0 'tail -f ~/socrates-training/training.log'

# Check GPU utilization
tnr ssh 0 'nvidia-smi'

# Check which checkpoints exist on remote
tnr ssh 0 'ls ~/socrates-training/models/socrates-qwen-full-dataset/'
```

### Step 5: Review Checkpoints Manually (Your Schedule)

As checkpoints download locally, you can review them whenever you want:

```bash
# List downloaded checkpoints
ls models/socrates-qwen-full-dataset/

# Evaluate a specific checkpoint manually
python scripts/evaluate.py \
  --model models/socrates-qwen-full-dataset/checkpoint-500 \
  --test-data data/socsci210_full/val.jsonl \
  --output evaluation_results/checkpoint-500 \
  --max-examples 5000
```

**Training continues uninterrupted** while you review checkpoints on your own schedule.

---

## Alternative: Full Validation Workflow (If Not Already Validated)

**Use this section ONLY if you haven't already validated hyperparameters with smaller training run**

Before committing to the full training run, we perform **two validation phases**:

1. **Phase 1: Hyperparameter Validation** (500 steps) - Validates model is learning
2. **Phase 2: Infrastructure Test** (50 steps) - Validates checkpoint system works
3. **Phase 3: Full Training** (~11,300 steps) - Only if both phases pass

**Total validation cost:** ~$65 (Phase 1: $45, Phase 2: $20)
**Total validation time:** ~6-8 hours
**Full training at risk:** $360, 50 hours

---

## Phase 1: Hyperparameter Validation (MUST DO FIRST)

**Purpose:** Validate that our fine-tuning configuration actually works before committing to expensive training.

**Question Answered:** "Is the model learning from our data with these hyperparameters?"

**Duration:** ~5-7 hours total
- Baseline evaluation: 2-3 hours (~$10)
- 500-step training: 1.5 hours (~$20)
- Checkpoint evaluation: 2-3 hours (~$15)

**Hardware:** 4x A100 80GB (same as full training)

**GO/NO-GO Decision:** Automated based on Wasserstein distance improvement

See `VALIDATION_CRITERIA.md` for detailed decision criteria.

### Phase 1 Workflow

```
Step 1: Baseline Evaluation
   ↓
Step 2: Train to 500 Steps (4.5% of full training)
   ↓
Step 3: Evaluate Checkpoint vs Baseline
   ↓
Step 4: Automated GO/NO-GO Decision
   ↓
GO → Proceed to Phase 2 (Infrastructure Test)
NO-GO → Debug hyperparameters, retry Phase 1
```

### Phase 1: Step-by-Step Instructions

#### 1.1: Baseline Evaluation

**Purpose:** Establish "zero learning" performance of untrained model.

```bash
# Run on local machine (or single GPU on Thunder)
./scripts/evaluate_baseline.sh

# Expected output:
# Baseline Model Evaluation
# Model: Qwen/Qwen2.5-14B-Instruct
# Validation data: data/socsci210_full/val.jsonl
# ...
# Baseline Evaluation Complete
# Key baseline metrics:
#   "wasserstein_distance": 0.35-0.45 (expected range)
#   "mae": 0.25-0.35
#   "correlation": 0.0-0.3
```

**Save the baseline metrics!** You'll compare against these.

```bash
# Verify baseline saved
cat evaluation_results/baseline/metrics.json | python -m json.tool
```

#### 1.2: Train to 500 Steps

**Create Thunder instance:**

```bash
export TNR_API_TOKEN="YOUR_API_KEY"

tnr create \
  --name "socrates-500step-validation" \
  --gpu A100-80GB:4 \
  --disk 500 \
  --region us-east
```

**Upload files:**

```bash
# Upload training code and data
tnr scp 0 scripts/ ~/socrates-training/scripts/
tnr scp 0 config_500step_validation.json ~/socrates-training/
tnr scp 0 deepspeed_config.json ~/socrates-training/
tnr scp 0 data/socsci210_full/ ~/socrates-training/data/socsci210_full/
tnr scp 0 requirements_full_training.txt ~/socrates-training/
```

**Install dependencies:**

```bash
tnr ssh 0

cd ~/socrates-training
python3 -m venv venv
source venv/bin/activate
pip install -r requirements_full_training.txt

exit
```

**Launch training:**

```bash
export HF_TOKEN="YOUR_HF_TOKEN"

tnr ssh 0 'cd ~/socrates-training && \
  source venv/bin/activate && \
  export HF_TOKEN="'$HF_TOKEN'" && \
  export CUDA_VISIBLE_DEVICES=0,1,2,3 && \
  nohup deepspeed --num_gpus=4 scripts/train_full_dataset.py \
    --config config_500step_validation.json \
    --deepspeed deepspeed_config.json \
    > training_500step.log 2>&1 &'
```

**Monitor training:**

```bash
# Watch progress
tnr ssh 0 'tail -f ~/socrates-training/training_500step.log'

# Check GPU utilization
tnr ssh 0 'nvidia-smi'

# Expected: ~1.5 hours, loss should decrease from ~2.5 → ~2.0
```

**Wait for completion** (~1.5 hours). Checkpoint will save at step 500.

#### 1.3: Download and Evaluate Checkpoint

**Download checkpoint:**

```bash
# Download checkpoint-500
tnr scp -r 0:~/socrates-training/models/socrates-qwen-500step-validation/checkpoint-500 \
  models/socrates-qwen-500step-validation/
```

**Run evaluation and GO/NO-GO decision:**

```bash
./scripts/evaluate_checkpoint.sh \
  models/socrates-qwen-500step-validation/checkpoint-500 \
  evaluation_results/baseline/metrics.json

# Automated output:
# ========================================
# Checkpoint Evaluation & GO/NO-GO Decision
# ========================================
# ...
# Wasserstein Distance:
#   Baseline:    0.400
#   Checkpoint:  0.370
#   Improvement: 7.5%
#
# ✅ GO - Improvement ≥5%
# Proceed to full training!
```

#### 1.4: Interpret Results

**If GO (✅ Improvement ≥5%):**
- Hyperparameters are working!
- Model is learning from data
- **Proceed to Phase 2: Infrastructure Test**
- Clean up validation instance: `tnr delete 0`

**If NO-GO (❌ Improvement <5%):**
- Hyperparameters need adjustment
- See `VALIDATION_CRITERIA.md` for debugging steps
- **DO NOT proceed to Phase 2 or full training**
- Debug, adjust config, retry Phase 1
- Clean up instance: `tnr delete 0`

---

## Phase 2: Infrastructure Test (50 Steps)

**IMPORTANT:** Only run Phase 2 if Phase 1 returned GO (✅)

**Purpose:** Validate infrastructure before full training.

This 50-step test verifies:
- Training script runs without errors
- Checkpoints save correctly on remote
- Checkpoint download system works
- Can resume from checkpoints
- Loss is decreasing properly
- GPU utilization is high
- No memory errors

**Expected Duration:** 10-15 minutes
**Expected Checkpoints:** 5 (at steps 10, 20, 30, 40, 50)
**Hardware:** 2x A100 40GB GPUs (cheaper than full 4xGPU setup)

---

## Phase 2 Prerequisites

1. **Phase 1 Complete:** Must have GO (✅) from hyperparameter validation
2. **Baseline Metrics:** Saved in `evaluation_results/baseline/metrics.json`
3. **Thunder Compute Account:** Account with API key
4. **Local Setup:** All files synced to local machine
5. **Full Dataset Prepared:** `data/socsci210_full/` ready (2.9M examples)

---

## Phase 2 Test Run Workflow

### Step 1: Create Thunder Instance

```bash
# Set API key (get from new Thunder account)
export TNR_API_TOKEN="YOUR_NEW_API_KEY"

# Create instance with 2x A100 40GB
tnr create \
  --name "socrates-test-run" \
  --gpu A100:2 \
  --disk 200 \
  --region us-east
```

### Step 2: Upload Files to Remote

Use the existing `run_remote_setup.sh` script, or manually:

```bash
# Upload code and data
tnr scp 0 scripts/ ~/socrates-training/scripts/
tnr scp 0 config_test_run.json ~/socrates-training/
tnr scp 0 deepspeed_config_test.json ~/socrates-training/
tnr scp 0 launch_test_run.sh ~/socrates-training/
tnr scp 0 data/socsci210_full/ ~/socrates-training/data/socsci210_full/
```

### Step 3: Install Dependencies (if needed)

```bash
tnr ssh 0

# Create venv and install dependencies
cd ~/socrates-training
python3 -m venv venv
source venv/bin/activate
pip install -r requirements_full_training.txt

exit
```

### Step 4: Launch Test Run

```bash
# Set HuggingFace token
export HF_TOKEN="YOUR_HF_TOKEN"

# Launch via SSH
tnr ssh 0 'cd ~/socrates-training && export HF_TOKEN="YOUR_HF_TOKEN" && bash launch_test_run.sh'
```

### Step 5: Start Checkpoint Monitor (Local)

In a separate terminal on your local machine:

```bash
# This will automatically download checkpoints as they're created
./scripts/monitor_checkpoints.sh 0 60  # Check every 60 seconds
```

Expected output:
```
================================
Checkpoint Monitor
================================
Instance ID: 0
Check interval: 60s
...
[2025-12-23 10:15:00] Found new checkpoint: checkpoint-10
[2025-12-23 10:15:00] Downloading checkpoint-10...
[2025-12-23 10:15:30] ✓ checkpoint-10 downloaded successfully
```

### Step 6: Monitor Training Progress

```bash
# Watch training log
tnr ssh 0 'tail -f ~/socrates-training/training_test.log'

# Check GPU utilization
tnr ssh 0 'nvidia-smi'

# Expected output every 10 steps:
# {'loss': 2.xxx, 'learning_rate': 1e-5, ...}
# Saving model checkpoint to .../checkpoint-10
```

### Step 7: Verify Checkpoints

After ~15 minutes, check local checkpoints:

```bash
ls -lh models/socrates-qwen-test-run/

# Expected:
# checkpoint-10/
# checkpoint-20/
# checkpoint-30/
# checkpoint-40/
# checkpoint-50/
```

Each checkpoint should contain:
- `adapter_config.json`
- `adapter_model.safetensors` (~130MB)
- `optimizer.pt` (~260MB)
- `trainer_state.json`
- `training_args.bin`

### Step 8: Test Manual Download (Optional)

```bash
# Download a specific checkpoint manually
./scripts/download_checkpoint.sh 30

# Verify it downloaded
ls -lh models/socrates-qwen-test-run/checkpoint-30/
```

### Step 9: Verify Loss Decreased

```bash
# Check training log for loss values
tnr ssh 0 'grep "loss" ~/socrates-training/training_test.log | tail -10'

# Expected: Loss should decrease from ~2.5 → ~2.0 (rough estimate)
# Example:
# Step 10: loss=2.456
# Step 20: loss=2.301
# Step 30: loss=2.198
# Step 40: loss=2.134
# Step 50: loss=2.089
```

---

## Success Criteria

- [ ] Training completed 50 steps without errors
- [ ] All 5 checkpoints saved on remote
- [ ] All 5 checkpoints downloaded locally
- [ ] Each checkpoint ~390MB (adapter + optimizer)
- [ ] Loss decreased over training
- [ ] GPU utilization >80% during training
- [ ] No CUDA out-of-memory errors
- [ ] Can list checkpoint contents
- [ ] Checkpoint monitor ran without errors

---

## Troubleshooting

### Training Fails to Start

```bash
# Check error log
tnr ssh 0 'tail -50 ~/socrates-training/training_test.log'

# Common issues:
# - Missing HF_TOKEN: Set export HF_TOKEN="..."
# - Missing data: Check data/socsci210_full/ exists
# - Missing dependencies: Install requirements_full_training.txt
```

### Checkpoints Not Downloading

```bash
# Check if checkpoints exist on remote
tnr ssh 0 'ls ~/socrates-training/models/socrates-qwen-test-run/'

# Manually download if needed
./scripts/download_checkpoint.sh 10

# Check monitor script is running
ps aux | grep monitor_checkpoints
```

### Low GPU Utilization

```bash
# Check GPU status
tnr ssh 0 'nvidia-smi'

# Should see:
# - GPU utilization: 80-100%
# - Memory used: ~60-70GB per GPU
```

---

## After Phase 2 Completes

### If Phase 2 Test PASSED ✅

**Success Criteria:**
- [ ] Training completed 50 steps without errors
- [ ] All 5 checkpoints saved on remote
- [ ] All 5 checkpoints downloaded locally
- [ ] Each checkpoint ~390MB (adapter + optimizer)
- [ ] Loss decreased over training
- [ ] GPU utilization >80% during training
- [ ] No CUDA out-of-memory errors
- [ ] Checkpoint monitor ran without errors

**Next Steps:**
1. Stop checkpoint monitor (Ctrl+C)
2. Verify all checkpoints downloaded
3. Clean up test instance: `tnr delete 0`
4. **Ready to launch Phase 3: Full Training**
   - Create new instance with 4x A100 80GB
   - Use `config_full_dataset.json`
   - Use `deepspeed_config.json`
   - Expected duration: ~50 hours (~2 days)
   - Expected checkpoints: ~23 (every 500 steps)
   - Expected cost: ~$360

### If Phase 2 Test FAILED ❌

**Common Issues:**
- Training script errors
- Checkpoint saving issues
- Memory errors
- Download system failures

**Actions:**
1. Review error logs
2. Fix identified issues (see Troubleshooting section)
3. Delete instance: `tnr delete 0`
4. Run Phase 2 test again with fixes
5. **DO NOT** proceed to full training until Phase 2 passes

---

## Files Created for Validation & Testing

### Phase 1: Hyperparameter Validation
```
scripts/evaluate_baseline.sh         - Baseline evaluation script
config_500step_validation.json       - 500-step validation config
scripts/evaluate_checkpoint.sh       - Checkpoint evaluation with GO/NO-GO
VALIDATION_CRITERIA.md               - Decision criteria documentation
```

### Phase 2: Infrastructure Test
```
config_test_run.json                 - Test configuration (50 steps)
deepspeed_config_test.json           - DeepSpeed config (2 GPUs)
launch_test_run.sh                   - Launch script for test
scripts/download_checkpoint.sh       - Download single checkpoint
scripts/monitor_checkpoints.sh       - Auto-download monitor
```

### Supporting Documentation
```
TEST_RUN_GUIDE.md                    - This file (complete workflow)
TRAINING_AUDIT.md                    - Training methodology verification
```

---

## Estimated Costs

**Phase 1: Hyperparameter Validation**
- Baseline evaluation (local or single GPU): ~$10 (2-3 hours)
- 500-step training (4x A100 80GB): ~$20 (1.5 hours)
- Checkpoint evaluation (local or single GPU): ~$15 (2-3 hours)
- **Phase 1 Total: ~$45**

**Phase 2: Infrastructure Test**
- Test run (2x A100 40GB): ~$20 (15 min × $80/hour)
- **Phase 2 Total: ~$20**

**Phase 3: Full Training**
- Full run (4x A100 80GB): ~$360 (95 hr × $3.80/hour)
- **Phase 3 Total: ~$360**

**Grand Total: ~$425**
- Without validation: $360 (risky)
- With validation: $425 (confident)
- **Validation premium: $65 for peace of mind**

---

## Complete Workflow Summary

### Recommended Order

```
1. Phase 1: Hyperparameter Validation (~5-7 hours, $45)
   ├─ 1.1: Baseline evaluation
   ├─ 1.2: Train to 500 steps
   ├─ 1.3: Evaluate checkpoint
   └─ 1.4: GO/NO-GO decision

   If GO ✅ → Proceed to Phase 2
   If NO-GO ❌ → Debug, adjust config, retry Phase 1

2. Phase 2: Infrastructure Test (~15 minutes, $20)
   ├─ 2.1: Create test instance (2x A100 40GB)
   ├─ 2.2: Upload files
   ├─ 2.3: Launch 50-step test
   ├─ 2.4: Monitor checkpoints auto-download
   └─ 2.5: Verify all systems working

   If PASS ✅ → Proceed to Phase 3
   If FAIL ❌ → Fix issues, retry Phase 2

3. Phase 3: Full Training (~50 hours, $360)
   ├─ 3.1: Create production instance (4x A100 80GB)
   ├─ 3.2: Upload all files
   ├─ 3.3: Launch full training (~11,300 steps)
   ├─ 3.4: Monitor progress (checkpoint every 500 steps, ~23 total)
   └─ 3.5: Download checkpoints as they complete
```

### Why This Workflow?

**Phase 1 validates WHAT:** Are hyperparameters working? Is model learning?
- Answers: "Should we train at all with these settings?"
- Cost: $45 to potentially save $360
- Time: 5-7 hours to validate before 2-day commitment

**Phase 2 validates HOW:** Does infrastructure work? Can we checkpoint safely?
- Answers: "Will the training run complete without crashes?"
- Cost: $20 to ensure no infrastructure failures
- Time: 15 minutes to catch any system issues

**Phase 3 is the REAL RUN:** Full training with validated config and infrastructure
- Confidence: High (passed both validation phases)
- Risk: Low (already validated hyperparameters and infrastructure)
- Cost: $360 well-spent

---

## Next Steps

See `TRAINING_AUDIT.md` for verification that our approach matches the SOCRATES paper.

See `VALIDATION_CRITERIA.md` for detailed GO/NO-GO decision criteria.

**Start with Phase 1:**
```bash
# Step 1: Baseline evaluation
./scripts/evaluate_baseline.sh

# Wait 2-3 hours, then proceed to 500-step training
# See Phase 1 section above for complete instructions
```
