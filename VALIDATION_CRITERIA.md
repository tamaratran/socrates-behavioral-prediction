# Validation Criteria - GO/NO-GO Decision at 500 Steps

## Overview

Before committing to the full 2-day training run ($360), we validate that our fine-tuning configuration is working by training to 500 steps (4.5% of full training) and comparing against an untrained baseline.

**Total Validation Cost:** ~$45
**Total Validation Time:** ~5-7 hours
**Full Training at Risk:** $360, 50 hours

---

## Validation Workflow

```
1. Baseline Evaluation (2-3 hours, ~$10)
   ↓
2. Train to 500 Steps (1.5 hours, ~$20)
   ↓
3. Checkpoint Evaluation (2-3 hours, ~$15)
   ↓
4. GO/NO-GO Decision (automated)
   ↓
5a. GO → Full Training (95 hours, ~$360)
5b. NO-GO → Debug & Retry
```

---

## Primary Success Criterion

**Wasserstein Distance Improvement ≥ 5%**

The SOCRATES paper uses Wasserstein distance as the primary evaluation metric (Section 3.2). This measures how well the model's predicted distributions align with actual human response distributions at the experimental condition level.

### Decision Rule

```bash
IMPROVEMENT = (1 - CHECKPOINT_WD / BASELINE_WD) × 100%

if IMPROVEMENT ≥ 5%:
    → GO: Proceed to full training
else:
    → NO-GO: Debug hyperparameters
```

---

## Expected Baseline Metrics

**Untrained Model:** Qwen/Qwen2.5-14B-Instruct (no fine-tuning)

Expected baseline performance (rough estimates based on zero learning):

| Metric | Expected Range | Notes |
|--------|---------------|-------|
| **Wasserstein Distance** | 0.30 - 0.45 | Primary metric - lower is better |
| **MAE** | 0.25 - 0.35 | Mean absolute error on [0,1] scale |
| **RMSE** | 0.30 - 0.40 | Root mean squared error |
| **Pearson Correlation** | 0.0 - 0.3 | Study-level correlation |

**Why these ranges?**
- Untrained model has no knowledge of behavioral prediction task
- May have some general world knowledge, but not calibrated to survey data
- Baseline establishes "zero learning" performance

---

## Expected 500-Step Metrics

**After 4.5% of Training:** Should show clear learning signal

Target improvements at 500 steps:

| Metric | Target | Minimum for GO |
|--------|--------|----------------|
| **Wasserstein Distance** | 5-10% better | ≥5% improvement |
| **MAE** | Decreasing | Any improvement |
| **Correlation** | Increasing | Any improvement |
| **Training Loss** | Decreasing | Consistent decrease |

### Example GO Scenario

```
Baseline:     Wasserstein = 0.400
Checkpoint:   Wasserstein = 0.370
Improvement:  7.5% → GO ✅

Supporting signals:
- MAE improved from 0.30 → 0.27
- Correlation improved from 0.15 → 0.35
- Training loss decreased from 2.5 → 2.1
```

### Example NO-GO Scenario

```
Baseline:     Wasserstein = 0.400
Checkpoint:   Wasserstein = 0.395
Improvement:  1.25% → NO-GO ❌

Warning signs:
- Minimal improvement in primary metric
- Training loss barely decreased (2.5 → 2.4)
- Possible learning rate too low or model not learning
```

---

## Green Lights (Proceed to Full Training)

**Primary:**
- ✅ Wasserstein distance improved ≥5%

**Supporting Signals (not required, but reassuring):**
- ✅ Training loss decreased consistently over 500 steps
- ✅ No signs of overfitting (validation loss not increasing)
- ✅ MAE and correlation improving
- ✅ Output format correct (valid JSON distributions)
- ✅ GPU utilization >80% during training
- ✅ Checkpoints saving correctly

---

## Red Flags (Debug Before Full Training)

**Critical (NO-GO):**
- ❌ Wasserstein distance improved <5%
- ❌ Wasserstein distance got worse
- ❌ Training loss not decreasing
- ❌ Model outputs malformed/unparseable

**Warning Signs (Investigate):**
- ⚠️ Training loss fluctuating wildly
- ⚠️ Validation loss increasing (overfitting)
- ⚠️ Correlation not improving
- ⚠️ Low GPU utilization (<50%)
- ⚠️ Training much slower than expected

---

## What to Check at 500 Steps

### 1. Automated Metrics (from evaluate_checkpoint.sh)

```bash
./scripts/evaluate_checkpoint.sh

# Outputs:
# Wasserstein Distance:
#   Baseline:    0.400
#   Checkpoint:  0.370
#   Improvement: 7.5%
#
# ✅ GO - Improvement ≥5%
# Proceed to full training!
```

### 2. Training Loss Curve

```bash
# Check training log for loss progression
grep "loss" training.log | tail -50

# Look for:
# - Consistent decrease (2.5 → 2.4 → 2.3 → 2.2 → 2.1)
# - No wild fluctuations
# - Smooth learning curve
```

### 3. Validation Metrics

```bash
cat models/socrates-qwen-500step-validation/checkpoint-500/evaluation/metrics.json | python -m json.tool

# Check:
# - wasserstein_distance < baseline
# - mae < baseline
# - correlation > baseline
# - All values reasonable (no NaN, Inf)
```

### 4. Sample Outputs

```bash
# Check a few predictions to ensure format is correct
cat models/socrates-qwen-500step-validation/checkpoint-500/evaluation/predictions.jsonl | head -5

# Each line should be valid JSON with:
# - "predicted_distribution": array summing to ~1.0
# - "ground_truth": array of actual responses
# - Reasonable predictions (not all 0s or 1s)
```

---

## If NO-GO: Debugging Steps

### Step 1: Analyze Training Loss

```bash
# Extract loss values
grep "loss" training.log | python -c "
import sys
losses = []
for line in sys.stdin:
    if 'loss' in line:
        # Extract loss value
        losses.append(float(line.split('loss')[1].split()[0]))
print(f'Initial: {losses[0]:.3f}')
print(f'Final:   {losses[-1]:.3f}')
print(f'Change:  {losses[0] - losses[-1]:.3f}')
"
```

**If loss barely decreased:**
- Increase learning rate (try 2e-5 or 5e-5)
- Check if model is frozen (ensure LoRA is enabled)
- Verify data is loading correctly

**If loss fluctuating:**
- Decrease learning rate (try 5e-6)
- Increase warmup ratio (try 0.1)
- Check for data quality issues

### Step 2: Check Output Quality

```bash
# Sample 10 predictions
python scripts/evaluate.py \
    --model models/socrates-qwen-500step-validation/checkpoint-500 \
    --test-data data/socsci210_full/val.jsonl \
    --output debug_samples \
    --max-examples 10

# Manually inspect outputs
cat debug_samples/predictions.jsonl | python -m json.tool | less
```

**Common Issues:**
- All predictions same → Model not learning diversity
- Invalid distributions (negatives, sum ≠ 1) → Generation issues
- Nonsense outputs → Model not understanding task

### Step 3: Hyperparameter Adjustments

**If learning too slow:**
```json
{
  "learning_rate": 2e-05,  // Increase from 1e-5
  "warmup_ratio": 0.03     // Decrease warmup
}
```

**If learning unstable:**
```json
{
  "learning_rate": 5e-06,  // Decrease from 1e-5
  "warmup_ratio": 0.1,     // Increase warmup
  "gradient_accumulation_steps": 16  // Increase for stability
}
```

**If potential overfitting:**
```json
{
  "lora_dropout": 0.1,     // Increase from 0.05
  "weight_decay": 0.15     // Increase from 0.1
}
```

### Step 4: Retry Validation

```bash
# With adjusted hyperparameters
# 1. Update config_500step_validation.json
# 2. Re-run training
# 3. Re-evaluate
# 4. Compare to baseline again
```

---

## If GO: Full Training Launch

**Prerequisites:**
- ✅ Wasserstein distance improved ≥5%
- ✅ Training loss decreased consistently
- ✅ No critical red flags
- ✅ Checkpoints downloading correctly
- ✅ Have ~$400 available in Thunder Compute account

**Launch Command:**
```bash
# Use exact same hyperparameters that succeeded at 500 steps
# Only change: max_steps from 500 → -1 (full training)

tnr create --name socrates-full-training --gpu A100-80GB:4 --disk 500 --region us-east
# Upload files
# Launch training with config_full_dataset.json
# Start checkpoint monitor
```

**Monitoring During Full Training:**
- Check progress every 4-6 hours
- Ensure checkpoints being saved (every 500 steps)
- Verify loss continues decreasing
- Download checkpoints as they complete
- Expected duration: ~95 hours (4 days)

---

## Comparison to SOCRATES Paper

**What the paper reports:**
- Uses Wasserstein distance as primary metric (Section 3.2)
- Does not describe intermediate validation
- Final results: Correlation 0.81-0.85 (Table 2)

**Our validation strategy:**
- Uses same Wasserstein distance metric
- Adds early validation checkpoint (not in paper)
- Conservative 5% improvement threshold
- Prevents wasting resources on failed configurations

**Why we're being more cautious:**
- Paper had institutional resources for multiple runs
- We have limited budget (~$400)
- Better to validate early than waste 4 days
- Industry best practice for expensive training runs

---

## Cost-Benefit Analysis

| Scenario | Cost | Outcome |
|----------|------|---------|
| **Skip validation, train succeeds** | $360 | ✅ Success (risky) |
| **Skip validation, train fails** | $360 | ❌ Total loss |
| **Validate GO → train succeeds** | $405 | ✅ Success (safe) |
| **Validate NO-GO → fix → train succeeds** | $450 | ✅ Success (debugged) |
| **Validate NO-GO → avoid bad train** | $45 | ✅ Saved $360 |

**Expected value of validation:** Strongly positive
- Adds $45 upfront cost
- Prevents potential $360 loss
- Increases confidence in hyperparameters
- Provides learning signal before long wait

---

## Summary

**GO Criteria:**
- Wasserstein distance ≥5% better than baseline
- Training loss decreasing consistently
- Outputs well-formed
- No critical errors

**NO-GO Actions:**
- Analyze training loss curve
- Check output quality
- Adjust hyperparameters
- Retry validation run
- Do NOT proceed to full training

**Philosophy:**
> "Measure twice, cut once" - Validate hyperparameters at 4.5% of training before committing to the full 95-hour run.
