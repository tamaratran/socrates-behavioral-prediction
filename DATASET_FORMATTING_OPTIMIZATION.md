# Dataset Formatting Inefficiency - Issue & Solution

## Problem Identified

During the SOCRATES training launch on Thunder Compute instance 0, dataset formatting occurred **after GPU provisioning**, wasting 2-3 minutes of expensive GPU time at $3.80/hour.

### Timeline Observed

1. **00:00** - GPUs provisioned (4x A100 80GB at $3.80/hour)
2. **00:00-02:30** - Dataset formatting runs (CPU-bound, GPUs idle at 0%)
3. **02:30+** - Training starts (GPUs reach 100% utilization)

### Cost Impact

- **Wasted GPU time per run**: 2-3 minutes
- **Cost per wasted minute**: $0.063/min ($3.80/hour ÷ 60)
- **Total waste per training launch**: $0.13-0.19

While small per-run, this compounds:
- 10 training runs = $1.30-1.90 wasted
- 100 training runs = $13-19 wasted
- For hyperparameter sweeps or multi-run experiments, this inefficiency scales

## Root Cause

In `scripts/train_full_dataset.py`, dataset formatting happens inside the training script:

```python
# scripts/train_full_dataset.py (CURRENT - INEFFICIENT)

def main():
    # ... model loading, GPU allocation ...

    # Load dataset
    train_dataset = load_from_disk(train_path)
    val_dataset = load_from_disk(val_path)

    # THIS IS WHERE THE INEFFICIENCY OCCURS
    # Formatting happens AFTER GPUs are already provisioned and idle
    train_dataset = train_dataset.map(formatting_function)  # ~2-3 minutes, CPU-bound
    val_dataset = val_dataset.map(formatting_function)

    # Training starts
    trainer.train()
```

### Why It Takes 2-3 Minutes

- **Dataset size**: 2,468,761 training examples
- **Processing speed**: ~20,000 examples/second (CPU-bound)
- **Calculation**: 2,468,761 ÷ 20,000 = ~123 seconds (2 minutes)
- **Validation dataset**: Additional ~10-15 seconds
- **Total**: 2-3 minutes with GPUs sitting idle

## Proposed Solutions

### Solution 1: Pre-format During Dataset Preparation (RECOMMENDED)

Move formatting to `scripts/prepare_full_data.py` **before** GPU provisioning:

```python
# scripts/prepare_full_data.py (OPTIMIZED)

def prepare_data(output_dir, seed=42):
    # ... existing dataset loading and splitting ...

    # NEW: Format datasets during preparation (CPU time, no GPU cost)
    print("Formatting training dataset...")
    train_dataset = train_dataset.map(formatting_function, batched=True, num_proc=4)

    print("Formatting validation dataset...")
    val_dataset = val_dataset.map(formatting_function, batched=True, num_proc=4)

    # Save formatted datasets
    train_dataset.save_to_disk(os.path.join(output_dir, 'train'))
    val_dataset.save_to_disk(os.path.join(output_dir, 'validation'))
```

**Benefits**:
- Formatting happens on local machine or CPU-only instance ($0/hour)
- Datasets cached and ready for immediate training
- No changes required to training script
- One-time cost per dataset version

**Implementation**:
1. Add formatting function to `prepare_full_data.py`
2. Run preparation once before GPU provisioning
3. Upload pre-formatted datasets to GPU instance
4. Remove `.map()` calls from `train_full_dataset.py`

### Solution 2: Use HuggingFace Datasets Caching

Enable and leverage HuggingFace's built-in caching:

```python
# scripts/train_full_dataset.py (ALTERNATIVE)

from datasets import load_from_disk

# First run: formatting happens and cache is created
train_dataset = load_from_disk(train_path)
train_dataset = train_dataset.map(
    formatting_function,
    batched=True,
    num_proc=4,
    cache_file_name="train_formatted_cache.arrow"  # Cache to disk
)

# Subsequent runs: instant load from cache (if dataset unchanged)
```

**Benefits**:
- Automatic caching after first run
- No code changes needed for multi-run scenarios
- Useful for hyperparameter sweeps

**Limitations**:
- First run still wastes GPU time
- Requires ephemeral storage (Thunder Compute instances lose data on shutdown)
- Cache invalidated if dataset or formatting function changes

### Solution 3: Pre-tokenize and Save as Arrow Files

Save fully tokenized datasets as Arrow files:

```python
# prepare_pretokenized_data.py

def prepare_pretokenized(output_dir):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")

    # Load and tokenize
    train_dataset = load_from_disk("data/socsci210_full/train")
    train_dataset = train_dataset.map(
        lambda x: tokenizer(x['text'], truncation=True, padding='max_length'),
        batched=True,
        num_proc=4
    )

    # Save as Arrow (fast binary format)
    train_dataset.save_to_disk(os.path.join(output_dir, 'train_tokenized'))
```

**Benefits**:
- Fastest possible loading (binary format, no processing)
- Suitable for repeated training runs with same dataset
- Reduces memory usage during training

**Limitations**:
- Largest disk space requirement
- Less flexible (tokenizer changes require re-tokenization)

## Recommended Implementation Plan

### Phase 1: Immediate Fix (Solution 1)

1. Modify `scripts/prepare_full_data.py`:
   - Import formatting function
   - Apply `.map(formatting_function)` before saving
   - Add `--format` flag (default: True)

2. Update training workflow:
   ```bash
   # Run ONCE on local machine (no GPU cost)
   python scripts/prepare_full_data.py --output data/formatted --seed 42

   # Upload formatted dataset to GPU instance
   tnr upload 0 data/formatted ~/socrates-training/data/socsci210_full

   # Launch training (GPUs immediately utilized)
   tnr ssh 0 'cd ~/socrates-training && deepspeed --num_gpus=4 scripts/train_full_dataset.py ...'
   ```

3. Modify `scripts/train_full_dataset.py`:
   - Remove `.map(formatting_function)` calls
   - Add validation check to ensure datasets are pre-formatted

### Phase 2: Long-term Optimization (Solution 3)

For production deployments with frequent retraining:
- Implement pre-tokenization pipeline
- Store tokenized datasets in cloud storage (S3, GCS)
- Download tokenized data directly to GPU instances

## Testing Validation

Before/after comparison:

```bash
# BEFORE (current):
# - Dataset formatting: 2:30 (GPUs idle, $0.16 wasted)
# - Training start: 2:30
# - Total GPU time to first training step: 2:30

# AFTER (optimized):
# - Dataset formatting: 0:00 (already done, no GPU cost)
# - Training start: 0:00
# - Total GPU time to first training step: 0:00
# - Savings: 2:30 ($0.16 per run)
```

## Current Status

- **Issue Identified**: 2024-12-23 (user feedback)
- **Current Training Run**: Instance 0, ongoing with inefficiency (step 180/9,644)
- **Implementation Status**: Documented, not yet implemented
- **Next Training Run**: Should implement Solution 1 before next launch

## References

- HuggingFace Datasets documentation: https://huggingface.co/docs/datasets/
- DeepSpeed optimization guide: https://www.deepspeed.ai/tutorials/
- Thunder Compute pricing: $3.80/hour for 4x A100 80GB

---

**User Feedback** (2024-12-23):
> "Why didn't we do this before we provisioned the GPUs? Or why didn't we just do this on the GPUs? Why does it take so long?"

This document addresses that critical observation and provides actionable solutions for future training runs.
