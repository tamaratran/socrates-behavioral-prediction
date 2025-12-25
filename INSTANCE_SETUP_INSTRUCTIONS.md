# Thunder Instance Setup Instructions

## Issue Summary

The Thunder CLI (`tnr`) requires interactive prompts that cannot be automated from non-interactive scripts. You'll need to create the instance manually via the Thunder web UI, then run the automated setup script.

## Step 1: Create Instance Manually

Go to https://www.thundercompute.com/ and create an instance with these specs:

- **GPU**: A100 80GB
- **Number of GPUs**: 4
- **Disk Size**: 500GB
- **Mode**: Production
- **Template**: Ubuntu + Python (or any Ubuntu-based template with Python 3.10+)

**Cost**: ~$7.16/hour for 4x A100 80GB GPUs

## Step 2: Get Instance ID

After creation, note the instance ID (usually 0 for first instance).

Check with:
```bash
export TNR_API_TOKEN="af22ef23aca94b95661ed2a3f5d8dbe94e78bc39d8c2a38c4c34ac25a77f3759"
/usr/local/bin/python3.11 -m thunder.thunder status
```

## Step 3: Run Automated Setup

From your local machine, run the automated setup script:

```bash
./scripts/setup_and_launch_training.sh [INSTANCE_ID]
```

This script will:
1. Upload all necessary files (scripts, configs, data)
2. Set up Python environment
3. Install dependencies
4. Download HuggingFace datasets
5. Launch training with DeepSpeed
6. Start GPU monitoring (auto-shutdown if GPUs don't engage within 5 minutes)

## Step 4: Monitor Training

Training status will be monitored automatically. The script will:
- Check GPU utilization every 30 seconds for 5 minutes
- Auto-delete instance if GPUs never reach >80% (saves $7.16/hour)
- Continue training if GPUs engage successfully

You can manually check progress anytime with:
```bash
export TNR_API_TOKEN="af22ef23aca94b95661ed2a3f5d8dbe94e78bc39d8c2a38c4c34ac25a77f3759"
/usr/local/bin/python3.11 -m thunder.thunder connect [INSTANCE_ID] "tail -50 ~/socrates-training/training.log"
```

## What Was Fixed

All the issues from the previous hung training have been resolved:

### 1. Data Loading Deadlock (scripts/train_full_dataset.py:83-94)
- **Problem**: All 6 DeepSpeed processes tried to load 2.4M-line JSONL files simultaneously
- **Fix**: Added LOCAL_RANK check so only rank 0 prints status (data still loads correctly on all processes)

### 2. Dataloader I/O Contention (scripts/train_full_dataset.py:191)
- **Problem**: 4 GPUs × 4 workers = 16 concurrent file readers causing contention
- **Fix**: Reduced to 2 workers per GPU (8 total threads)

### 3. No Cost Protection (scripts/monitor_gpu_startup.sh - NEW FILE)
- **Problem**: Instance could run idle at $7.16/hour if training hung
- **Fix**: Created 5-minute GPU validation monitor that auto-deletes instance if training doesn't start

### 4. GPU Count Mismatch (scripts/launch_full_training.sh:23)
- **Status**: Already correct (NUM_GPUS=4)

## Expected Timeline

- **Setup**: ~10-15 minutes (data download)
- **First checkpoint**: ~2 hours (step 500)
- **Full training**: ~95 hours (4 days)
- **Total cost**: ~$360 for full training

## Safety Features

1. **GPU Startup Monitor**: Auto-shutdown if GPUs don't engage within 5 minutes
   - Worst case waste: ~$0.60 (5 minutes at $7.16/hour)
   - Prevents: Potential $360 loss from hung training

2. **Checkpoint Downloads**: Checkpoints saved every 500 steps
   - Stored locally to prevent data loss if instance crashes
   - Can resume training from any checkpoint

3. **Validation Criteria**: Per VALIDATION_CRITERIA.md
   - Wasserstein distance must improve ≥5% at step 500
   - If not, debug hyperparameters before full training

## Manual Alternative

If you prefer manual setup, see `scripts/setup_and_launch_training.sh` for the exact commands to run.

## Troubleshooting

**If training hangs again:**
```bash
# Check process status
tnr ssh [INSTANCE_ID] "ps aux | grep train_full_dataset"

# Check GPU utilization
tnr ssh [INSTANCE_ID] "nvidia-smi"

# Check training log
tnr ssh [INSTANCE_ID] "tail -100 ~/socrates-training/training.log"
```

**If GPU monitor deletes instance but training was actually starting:**
- This means it took >5 minutes to reach 80% GPU utilization
- Increase `MAX_CHECKS` in `scripts/monitor_gpu_startup.sh` (currently 10 checks × 30s = 5 minutes)
- Re-create instance and try again

**Cost went to zero but training still running:**
- Instance was deleted (ephemeral storage means all data lost)
- Check `tnr status` to confirm
- If deleted, re-create and restart from latest checkpoint (if any were downloaded)
