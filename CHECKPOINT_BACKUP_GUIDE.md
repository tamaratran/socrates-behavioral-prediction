# SOCRATES Checkpoint Backup Guide

Complete guide for backing up and restoring SOCRATES training checkpoints between Thunder Compute GPU servers and your local laptop.

## Overview

This system implements a two-tier checkpoint strategy:

1. **GPU Server**: Saves full DeepSpeed checkpoints every 500 steps, keeps only the 2-3 most recent
2. **Local Laptop**: Backs up checkpoints for long-term storage and disaster recovery

## Checkpoint Configuration

### Current Settings

**GPU Server Configuration** (`config_full_dataset.json` and `scripts/train_full_dataset.py`):
- **Save frequency**: Every 500 steps (~25 minutes at 2.4 it/s)
- **Retention**: Keep only 3 most recent checkpoints
- **Auto-deletion**: Older checkpoints automatically deleted after successful save
- **LoRA artifacts**: Included in every checkpoint (adapter_model.safetensors, adapter_config.json)

### Checkpoint Structure

Each full DeepSpeed checkpoint directory (`checkpoint-XXX/`) contains:

```
checkpoint-500/
├── adapter_model.safetensors    # LoRA adapter weights (~131 MB)
├── adapter_config.json          # LoRA configuration
├── optimizer.pt                 # Optimizer state (~263 MB)
├── scheduler.pt                 # LR scheduler state (~1.4 KB)
├── trainer_state.json           # Training metadata (step, epoch, etc.)
├── training_args.bin            # HuggingFace training arguments
├── pytorch_model.bin            # (optional) Model state dict
└── rng_state_*.pth              # (optional) Random number generator states
```

**Total size per checkpoint**: ~410 MB
**Total storage with 3 checkpoints**: ~1.2 GB

### Save Performance Impact

During checkpoint saving (~5-10 seconds):
- Training pauses briefly while state is serialized
- GPUs remain allocated but idle during save
- Minimal impact: ~5-10 seconds every 500 steps = ~0.03% overhead

## Backup Workflow

### 1. Backing Up from GPU to Laptop

Use the `backup_checkpoints_from_laptop.sh` script to download checkpoints:

```bash
# Default: Backup 2 most recent checkpoints from instance 0
./scripts/backup_checkpoints_from_laptop.sh

# Backup from specific instance
./scripts/backup_checkpoints_from_laptop.sh --instance 1

# Backup more checkpoints (e.g., 3)
./scripts/backup_checkpoints_from_laptop.sh --max-checkpoints 3

# Preview what would be downloaded (dry run)
./scripts/backup_checkpoints_from_laptop.sh --dry-run

# Custom local directory
./scripts/backup_checkpoints_from_laptop.sh --local-dir ~/my-backups/socrates
```

**Script behavior:**
- Lists all available checkpoints on GPU server
- Selects N most recent checkpoints (default: 2)
- Downloads via Thunder CLI using tarball compression
- Preserves directory structure
- Verifies checkpoint integrity (checks for adapter + optimizer files)
- Also downloads config files (config_full_dataset.json, deepspeed_config.json)

**When to backup:**
- After every checkpoint milestone (e.g., checkpoint-1000, checkpoint-2000)
- Before shutting down GPU instance
- After completing training epochs
- When approaching Thunder Compute session limits

**Backup frequency recommendation:**
- Manual: After each major milestone (every 1-2 hours of training)
- Automated: Run cron job every 30 minutes during active training

### 2. Verifying Backups

After backup, verify checkpoint integrity:

```bash
# Check backup directory
ls -lh ./backups/socrates-checkpoints/

# Verify checkpoint structure
find ./backups/socrates-checkpoints/checkpoint-500 -type f | head -20

# Confirm sizes
du -sh ./backups/socrates-checkpoints/*
```

Expected output:
```
410M    checkpoint-500
410M    checkpoint-1000
```

## Restore Workflow

### 1. Restoring Checkpoints to New GPU Instance

Use the `restore_checkpoints_to_server.sh` script to upload checkpoints:

```bash
# Restore latest checkpoint to instance 0
./scripts/restore_checkpoints_to_server.sh

# Restore to specific instance
./scripts/restore_checkpoints_to_server.sh --instance 1

# Restore specific checkpoint
./scripts/restore_checkpoints_to_server.sh --checkpoint checkpoint-1500

# Preview what would be uploaded (dry run)
./scripts/restore_checkpoints_to_server.sh --dry-run

# Custom local backup directory
./scripts/restore_checkpoints_to_server.sh --local-dir ~/my-backups/socrates
```

**Script behavior:**
- Auto-selects latest checkpoint if not specified
- Verifies checkpoint structure before upload
- Warns if optimizer state is missing (adapter-only checkpoint)
- Creates tarball for efficient transfer
- Uploads via Thunder CLI
- Extracts on remote server
- Verifies remote checkpoint integrity
- Uploads config files (config_full_dataset.json, deepspeed_config.json)

### 2. Resuming Training from Checkpoint

After restoring checkpoint to GPU server, resume training:

```bash
# Connect to Thunder instance
tnr ssh 0

# Navigate to training directory
cd ~/socrates-training
source venv/bin/activate

# Set environment variables
export HF_TOKEN="your_huggingface_token_here"
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Resume training with DeepSpeed
deepspeed --num_gpus=4 scripts/train_full_dataset.py \
  --config config_full_dataset.json \
  --deepspeed deepspeed_config.json \
  --resume-from-checkpoint models/socrates-qwen-full-dataset/checkpoint-500
```

**What happens when resuming:**
- Training continues from **exact step** saved in checkpoint
- Optimizer state restored (momentum, adaptive learning rates)
- LR scheduler state restored (warmup, cosine decay position)
- Global step counter continues from checkpoint
- Loss curve is continuous (no jump or reset)
- RNG states restored for reproducible data shuffling

### 3. Important: Resume vs. Fine-tune from Weights

**Resume** (full DeepSpeed checkpoint with optimizer state):
```bash
--resume-from-checkpoint models/socrates-qwen-full-dataset/checkpoint-500
```
- Continues training from exact state (same step, same LR, same optimizer momentum)
- Used for: Recovering from crashes, continuing interrupted training

**Fine-tune** (adapter weights only, no optimizer state):
```bash
# Load adapter weights into new training run
# Optimizer/scheduler start fresh
```
- Starts new training with loaded weights (fresh optimizer, reset step counter)
- Used for: Transfer learning, different hyperparameters

## Checkpoint Management Best Practices

### On GPU Server

1. **Monitor disk usage:**
   ```bash
   # Check checkpoint directory size
   du -sh ~/socrates-training/models/socrates-qwen-full-dataset/

   # List checkpoints
   ls -lh ~/socrates-training/models/socrates-qwen-full-dataset/
   ```

2. **Verify checkpoints are being saved:**
   ```bash
   # Check training log for save events
   grep "Saving" ~/socrates-training/training.log | tail -10

   # Verify 3 checkpoints exist
   ls ~/socrates-training/models/socrates-qwen-full-dataset/ | grep checkpoint | wc -l
   ```

3. **Auto-deletion is working:**
   - HuggingFace Trainer automatically deletes checkpoints beyond `save_total_limit=3`
   - Oldest checkpoint deleted AFTER newest is successfully written
   - Ensures you always have at least 2 checkpoints available

### On Local Laptop

1. **Organize backups by run:**
   ```bash
   mkdir -p backups/socrates-run-20251224
   ./scripts/backup_checkpoints_from_laptop.sh \
     --local-dir backups/socrates-run-20251224
   ```

2. **Backup metadata:**
   - Config files automatically downloaded with checkpoints
   - Consider saving training logs:
     ```bash
     tnr ssh 0 'cat ~/socrates-training/training.log' > backups/training-20251224.log
     ```

3. **Storage management:**
   - Each checkpoint: ~410 MB
   - Full training run (5-10 checkpoints): ~2-4 GB
   - Archive old runs: `tar -czf socrates-run-20251224.tar.gz backups/socrates-run-20251224/`

## Disaster Recovery Scenarios

### Scenario 1: GPU Instance Crashes Mid-Training

**Problem**: Thunder instance crashes at step 2847, last checkpoint was step 2500

**Solution**:
1. Verify backup exists on laptop: `ls backups/socrates-checkpoints/checkpoint-2500`
2. Start new Thunder instance: `tnr up 0`
3. Set up environment (run setup scripts)
4. Restore checkpoint: `./scripts/restore_checkpoints_to_server.sh --checkpoint checkpoint-2500`
5. Resume training from step 2500
6. **Loss**: 347 steps (~17 minutes of training at 2.4 it/s)

### Scenario 2: Accidental Checkpoint Deletion

**Problem**: Deleted wrong checkpoint directory on GPU server

**Solution**:
1. Stop training immediately
2. Restore from laptop backup: `./scripts/restore_checkpoints_to_server.sh`
3. Resume training

### Scenario 3: Thunder Instance Deleted Before Backup

**Problem**: Instance deleted/expired without backing up checkpoints

**Solution**:
- **No recovery possible** - checkpoints lost
- **Prevention**: Always backup before deleting instance
- **Mitigation**: Set up automated hourly backups during long training runs

### Scenario 4: Corrupted Checkpoint

**Problem**: Checkpoint saved but files are corrupted

**Solution**:
1. Training will error when loading: "Error loading checkpoint"
2. Restore previous checkpoint from backup: `./scripts/restore_checkpoints_to_server.sh --checkpoint checkpoint-2000`
3. Resume from earlier checkpoint

## Troubleshooting

### Backup Script Issues

**Problem**: "No checkpoints found on server"
```bash
# Check if training has reached step 500
tnr ssh 0 'ls ~/socrates-training/models/socrates-qwen-full-dataset/'

# Check training progress
tnr ssh 0 'tail -50 ~/socrates-training/training.log | grep Step'
```

**Problem**: "Thunder connection failed"
```bash
# Verify Thunder CLI is working
tnr status

# Check instance is running
tnr ssh 0 'echo Connection successful'

# Verify TNR_API_TOKEN is set
echo $TNR_API_TOKEN
```

**Problem**: Backup download is slow
- Checkpoints are ~410 MB each, expect 1-3 minutes per checkpoint
- Thunder CLI uses compression (gzip) to reduce transfer size
- Network speed dependent (check Thunder network status)

### Restore Script Issues

**Problem**: "Checkpoint missing optimizer state"
- Checkpoint may be adapter-only (not full DeepSpeed checkpoint)
- Can still restore but training will restart optimizer from scratch
- Check if checkpoint directory contains `optimizer.pt`

**Problem**: Upload fails with "Permission denied"
```bash
# Verify remote directory exists and is writable
tnr ssh 0 'mkdir -p ~/socrates-training/models/socrates-qwen-full-dataset'
tnr ssh 0 'ls -ld ~/socrates-training/models/socrates-qwen-full-dataset'
```

### Resume Training Issues

**Problem**: "Checkpoint not found" when resuming
```bash
# Verify checkpoint path on server
tnr ssh 0 'ls ~/socrates-training/models/socrates-qwen-full-dataset/checkpoint-500'

# Use absolute path in resume command
--resume-from-checkpoint ~/socrates-training/models/socrates-qwen-full-dataset/checkpoint-500
```

**Problem**: Training resumes but loss jumps significantly
- This is normal for first few steps after resume (batch variance)
- Loss should stabilize within 10-20 steps
- If loss remains high, checkpoint may be corrupted (restore from backup)

**Problem**: "Mismatch in model configuration"
- Config files on server don't match checkpoint
- Restore config files from backup:
  ```bash
  tnr ssh 0 'cat > ~/socrates-training/config_full_dataset.json' < backups/socrates-checkpoints/config_full_dataset.json
  ```

## Checkpoint Sizes and Storage Planning

### Per-Checkpoint Breakdown

| Component | Size | Required for Resume? |
|-----------|------|---------------------|
| adapter_model.safetensors | ~131 MB | Yes (LoRA weights) |
| optimizer.pt | ~263 MB | Yes (exact resume) |
| scheduler.pt | ~1.4 KB | Yes (LR schedule) |
| trainer_state.json | ~5 KB | Yes (step counter) |
| adapter_config.json | ~500 B | Yes (LoRA config) |
| training_args.bin | ~5 KB | No (can regenerate) |
| **Total per checkpoint** | **~410 MB** | - |

### Storage Requirements

**GPU Server** (with `save_total_limit=3`):
- 3 checkpoints × 410 MB = ~1.2 GB
- Auto-managed, no user intervention needed

**Local Laptop** (backup strategy):
- Minimal (last 2 checkpoints): 820 MB per training run
- Conservative (checkpoints every 1000 steps, full run): ~4 GB for 10K steps
- Archive (compressed): ~2 GB for full run (tar.gz)

**Recommended laptop backup retention:**
- Active training runs: Keep all checkpoints
- Completed runs: Keep final checkpoint + intermediate milestones
- Archived runs: Compress to tar.gz (50% space savings)

## Advanced Topics

### Automated Backup via Cron

Run backup automatically every 30 minutes during training:

```bash
# Add to laptop crontab (crontab -e)
*/30 * * * * cd /path/to/prediction-agents-copy && ./scripts/backup_checkpoints_from_laptop.sh >> logs/backup.log 2>&1
```

### Selective Checkpoint Restoration

Restore only LoRA adapter (skip optimizer for faster upload):

```bash
# Extract adapter files only
cd backups/socrates-checkpoints/checkpoint-500
tar -czf adapter_only.tar.gz adapter_model.safetensors adapter_config.json

# Upload and use for inference or fine-tuning (not exact resume)
```

### Multi-Instance Backup

Backup checkpoints from multiple Thunder instances:

```bash
# Backup from instance 0 and 1
for i in 0 1; do
  ./scripts/backup_checkpoints_from_laptop.sh \
    --instance $i \
    --local-dir backups/instance-$i
done
```

### Checkpoint Comparison

Compare two checkpoints to see training progress:

```bash
# View step numbers
cat backups/socrates-checkpoints/checkpoint-500/trainer_state.json | grep global_step
cat backups/socrates-checkpoints/checkpoint-1000/trainer_state.json | grep global_step

# View loss history
cat backups/socrates-checkpoints/checkpoint-1000/trainer_state.json | jq '.log_history[] | select(.loss) | {step: .step, loss: .loss}'
```

## Quick Reference

### Common Commands

```bash
# Backup latest checkpoints
./scripts/backup_checkpoints_from_laptop.sh

# Restore latest checkpoint to instance 0
./scripts/restore_checkpoints_to_server.sh

# Resume training from checkpoint
deepspeed --num_gpus=4 scripts/train_full_dataset.py \
  --config config_full_dataset.json \
  --deepspeed deepspeed_config.json \
  --resume-from-checkpoint models/socrates-qwen-full-dataset/checkpoint-XXX

# Check checkpoint status on server
tnr ssh 0 'ls -lh ~/socrates-training/models/socrates-qwen-full-dataset/'

# Verify backup integrity
du -sh ./backups/socrates-checkpoints/*
find ./backups/socrates-checkpoints/checkpoint-500 -type f
```

### Pre-Shutdown Checklist

Before deleting Thunder instance:
- [ ] Verify training is complete or paused
- [ ] Check latest checkpoint number
- [ ] Run backup script
- [ ] Verify backup completed successfully
- [ ] Check backup size matches expected (~410 MB per checkpoint)
- [ ] Delete Thunder instance

## Support and Documentation

- **Training script**: `scripts/train_full_dataset.py:234` (resume logic)
- **Config file**: `config_full_dataset.json:18-20` (save settings)
- **Backup script**: `scripts/backup_checkpoints_from_laptop.sh`
- **Restore script**: `scripts/restore_checkpoints_to_server.sh`
- **Thunder CLI docs**: https://docs.thundercompute.com/

## Changelog

- **2024-12-24**: Initial implementation with backup/restore scripts
  - Set `save_steps: 500`, `save_total_limit: 3`
  - Created laptop backup workflow via Thunder CLI
  - Documented full checkpoint structure and resume process
