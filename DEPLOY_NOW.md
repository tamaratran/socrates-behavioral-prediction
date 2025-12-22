# Ready to Deploy - Thunder Compute Launch Guide

## ✅ Local Setup Complete

All preparation work is finished:
- ✅ 2.9M examples prepared and formatted
- ✅ Training scripts configured for 8x A100s
- ✅ DeepSpeed ZeRO-2 configured
- ✅ Documentation complete
- ✅ Deployment package created: **socrates-full-training.tar.gz (54MB)**

**Next step: Deploy to Thunder Compute and launch training**

---

## 🚀 Thunder Compute Deployment (30 Minutes)

### Step 1: Create Thunder Compute Instance (5 min)

1. Go to https://thundercompute.com
2. Create new instance:
   - **GPUs**: 8x A100 80GB
   - **OS**: Ubuntu 22.04 LTS
   - **Storage**: 500GB SSD
   - **Region**: US (lowest latency)
3. Wait for instance to provision
4. Note the SSH command provided (e.g., `ssh -i key.pem ubuntu@<ip>`)

### Step 2: Transfer Files (10-15 min)

**Option A: Direct SCP (simplest)**

```bash
# From your local machine
cd "/Users/tamaratran/Desktop/Desktop - MacBook Pro (8)/prediction-agents-copy"

# Upload archive (54MB - fast!)
scp -i <your-key.pem> socrates-full-training.tar.gz ubuntu@<instance-ip>:~/
```

**Option B: Rsync (resumable)**

```bash
rsync -avz --progress -e "ssh -i <your-key.pem>" \
  socrates-full-training.tar.gz \
  ubuntu@<instance-ip>:~/
```

### Step 3: Set Up Environment (10 min)

SSH into Thunder Compute:

```bash
ssh -i <your-key.pem> ubuntu@<instance-ip>
```

Then run these commands:

```bash
# Extract files
tar -xzf socrates-full-training.tar.gz
cd ~/
mkdir -p socrates-training
mv scripts config_full_dataset.json deepspeed_config.json \
   requirements_full_training.txt *.md data/ socrates-training/
cd socrates-training

# Create Python environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies (includes DeepSpeed)
pip install --upgrade pip
pip install -r requirements_full_training.txt

# Verify installations
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import deepspeed; print(f'DeepSpeed: {deepspeed.__version__}')"
nvidia-smi  # Verify 8 GPUs detected
```

### Step 4: Set HuggingFace Token (2 min)

You need a HuggingFace token to download Qwen2.5-14B-Instruct:

```bash
# Get token from: https://huggingface.co/settings/tokens
export HF_TOKEN="hf_your_token_here"

# Or login via CLI
pip install huggingface-hub
huggingface-cli login
```

### Step 5: Launch Training (2 min)

```bash
# Start tmux session (persists if SSH disconnects)
tmux new -s training

# Launch training with WandB logging
bash scripts/run_full_training.sh --wandb

# Detach from tmux: Press Ctrl+B, then D
# Reattach later: tmux attach -t training
```

---

## 📊 What Happens Next

### Training Timeline (25-30 hours)

1. **Model Download** (30-60 min)
   - Downloads Qwen2.5-14B-Instruct (~28GB)
   - One-time download, cached for future runs

2. **Training** (25-30 hours)
   - ~2,800 training steps
   - Checkpoint every 500 steps (~280 checkpoints)
   - Evaluation every 500 steps
   - Total GPU-hours: 200-240 (8 GPUs × 25-30 hrs)

3. **Cost**: ~$220-264 total
   - Rate: $8.80/hour (8 GPUs × $1.10/GPU/hr)
   - Duration: 25-30 hours

### Monitoring Training

**WandB Dashboard** (recommended):
- Real-time metrics: loss, learning rate, throughput
- GPU utilization graphs
- Training progress visualization
- Access at: https://wandb.ai

**Command Line**:
```bash
# Reattach to tmux session
tmux attach -t training

# In another SSH session - watch GPUs
watch -n 1 nvidia-smi

# Check checkpoints being saved
ls -lh models/socrates-qwen-full-dataset/checkpoint-*/
```

**Logs**:
```bash
# View training logs
tail -f models/socrates-qwen-full-dataset/runs/*/events.out.tfevents.*

# Check trainer state
cat models/socrates-qwen-full-dataset/trainer_state.json | jq
```

---

## 🎯 Expected Results

### Training Metrics

| Metric | Value |
|--------|-------|
| **Correlation** | >85% (target) |
| **MAE** | <3.5 (target) |
| **Training Examples** | 2,468,761 |
| **Validation Examples** | 188,733 |
| **Test Examples** | 243,896 |
| **Training Steps** | ~2,800 |
| **Checkpoints** | ~280 |

### Comparison to Baseline

| Metric | 1% Baseline | Full Dataset | Improvement |
|--------|-------------|--------------|-------------|
| Data | 29K | 2.9M | 100x |
| Correlation | 81.5% | >85% | +3.5% |
| MAE | 3.72 | <3.5 | -0.22 |
| Time | 2 hours | 25-30 hours | 12-15x |

---

## ⚡ Quick Commands Reference

```bash
# Check training status
tmux attach -t training

# Monitor GPUs
nvidia-smi

# Check disk space (need ~150GB free)
df -h

# Count checkpoints saved
ls models/socrates-qwen-full-dataset/checkpoint-* | wc -l

# View latest metrics
cat models/socrates-qwen-full-dataset/trainer_state.json | jq '.log_history[-5:]'

# Estimate time remaining (based on current step)
# Formula: (2800 - current_step) * seconds_per_step / 3600
```

---

## 🔧 Troubleshooting

### Issue: Out of Memory

**Solution**: Reduce per-device batch size

```bash
# Edit config_full_dataset.json
{
  "training": {
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 32
  }
}
```

### Issue: NCCL Timeout (Multi-GPU Communication)

**Solution**: Increase timeout

```bash
export NCCL_TIMEOUT=7200
export NCCL_DEBUG=INFO
bash scripts/run_full_training.sh --wandb
```

### Issue: Training Interrupted

**Solution**: Resume from last checkpoint

```bash
# Find latest checkpoint
ls -t models/socrates-qwen-full-dataset/checkpoint-* | head -1

# Resume
bash scripts/run_full_training.sh --resume models/socrates-qwen-full-dataset/checkpoint-2500
```

### Issue: Disk Full

**Solution**: Keep fewer checkpoints

```bash
# Edit config_full_dataset.json
{
  "training": {
    "save_total_limit": 2  // Only keep last 2 checkpoints
  }
}

# Or manually delete old checkpoints
rm -rf models/socrates-qwen-full-dataset/checkpoint-{500..2000}
```

---

## 📥 After Training Completes

### Download Model (5-10 GB)

```bash
# From your local machine
scp -i <your-key.pem> -r \
  ubuntu@<instance-ip>:~/socrates-training/models/socrates-qwen-full-dataset \
  "/Users/tamaratran/Desktop/Desktop - MacBook Pro (8)/prediction-agents-copy/models/"

# Or use rsync (resumable)
rsync -avz --progress -e "ssh -i <your-key.pem>" \
  ubuntu@<instance-ip>:~/socrates-training/models/socrates-qwen-full-dataset/ \
  "/Users/tamaratran/Desktop/Desktop - MacBook Pro (8)/prediction-agents-copy/models/socrates-qwen-full-dataset/"
```

### Run Evaluation

```bash
# On local machine or Thunder Compute
python scripts/evaluate.py \
  --model models/socrates-qwen-full-dataset \
  --data data/socsci210_full/test.jsonl \
  --output results/full_dataset_eval.json
```

### Clean Up Thunder Compute

```bash
# Delete instance via Thunder Compute dashboard
# Or keep for future experiments (charged per hour)
```

---

## 📋 Pre-Launch Checklist

Before launching training, verify:

- [ ] Thunder Compute instance created (8x A100 80GB)
- [ ] Files transferred successfully (54MB archive)
- [ ] Python environment set up (`source venv/bin/activate`)
- [ ] Dependencies installed (`pip list | grep torch`)
- [ ] 8 GPUs detected (`nvidia-smi`)
- [ ] HuggingFace token set (`echo $HF_TOKEN`)
- [ ] Enough disk space (`df -h` shows >150GB free)
- [ ] WandB configured (optional, `wandb login`)
- [ ] tmux installed (`tmux -V`)
- [ ] Training script is executable (`ls -l scripts/run_full_training.sh`)

---

## 🎓 Documentation Reference

For more details, see:

- **FULL_TRAINING_README.md** - Comprehensive training guide
- **THUNDER_COMPUTE_DEPLOYMENT.md** - Detailed deployment steps
- **IMPLEMENTATION_DECISIONS.md** - All planning decisions and rationale
- **QUICK_START.md** - Quick reference commands
- **SETUP_SUMMARY.md** - Current status overview

---

## 📞 Support

If you encounter issues:

1. Check logs in `models/socrates-qwen-full-dataset/`
2. Verify GPU memory with `nvidia-smi`
3. Review WandB dashboard for training curves
4. Consult FULL_TRAINING_README.md troubleshooting section

---

## 🚀 Ready to Launch!

**Total time from now to training start: ~30 minutes**

1. Create Thunder Compute instance (5 min)
2. Transfer files (10-15 min)
3. Set up environment (10 min)
4. Set HuggingFace token (2 min)
5. Launch training (2 min)

**Then wait 25-30 hours for training to complete.**

**Estimated total cost: $220-264**

---

**Current Status**: Everything is prepared and ready for deployment.

**Next Action**: Create Thunder Compute instance and follow Step 1 above.

**Good luck with the training!** 🎉
