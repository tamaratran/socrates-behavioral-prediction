# Thunder Compute Deployment Guide

This guide covers deploying and running the full dataset training on Thunder Compute with 8x A100 80GB GPUs.

## 📦 What to Transfer

After local data preparation completes, you'll need to transfer these files to Thunder Compute:

```
prediction-agents-copy/
├── scripts/
│   ├── train_full_dataset.py
│   └── run_full_training.sh
├── config_full_dataset.json
├── deepspeed_config.json
├── requirements_full_training.txt
└── data/socsci210_full/
    ├── train.jsonl (~2.4M examples)
    ├── val.jsonl (~280K examples)
    ├── test.jsonl (~280K examples)
    └── metadata.json
```

**Total size**: ~15-20 GB (mostly the data files)

## 🚀 Thunder Compute Setup

### Step 1: Create Instance

1. Go to [Thunder Compute](https://thundercompute.com)
2. Create a new instance:
   - **GPUs**: 8x A100 80GB
   - **OS**: Ubuntu 22.04 LTS
   - **Storage**: 500GB SSD (for model downloads + checkpoints)
   - **Region**: Choose closest to you

### Step 2: Connect via SSH

```bash
# Thunder will provide SSH command like:
ssh -i your_key.pem ubuntu@<instance-ip>
```

### Step 3: Install System Dependencies

```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install Python 3.10+
sudo apt-get install -y python3.10 python3.10-venv python3-pip

# Install CUDA toolkit (if not pre-installed)
nvidia-smi  # Verify GPUs are detected

# Install git (for cloning)
sudo apt-get install -y git
```

### Step 4: Set Up Python Environment

```bash
# Create project directory
mkdir -p ~/socrates-training
cd ~/socrates-training

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### Step 5: Transfer Files

**Option A: Using SCP (from your local machine)**

```bash
# From your local terminal (NOT Thunder Compute)
cd "/Users/tamaratran/Desktop/Desktop - MacBook Pro (8)/prediction-agents-copy"

# Transfer entire directory
scp -i your_key.pem -r \
  scripts/ config_full_dataset.json deepspeed_config.json requirements_full_training.txt data/ \
  ubuntu@<instance-ip>:~/socrates-training/
```

**Option B: Using rsync (faster for large files)**

```bash
# From your local machine
rsync -avz --progress -e "ssh -i your_key.pem" \
  --exclude '.git' --exclude '*.pyc' --exclude '__pycache__' \
  "/Users/tamaratran/Desktop/Desktop - MacBook Pro (8)/prediction-agents-copy/" \
  ubuntu@<instance-ip>:~/socrates-training/
```

**Option C: Using Cloud Storage (S3/GCS)**

```bash
# On local machine - upload to S3/GCS
aws s3 cp data/ s3://your-bucket/socsci210_full/ --recursive

# On Thunder Compute - download
aws s3 cp s3://your-bucket/socsci210_full/ data/socsci210_full/ --recursive
```

### Step 6: Install Python Dependencies

```bash
# On Thunder Compute
cd ~/socrates-training
source venv/bin/activate

# Install dependencies (now DeepSpeed will work on Linux/CUDA)
pip install -r requirements_full_training.txt

# Verify installations
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import deepspeed; print(f'DeepSpeed: {deepspeed.__version__}')"
```

### Step 7: Set HuggingFace Token

You need a HuggingFace token to download the Qwen model.

1. Get token from: https://huggingface.co/settings/tokens
2. Set environment variable:

```bash
# Add to ~/.bashrc for persistence
echo 'export HF_TOKEN="hf_your_token_here"' >> ~/.bashrc
source ~/.bashrc

# Or just for this session
export HF_TOKEN="hf_your_token_here"

# Login via CLI (alternative)
pip install huggingface-hub
huggingface-cli login
```

### Step 8: Optional - Set Up Weights & Biases

```bash
# Get API key from: https://wandb.ai/settings
export WANDB_API_KEY="your_wandb_key"

# Or login interactively
wandb login
```

## 🎯 Running Training

### Quick Start

```bash
cd ~/socrates-training
source venv/bin/activate

# Run training with WandB logging
bash scripts/run_full_training.sh --wandb
```

### Monitor Progress

**Option 1: Watch logs in terminal**

```bash
# The training script outputs to stdout
# Use tmux/screen to keep it running if SSH disconnects
tmux new -s training
bash scripts/run_full_training.sh --wandb
# Detach: Ctrl+B, then D
# Reattach: tmux attach -t training
```

**Option 2: Monitor with Weights & Biases**

Go to https://wandb.ai and view real-time metrics

**Option 3: Check GPU usage**

```bash
# In another terminal
watch -n 1 nvidia-smi
```

**Option 4: Monitor checkpoints**

```bash
# Check saved checkpoints
ls -lh models/socrates-qwen-full-dataset/checkpoint-*/

# Check disk usage
df -h
```

## 📊 Expected Training Metrics

### Timeline

- **Setup time**: ~30-60 minutes (download Qwen model ~28GB)
- **Training time**: ~25-30 hours
- **Checkpoints**: Every 500 steps (~280 total)
- **Total steps**: ~2,800 steps (2.4M examples / 1024 batch size)

### Resource Usage

- **GPU Memory**: ~70-75GB per A100 (out of 80GB)
- **RAM**: ~100-150GB system RAM
- **Disk**: ~150GB total
  - 28GB: Qwen base model
  - 15GB: Training data
  - 100GB: Checkpoints (will grow during training)
  - 5GB: Environment/dependencies

### Costs

- **Training**: ~$220-264 ($1.10/GPU/hr × 8 GPUs × 25-30 hrs)
- **Data transfer**: Minimal (~20GB upload)
- **Storage**: Included with instance

## 🔧 Troubleshooting

### Out of Disk Space

```bash
# Remove old checkpoints (keep only last 3)
# Edit config_full_dataset.json before starting:
"save_total_limit": 3

# Or manually delete old checkpoints
rm -rf models/socrates-qwen-full-dataset/checkpoint-{500..1500}
```

### Training Interrupted

```bash
# Resume from last checkpoint
bash scripts/run_full_training.sh --resume models/socrates-qwen-full-dataset/checkpoint-2500
```

### GPU Out of Memory

```bash
# Reduce per-device batch size
# Edit config_full_dataset.json:
"per_device_train_batch_size": 4,
"gradient_accumulation_steps": 32
```

### Slow Download of Qwen Model

```bash
# Pre-download the model
python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
           AutoTokenizer.from_pretrained('Qwen/Qwen2.5-14B-Instruct'); \
           print('Model cached!')"
```

### DeepSpeed NCCL Errors

```bash
# Increase timeout
export NCCL_TIMEOUT=7200
export NCCL_DEBUG=INFO  # For debugging

# Or edit deepspeed_config.json to add:
"communication_data_type": "fp32"
```

## 📥 Downloading Results

After training completes:

```bash
# From local machine
scp -i your_key.pem -r \
  ubuntu@<instance-ip>:~/socrates-training/models/socrates-qwen-full-dataset \
  "/Users/tamaratran/Desktop/Desktop - MacBook Pro (8)/prediction-agents-copy/models/"
```

Or use `rsync` for resumable transfer:

```bash
rsync -avz --progress -e "ssh -i your_key.pem" \
  ubuntu@<instance-ip>:~/socrates-training/models/socrates-qwen-full-dataset/ \
  "/Users/tamaratran/Desktop/Desktop - MacBook Pro (8)/prediction-agents-copy/models/socrates-qwen-full-dataset/"
```

## 🧹 Cleanup

```bash
# On Thunder Compute - after downloading results
cd ~/socrates-training

# Remove large files but keep configs/scripts
rm -rf data/socsci210_full/*.jsonl
rm -rf models/socrates-qwen-full-dataset/checkpoint-*

# Or destroy the instance entirely via Thunder Compute dashboard
```

## 💡 Pro Tips

### 1. Use tmux for Long-Running Jobs

```bash
# Start tmux session
tmux new -s training

# Run training
bash scripts/run_full_training.sh --wandb

# Detach (training keeps running)
# Press: Ctrl+B, then D

# Reattach later
tmux attach -t training
```

### 2. Monitor Costs

Thunder Compute shows real-time costs. For ~25 hours:
- Cost/hour: $8.80 (8 GPUs × $1.10)
- Total: $220-264

### 3. Use Spot Instances (if available)

Thunder may offer spot instances at ~45% discount. Use for non-critical jobs where interruptions are acceptable.

### 4. Snapshot the Instance

After setup, create a snapshot so you can relaunch with everything pre-installed.

## 📝 Pre-Launch Checklist

- [ ] Thunder Compute instance created (8x A100 80GB)
- [ ] SSH access working
- [ ] Python environment set up
- [ ] Dependencies installed (including DeepSpeed)
- [ ] Data files transferred (~15-20GB)
- [ ] HuggingFace token set
- [ ] Weights & Biases configured (optional)
- [ ] `nvidia-smi` shows all 8 GPUs
- [ ] Enough disk space (~500GB recommended)
- [ ] tmux/screen installed for persistent sessions

## 🚀 Launch Command

```bash
# Final command to start training
cd ~/socrates-training
source venv/bin/activate
tmux new -s training
bash scripts/run_full_training.sh --wandb
```

Press `Ctrl+B` then `D` to detach and let it run!

---

**Questions?** Check logs in `models/socrates-qwen-full-dataset/` or monitoring on WandB.
