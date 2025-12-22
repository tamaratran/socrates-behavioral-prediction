# Full Dataset Training Guide - SOCRATES

This guide covers training the SOCRATES behavioral prediction model on the **complete SocSci210 dataset (2.9M examples)** using 8x A100 80GB GPUs on Thunder Compute.

## 📊 Overview

- **Dataset**: SocSci210 (2.9M examples from 210+ studies)
- **Model**: Qwen2.5-14B-Instruct with QLoRA
- **Hardware**: 8x A100 80GB GPUs
- **Training Time**: ~25-30 hours
- **Estimated Cost**: ~$220-264 on Thunder Compute
- **Methodology**: SOCRATES paper (arxiv:2509.05830)

## 🔧 Prerequisites

### 1. Hardware Requirements

- **Recommended**: 8x A100 80GB GPUs (Thunder Compute, Lambda Labs, etc.)
- **Minimum**: 4x A100 80GB GPUs (adjust config accordingly)
- **Storage**: ~200GB for dataset + model checkpoints

### 2. Software Requirements

```bash
# Python 3.9+
python --version

# Install required packages
pip install torch transformers peft datasets trl accelerate deepspeed bitsandbytes wandb
```

### 3. Environment Setup

```bash
# Set Hugging Face token (for downloading models)
export HF_TOKEN="your_huggingface_token"

# Optional: Weights & Biases for logging
export WANDB_API_KEY="your_wandb_key"
```

## 📁 Project Structure

```
prediction-agents-copy/
├── scripts/
│   ├── prepare_full_data.py      # Data preparation (2.9M examples)
│   ├── train_full_dataset.py     # Multi-GPU training script
│   └── run_full_training.sh      # Training launcher (recommended)
├── config_full_dataset.json      # Hyperparameters for full dataset
├── deepspeed_config.json         # DeepSpeed ZeRO-2 configuration
├── data/
│   └── socsci210_full/           # Will be created by prepare_full_data.py
└── models/
    └── socrates-qwen-full-dataset/  # Training output
```

## 🚀 Quick Start

### Option 1: One-Command Training (Recommended)

```bash
# Prepare data + train in one command
bash scripts/run_full_training.sh --prepare-data --wandb
```

### Option 2: Step-by-Step

#### Step 1: Prepare Dataset

```bash
python scripts/prepare_full_data.py \
    --output data/socsci210_full \
    --seed 42
```

**Expected output:**
- `data/socsci210_full/train.jsonl` (~2.4M examples)
- `data/socsci210_full/val.jsonl` (~280K examples)
- `data/socsci210_full/test.jsonl` (~280K examples)
- `data/socsci210_full/metadata.json`

**Time**: 30-60 minutes

#### Step 2: Launch Training

```bash
bash scripts/run_full_training.sh --wandb
```

**Or use DeepSpeed directly:**

```bash
deepspeed --num_gpus=8 scripts/train_full_dataset.py \
    --config config_full_dataset.json \
    --deepspeed deepspeed_config.json \
    --use-wandb
```

## ⚙️ Configuration Details

### Hyperparameters (`config_full_dataset.json`)

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Model** | Qwen2.5-14B-Instruct | 14B parameter base model |
| **LoRA Rank** | 32 | Increased from 16 (paper used 16 for 1% data) |
| **LoRA Alpha** | 64 | 2x rank (standard convention) |
| **Epochs** | 1 | Single pass through 2.9M examples |
| **Global Batch Size** | 1024 | 4x larger than paper (256) |
| **Per-Device Batch Size** | 8 | Per GPU |
| **Gradient Accumulation** | 16 | Steps before optimizer update |
| **Learning Rate** | 1e-5 | Same as paper |
| **Warmup Ratio** | 0.05 | 5% of training |
| **LR Schedule** | Cosine | With warmup |
| **Weight Decay** | 0.1 | Regularization |
| **Sequence Length** | 2048 | Max tokens |

### Hardware Configuration

```
8 GPUs × 8 per-device batch × 16 grad accumulation = 1024 global batch size
```

**Training Steps:**
- ~2,800 steps per epoch (2.4M train examples / 1024 batch size)
- ~280 checkpoints saved (every 500 steps)
- ~280 evaluations (every 500 steps)

### DeepSpeed ZeRO-2

- **Stage 2**: Partitions optimizer states and gradients across GPUs
- **BF16**: Mixed precision training
- **No Offload**: All computation on GPU (A100s have enough memory)
- **Gradient Checkpointing**: Enabled for memory efficiency

## 📈 Expected Results

### Baseline (1% dataset, 29K examples)
- **Correlation**: 81.5% with human responses
- **MAE**: 3.72
- **Training Time**: 2 hours on 1x A100

### Expected (100% dataset, 2.9M examples)
- **Correlation**: >85% (estimated improvement from more data)
- **MAE**: <3.5 (estimated)
- **Training Time**: 25-30 hours on 8x A100s
- **Cost**: ~$220-264 on Thunder Compute

## 🔄 Resuming Training

If training is interrupted:

```bash
# Find latest checkpoint
ls models/socrates-qwen-full-dataset/checkpoint-*

# Resume from checkpoint
bash scripts/run_full_training.sh --resume models/socrates-qwen-full-dataset/checkpoint-2000
```

## 📊 Monitoring Training

### Option 1: Weights & Biases

```bash
bash scripts/run_full_training.sh --wandb
```

View metrics at https://wandb.ai

### Option 2: Local Logs

```bash
# Training logs are in output directory
tail -f models/socrates-qwen-full-dataset/trainer_state.json
```

### Option 3: TensorBoard (if configured)

```bash
tensorboard --logdir models/socrates-qwen-full-dataset
```

## 🧪 Evaluation

After training completes:

```bash
# Evaluate on test set
python scripts/evaluate.py \
    --model models/socrates-qwen-full-dataset \
    --data data/socsci210_full/test.jsonl

# Interactive testing
python scripts/test_inference.py \
    --model models/socrates-qwen-full-dataset
```

## 🎯 Optimization Tips

### Increase Training Speed

1. **Reduce evaluation frequency**:
   - Edit `config_full_dataset.json`: `"eval_steps": 1000` (default: 500)
   - Saves time on validation passes

2. **Increase per-device batch size** (if memory allows):
   - `"per_device_train_batch_size": 16` (reduce gradient accumulation to 8)
   - Reduces communication overhead

3. **Disable checkpointing** (NOT recommended):
   - `"save_steps": 10000` (only save at end)

### Reduce Cost

1. **Use fewer GPUs**:
   - Modify `run_full_training.sh`: `NUM_GPUS=4`
   - Adjust gradient accumulation: `32` (to maintain batch size 1024)
   - Training time: ~50 hours instead of 25

2. **Smaller batch size**:
   - `"global_batch_size": 512`
   - Faster but potentially worse convergence

3. **Use spot instances**:
   - Thunder Compute spot: ~$0.60/GPU/hr (vs $1.10 on-demand)
   - Can save 45% but may be interrupted

## 🐛 Troubleshooting

### Out of Memory (OOM)

```bash
# Reduce per-device batch size
# In config_full_dataset.json:
"per_device_train_batch_size": 4,
"gradient_accumulation_steps": 32
```

### Slow Data Loading

```bash
# Increase data loader workers
# In train_full_dataset.py:
"dataloader_num_workers": 8
```

### NCCL Timeout (multi-GPU communication)

```bash
# Increase timeout
export NCCL_TIMEOUT=7200
bash scripts/run_full_training.sh
```

### DeepSpeed Not Found

```bash
pip install deepspeed
# If compilation fails, use pre-built wheels:
pip install deepspeed --no-build-isolation
```

## 📝 Differences from Paper

| Aspect | Paper (1% data) | This Config (100% data) |
|--------|----------------|------------------------|
| Dataset Size | 29K examples | 2.9M examples |
| LoRA Rank | 16 | 32 (more capacity) |
| Global Batch Size | 256 | 1024 (larger dataset) |
| Training Time | 2 hours | 25-30 hours |
| Cost | ~$2 | ~$220-264 |

## 🔗 Resources

- **Paper**: [SOCRATES (arxiv:2509.05830)](https://arxiv.org/abs/2509.05830)
- **Dataset**: [socratesft/SocSci210](https://huggingface.co/datasets/socratesft/SocSci210)
- **Base Model**: [Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct)
- **Thunder Compute**: [thundercompute.com](https://thundercompute.com)

## 📧 Support

If you encounter issues:

1. Check logs in `models/socrates-qwen-full-dataset/`
2. Verify GPU memory with `nvidia-smi`
3. Test with smaller dataset first (`--subset 0.1` for 10%)

## 🎓 Citation

If you use this code or model:

```bibtex
@article{socrates2024,
  title={SOCRATES: Finetuning LLMs for Human Behavior Prediction in Social Science},
  journal={arXiv preprint arxiv:2509.05830},
  year={2024}
}
```

---

**Good luck with your training! 🚀**
