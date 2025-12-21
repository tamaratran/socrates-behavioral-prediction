# SOCRATES Finetuning - Quick Start Guide

Get started with finetuning LLMs for behavioral prediction in 3 steps.

## Prerequisites

- Python 3.9+
- CUDA-compatible GPU (or cloud GPU access via RunPod/Lambda)
- Hugging Face account (for dataset access)

## Setup (5 minutes)

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up Hugging Face token
# Get your token from: https://huggingface.co/settings/tokens
echo "HUGGINGFACE_TOKEN=your_token_here" >> .env
```

## Run Your First Experiment (1% subset, ~$2-3 cost)

### Step 1: Prepare Data (~2 minutes)

```bash
python scripts/prepare_data.py --subset 0.01 --output data/socsci210_1pct
```

This downloads 1% of SocSci210 dataset (~29K examples) and creates train/val/test splits.

**Expected output:**
```
Loading SocSci210 dataset...
Using 1% of the data
Full dataset size: 2900000 examples
Using subset: 29000 examples
...
Train: 23200, Val: 2900, Test: 2900
Dataset preparation complete!
```

### Step 2: Finetune Model (~30-60 minutes on A100)

**Option A: Cloud GPU (Recommended for first run)**

1. Go to [RunPod](https://runpod.io) or [Lambda Labs](https://lambdalabs.com)
2. Rent an A100 40GB GPU (~$1.50/hour)
3. Upload your code or clone from git
4. Run:

```bash
python scripts/finetune.py \
  --model "Qwen/Qwen2.5-14B-Instruct" \
  --data data/socsci210_1pct \
  --output models/socrates-qwen-1pct \
  --use-qlora \
  --epochs 3
```

**Option B: Local GPU (if you have RTX 4090 or better)**

Same command as above, run locally.

**Expected output:**
```
Loading model: Qwen/Qwen2.5-14B-Instruct
QLoRA: True
Model loaded. Trainable parameters: 14,715,822,080
LoRA applied. Trainable parameters: 67,108,864
...
Training complete!
Model saved to: models/socrates-qwen-1pct
```

**Cost:** ~$1.50-3 for 1 hour of training

### Step 3: Evaluate Model (~10 minutes)

```bash
python scripts/evaluate.py --model models/socrates-qwen-1pct
```

**Expected output:**
```
Evaluation Results
==================================================
Number of numeric predictions: 2850/2900
Mean Absolute Error (MAE): X.XXXX
Root Mean Squared Error (RMSE): X.XXXX
Correlation: 0.XXX
```

### Step 4: Test Your Model Interactively

```bash
# Run a predefined example
python scripts/test_inference.py --model models/socrates-qwen-1pct --example 1

# Or interactive mode
python scripts/test_inference.py --model models/socrates-qwen-1pct
```

## Scale Up (if results look good)

### 10% Subset (~$6-12, 4-8 hours)

```bash
# Prepare data
python scripts/prepare_data.py --subset 0.1 --output data/socsci210_10pct

# Finetune
python scripts/finetune.py \
  --model "Qwen/Qwen2.5-14B-Instruct" \
  --data data/socsci210_10pct \
  --output models/socrates-qwen-10pct \
  --use-qlora

# Evaluate
python scripts/evaluate.py --model models/socrates-qwen-10pct
```

### Full Dataset (~$60-120, 40-80 hours)

Only do this if 10% subset shows significant improvement.

```bash
python scripts/prepare_data.py --subset 1.0 --output data/socsci210_full
python scripts/finetune.py --model "Qwen/Qwen2.5-14B-Instruct" --data data/socsci210_full --output models/socrates-qwen-full --use-qlora
```

## Cost Summary

| Experiment | Subset | Examples | Training Time | GPU Cost | Total Cost |
|------------|--------|----------|---------------|----------|------------|
| **Quick Test** | 1% | 29K | ~1 hour | $1.50/hr | **$1.50-3** |
| **Recommended** | 10% | 290K | ~4-8 hours | $1.50/hr | **$6-12** |
| **Full (if needed)** | 100% | 2.9M | ~40-80 hours | $1.50/hr | **$60-120** |

*Based on RunPod/Lambda Labs A100 GPU pricing*

## GPU Rental Services

| Service | GPU | Cost/hour | Link |
|---------|-----|-----------|------|
| RunPod | A100 40GB | ~$1.39 | https://runpod.io |
| Lambda Labs | A100 40GB | ~$1.10 | https://lambdalabs.com |
| Vast.ai | A100 40GB | ~$0.80+ | https://vast.ai |

## Troubleshooting

### Out of Memory Error
- Reduce `--batch-size` (try 2 or 1)
- Increase `gradient_accumulation_steps` to 8 or 16
- Use smaller model (Llama 3 8B instead of Qwen 14B)

### Dataset Download Issues
- Make sure your HuggingFace token is set in `.env`
- Check: `huggingface-cli login`

### Slow Training
- Verify GPU is being used: `nvidia-smi`
- Make sure `--use-qlora` flag is set
- Consider using Unsloth for 2-3x speedup (see requirements.txt)

## Next Steps

1. **Compare with base model:** Run evaluation on base model without finetuning
2. **Try different models:** Test Llama 3 8B vs Qwen 2.5 14B
3. **Experiment with hyperparameters:** Learning rate, epochs, LoRA rank
4. **Analyze results by study type:** Look at per-study performance metrics

## Questions?

- Check the main [README.md](README.md) for detailed documentation
- Review the [SOCRATES paper](papers/2509.05830v2.pdf)
- Visit the [project website](https://stanfordhci.github.io/socrates/)
