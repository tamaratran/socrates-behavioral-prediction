# Quick Start Guide - Full Dataset Training

## ✅ What's Been Done

1. ✅ Dependencies installed (transformers, peft, trl, etc.)
2. ✅ Data preparation running (2.9M examples)
3. ✅ Training scripts created and configured
4. ✅ DeepSpeed config ready for 8x A100s

## 📋 Current Status

**Data Preparation:** In progress (~30-45 min total)
- Downloading SocSci210 dataset
- Formatting 2.9M examples
- Creating study-level train/val/test splits

**Output:**
```
data/socsci210_full/
├── train.jsonl       (~2.4M examples, ~12GB)
├── val.jsonl         (~280K examples, ~1.4GB)
├── test.jsonl        (~280K examples, ~1.4GB)
└── metadata.json     (split information)
```

## 🚀 Next Steps

### 1. Wait for Data Preparation to Complete

Check progress:
```bash
# In another terminal
ls -lh data/socsci210_full/
```

### 2. Transfer to Thunder Compute

**What to upload:**
- All scripts and configs (~1MB)
- Prepared data (~15GB)

**Transfer methods:** See `THUNDER_COMPUTE_DEPLOYMENT.md`

### 3. On Thunder Compute

```bash
# Install dependencies
pip install -r requirements_full_training.txt

# Set HuggingFace token
export HF_TOKEN="your_token_here"

# Launch training
bash scripts/run_full_training.sh --wandb
```

## 💰 Cost Summary

| Item | Amount |
|------|--------|
| **Training** | $220-264 |
| GPU hours | 200-240 hrs (8 GPUs × 25-30 hrs) |
| Rate | $1.10/GPU/hr (Thunder Compute) |
| Storage | ~$5/month (500GB) |
| Data transfer | Minimal (~15GB upload) |
| **Total** | **~$225-270** |

## 📊 Expected Results

| Metric | 1% Baseline | Full Dataset (Expected) |
|--------|-------------|------------------------|
| Training examples | 29K | 2.9M |
| Correlation | 81.5% | >85% |
| MAE | 3.72 | <3.5 |
| Training time | 2 hours | 25-30 hours |

## 🔗 Documentation

- **Full guide:** `FULL_TRAINING_README.md`
- **Thunder deployment:** `THUNDER_COMPUTE_DEPLOYMENT.md`
- **Configuration:** `config_full_dataset.json`

## ⚡ Quick Commands

```bash
# Check data preparation status
ls -lh data/socsci210_full/

# Verify data
head -n 1 data/socsci210_full/train.jsonl | jq

# See training config
cat config_full_dataset.json | jq .training

# Test scripts are executable
ls -l scripts/*.sh

# Package for transfer (after data prep completes)
tar -czf socrates-full-training.tar.gz \
  scripts/ config_full_dataset.json deepspeed_config.json \
  requirements_full_training.txt data/socsci210_full/
```

## 🎯 Training Configuration

```json
{
  "model": "Qwen2.5-14B-Instruct",
  "dataset": "2.9M examples (210 studies)",
  "split": "Study-level (paper methodology)",
  "lora_rank": 32,
  "batch_size": 1024,
  "learning_rate": 1e-5,
  "epochs": 1,
  "hardware": "8x A100 80GB"
}
```

## ❓ Troubleshooting

**Data prep stuck?**
```bash
# Check background process
ps aux | grep prepare_full_data
```

**Out of disk space?**
```bash
# Check available space
df -h
```

**Need to restart data prep?**
```bash
# Kill and restart
pkill -f prepare_full_data
python scripts/prepare_full_data.py --output data/socsci210_full
```

---

**Ready to deploy when data preparation completes!** 🚀
