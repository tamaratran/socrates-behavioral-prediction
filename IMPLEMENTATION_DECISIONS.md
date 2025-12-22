# Full Dataset Fine-Tuning - Implementation Decisions

## Project Overview

**Goal**: Scale SOCRATES behavioral prediction model from 1% baseline (29K examples) to full SocSci210 dataset (2.9M examples) for improved prediction accuracy.

**Date Started**: December 2024
**Paper Reference**: SOCRATES - arxiv:2509.05830
**Dataset**: socratesft/SocSci210 (2.9M examples from 210+ studies)

---

## Planning Decisions

### 1. Hardware Selection

**Decision**: 8x A100 80GB GPUs on Thunder Compute

**Rationale**:
- Need multi-GPU setup for 2.9M examples
- A100 80GB provides enough memory for large batch sizes
- Thunder Compute offers competitive pricing ($1.10/GPU/hr)
- 8 GPUs allows efficient data parallelism with DeepSpeed

**Alternatives Considered**:
- 4x A100: Slower (50 hours vs 25-30 hours)
- Smaller GPUs (A10): Insufficient memory for large batches
- More GPUs (16x): Diminishing returns on speedup

**Cost Analysis**:
- Training: $220-264 for 25-30 hours
- Storage: ~$5/month for 500GB
- Data transfer: Minimal (~20GB upload)
- **Total**: ~$225-270

### 2. Training Configuration

**Decision**: Optimize for maximum performance with 1 epoch

**Key Hyperparameters**:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Model** | Qwen2.5-14B-Instruct | Same as paper, proven performance |
| **LoRA Rank** | 32 | 2x increase from baseline (16) for more capacity with 100x data |
| **LoRA Alpha** | 64 | Maintains 2x rank ratio (standard convention) |
| **Global Batch Size** | 1024 | 4x larger than baseline (256) to leverage 8 GPUs |
| **Per-Device Batch** | 8 | Fits in A100 80GB memory |
| **Gradient Accumulation** | 16 | Achieves 1024 global batch (8 GPUs × 8 batch × 16 accum) |
| **Learning Rate** | 1e-5 | Same as paper |
| **Epochs** | 1 | Start conservative, can increase if needed |
| **Warmup Ratio** | 0.05 | 5% warmup for stable training |
| **LR Schedule** | Cosine | With warmup |
| **Sequence Length** | 2048 | Max tokens per example |

**Comparison to 1% Baseline**:

```
Dataset Size:     29K → 2.9M examples (100x increase)
LoRA Rank:        16 → 32 (2x increase for more capacity)
LoRA Alpha:       32 → 64 (maintains ratio)
Batch Size:       256 → 1024 (4x increase)
Per-Device Batch: 4 → 8 (2x increase)
Grad Accumulation: 64 → 16 (fewer steps needed with more GPUs)
GPUs:             1 → 8 (multi-GPU with DeepSpeed)
Training Time:    2 hours → 25-30 hours
Cost:             ~$2 → ~$220-264
```

### 3. Data Splitting Methodology

**Decision**: Use study-level splits (paper methodology)

**Implementation**:
- Split dataset by study_id, not individual examples
- Ensures no data leakage between train/val/test
- Matches SOCRATES paper Section 5.3

**Split Ratios**:
- Train: 170 studies (85%) = 2,468,761 examples
- Validation: 20 studies (10%) = 188,733 examples
- Test: 20 studies (10%) = 243,896 examples
- **Total**: 210 studies, 2,901,390 examples

**Why Study-Level Splits Matter**:
- Prevents overfitting to specific studies
- Tests generalization to new experimental designs
- More challenging than random splits
- Aligns with paper methodology for fair comparison

### 4. Infrastructure Choices

**Decision**: DeepSpeed ZeRO-2 for distributed training

**Configuration**:
```json
{
  "zero_optimization": {
    "stage": 2,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "allgather_bucket_size": 200000000,
    "reduce_bucket_size": 200000000
  },
  "bf16": {"enabled": true},
  "gradient_clipping": 1.0
}
```

**Rationale**:
- **ZeRO-2**: Partitions optimizer states and gradients across GPUs
- **BF16**: Mixed precision for speed without accuracy loss
- **No CPU Offload**: A100s have enough memory
- **Gradient Checkpointing**: Enabled for memory efficiency

**Alternatives Considered**:
- ZeRO-3: Unnecessary overhead for our model size
- FP16: BF16 more stable for large models
- FSDP: DeepSpeed more mature for transformers

### 5. Monitoring and Logging

**Decision**: Weights & Biases for experiment tracking

**What We Track**:
- Training/validation loss
- Learning rate schedule
- GPU memory usage
- Throughput (examples/sec)
- Checkpoint saves every 500 steps

**Evaluation Strategy**:
- Evaluate every 500 steps
- Save checkpoints every 500 steps
- Keep last 3 checkpoints (save_total_limit: 3)
- ~280 total checkpoints over training

---

## Implementation Details

### File Structure Created

```
prediction-agents-copy/
├── scripts/
│   ├── prepare_full_data.py       # Data preparation script
│   ├── train_full_dataset.py      # Multi-GPU training script
│   └── run_full_training.sh       # All-in-one launcher
├── config_full_dataset.json       # Hyperparameters
├── deepspeed_config.json          # DeepSpeed configuration
├── requirements_full_training.txt # Python dependencies
├── FULL_TRAINING_README.md        # Comprehensive guide
├── THUNDER_COMPUTE_DEPLOYMENT.md  # Deployment instructions
├── QUICK_START.md                 # Quick reference
├── SETUP_SUMMARY.md               # Setup overview
├── IMPLEMENTATION_DECISIONS.md    # This file
└── data/socsci210_full/           # Prepared dataset
    ├── train.jsonl (2.2GB)
    ├── val.jsonl (206MB)
    ├── test.jsonl (245MB)
    └── metadata.json
```

### Data Preparation

**Script**: `scripts/prepare_full_data.py`

**Process**:
1. Download SocSci210 from HuggingFace (socratesft/SocSci210)
2. Format into Alpaca instruction-following format:
   ```json
   {
     "instruction": "You are a behavioral prediction model...",
     "input": "Demographics: ... Scenario: ...",
     "output": "Response: <answer>",
     "study_id": "...",
     "condition_num": ...,
     "sample_id": ...,
     "participant": ...
   }
   ```
3. Create study-level splits (170/20/20 studies)
4. Save as JSONL files with metadata

**Execution**:
- Completed successfully on December 21, 2024
- Processing time: ~30 minutes
- Output size: ~2.6GB total

**Verification**:
- ✅ All 2,901,390 examples processed
- ✅ Study-level splits verified (no overlap)
- ✅ Data format validated (Alpaca instruction format)
- ✅ Metadata saved with study assignments

### Training Script

**Script**: `scripts/train_full_dataset.py`

**Key Features**:
- **QLoRA**: 4-bit quantization (NF4) with LoRA adapters
- **DeepSpeed**: Multi-GPU distributed training
- **Gradient Checkpointing**: Memory efficiency
- **Data Parallelism**: Automatic across 8 GPUs
- **Evaluation**: Every 500 steps on validation set
- **Checkpointing**: Save every 500 steps, keep last 3

**Technical Highlights**:

```python
# QLoRA Configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# LoRA Configuration
peft_config = LoraConfig(
    r=32,                  # Rank
    lora_alpha=64,         # Scaling
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# Training Arguments
training_args = TrainingArguments(
    per_device_train_batch_size=8,
    gradient_accumulation_steps=16,
    num_train_epochs=1,
    learning_rate=1e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    logging_steps=10,
    save_steps=500,
    eval_steps=500,
    deepspeed="deepspeed_config.json",
    bf16=True,
    gradient_checkpointing=True,
    save_total_limit=3
)
```

### Launcher Script

**Script**: `scripts/run_full_training.sh`

**Functionality**:
- Environment validation (CUDA, GPUs, HF token)
- Data verification (files exist, correct format)
- GPU count detection
- DeepSpeed launch
- Cost estimation display
- WandB integration (optional)
- Resume from checkpoint support

**Usage**:
```bash
# Standard launch
bash scripts/run_full_training.sh --wandb

# Resume from checkpoint
bash scripts/run_full_training.sh --resume models/socrates-qwen-full-dataset/checkpoint-2000

# With data preparation
bash scripts/run_full_training.sh --prepare-data --wandb
```

---

## Expected Outcomes

### Performance Targets

**Baseline (1% dataset)**:
- Correlation with human responses: 81.5%
- Mean Absolute Error (MAE): 3.72
- Training examples: 29K

**Expected (100% dataset)**:
- Correlation: >85% (conservative estimate)
- MAE: <3.5 (conservative estimate)
- Training examples: 2.9M

**Improvement Factors**:
- 100x more training data
- 2x more model capacity (LoRA rank)
- 4x larger batch size
- Better generalization from more studies

### Training Timeline

**Phase 1: Setup** (1-2 hours)
- Instance provisioning
- Environment setup
- Dependency installation
- Data transfer (~20GB)
- HuggingFace token configuration

**Phase 2: Model Download** (30-60 minutes)
- Download Qwen2.5-14B-Instruct (~28GB)
- Cache tokenizer and model

**Phase 3: Training** (25-30 hours)
- ~2,800 training steps
- ~280 checkpoints saved
- ~280 validation evaluations
- Total GPU-hours: 200-240 (8 GPUs × 25-30 hrs)

**Phase 4: Evaluation** (1-2 hours)
- Test set evaluation
- Metrics computation
- Results analysis

**Total**: ~28-35 hours from start to finish

### Resource Requirements

**Compute**:
- 8x NVIDIA A100 80GB GPUs
- ~70-75GB GPU memory per device
- ~100-150GB system RAM

**Storage**:
- 28GB: Qwen base model
- 15GB: Training data
- 100GB: Checkpoints (grows during training)
- 5GB: Environment/dependencies
- **Total**: ~150GB disk space

**Network**:
- 20GB upload (data + scripts)
- 5GB download (final model)
- Inter-GPU: High-bandwidth NVLink

---

## Deployment Plan

### Thunder Compute Deployment

**Steps**:

1. **Create Instance**
   - 8x A100 80GB GPUs
   - Ubuntu 22.04 LTS
   - 500GB SSD storage
   - Region: US (lowest latency)

2. **Transfer Files**
   - Option A: SCP (~20GB)
   - Option B: rsync (resumable)
   - Option C: Cloud storage (S3/GCS)

3. **Environment Setup**
   ```bash
   # Install Python dependencies
   pip install -r requirements_full_training.txt

   # Set HuggingFace token
   export HF_TOKEN="your_token_here"

   # Login to WandB
   wandb login
   ```

4. **Launch Training**
   ```bash
   # In tmux session (persist through SSH disconnects)
   tmux new -s training
   bash scripts/run_full_training.sh --wandb
   ```

5. **Monitor Progress**
   - WandB dashboard (real-time metrics)
   - `nvidia-smi` (GPU usage)
   - `ls models/*/checkpoint-*` (saved checkpoints)

6. **Download Results**
   ```bash
   # After training completes
   rsync -avz ubuntu@instance:~/socrates-training/models/ ./models/
   ```

### Pre-Launch Checklist

- [ ] Thunder Compute instance created
- [ ] SSH access verified
- [ ] Python environment set up
- [ ] Dependencies installed (including DeepSpeed)
- [ ] Data transferred (~20GB)
- [ ] HuggingFace token configured
- [ ] WandB configured (optional)
- [ ] 8 GPUs detected (`nvidia-smi`)
- [ ] Sufficient disk space (~500GB)
- [ ] tmux/screen installed

---

## Troubleshooting Guide

### Common Issues

**1. Out of Memory (OOM)**
```json
// Reduce per-device batch size in config_full_dataset.json
"per_device_train_batch_size": 4,
"gradient_accumulation_steps": 32
```

**2. NCCL Timeout (Multi-GPU Communication)**
```bash
export NCCL_TIMEOUT=7200
export NCCL_DEBUG=INFO  # For debugging
```

**3. Slow Model Download**
```bash
# Pre-download model
python -c "from transformers import AutoTokenizer; \
           AutoTokenizer.from_pretrained('Qwen/Qwen2.5-14B-Instruct')"
```

**4. Training Interrupted**
```bash
# Resume from last checkpoint
bash scripts/run_full_training.sh --resume models/socrates-qwen-full-dataset/checkpoint-2500
```

**5. Disk Space Issues**
```json
// Keep fewer checkpoints in config_full_dataset.json
"save_total_limit": 2  // Only keep last 2 checkpoints
```

---

## Evaluation Strategy

### Metrics to Track

**Primary Metrics** (match paper):
- Pearson correlation with human responses
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

**Secondary Metrics**:
- Per-study correlation (generalization)
- Per-question-type accuracy
- Calibration (predicted vs actual distributions)

### Evaluation Script

```bash
python scripts/evaluate.py \
    --model models/socrates-qwen-full-dataset \
    --data data/socsci210_full/test.jsonl \
    --output results/full_dataset_eval.json
```

**Expected Output**:
```json
{
  "correlation": 0.85,
  "mae": 3.4,
  "rmse": 4.2,
  "n_examples": 243896,
  "per_study_correlations": {...}
}
```

### Comparison to Baseline

| Metric | 1% Baseline | Full Dataset | Improvement |
|--------|-------------|--------------|-------------|
| Correlation | 81.5% | >85% | +3.5% |
| MAE | 3.72 | <3.5 | -0.22 |
| RMSE | ~4.5 | <4.2 | -0.3 |
| Training Data | 29K | 2.9M | 100x |
| Training Time | 2 hours | 25-30 hours | 12-15x |
| Cost | ~$2 | ~$220-264 | 110-132x |

---

## Future Improvements

### Potential Next Steps

1. **Multi-Epoch Training**
   - Try 2-3 epochs if 1 epoch shows good convergence
   - Monitor for overfitting with validation loss

2. **Hyperparameter Tuning**
   - Learning rate sweep (5e-6, 1e-5, 2e-5)
   - LoRA rank variations (16, 32, 64)
   - Batch size experiments (512, 1024, 2048)

3. **Model Ablations**
   - Different base models (Llama 3, Mistral)
   - Full fine-tuning vs QLoRA comparison
   - Different LoRA target modules

4. **Data Augmentation**
   - Paraphrasing scenarios
   - Synthetic demographic variations
   - Cross-study transfer learning

5. **Evaluation Extensions**
   - Human evaluation on held-out studies
   - Adversarial examples
   - Out-of-distribution generalization

---

## References

**Paper**: SOCRATES: Finetuning LLMs for Human Behavior Prediction in Social Science
- arXiv: 2509.05830
- Authors: [From paper]
- Published: 2024

**Dataset**: SocSci210
- HuggingFace: socratesft/SocSci210
- Size: 2.9M examples from 210+ studies
- License: [As per HuggingFace]

**Base Model**: Qwen2.5-14B-Instruct
- HuggingFace: Qwen/Qwen2.5-14B-Instruct
- Size: 14B parameters
- License: Apache 2.0

**Infrastructure**: Thunder Compute
- Website: thundercompute.com
- Pricing: $1.10/GPU/hr for A100 80GB
- Region: US

---

## Decision Log

This section tracks key decisions made during planning:

1. **Hardware**: 8x A100 80GB (not 4x) - for faster training
2. **Epochs**: Start with 1 (not 2-3) - conservative, can increase
3. **LoRA Rank**: 32 (not 16 or 64) - balanced capacity increase
4. **Batch Size**: 1024 (not 512 or 2048) - optimal for 8 GPUs
5. **Data Splits**: Study-level (not random) - matches paper
6. **DeepSpeed**: ZeRO-2 (not ZeRO-3) - sufficient for model size
7. **Platform**: Thunder Compute (not Lambda/AWS) - best pricing
8. **Monitoring**: WandB (not TensorBoard) - better for distributed training

---

**Document Created**: December 21, 2024
**Last Updated**: December 21, 2024
**Status**: Data preparation complete, ready for deployment
**Next Step**: Transfer to Thunder Compute and launch training
