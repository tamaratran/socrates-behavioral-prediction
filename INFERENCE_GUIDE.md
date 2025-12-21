# SOCRATES Model Inference Guide

## Quick Start

Your fine-tuned SOCRATES model is ready! Here's how to use it:

### 1. Model Location
```
models/socrates-qwen-paper-replication/
├── adapter_model.safetensors  (131MB - your fine-tuned LoRA adapter)
├── adapter_config.json
├── training_metadata.json     (created - contains base model reference)
└── ... (tokenizer files)
```

### 2. Run Inference

The inference scripts will automatically:
- Detect the LoRA adapter
- Download the base model (Qwen2.5-14B-Instruct, ~28GB) if not cached
- Load your fine-tuned adapter on top

#### Option A: Interactive Testing (Recommended)

**Test with predefined example:**
```bash
cd prediction-agents-copy
python scripts/test_inference.py \
  --model models/socrates-qwen-paper-replication \
  --example 1
```

**Interactive mode (prompt for input):**
```bash
python scripts/test_inference.py \
  --model models/socrates-qwen-paper-replication
```

**Custom prediction:**
```bash
python scripts/test_inference.py \
  --model models/socrates-qwen-paper-replication \
  --demographics "Age: 25-34; Gender: Female; Education: Bachelor's" \
  --scenario "Should the minimum wage be increased to $15/hour? (1-7 scale, 1=Strongly oppose, 7=Strongly support)"
```

#### Option B: Full Evaluation

Run evaluation on the test set (2,902 examples):
```bash
python scripts/evaluate.py \
  --model models/socrates-qwen-paper-replication \
  --test-data data/socsci210_1pct/test.jsonl \
  --output results/evaluation_results.json
```

This will generate:
- Metrics: MAE, RMSE, correlation
- Predictions file (JSON)
- Scatter plot visualization

### 3. Input Format

The model expects:
```
Demographics: Age: X; Gender: Y; Education: Z; ...

Scenario: [Question text and response options]
```

### 4. Output Format

The model generates text like:
```
Response: 3
```

Where the number corresponds to the response option in the scenario.

### 5. Expected Performance

Based on the SOCRATES paper (arxiv:2509.05830):
- **Distribution Alignment:** ~0.151 Wasserstein distance
- **Improvement vs Base:** 26% better
- **vs GPT-4o:** 13% better

### 6. Hardware Requirements

**For inference:**
- GPU with ~28-32GB VRAM (recommended)
  - NVIDIA A100, V100 32GB, or similar
- Or: CPU inference (much slower, 1-2 min per prediction)

**Memory breakdown:**
- Base model in bfloat16: ~28GB
- LoRA adapter: ~131MB
- Activation memory: ~2-4GB

### 7. Predefined Examples

The script includes 3 built-in examples:

**Example 1: Political Opinion**
- Demographics: 35-year-old male with Bachelor's
- Scenario: Support for federal food stamp spending
- Expected response: 1-7 scale

**Example 2: Economic Policy**
- Demographics: 28-year-old female with Master's
- Scenario: Tax increase on wealthy
- Expected response: 1-5 scale

**Example 3: Social Issue**
- Demographics: 42-year-old with high school education
- Scenario: Climate change concern
- Expected response: 1-7 scale

### 8. Troubleshooting

**"Out of memory" error:**
- The model requires significant GPU memory
- Try running on a machine with more VRAM
- Or modify the script to use 4-bit quantization (reduces quality slightly)

**Model download slow:**
- The base model is ~28GB
- First run will download it from Hugging Face
- Cached in `~/.cache/huggingface/` for future use

**Import errors:**
- Make sure you have all dependencies:
  ```bash
  pip install torch transformers peft accelerate datasets scikit-learn matplotlib
  ```

### 9. Advanced Usage

**Use with different base model versions:**
```bash
python scripts/test_inference.py \
  --model models/socrates-qwen-paper-replication \
  --base-model "Qwen/Qwen2.5-14B-Instruct" \
  --example 1
```

**Batch predictions from file:**
Create a JSONL file with your test cases, then use `evaluate.py`:
```bash
python scripts/evaluate.py \
  --model models/socrates-qwen-paper-replication \
  --test-data my_test_cases.jsonl
```

### 10. Model Details

**Training Details:**
- Paper: arxiv:2509.05830
- Base Model: Qwen/Qwen2.5-14B-Instruct
- Training examples: 23,210 (1% of SocSci210 dataset)
- Validation examples: 2,901
- Method: Supervised Fine-Tuning with QLoRA
- Epochs: 1
- Batch size: 256 (effective)
- Learning rate: 1e-05
- Optimizer: AdamW (PyTorch)
- Training time: 2 hours 9 minutes on A100 80GB

**Adapter Details:**
- Type: LoRA (Low-Rank Adaptation)
- Rank: 16
- Alpha: 32
- Dropout: 0.05
- Target modules: All attention and MLP layers
- Trainable params: 68.8M (0.46% of full model)

---

## Next Steps

1. **Test the model** with the interactive script
2. **Run evaluation** to see performance metrics
3. **Try custom scenarios** relevant to your use case
4. **Compare results** with the paper's reported performance

For questions or issues, refer to the paper: https://arxiv.org/abs/2509.05830
