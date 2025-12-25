---
base_model: Qwen/Qwen2.5-14B-Instruct
library_name: peft
tags:
- socrates
- behavioral-prediction
- lora
- qwen
- test-checkpoint
license: apache-2.0
---

# SOCRATES QLoRA Test Checkpoint (10 steps)

This is an early test checkpoint from fine-tuning Qwen2.5-14B-Instruct using QLoRA for behavioral prediction tasks, based on the SOCRATES paper methodology.

⚠️ **Note**: This is a test checkpoint with only 10 training steps - not intended for production use. It serves as a validation checkpoint for the training infrastructure.

## Model Details

### Model Description

This checkpoint contains a QLoRA (Quantized Low-Rank Adaptation) adapter fine-tuned on the Qwen2.5-14B-Instruct base model. The model is being trained to predict human behavioral outcomes based on scenario descriptions, following the SOCRATES (Systematic Cognitive Reasoning for Adaptive Task Execution in Scenarios) methodology.

- **Developed by:** Tamara Tran
- **Model type:** Causal Language Model with LoRA adapter
- **Language:** English
- **Base model:** Qwen/Qwen2.5-14B-Instruct (14B parameters)
- **Fine-tuning method:** QLoRA (4-bit quantization + LoRA)
- **License:** Apache 2.0
- **Checkpoint:** Step 10 (early test checkpoint)

### Repository Links

- **GitHub:** [tamaratran/socrates-behavioral-prediction](https://github.com/tamaratran/socrates-behavioral-prediction)

## Training Details

### Training Configuration

**LoRA Hyperparameters:**
- Rank (r): 32
- Alpha: 64
- Dropout: 0.05
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- Trainable parameters: ~263M (LoRA adapters only)

**Training Setup:**
- Training steps: 10 (test run)
- Epochs: 1.0
- Batch size: 2
- Learning rate: 1e-5 (with cosine decay)
- Precision: bfloat16
- Framework: DeepSpeed ZeRO-2

**Hardware:**
- GPUs: 4x A100 80GB (Thunder Compute)
- Distributed training: DeepSpeed

### Training Results

**Final Metrics (Step 10):**
- Training loss: 4.648
- Evaluation loss: 4.582
- Mean token accuracy: 33.3%
- Entropy: 1.961 bits/token
- Gradient norm: 9.09

**Training Progression:**
| Step | Loss  | Eval Loss | Token Accuracy | Learning Rate |
|------|-------|-----------|----------------|---------------|
| 1    | 5.282 | -         | 27.8%          | 0.0           |
| 5    | 5.175 | -         | 27.8%          | 7.5e-6        |
| 10   | 4.648 | 4.582     | 33.3%          | 1.25e-6       |

The model showed steady loss reduction over 10 steps, with token accuracy improving from 27.8% to 33.3%.

### Training Data

The model was trained on behavioral prediction scenarios derived from the SOCRATES dataset, focusing on predicting human decisions and outcomes in various social and cognitive scenarios.

## Usage

### Loading the Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-14B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(
    base_model,
    "tamaratran/socrates-qwen-test-checkpoint"
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("tamaratran/socrates-qwen-test-checkpoint")

# Inference
messages = [
    {"role": "user", "content": "Your behavioral prediction prompt here"}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Merging Adapter (Optional)

```python
# Merge LoRA weights with base model
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./merged_model")
```

## Limitations

- **Early checkpoint**: Only 10 training steps - not representative of final model performance
- **Limited training**: Model has not converged and requires significantly more training
- **Test purpose only**: This checkpoint is primarily for infrastructure validation
- **Base model limitations**: Inherits biases and limitations from Qwen2.5-14B-Instruct

## Intended Use

This checkpoint is intended for:
- Testing and validating the training pipeline
- Demonstrating the checkpoint upload process
- Early-stage model development and debugging
- **Not for production use or deployment**

## Technical Specifications

### Model Architecture
- **Base:** Qwen2.5-14B-Instruct (14B parameters)
- **Adapter:** LoRA (Low-Rank Adaptation)
- **Quantization:** 4-bit (QLoRA)
- **Precision:** bfloat16

### Compute Infrastructure

**Hardware:**
- 4x NVIDIA A100 80GB GPUs
- Platform: Thunder Compute

**Software:**
- Transformers: 4.x
- PEFT: 0.7.1
- DeepSpeed: Latest
- PyTorch: 2.x
- CUDA: 12.x

## Citation

If you use this work, please cite:

```bibtex
@software{socrates_qwen_checkpoint,
  author = {Tran, Tamara},
  title = {SOCRATES QLoRA Test Checkpoint},
  year = {2025},
  url = {https://huggingface.co/tamaratran/socrates-qwen-test-checkpoint}
}
```

## Contact

For questions or issues, please open an issue on the [GitHub repository](https://github.com/tamaratran/socrates-behavioral-prediction).

---

**Framework versions:**
- PEFT: 0.7.1
- Transformers: 4.x
- PyTorch: 2.x
