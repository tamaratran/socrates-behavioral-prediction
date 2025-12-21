#!/usr/bin/env python3
"""
Finetuning Script for SOCRATES Replication
Uses QLoRA to finetune Llama 3 or Qwen 2.5 on SocSci210 dataset
"""

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def format_prompt(example):
    """Format example as a prompt for instruction finetuning"""
    return f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""


def prepare_training_data(data_dir: str):
    """Load and format training data"""
    print(f"Loading data from {data_dir}...")

    # Load datasets
    train_dataset = load_dataset('json', data_files=f"{data_dir}/train.jsonl", split="train")
    val_dataset = load_dataset('json', data_files=f"{data_dir}/val.jsonl", split="train")

    print(f"Train examples: {len(train_dataset)}")
    print(f"Val examples: {len(val_dataset)}")

    return train_dataset, val_dataset


def setup_model_and_tokenizer(model_name: str, use_qlora: bool = True):
    """
    Load model and tokenizer with QLoRA configuration

    Args:
        model_name: HuggingFace model name (e.g., "Qwen/Qwen2.5-14B-Instruct")
        use_qlora: Whether to use 4-bit quantization with LoRA
    """
    print(f"Loading model: {model_name}")
    print(f"QLoRA: {use_qlora}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Set padding token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Configure quantization for QLoRA
    if use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

    print(f"Model loaded. Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    return model, tokenizer


def setup_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: list = None
):
    """
    Configure LoRA parameters

    Args:
        r: LoRA rank
        lora_alpha: LoRA alpha scaling
        lora_dropout: Dropout rate
        target_modules: Which modules to apply LoRA to (None = auto-detect)
    """
    if target_modules is None:
        # Common target modules for Llama/Qwen architectures
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    return lora_config


def finetune(
    model_name: str = "Qwen/Qwen2.5-14B-Instruct",
    data_dir: str = "data/socsci210_1pct",
    output_dir: str = "models/socrates-qwen-1pct",
    use_qlora: bool = True,
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    max_seq_length: int = 2048,
    gradient_accumulation_steps: int = 4,
    warmup_steps: int = 10,
    save_steps: int = 100,
    logging_steps: int = 10,
    use_wandb: bool = False,
):
    """
    Main finetuning function

    Args:
        model_name: HuggingFace model name
        data_dir: Directory containing prepared data
        output_dir: Directory to save finetuned model
        use_qlora: Whether to use QLoRA
        epochs: Number of training epochs
        batch_size: Per-device batch size
        learning_rate: Learning rate
        max_seq_length: Maximum sequence length
        gradient_accumulation_steps: Gradient accumulation steps
        warmup_steps: Number of warmup steps
        save_steps: Save checkpoint every N steps
        logging_steps: Log every N steps
        use_wandb: Whether to use Weights & Biases for logging
    """
    # Load data
    train_dataset, val_dataset = prepare_training_data(data_dir)

    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(model_name, use_qlora)

    # Setup LoRA
    if use_qlora:
        lora_config = setup_lora_config()
        model = get_peft_model(model, lora_config)
        print(f"LoRA applied. Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        model.print_trainable_parameters()

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=save_steps,
        warmup_steps=warmup_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        evaluation_strategy="steps",
        save_strategy="steps",
        bf16=True,  # Use bfloat16 for training
        optim="paged_adamw_8bit" if use_qlora else "adamw_torch",
        report_to="wandb" if use_wandb else "none",
        run_name=f"socrates-{model_name.split('/')[-1]}-{Path(data_dir).name}",
        remove_unused_columns=False,
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        formatting_func=format_prompt,
        packing=False,  # Don't pack sequences
    )

    # Train
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50 + "\n")

    trainer.train()

    # Save final model
    print(f"\nSaving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save training metadata
    metadata = {
        'model_name': model_name,
        'data_dir': data_dir,
        'use_qlora': use_qlora,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'max_seq_length': max_seq_length,
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
    }

    with open(Path(output_dir) / "training_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "="*50)
    print("Training complete!")
    print(f"Model saved to: {output_dir}")
    print("="*50 + "\n")

    print("Next steps:")
    print(f"  1. Evaluate model: python scripts/evaluate.py --model {output_dir}")
    print(f"  2. Test inference: python scripts/test_inference.py --model {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Finetune LLM on SocSci210 dataset using QLoRA"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-14B-Instruct",
        help="Base model to finetune (default: Qwen/Qwen2.5-14B-Instruct)"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/socsci210_1pct",
        help="Directory containing prepared data"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/socrates-qwen-1pct",
        help="Output directory for finetuned model"
    )
    parser.add_argument(
        "--use-qlora",
        action="store_true",
        default=True,
        help="Use QLoRA for memory-efficient training"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per-device batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Use Weights & Biases for logging"
    )

    args = parser.parse_args()

    finetune(
        model_name=args.model,
        data_dir=args.data,
        output_dir=args.output,
        use_qlora=args.use_qlora,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        use_wandb=args.use_wandb,
    )


if __name__ == "__main__":
    main()
