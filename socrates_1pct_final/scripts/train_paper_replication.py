#!/usr/bin/env python3
"""
SOCRATES Paper Replication Training Script
Exact implementation of arxiv:2509.05830 methodology
"""

import json
import argparse
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


def load_config(config_path="config_paper.json"):
    """Load training configuration"""
    with open(config_path) as f:
        return json.load(f)


def format_prompt(example):
    """Format example following paper's prompt structure"""
    return f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""


def main():
    parser = argparse.ArgumentParser(description="SOCRATES Paper Replication Training")
    parser.add_argument("--config", default="config_paper.json", help="Config file path")
    parser.add_argument("--use-wandb", action="store_true", help="Use Weights & Biases logging")
    args = parser.parse_args()

    # Load configuration
    print("="*80)
    print("SOCRATES PAPER REPLICATION TRAINING")
    print("Paper: arxiv:2509.05830")
    print("="*80)

    config = load_config(args.config)
    print(f"\nConfiguration: {args.config}")
    print(f"Model: {config['model_name']}")
    print(f"Dataset: {config['data_dir']}")
    print(f"Output: {config['output_dir']}\n")

    # Load datasets
    print("Loading data...")
    train_dataset = load_dataset('json', data_files=f"{config['data_dir']}/train.jsonl", split="train")
    val_dataset = load_dataset('json', data_files=f"{config['data_dir']}/val.jsonl", split="train")
    print(f"✓ Train: {len(train_dataset)} examples")
    print(f"✓ Val: {len(val_dataset)} examples\n")

    # Load tokenizer
    print(f"Loading tokenizer: {config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print("✓ Tokenizer loaded\n")

    # Configure quantization (QLoRA)
    print("Configuring QLoRA...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model
    print(f"Loading model: {config['model_name']}")
    model = AutoModelForCausalLM.from_pretrained(
        config['model_name'],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    print("✓ Model loaded\n")

    # Configure LoRA
    print("Applying LoRA...")
    lora_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['lora_alpha'],
        target_modules=config['lora']['target_modules'],
        lora_dropout=config['lora']['lora_dropout'],
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print()

    # Training arguments (paper-exact)
    train_config = config['training']
    training_args = TrainingArguments(
        output_dir=config['output_dir'],
        num_train_epochs=train_config['num_epochs'],
        per_device_train_batch_size=train_config['per_device_train_batch_size'],
        per_device_eval_batch_size=train_config['per_device_train_batch_size'],
        gradient_accumulation_steps=train_config['gradient_accumulation_steps'],
        learning_rate=train_config['learning_rate'],
        weight_decay=train_config['weight_decay'],
        warmup_ratio=train_config['warmup_ratio'],
        lr_scheduler_type=train_config['lr_scheduler_type'],
        logging_steps=train_config['logging_steps'],
        save_steps=train_config['save_steps'],
        eval_steps=train_config['save_steps'],
        save_total_limit=3,
        load_best_model_at_end=True,
        eval_strategy="steps",
        save_strategy="steps",
        bf16=train_config['bf16'],
        optim=train_config['optim'],
        report_to="wandb" if args.use_wandb else "none",
        run_name=f"socrates-paper-replication-{Path(config['data_dir']).name}",
        remove_unused_columns=False,
    )

    # Create trainer
    print("Creating trainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        formatting_func=format_prompt,
    )
    print("✓ Trainer created\n")

    # Display training info
    print("="*80)
    print("TRAINING CONFIGURATION (Paper-Exact)")
    print("="*80)
    print(f"Epochs: {train_config['num_epochs']}")
    print(f"Global Batch Size: {train_config['global_batch_size']}")
    print(f"  → Per-device: {train_config['per_device_train_batch_size']}")
    print(f"  → Gradient accumulation: {train_config['gradient_accumulation_steps']}")
    print(f"Learning Rate: {train_config['learning_rate']}")
    print(f"LR Scheduler: {train_config['lr_scheduler_type']} (warmup: {train_config['warmup_ratio']})")
    print(f"Weight Decay: {train_config['weight_decay']}")
    print(f"Max Sequence Length: {train_config['max_seq_length']}")
    print(f"Optimizer: {train_config['optim']}")
    print("="*80)
    print()

    # Train
    print("Starting training...")
    print("="*80)
    trainer.train()

    # Save
    print("\n" + "="*80)
    print("Training complete! Saving model...")
    trainer.save_model(config['output_dir'])
    tokenizer.save_pretrained(config['output_dir'])

    # Save config and metadata
    with open(Path(config['output_dir']) / "training_config.json", 'w') as f:
        json.dump(config, f, indent=2)

    metadata = {
        'paper': 'arxiv:2509.05830',
        'model': config['model_name'],
        'train_examples': len(train_dataset),
        'val_examples': len(val_dataset),
        'epochs': train_config['num_epochs'],
        'global_batch_size': train_config['global_batch_size'],
    }
    with open(Path(config['output_dir']) / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Model saved to: {config['output_dir']}")
    print("="*80)
    print("\nNext steps:")
    print(f"  1. Evaluate: python scripts/evaluate.py --model {config['output_dir']}")
    print(f"  2. Test: python scripts/test_inference.py --model {config['output_dir']}")
    print("="*80)


if __name__ == "__main__":
    main()
