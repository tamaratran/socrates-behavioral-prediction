#!/usr/bin/env python3
"""
Full Dataset Training Script for SOCRATES
Optimized for multi-GPU training (8x A100 80GB) with DeepSpeed ZeRO-2
Trains on complete SocSci210 dataset (2.9M examples)
"""

import json
import argparse
import os
from pathlib import Path
from datetime import datetime
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


def load_config(config_path="config_full_dataset.json"):
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


def print_gpu_memory():
    """Print GPU memory usage"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"  GPU {i}: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")


def main():
    parser = argparse.ArgumentParser(description="Full Dataset Training with Multi-GPU")
    parser.add_argument("--config", default="config_full_dataset.json", help="Config file path")
    parser.add_argument("--use-wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--deepspeed", default="deepspeed_config.json", help="DeepSpeed config file")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    # Load configuration
    print("="*80)
    print("SOCRATES FULL DATASET TRAINING")
    print("Multi-GPU with DeepSpeed ZeRO-2")
    print("="*80)

    config = load_config(args.config)
    print(f"\nConfiguration: {args.config}")
    print(f"Model: {config['model_name']}")
    print(f"Dataset: {config['data_dir']}")
    print(f"Output: {config['output_dir']}")
    print(f"DeepSpeed: {args.deepspeed}\n")

    # Check GPU availability
    print("Hardware Configuration:")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print()

    # Load datasets
    print("Loading data...")
    train_dataset = load_dataset('json', data_files=f"{config['data_dir']}/train.jsonl", split="train")
    val_dataset = load_dataset('json', data_files=f"{config['data_dir']}/val.jsonl", split="train")
    print(f"✓ Train: {len(train_dataset):,} examples")
    print(f"✓ Val: {len(val_dataset):,} examples")

    # Calculate training stats
    train_config = config['training']
    steps_per_epoch = len(train_dataset) // train_config['global_batch_size']
    total_steps = steps_per_epoch * train_config['num_epochs']
    print(f"\nTraining Statistics:")
    print(f"  Steps per epoch: {steps_per_epoch:,}")
    print(f"  Total training steps: {total_steps:,}")
    print(f"  Global batch size: {train_config['global_batch_size']}")
    print(f"  Estimated time (1 step ≈ 3s): {total_steps * 3 / 3600:.1f} hours\n")

    # Load tokenizer
    print(f"Loading tokenizer: {config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print("✓ Tokenizer loaded\n")

    # Configure quantization (QLoRA)
    print("Configuring QLoRA (4-bit quantization)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model
    print(f"Loading model: {config['model_name']}")
    print("This may take several minutes...")
    model = AutoModelForCausalLM.from_pretrained(
        config['model_name'],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    print("✓ Model loaded\n")

    print("Initial GPU Memory:")
    print_gpu_memory()
    print()

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
    print("\nTrainable Parameters:")
    model.print_trainable_parameters()
    print()

    # Training arguments
    output_dir = config['output_dir']
    run_name = f"socrates-full-{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_config['num_epochs'],
        per_device_train_batch_size=train_config['per_device_train_batch_size'],
        per_device_eval_batch_size=train_config['per_device_eval_batch_size'],
        gradient_accumulation_steps=train_config['gradient_accumulation_steps'],
        learning_rate=train_config['learning_rate'],
        weight_decay=train_config['weight_decay'],
        warmup_ratio=train_config['warmup_ratio'],
        lr_scheduler_type=train_config['lr_scheduler_type'],
        logging_steps=train_config['logging_steps'],
        save_steps=train_config['save_steps'],
        eval_steps=train_config['eval_steps'],
        save_total_limit=5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        eval_strategy="steps",
        save_strategy="steps",
        bf16=train_config['bf16'],
        optim=train_config['optim'],
        report_to="wandb" if args.use_wandb else "none",
        run_name=run_name,
        remove_unused_columns=False,
        # Multi-GPU settings
        ddp_find_unused_parameters=False,
        # DeepSpeed
        deepspeed=args.deepspeed if os.path.exists(args.deepspeed) else None,
        # Gradient checkpointing for memory efficiency
        gradient_checkpointing=train_config.get('gradient_checkpointing', True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Data handling
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

    # Create trainer
    print("Creating SFT Trainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        formatting_func=format_prompt,
        max_seq_length=train_config['max_seq_length'],
    )
    print("✓ Trainer created\n")

    # Display training configuration
    print("="*80)
    print("TRAINING CONFIGURATION")
    print("="*80)
    print(f"Model: {config['model_name']}")
    print(f"Dataset: {len(train_dataset):,} train, {len(val_dataset):,} val")
    print(f"\nHyperparameters:")
    print(f"  Epochs: {train_config['num_epochs']}")
    print(f"  Global Batch Size: {train_config['global_batch_size']}")
    print(f"    → Per-device: {train_config['per_device_train_batch_size']}")
    print(f"    → Gradient accumulation: {train_config['gradient_accumulation_steps']}")
    print(f"    → Effective GPUs: {train_config['global_batch_size'] // (train_config['per_device_train_batch_size'] * train_config['gradient_accumulation_steps'])}")
    print(f"  Learning Rate: {train_config['learning_rate']}")
    print(f"  LR Scheduler: {train_config['lr_scheduler_type']}")
    print(f"  Warmup: {train_config['warmup_ratio']} ({int(total_steps * train_config['warmup_ratio'])} steps)")
    print(f"  Weight Decay: {train_config['weight_decay']}")
    print(f"  Max Sequence Length: {train_config['max_seq_length']}")
    print(f"  Optimizer: {train_config['optim']}")
    print(f"\nLoRA:")
    print(f"  Rank (r): {config['lora']['r']}")
    print(f"  Alpha: {config['lora']['lora_alpha']}")
    print(f"  Dropout: {config['lora']['lora_dropout']}")
    print(f"\nCheckpointing:")
    print(f"  Save every: {train_config['save_steps']} steps")
    print(f"  Eval every: {train_config['eval_steps']} steps")
    print(f"  Total checkpoints: ~{total_steps // train_config['save_steps']}")
    print("="*80)
    print()

    # Train
    print("Starting training...")
    print("="*80)
    start_time = datetime.now()

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    end_time = datetime.now()
    training_time = (end_time - start_time).total_seconds() / 3600

    # Save final model
    print("\n" + "="*80)
    print("Training complete!")
    print(f"Training time: {training_time:.2f} hours")
    print("="*80)

    print("\nSaving final model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✓ Model saved to: {output_dir}")

    # Save training metadata
    metadata = {
        'paper': 'arxiv:2509.05830',
        'model': config['model_name'],
        'dataset': 'socratesft/SocSci210',
        'train_examples': len(train_dataset),
        'val_examples': len(val_dataset),
        'epochs': train_config['num_epochs'],
        'global_batch_size': train_config['global_batch_size'],
        'total_steps': total_steps,
        'training_time_hours': round(training_time, 2),
        'lora_rank': config['lora']['r'],
        'learning_rate': train_config['learning_rate'],
        'run_name': run_name,
        'timestamp': datetime.now().isoformat(),
        'hardware': {
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'gpu_type': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'
        }
    }

    with open(Path(output_dir) / "training_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    with open(Path(output_dir) / "training_config.json", 'w') as f:
        json.dump(config, f, indent=2)

    print("✓ Metadata saved")

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Model saved to: {output_dir}")
    print(f"Training time: {training_time:.2f} hours")
    print(f"Steps completed: {total_steps:,}")
    print("\nNext steps:")
    print(f"  1. Evaluate: python scripts/evaluate.py --model {output_dir}")
    print(f"  2. Test inference: python scripts/test_inference.py --model {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
