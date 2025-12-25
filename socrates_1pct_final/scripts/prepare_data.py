#!/usr/bin/env python3
"""
Data Preparation Script for SOCRATES Finetuning
Downloads and preprocesses the SocSci210 dataset for behavior prediction
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


def format_demographics(demo: Dict[str, Any]) -> str:
    """Format demographic information as a readable string"""
    demo_text = []

    # Key demographic fields
    if 'age' in demo and demo['age']:
        demo_text.append(f"Age: {demo['age']}")
    if 'gender' in demo and demo['gender']:
        demo_text.append(f"Gender: {demo['gender']}")
    if 'education' in demo and demo['education']:
        demo_text.append(f"Education: {demo['education']}")
    if 'income' in demo and demo['income']:
        demo_text.append(f"Income: {demo['income']}")
    if 'employment' in demo and demo['employment']:
        demo_text.append(f"Employment: {demo['employment']}")
    if 'party_id' in demo and demo['party_id']:
        demo_text.append(f"Political affiliation: {demo['party_id']}")
    if 'ideology' in demo and demo['ideology']:
        demo_text.append(f"Ideology: {demo['ideology']}")

    return "; ".join(demo_text) if demo_text else "No demographic information"


def format_training_example(example: Dict[str, Any]) -> Dict[str, str]:
    """
    Format a single example for instruction finetuning

    Returns a dict with 'instruction', 'input', and 'output' fields
    following the Alpaca format
    """
    # Format demographics
    demo_str = format_demographics(example['demographic'])

    # Extract stimuli (the experimental scenario/question)
    stimuli = example['stimuli'].strip()

    # Extract response
    response = str(example['response'])

    # Create instruction-following format
    instruction = (
        "You are a behavioral prediction model. Given demographic information about a participant "
        "and an experimental scenario, predict how they would respond."
    )

    input_text = f"Demographics: {demo_str}\n\nScenario: {stimuli}"

    output_text = f"Response: {response}"

    return {
        'instruction': instruction,
        'input': input_text,
        'output': output_text,
        'study_id': example.get('study_id', ''),
        'condition_num': example.get('condition_num', ''),
    }


def prepare_dataset(
    subset_fraction: float = 0.01,
    output_dir: str = "data/socsci210",
    train_split: float = 0.8,
    seed: int = 42,
    study_level_split: bool = True
):
    """
    Download and prepare the SocSci210 dataset

    Args:
        subset_fraction: Fraction of dataset to use (0.01 = 1%, 1.0 = 100%)
        output_dir: Directory to save processed data
        train_split: Fraction for training (rest is split between val/test)
        seed: Random seed for reproducibility
        study_level_split: If True, split by study_id (as in SOCRATES paper).
                          If False, split randomly across all examples.
    """
    print(f"Loading SocSci210 dataset...")
    print(f"Using {subset_fraction*100}% of the data")

    # Load dataset from Hugging Face
    dataset = load_dataset("socratesft/SocSci210", split="train")

    print(f"Full dataset size: {len(dataset)} examples")

    # Take subset if requested
    if subset_fraction < 1.0:
        num_examples = int(len(dataset) * subset_fraction)
        dataset = dataset.shuffle(seed=seed).select(range(num_examples))
        print(f"Using subset: {len(dataset)} examples")

    # Format all examples
    print("Formatting examples...")
    formatted_data = []
    for example in tqdm(dataset):
        try:
            formatted = format_training_example(example)
            formatted_data.append(formatted)
        except Exception as e:
            print(f"Error formatting example: {e}")
            continue

    print(f"Successfully formatted {len(formatted_data)} examples")

    # Create train/val/test splits
    print("Creating splits...")
    df = pd.DataFrame(formatted_data)

    if study_level_split:
        # STUDY-LEVEL SPLIT (as in SOCRATES paper Section 5.3)
        # Split by study_id to ensure complete separation of experimental scenarios
        print("Using study-level split (SOCRATES paper methodology)")

        import numpy as np
        unique_studies = df['study_id'].unique()
        n_studies = len(unique_studies)
        print(f"Found {n_studies} unique studies")

        # Shuffle studies
        np.random.seed(seed)
        np.random.shuffle(unique_studies)

        # Paper uses 170 train / 40 test split (170/210 = 0.81)
        # Apply same ratio regardless of subset size
        train_ratio = 170 / 210  # ≈ 0.81
        n_train_studies = int(n_studies * train_ratio)
        n_val_studies = int((n_studies - n_train_studies) / 2)

        train_studies = set(unique_studies[:n_train_studies])
        val_studies = set(unique_studies[n_train_studies:n_train_studies + n_val_studies])
        test_studies = set(unique_studies[n_train_studies + n_val_studies:])

        print(f"Study split: {len(train_studies)} train, {len(val_studies)} val, {len(test_studies)} test")

        # Split dataframe by study membership
        train_df = df[df['study_id'].isin(train_studies)].reset_index(drop=True)
        val_df = df[df['study_id'].isin(val_studies)].reset_index(drop=True)
        test_df = df[df['study_id'].isin(test_studies)].reset_index(drop=True)

        # Verify no overlap
        train_study_set = set(train_df['study_id'].unique())
        test_study_set = set(test_df['study_id'].unique())
        overlap = train_study_set & test_study_set
        assert len(overlap) == 0, f"ERROR: {len(overlap)} studies overlap between train and test!"
        print("✓ Verified: Zero study overlap between train and test")

    else:
        # RANDOM SPLIT (NOT paper methodology - for comparison only)
        print("Using random split (NOT paper methodology)")
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

        train_size = int(len(df) * train_split)
        val_size = int(len(df) * (1 - train_split) / 2)

        train_df = df[:train_size]
        val_df = df[train_size:train_size + val_size]
        test_df = df[train_size + val_size:]

    print(f"Example split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save to JSON files (format compatible with most finetuning tools)
    print(f"Saving to {output_dir}...")

    train_df.to_json(output_path / "train.jsonl", orient="records", lines=True)
    val_df.to_json(output_path / "val.jsonl", orient="records", lines=True)
    test_df.to_json(output_path / "test.jsonl", orient="records", lines=True)

    # Save metadata
    metadata = {
        'total_examples': len(formatted_data),
        'subset_fraction': subset_fraction,
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df),
        'train_split': train_split,
        'seed': seed,
        'source_dataset': 'socratesft/SocSci210',
        'study_level_split': study_level_split
    }

    if study_level_split:
        metadata['train_studies'] = sorted(list(train_df['study_id'].unique()))
        metadata['val_studies'] = sorted(list(val_df['study_id'].unique()))
        metadata['test_studies'] = sorted(list(test_df['study_id'].unique()))
        metadata['n_train_studies'] = len(metadata['train_studies'])
        metadata['n_val_studies'] = len(metadata['val_studies'])
        metadata['n_test_studies'] = len(metadata['test_studies'])

    with open(output_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\nDataset preparation complete!")
    print(f"Files saved to: {output_dir}")
    print("\nNext steps:")
    print(f"  1. Review the data: cat {output_dir}/train.jsonl | head -1 | jq")
    print(f"  2. Start finetuning: python scripts/finetune.py --data {output_dir}")

    # Show example
    print("\nExample formatted entry:")
    print(json.dumps(formatted_data[0], indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Prepare SocSci210 dataset for finetuning"
    )
    parser.add_argument(
        "--subset",
        type=float,
        default=0.01,
        help="Fraction of dataset to use (default: 0.01 = 1%%)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/socsci210_1pct",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Training split ratio (default: 0.8)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--random-split",
        action="store_true",
        help="Use random split instead of study-level split (NOT recommended)"
    )

    args = parser.parse_args()

    prepare_dataset(
        subset_fraction=args.subset,
        output_dir=args.output,
        train_split=args.train_split,
        seed=args.seed,
        study_level_split=not args.random_split
    )


if __name__ == "__main__":
    main()
