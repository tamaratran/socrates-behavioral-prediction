#!/usr/bin/env python3
"""
Full Dataset Preparation Script for SOCRATES
Downloads and preprocesses the complete SocSci210 dataset (2.9M examples)
Uses official dataset splits for train/val/test
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

    # Pre-format the full text for training (avoids GPU-based formatting later)
    formatted_text = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output_text}"""

    return {
        'instruction': instruction,
        'input': input_text,
        'output': output_text,
        'text': formatted_text,  # Pre-formatted field for efficient training
        'study_id': example.get('study_id', ''),
        'condition_num': example.get('condition_num', ''),
        'sample_id': example.get('sample_id', ''),
        'participant': example.get('participant', ''),
    }


def prepare_full_dataset(
    output_dir: str = "data/socsci210_full",
    seed: int = 42,
    use_official_splits: bool = True,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1
):
    """
    Download and prepare the FULL SocSci210 dataset (2.9M examples)

    Args:
        output_dir: Directory to save processed data
        seed: Random seed for reproducibility
        use_official_splits: Use dataset's official split methodology
        train_ratio: Training ratio if creating custom splits
        val_ratio: Validation ratio if creating custom splits
    """
    print("="*80)
    print("FULL SOCSCI210 DATASET PREPARATION")
    print("="*80)
    print(f"Loading complete SocSci210 dataset (2.9M examples)...")

    # Load full dataset from Hugging Face
    dataset = load_dataset("socratesft/SocSci210", split="train")

    print(f"✓ Dataset loaded: {len(dataset):,} examples")
    print(f"  Memory usage: ~{len(dataset) * 0.5 / 1024:.1f} MB estimated\n")

    # Format all examples
    print("Formatting examples for instruction finetuning...")
    formatted_data = []
    failed_count = 0

    for example in tqdm(dataset, desc="Formatting"):
        try:
            formatted = format_training_example(example)
            formatted_data.append(formatted)
        except Exception as e:
            failed_count += 1
            if failed_count < 10:  # Only print first few errors
                print(f"  Warning: Error formatting example: {e}")
            continue

    print(f"✓ Successfully formatted {len(formatted_data):,} examples")
    if failed_count > 0:
        print(f"  ⚠ {failed_count} examples failed to format")

    # Create DataFrame
    print("\nCreating DataFrame...")
    df = pd.DataFrame(formatted_data)

    # Create train/val/test splits
    print("Creating splits...")

    if use_official_splits:
        # STUDY-LEVEL SPLIT (SOCRATES paper methodology)
        print("Using study-level split (SOCRATES paper Section 5.3)")

        import numpy as np
        unique_studies = df['study_id'].unique()
        n_studies = len(unique_studies)
        print(f"  Found {n_studies} unique studies")

        # Shuffle studies
        np.random.seed(seed)
        shuffled_studies = unique_studies.copy()
        np.random.shuffle(shuffled_studies)

        # Paper uses 170 train / 20 val / 20 test (from 210 total)
        # Scale to actual number of studies in full dataset
        paper_train_ratio = 170 / 210  # 0.81
        paper_val_ratio = 20 / 210      # 0.095

        n_train_studies = int(n_studies * paper_train_ratio)
        n_val_studies = int(n_studies * paper_val_ratio)

        train_studies = set(shuffled_studies[:n_train_studies])
        val_studies = set(shuffled_studies[n_train_studies:n_train_studies + n_val_studies])
        test_studies = set(shuffled_studies[n_train_studies + n_val_studies:])

        print(f"  Study distribution:")
        print(f"    Train: {len(train_studies)} studies ({len(train_studies)/n_studies*100:.1f}%)")
        print(f"    Val:   {len(val_studies)} studies ({len(val_studies)/n_studies*100:.1f}%)")
        print(f"    Test:  {len(test_studies)} studies ({len(test_studies)/n_studies*100:.1f}%)")

        # Split dataframe by study membership
        train_df = df[df['study_id'].isin(train_studies)].reset_index(drop=True)
        val_df = df[df['study_id'].isin(val_studies)].reset_index(drop=True)
        test_df = df[df['study_id'].isin(test_studies)].reset_index(drop=True)

        # Verify no overlap
        train_study_set = set(train_df['study_id'].unique())
        val_study_set = set(val_df['study_id'].unique())
        test_study_set = set(test_df['study_id'].unique())

        assert len(train_study_set & val_study_set) == 0, "ERROR: Studies overlap between train and val!"
        assert len(train_study_set & test_study_set) == 0, "ERROR: Studies overlap between train and test!"
        assert len(val_study_set & test_study_set) == 0, "ERROR: Studies overlap between val and test!"

        print("  ✓ Verified: Zero study overlap between splits")

    else:
        # RANDOM SPLIT (for comparison/ablation only)
        print("Using random split (NOT paper methodology)")
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

        train_size = int(len(df) * train_ratio)
        val_size = int(len(df) * val_ratio)

        train_df = df[:train_size]
        val_df = df[train_size:train_size + val_size]
        test_df = df[train_size + val_size:]

    print(f"\n{'='*80}")
    print("SPLIT SIZES")
    print(f"{'='*80}")
    print(f"  Train: {len(train_df):>10,} examples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val:   {len(val_df):>10,} examples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(test_df):>10,} examples ({len(test_df)/len(df)*100:.1f}%)")
    print(f"  Total: {len(df):>10,} examples")
    print(f"{'='*80}\n")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save to JSONL files
    print(f"Saving to {output_dir}...")

    print("  Writing train.jsonl...")
    train_df.to_json(output_path / "train.jsonl", orient="records", lines=True)

    print("  Writing val.jsonl...")
    val_df.to_json(output_path / "val.jsonl", orient="records", lines=True)

    print("  Writing test.jsonl...")
    test_df.to_json(output_path / "test.jsonl", orient="records", lines=True)

    # Save metadata
    metadata = {
        'dataset_name': 'SocSci210',
        'dataset_source': 'socratesft/SocSci210',
        'total_examples': len(formatted_data),
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df),
        'seed': seed,
        'use_official_splits': use_official_splits,
        'split_methodology': 'study_level' if use_official_splits else 'random',
        'paper_reference': 'arxiv:2509.05830'
    }

    if use_official_splits:
        metadata['train_studies'] = sorted(list(train_df['study_id'].unique()))
        metadata['val_studies'] = sorted(list(val_df['study_id'].unique()))
        metadata['test_studies'] = sorted(list(test_df['study_id'].unique()))
        metadata['n_train_studies'] = len(metadata['train_studies'])
        metadata['n_val_studies'] = len(metadata['val_studies'])
        metadata['n_test_studies'] = len(metadata['test_studies'])

    print("  Writing metadata.json...")
    with open(output_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*80}")
    print("DATASET PREPARATION COMPLETE!")
    print(f"{'='*80}")
    print(f"Files saved to: {output_dir}")
    print(f"  - train.jsonl ({len(train_df):,} examples)")
    print(f"  - val.jsonl ({len(val_df):,} examples)")
    print(f"  - test.jsonl ({len(test_df):,} examples)")
    print(f"  - metadata.json")
    print(f"{'='*80}\n")

    print("Next steps:")
    print(f"  1. Review data: head -n 1 {output_dir}/train.jsonl | jq")
    print(f"  2. Start training: python scripts/train_full_dataset.py")
    print()

    # Show example
    print("Example formatted entry:")
    print(json.dumps(formatted_data[0], indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Prepare FULL SocSci210 dataset (2.9M examples) for finetuning"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/socsci210_full",
        help="Output directory for processed data"
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
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training ratio for random split (default: 0.8)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation ratio for random split (default: 0.1)"
    )

    args = parser.parse_args()

    prepare_full_dataset(
        output_dir=args.output,
        seed=args.seed,
        use_official_splits=not args.random_split,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio
    )


if __name__ == "__main__":
    main()
