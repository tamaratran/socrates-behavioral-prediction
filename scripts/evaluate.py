#!/usr/bin/env python3
"""
Evaluation Script for SOCRATES Finetuned Models
Compares finetuned model against base model on test set
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import seaborn as sns


def load_model_and_tokenizer(model_path: str, base_model: str = None):
    """
    Load finetuned model and tokenizer

    Args:
        model_path: Path to finetuned model (with LoRA adapters)
        base_model: Base model name (if loading LoRA adapters)
    """
    print(f"Loading model from {model_path}...")

    # Check if this is a LoRA model or full model
    model_path = Path(model_path)
    if (model_path / "adapter_config.json").exists():
        # This is a LoRA model
        if base_model is None:
            # Try to read base model from metadata
            metadata_path = model_path / "training_metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
                    base_model = metadata.get('model_name')

        if base_model is None:
            raise ValueError("LoRA model detected but base_model not specified and not found in metadata")

        print(f"Loading base model: {base_model}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        print(f"Loading LoRA adapters from: {model_path}")
        model = PeftModel.from_pretrained(model, str(model_path))
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    else:
        # This is a full model
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer


def format_prompt(example):
    """Format example as a prompt (same as in finetuning)"""
    return f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
"""


def predict_response(model, tokenizer, example, max_new_tokens=100):
    """
    Generate prediction for a single example

    Returns the predicted response text
    """
    prompt = format_prompt(example)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding for consistency
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the generated part (skip the prompt)
    generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    return generated.strip()


def extract_numeric_response(text: str) -> float:
    """
    Extract numeric value from response text

    Handles formats like:
    - "Response: 5"
    - "5"
    - "The answer is 3.5"
    """
    import re

    # Try to find numbers in the text
    numbers = re.findall(r'-?\d+\.?\d*', text)

    if numbers:
        return float(numbers[0])
    else:
        return None


def compute_wasserstein_distance(results: List[Dict]) -> Dict:
    """
    Compute Wasserstein distance as described in SOCRATES paper Section 3.2

    For each (study_id, condition_num) pair, compares the distribution of
    predicted responses vs actual responses. This measures how well the model
    captures the distribution of human behavior under each experimental condition.

    Returns:
        Dictionary with overall Wasserstein distance and per-study breakdown
    """
    from collections import defaultdict

    # Group predictions by (study_id, condition_num)
    condition_groups = defaultdict(lambda: {'true': [], 'pred': []})

    for result in results:
        if result['true_numeric'] is not None and result['pred_numeric'] is not None:
            study_id = result['study_id']
            condition = result['condition_num']
            key = (study_id, condition)

            condition_groups[key]['true'].append(result['true_numeric'])
            condition_groups[key]['pred'].append(result['pred_numeric'])

    # Compute Wasserstein distance for each condition
    distances = []
    per_study_distances = defaultdict(list)

    for (study_id, condition), responses in condition_groups.items():
        true_dist = np.array(responses['true'])
        pred_dist = np.array(responses['pred'])

        if len(true_dist) < 2 or len(pred_dist) < 2:
            # Need at least 2 samples for meaningful distribution comparison
            continue

        # Normalize distributions to [0, 1] as described in paper
        # (r - rmin) / (rmax - rmin)
        true_min, true_max = true_dist.min(), true_dist.max()
        pred_min, pred_max = pred_dist.min(), pred_dist.max()

        # Handle edge case where all values are the same
        if true_max - true_min > 0:
            true_normalized = (true_dist - true_min) / (true_max - true_min)
        else:
            true_normalized = np.zeros_like(true_dist)

        if pred_max - pred_min > 0:
            pred_normalized = (pred_dist - pred_min) / (pred_max - pred_min)
        else:
            pred_normalized = np.zeros_like(pred_dist)

        # Compute Wasserstein distance (Earth Mover's Distance)
        distance = wasserstein_distance(true_normalized, pred_normalized)
        distances.append(distance)
        per_study_distances[study_id].append(distance)

    # Average across all conditions
    overall_wasserstein = np.mean(distances) if distances else float('nan')

    # Average per study
    study_wasserstein = {}
    for study_id, study_distances in per_study_distances.items():
        study_wasserstein[study_id] = np.mean(study_distances)

    return {
        'wasserstein_distance': float(overall_wasserstein),
        'num_conditions_evaluated': len(distances),
        'num_studies_evaluated': len(per_study_distances),
        'per_study_wasserstein': {k: float(v) for k, v in study_wasserstein.items()},
    }


def evaluate_model(
    model_path: str,
    test_data_path: str,
    base_model: str = None,
    output_dir: str = None,
    max_examples: int = None,
):
    """
    Evaluate finetuned model on test set

    Args:
        model_path: Path to finetuned model
        test_data_path: Path to test data (JSONL file)
        base_model: Base model name (for LoRA models)
        output_dir: Directory to save results
        max_examples: Maximum number of examples to evaluate (None = all)
    """
    # Load model
    model, tokenizer = load_model_and_tokenizer(model_path, base_model)

    # Load test data
    print(f"Loading test data from {test_data_path}...")
    test_dataset = load_dataset('json', data_files=test_data_path, split="train")

    if max_examples:
        test_dataset = test_dataset.select(range(min(max_examples, len(test_dataset))))

    print(f"Evaluating on {len(test_dataset)} examples...")

    # Run predictions
    results = []
    predictions = []
    ground_truth = []

    for example in tqdm(test_dataset, desc="Generating predictions"):
        # Generate prediction
        pred_text = predict_response(model, tokenizer, example)

        # Extract numeric response
        pred_numeric = extract_numeric_response(pred_text)
        true_response = example['output'].replace("Response: ", "").strip()
        true_numeric = extract_numeric_response(true_response)

        results.append({
            'study_id': example.get('study_id', ''),
            'condition_num': example.get('condition_num', ''),
            'input': example['input'],
            'true_response': true_response,
            'predicted_response': pred_text,
            'true_numeric': true_numeric,
            'pred_numeric': pred_numeric,
        })

        if pred_numeric is not None and true_numeric is not None:
            predictions.append(pred_numeric)
            ground_truth.append(true_numeric)

    # Calculate metrics
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)

    if len(predictions) > 0:
        mae = mean_absolute_error(ground_truth, predictions)
        rmse = np.sqrt(mean_squared_error(ground_truth, predictions))
        correlation = np.corrcoef(ground_truth, predictions)[0, 1]

        # Compute Wasserstein distance (paper's primary metric)
        print("Computing Wasserstein distance...")
        wasserstein_metrics = compute_wasserstein_distance(results)

        print(f"Number of numeric predictions: {len(predictions)}/{len(results)}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"Correlation: {correlation:.4f}")
        print(f"Wasserstein Distance: {wasserstein_metrics['wasserstein_distance']:.4f}")
        print(f"  - Evaluated across {wasserstein_metrics['num_conditions_evaluated']} conditions")
        print(f"  - Across {wasserstein_metrics['num_studies_evaluated']} studies")

        metrics = {
            'num_predictions': len(predictions),
            'total_examples': len(results),
            'mae': float(mae),
            'rmse': float(rmse),
            'correlation': float(correlation),
            'wasserstein_distance': wasserstein_metrics['wasserstein_distance'],
            'wasserstein_num_conditions': wasserstein_metrics['num_conditions_evaluated'],
            'wasserstein_num_studies': wasserstein_metrics['num_studies_evaluated'],
            'wasserstein_per_study': wasserstein_metrics['per_study_wasserstein'],
        }
    else:
        print("WARNING: No numeric predictions could be extracted!")
        metrics = {
            'num_predictions': 0,
            'total_examples': len(results),
            'error': 'No numeric predictions extracted'
        }

    # Save results
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save detailed results
        with open(output_path / "predictions.json", 'w') as f:
            json.dump(results, f, indent=2)

        # Save metrics
        with open(output_path / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)

        # Create visualization if we have numeric data
        if len(predictions) > 0:
            plt.figure(figsize=(10, 6))
            plt.scatter(ground_truth, predictions, alpha=0.5)
            plt.plot([min(ground_truth), max(ground_truth)],
                    [min(ground_truth), max(ground_truth)],
                    'r--', label='Perfect prediction')
            plt.xlabel('True Response')
            plt.ylabel('Predicted Response')
            plt.title(f'Predictions vs Ground Truth (Correlation: {correlation:.3f})')
            plt.legend()
            plt.savefig(output_path / "predictions_scatter.png", dpi=300, bbox_inches='tight')
            print(f"\nVisualization saved to: {output_path}/predictions_scatter.png")

        print(f"\nResults saved to: {output_dir}")

    # Show some example predictions
    print("\n" + "="*50)
    print("Sample Predictions")
    print("="*50)
    for i in range(min(3, len(results))):
        print(f"\nExample {i+1}:")
        print(f"Input: {results[i]['input'][:200]}...")
        print(f"True: {results[i]['true_response']}")
        print(f"Predicted: {results[i]['predicted_response']}")

    return metrics, results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate finetuned SOCRATES model"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to finetuned model"
    )
    parser.add_argument(
        "--test-data",
        type=str,
        help="Path to test data JSONL file (default: data/socsci210_*/test.jsonl)"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        help="Base model name (for LoRA models, auto-detected if possible)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output directory for results (default: {model_path}/evaluation)"
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        help="Maximum number of examples to evaluate (default: all)"
    )

    args = parser.parse_args()

    # Auto-detect test data path if not provided
    if args.test_data is None:
        model_path = Path(args.model)
        metadata_path = model_path / "training_metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
                data_dir = metadata.get('data_dir')
                if data_dir:
                    args.test_data = f"{data_dir}/test.jsonl"

        if args.test_data is None:
            raise ValueError("--test-data not provided and could not auto-detect from model metadata")

    # Auto-set output directory
    if args.output is None:
        args.output = str(Path(args.model) / "evaluation")

    print(f"Model: {args.model}")
    print(f"Test data: {args.test_data}")
    print(f"Output: {args.output}\n")

    evaluate_model(
        model_path=args.model,
        test_data_path=args.test_data,
        base_model=args.base_model,
        output_dir=args.output,
        max_examples=args.max_examples,
    )


if __name__ == "__main__":
    main()
