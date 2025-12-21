#!/usr/bin/env python3
"""
Interactive Testing Script for SOCRATES Models
Quick testing of finetuned models with custom inputs
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_model_and_tokenizer(model_path: str, base_model: str = None):
    """Load finetuned model and tokenizer"""
    print(f"Loading model from {model_path}...")

    model_path = Path(model_path)

    # Check if this is a LoRA model
    if (model_path / "adapter_config.json").exists():
        if base_model is None:
            metadata_path = model_path / "training_metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
                    base_model = metadata.get('model_name')

        if base_model is None:
            raise ValueError("LoRA model detected but base_model not specified")

        print(f"Loading base model: {base_model}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        print(f"Loading LoRA adapters...")
        model = PeftModel.from_pretrained(model, str(model_path))
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    else:
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
    print("Model loaded successfully!\n")
    return model, tokenizer


def create_prompt(demographics: str, scenario: str) -> str:
    """Create a formatted prompt for the model"""
    instruction = (
        "You are a behavioral prediction model. Given demographic information about a participant "
        "and an experimental scenario, predict how they would respond."
    )

    input_text = f"Demographics: {demographics}\n\nScenario: {scenario}"

    return f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""


def predict(model, tokenizer, demographics: str, scenario: str, max_new_tokens: int = 150):
    """Generate prediction for given demographics and scenario"""
    prompt = create_prompt(demographics, scenario)

    print("="*70)
    print("PROMPT:")
    print("="*70)
    print(prompt)
    print("="*70)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the generated part
    generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    print("\nPREDICTION:")
    print("="*70)
    print(generated.strip())
    print("="*70)

    return generated.strip()


def interactive_mode(model, tokenizer):
    """Run interactive testing mode"""
    print("\n" + "="*70)
    print("INTERACTIVE TESTING MODE")
    print("="*70)
    print("\nEnter demographics and scenarios to get predictions.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        print("\n" + "-"*70)

        # Get demographics
        print("\nEnter demographics (e.g., 'Age: 25-34; Gender: Female; Education: Bachelor's'):")
        demographics = input("> ").strip()

        if demographics.lower() in ['exit', 'quit', 'q']:
            break

        # Get scenario
        print("\nEnter experimental scenario/question:")
        scenario = input("> ").strip()

        if scenario.lower() in ['exit', 'quit', 'q']:
            break

        # Generate prediction
        print("\nGenerating prediction...\n")
        predict(model, tokenizer, demographics, scenario)


def run_example(model, tokenizer, example_num: int = 1):
    """Run a predefined example"""
    examples = [
        {
            "demographics": "Age: 25-34; Gender: Male; Education: Bachelor's; Income: $50-75k; Political affiliation: Democrat",
            "scenario": "A local government is proposing to increase property taxes by 10% to fund public schools. Do you support or oppose this measure?"
        },
        {
            "demographics": "Age: 55-64; Gender: Female; Education: High school; Income: $30-50k; Employment: Retired",
            "scenario": "A pharmaceutical company has developed a new COVID-19 vaccine. How likely are you to get vaccinated when it becomes available? (1 = Very unlikely, 7 = Very likely)"
        },
        {
            "demographics": "Age: 18-24; Gender: Non-binary; Education: Some college; Ideology: Liberal",
            "scenario": "Congress is debating a bill to increase the federal minimum wage to $15/hour. Do you approve or disapprove of this bill?"
        }
    ]

    if example_num > len(examples):
        print(f"Example {example_num} not found. Only {len(examples)} examples available.")
        return

    example = examples[example_num - 1]

    print(f"\n{'='*70}")
    print(f"EXAMPLE {example_num}")
    print('='*70)

    predict(model, tokenizer, example["demographics"], example["scenario"])


def main():
    parser = argparse.ArgumentParser(
        description="Test SOCRATES finetuned model interactively"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to finetuned model"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        help="Base model name (for LoRA models)"
    )
    parser.add_argument(
        "--example",
        type=int,
        help="Run a predefined example (1-3)"
    )
    parser.add_argument(
        "--demographics",
        type=str,
        help="Demographics for single prediction"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        help="Scenario for single prediction"
    )

    args = parser.parse_args()

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model, args.base_model)

    # Run appropriate mode
    if args.example:
        run_example(model, tokenizer, args.example)
    elif args.demographics and args.scenario:
        predict(model, tokenizer, args.demographics, args.scenario)
    else:
        interactive_mode(model, tokenizer)


if __name__ == "__main__":
    main()
