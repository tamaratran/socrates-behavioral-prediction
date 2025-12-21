#!/usr/bin/env python3
"""
CPU-friendly inference script for SOCRATES model
Uses 8-bit quantization to reduce memory requirements
"""

import argparse
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


def load_model_and_tokenizer(model_path, base_model=None):
    """Load model with 8-bit quantization for CPU/low memory"""
    model_path = Path(model_path)

    # Check if this is a LoRA model
    if (model_path / "adapter_config.json").exists():
        # Auto-detect base model from metadata
        if base_model is None:
            metadata_path = model_path / "training_metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
                    base_model = metadata.get('model_name')

        if base_model is None:
            raise ValueError("LoRA model detected but base_model not specified")

        print(f"Loading base model with 8-bit quantization: {base_model}")
        print("This may take a few minutes...")

        # Configure 8-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )

        # Load base model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

        print("Loading LoRA adapters...")
        model = PeftModel.from_pretrained(model, str(model_path))
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    else:
        # Regular model
        print(f"Loading model: {model_path}")
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            quantization_config=bnb_config,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    print("✓ Model loaded successfully")
    return model, tokenizer


def create_prompt(demographics, scenario):
    """Create formatted prompt"""
    instruction = "You are a behavioral prediction model. Given demographic information about a participant and an experimental scenario, predict how they would respond."

    input_text = f"""Demographics: {demographics}

Scenario: {scenario}"""

    prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
    return prompt


def predict(model, tokenizer, demographics, scenario):
    """Generate prediction"""
    prompt = create_prompt(demographics, scenario)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    print("\nGenerating prediction...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.6,
            top_p=0.9,
            do_sample=False,  # Greedy for deterministic results
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract just the response part
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()

    return response


# Predefined examples
EXAMPLES = {
    1: {
        "demographics": "Age: 35; Gender: Male; Education: Bachelor's degree",
        "scenario": "Should the federal government increase spending on food stamps? (1=Strongly oppose, 7=Strongly support)"
    },
    2: {
        "demographics": "Age: 28; Gender: Female; Education: Master's degree",
        "scenario": "Should taxes be increased on people earning over $200,000 per year? (1=Definitely no, 5=Definitely yes)"
    },
    3: {
        "demographics": "Age: 42; Gender: Male; Education: High school",
        "scenario": "How concerned are you about climate change? (1=Not at all concerned, 7=Extremely concerned)"
    }
}


def main():
    parser = argparse.ArgumentParser(description="CPU-friendly SOCRATES Model Inference")
    parser.add_argument("--model", required=True, help="Path to model directory")
    parser.add_argument("--base-model", help="Base model name (auto-detected if not provided)")
    parser.add_argument("--example", type=int, choices=[1, 2, 3], help="Use predefined example (1-3)")
    parser.add_argument("--demographics", help="Custom demographics string")
    parser.add_argument("--scenario", help="Custom scenario string")
    args = parser.parse_args()

    # Load model
    print("=" * 80)
    print("SOCRATES MODEL INFERENCE (CPU-friendly)")
    print("=" * 80)
    print()

    model, tokenizer = load_model_and_tokenizer(args.model, args.base_model)
    print()

    # Determine input source
    if args.example:
        print(f"Using predefined example {args.example}:")
        demographics = EXAMPLES[args.example]["demographics"]
        scenario = EXAMPLES[args.example]["scenario"]
        print(f"Demographics: {demographics}")
        print(f"Scenario: {scenario}")
    elif args.demographics and args.scenario:
        demographics = args.demographics
        scenario = args.scenario
    else:
        # Interactive mode
        print("Interactive mode - enter details:")
        demographics = input("Demographics: ")
        scenario = input("Scenario: ")

    # Generate prediction
    response = predict(model, tokenizer, demographics, scenario)

    # Display result
    print("\n" + "=" * 80)
    print("PREDICTION")
    print("=" * 80)
    print(response)
    print("=" * 80)


if __name__ == "__main__":
    main()
