#!/usr/bin/env python3
"""
Create Smoke Test Dataset
Extracts 10 examples from existing dataset for quick local validation
"""

import json
import os
from pathlib import Path

# Configuration
SOURCE_DIR = "data/socsci210_1pct"
OUTPUT_DIR = "data/smoke_test"
NUM_EXAMPLES = 10

def create_smoke_test_data():
    """Extract small subset of data for smoke testing"""
    print("=" * 80)
    print("CREATING SMOKE TEST DATASET")
    print("=" * 80)
    print(f"Source: {SOURCE_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Examples per split: {NUM_EXAMPLES}")
    print()

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Process each split
    for split in ["train", "val"]:
        source_file = f"{SOURCE_DIR}/{split}.jsonl"
        output_file = f"{OUTPUT_DIR}/{split}.jsonl"

        print(f"Processing {split}...")

        # Read first N examples from source
        examples = []
        with open(source_file, 'r') as f:
            for i, line in enumerate(f):
                if i >= NUM_EXAMPLES:
                    break
                examples.append(json.loads(line))

        # Write to output
        with open(output_file, 'w') as f:
            for example in examples:
                f.write(json.dumps(example) + '\n')

        print(f"  ✓ Wrote {len(examples)} examples to {output_file}")

    # Copy metadata
    metadata_source = f"{SOURCE_DIR}/metadata.json"
    metadata_output = f"{OUTPUT_DIR}/metadata.json"

    if os.path.exists(metadata_source):
        with open(metadata_source, 'r') as f:
            metadata = json.load(f)

        # Update counts for smoke test
        metadata['description'] = "Smoke test dataset (10 examples per split) for local validation"
        metadata['train_examples'] = NUM_EXAMPLES
        metadata['val_examples'] = NUM_EXAMPLES

        with open(metadata_output, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"  ✓ Updated metadata: {metadata_output}")

    print()
    print("=" * 80)
    print("SMOKE TEST DATA READY")
    print("=" * 80)
    print(f"Location: {OUTPUT_DIR}/")
    print(f"Files: train.jsonl ({NUM_EXAMPLES} examples), val.jsonl ({NUM_EXAMPLES} examples)")
    print()

if __name__ == "__main__":
    create_smoke_test_data()
