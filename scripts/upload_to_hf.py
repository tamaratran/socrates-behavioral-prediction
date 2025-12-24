#!/usr/bin/env python3
"""
Upload trained QLoRA adapter to HuggingFace Hub as a private repository.
"""

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, upload_folder

def upload_model_to_hub(
    model_path: str,
    repo_id: str,
    token: str,
    private: bool = True
):
    """
    Upload model to HuggingFace Hub.

    Args:
        model_path: Path to model directory
        repo_id: Repository ID (username/repo-name)
        token: HuggingFace write token
        private: Whether to make repository private
    """
    print(f"Uploading model from: {model_path}")
    print(f"Repository: {repo_id}")
    print(f"Private: {private}")
    print()

    # Check if model path exists
    if not Path(model_path).exists():
        raise ValueError(f"Model path does not exist: {model_path}")

    # Initialize HF API
    api = HfApi()

    # Create repository if it doesn't exist
    try:
        print("Creating repository...")
        api.create_repo(
            repo_id=repo_id,
            token=token,
            private=private,
            repo_type="model",
            exist_ok=True
        )
        print(f"✓ Repository created/verified: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"Error creating repository: {e}")
        raise

    # Upload model files
    print("\nUploading model files...")
    print("This may take a few minutes...")

    try:
        upload_folder(
            folder_path=model_path,
            repo_id=repo_id,
            token=token,
            repo_type="model",
            ignore_patterns=[
                "checkpoint-*/*",  # Exclude training checkpoints
                "*.log",
                "*.txt",
                "__pycache__",
                ".git*"
            ]
        )
        print(f"\n✓ Upload complete!")
        print(f"\nModel available at: https://huggingface.co/{repo_id}")
        print(f"Repository is {'PRIVATE' if private else 'PUBLIC'}")

    except Exception as e:
        print(f"\n✗ Upload failed: {e}")
        raise

def main():
    # Configuration
    MODEL_PATH = "models/socrates-qwen-paper-replication"
    REPO_ID = "tamaratran/socrates-qwen-1pct-lora"

    # Get token from command line or environment
    if len(sys.argv) > 1:
        token = sys.argv[1]
    else:
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        if not token:
            print("Error: No HuggingFace token provided")
            print("Usage: python upload_to_hf.py <HF_TOKEN>")
            print("Or set HF_TOKEN environment variable")
            sys.exit(1)

    # Upload model
    upload_model_to_hub(
        model_path=MODEL_PATH,
        repo_id=REPO_ID,
        token=token,
        private=True
    )

if __name__ == "__main__":
    main()
