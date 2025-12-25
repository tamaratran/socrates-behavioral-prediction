#!/usr/bin/env python3
"""
Upload checkpoint to HuggingFace Hub as a private repository.
"""

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, upload_folder

def upload_checkpoint_to_hub(
    checkpoint_path: str,
    repo_id: str,
    token: str,
    private: bool = True
):
    """
    Upload checkpoint to HuggingFace Hub.

    Args:
        checkpoint_path: Path to checkpoint directory
        repo_id: Repository ID (username/repo-name)
        token: HuggingFace write token
        private: Whether to make repository private
    """
    print(f"Uploading checkpoint from: {checkpoint_path}")
    print(f"Repository: {repo_id}")
    print(f"Private: {private}")
    print()

    # Check if checkpoint path exists
    if not Path(checkpoint_path).exists():
        raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")

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

    # Upload checkpoint files
    print("\nUploading checkpoint files...")
    print("This may take a few minutes...")

    try:
        upload_folder(
            folder_path=checkpoint_path,
            repo_id=repo_id,
            token=token,
            repo_type="model",
            ignore_patterns=[
                "optimizer.pt",        # Training-only file
                "scheduler.pt",        # Training-only file
                "rng_state_*.pth",     # Training-only RNG states
                "global_step*/*",      # DeepSpeed training artifacts
                "*.log",
                "__pycache__",
                ".git*"
            ]
        )
        print(f"\n✓ Upload complete!")
        print(f"\nCheckpoint available at: https://huggingface.co/{repo_id}")
        print(f"Repository is {'PRIVATE' if private else 'PUBLIC'}")

    except Exception as e:
        print(f"\n✗ Upload failed: {e}")
        raise

def main():
    # Configuration
    CHECKPOINT_PATH = "backups/test-checkpoints/checkpoint-10"
    REPO_ID = "tamaratran/socrates-qwen-test-checkpoint"

    # Get token from command line or environment
    if len(sys.argv) > 1:
        token = sys.argv[1]
    else:
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        if not token:
            print("Error: No HuggingFace token provided")
            print("Usage: python upload_checkpoint_to_hf.py <HF_TOKEN>")
            print("Or set HF_TOKEN environment variable")
            sys.exit(1)

    # Upload checkpoint
    upload_checkpoint_to_hub(
        checkpoint_path=CHECKPOINT_PATH,
        repo_id=REPO_ID,
        token=token,
        private=True
    )

if __name__ == "__main__":
    main()
