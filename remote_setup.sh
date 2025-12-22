#!/bin/bash
set -e

echo "=== Extracting deployment package ==="
cd ~
tar -xzf socrates-full-training.tar.gz

echo "=== Organizing files ==="
mkdir -p socrates-training
mv scripts config_full_dataset.json deepspeed_config.json requirements_full_training.txt *.md data socrates-training/ 2>/dev/null || true
cd socrates-training

echo "=== Creating Python virtual environment ==="
python3 -m venv venv
source venv/bin/activate

echo "=== Upgrading pip ==="
pip install --upgrade pip

echo "=== Installing dependencies ==="
pip install -r requirements_full_training.txt

echo "=== Verifying installations ==="
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA devices: {torch.cuda.device_count()}')"
python -c "import deepspeed; print(f'DeepSpeed: {deepspeed.__version__}')"

echo "=== Adjusting config for 4 GPUs ==="
# Update config_full_dataset.json for 4 GPUs
python3 << 'PYTHON_EOF'
import json

with open('config_full_dataset.json', 'r') as f:
    config = json.load(f)

# Adjust for 4 GPUs instead of 8
config['training']['gradient_accumulation_steps'] = 32  # Changed from 16
print("Updated gradient_accumulation_steps from 16 to 32 for 4 GPUs")

with open('config_full_dataset.json', 'w') as f:
    json.dump(config, f, indent=2)

print("Configuration adjusted for 4 GPUs!")
PYTHON_EOF

echo "=== Setup complete! ==="
echo ""
echo "To start training, run:"
echo "cd ~/socrates-training"
echo "source venv/bin/activate"
echo "bash scripts/run_full_training.sh --wandb"
