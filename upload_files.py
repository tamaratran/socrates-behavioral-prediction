#!/usr/bin/env python3
"""
Upload deepspeed config and training script to Thunder instance 1
"""
import os
import subprocess

os.environ['TNR_API_TOKEN'] = 'af22ef23aca94b95661ed2a3f5d8dbe94e78bc39d8c2a38c4c34ac25a77f3759'

# Read deepspeed config
with open('deepspeed_config.json', 'r') as f:
    deepspeed_config = f.read()

# Read training script
with open('scripts/train_full_dataset.py', 'r') as f:
    train_script = f.read()

print("Uploading deepspeed_config.json...")
proc = subprocess.Popen(
    ['/usr/local/bin/python3.11', '-m', 'thunder.thunder', 'connect', '1'],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)
stdout, stderr = proc.communicate(f"cat > ~/socrates-training/deepspeed_config.json\n{deepspeed_config}\n")
if proc.returncode != 0:
    print(f"Error uploading deepspeed_config.json: {stderr}")
else:
    print("✓ deepspeed_config.json uploaded")

print("\nUploading train_full_dataset.py...")
proc = subprocess.Popen(
    ['/usr/local/bin/python3.11', '-m', 'thunder.thunder', 'connect', '1'],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)
stdout, stderr = proc.communicate(f"cat > ~/socrates-training/scripts/train_full_dataset.py\n{train_script}\n")
if proc.returncode != 0:
    print(f"Error uploading train_full_dataset.py: {stderr}")
else:
    print("✓ train_full_dataset.py uploaded")

print("\nVerifying uploads...")
proc = subprocess.Popen(
    ['/usr/local/bin/python3.11', '-m', 'thunder.thunder', 'connect', '1'],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)
stdout, stderr = proc.communicate("ls -lh ~/socrates-training/ ~/socrates-training/scripts/\n")
print(stdout)
