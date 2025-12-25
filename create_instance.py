#!/usr/bin/env python3
"""
Create Thunder Compute instance programmatically
Avoids interactive CLI prompts
"""

import os
import sys
import subprocess

# Thunder API token
TOKEN = "af22ef23aca94b95661ed2a3f5d8dbe94e78bc39d8c2a38c4c34ac25a77f3759"

# Set environment variable
os.environ["TNR_API_TOKEN"] = TOKEN

# Create instance using Thunder CLI with template to avoid prompts
# Use Ubuntu + Python template
cmd = [
    "/usr/local/bin/python3.11",
    "-m", "thunder.thunder",
    "create",
    "--gpu", "a100xl",
    "--num-gpus", "4",
    "--disk-size-gb", "500",
    "--mode", "production",
    "--template", "ubuntu-python"  # Try common template
]

print(f"Creating Thunder instance with command:")
print(" ".join(cmd))
print()

try:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    print("STDOUT:")
    print(result.stdout)
    print("\nSTDERR:")
    print(result.stderr)
    print(f"\nReturn code: {result.returncode}")

    if result.returncode == 0:
        print("\n✓ Instance created successfully!")
    else:
        print("\n✗ Instance creation failed")
        sys.exit(1)

except subprocess.TimeoutExpired:
    print("Command timed out after 5 minutes")
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
