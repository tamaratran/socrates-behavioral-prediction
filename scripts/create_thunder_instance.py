#!/usr/bin/env python3
"""
Create Thunder Compute Instance via CLI
Uses Thunder CLI directly via subprocess (like Hewitt-Agents does)
"""

import sys
import os
import time
import subprocess

# Configuration
TOKEN = "af22ef23aca94b95661ed2a3f5d8dbe94e78bc39d8c2a38c4c34ac25a77f3759"
GPU_TYPE = "a100xl"  # A100 80GB
NUM_GPUS = 4
VCPUS = 32  # Maximum allowed value (4, 8, 16, or 32)
DISK_SIZE_GB = 500
TEMPLATE = "base"  # Base template (Ubuntu + PyTorch + CUDA)
MODE = "production"

def create_instance():
    """Create Thunder Compute instance using CLI"""
    print("=" * 80)
    print("THUNDER COMPUTE INSTANCE CREATION")
    print("=" * 80)
    print(f"GPU Type: {GPU_TYPE}")
    print(f"Number of GPUs: {NUM_GPUS}")
    print(f"vCPUs: {VCPUS}")
    print(f"Disk Size: {DISK_SIZE_GB} GB")
    print(f"Mode: {MODE}")
    print(f"Template: {TEMPLATE}")
    print("")

    print("Creating instance via Thunder CLI...")

    # Setup environment with token
    env = os.environ.copy()
    env["TNR_API_TOKEN"] = TOKEN

    try:
        # Call Thunder CLI directly with subprocess (like Hewitt-Agents)
        # Note: vCPUs cannot be set in production mode - automatically determined by GPU selection
        result = subprocess.run([
            '/usr/local/bin/python3.11', '-m', 'thunder.thunder', 'create',
            '--template', TEMPLATE,
            '--gpu', GPU_TYPE,
            '--num-gpus', str(NUM_GPUS),
            '--disk-size-gb', str(DISK_SIZE_GB),
            '--mode', MODE
        ],
        input='Y\n',  # Auto-confirm prompts
        capture_output=True,
        text=True,
        timeout=120,
        env=env
        )

        print("STDOUT:")
        print(result.stdout)
        print("\nSTDERR:")
        print(result.stderr)
        print(f"\nReturn code: {result.returncode}")

        if result.returncode != 0:
            print(f"✗ ERROR: Instance creation failed")
            sys.exit(1)

        # Wait for instance to appear
        print("\n" + "=" * 80)
        print("Waiting for instance to initialize...")
        print("=" * 80)
        time.sleep(30)

        # Get instance ID from status
        status_result = subprocess.run(
            ['/usr/local/bin/python3.11', '-m', 'thunder.thunder', 'status'],
            capture_output=True,
            text=True,
            env=env
        )

        # Extract instance ID (usually 0 for most recent)
        if "0" in status_result.stdout and "RUNNING" in status_result.stdout:
            instance_id = "0"
        else:
            # Try to parse instance ID from output
            lines = status_result.stdout.split('\n')
            for line in lines:
                if "RUNNING" in line or "STARTING" in line:
                    parts = line.split()
                    if parts and parts[0].isdigit():
                        instance_id = parts[0]
                        break
            else:
                instance_id = "0"  # Default to 0

        print(f"✓ Instance created successfully!")
        print(f"Instance ID: {instance_id}")
        print("")

        # Wait for instance to be fully RUNNING
        print("Waiting for instance to reach RUNNING state...")
        max_wait = 300  # 5 minutes
        start_time = time.time()

        while time.time() - start_time < max_wait:
            result = subprocess.run(
                ['/usr/local/bin/python3.11', '-m', 'thunder.thunder', 'status'],
                capture_output=True,
                text=True,
                env=env
            )

            if "RUNNING" in result.stdout:
                print(f"✓ Instance {instance_id} is RUNNING")
                break

            time.sleep(10)
            print("  Still starting...")
        else:
            print(f"⚠ WARNING: Instance may not be fully ready")

        print("")
        print("=" * 80)
        print("INSTANCE READY")
        print("=" * 80)
        print(f"Instance ID: {instance_id}")
        print(f"Cost: ~$7.16/hour (4x A100 80GB)")
        print("")
        print("Next steps:")
        print(f"  1. Connect: tnr ssh {instance_id}")
        print(f"  2. Run setup: ./scripts/setup_and_launch_training.sh {instance_id}")
        print("=" * 80)

        # Save instance ID to file for next scripts
        with open(".thunder_instance_id", "w") as f:
            f.write(instance_id)

        return instance_id

    except subprocess.TimeoutExpired:
        print(f"✗ ERROR: Command timed out after 120 seconds")
        sys.exit(1)
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    instance_id = create_instance()
    print(f"\nInstance ID: {instance_id}")
