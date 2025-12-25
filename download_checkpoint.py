#!/usr/bin/env python3
"""
Download checkpoint tarball from Thunder instance 1 to local laptop
"""
import os
import subprocess
import sys

os.environ['TNR_API_TOKEN'] = 'af22ef23aca94b95661ed2a3f5d8dbe94e78bc39d8c2a38c4c34ac25a77f3759'

print("Downloading checkpoint-10.tar.gz from Thunder instance 1...")
print("This may take 2-5 minutes for 1.8GB file...")

# Create backup directory
os.makedirs('./backups/test-checkpoints', exist_ok=True)

# Download via Thunder connect - read remote file and write to local
proc = subprocess.Popen(
    ['/usr/local/bin/python3.11', '-m', 'thunder.thunder', 'connect', '1'],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=False  # Binary mode for tarball
)

# Read the tarball from remote
stdout, stderr = proc.communicate(b"cat /tmp/checkpoint-10.tar.gz\n")

if proc.returncode != 0:
    print(f"Error downloading checkpoint: {stderr.decode()}")
    sys.exit(1)

# Write to local file
with open('./backups/test-checkpoints/checkpoint-10.tar.gz', 'wb') as f:
    f.write(stdout)

print("\n✓ Checkpoint downloaded successfully!")

# Verify size
import os
size_mb = os.path.getsize('./backups/test-checkpoints/checkpoint-10.tar.gz') / (1024 * 1024)
print(f"Downloaded file size: {size_mb:.1f} MB")

if size_mb < 100:
    print("⚠ Warning: File size seems too small, download may have failed")
    sys.exit(1)

print("\nCheckpoint saved to: ./backups/test-checkpoints/checkpoint-10.tar.gz")
