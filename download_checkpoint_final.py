#!/usr/bin/env python3
"""
Final checkpoint download - only trim Thunder banner from beginning, not end
"""
import os
import subprocess
import sys

os.environ['TNR_API_TOKEN'] = 'af22ef23aca94b95661ed2a3f5d8dbe94e78bc39d8c2a38c4c34ac25a77f3759'

INSTANCE = '1'
REMOTE_FILE = '/tmp/checkpoint-10.tar.gz'
LOCAL_FILE = './backups/test-checkpoints/checkpoint-10.tar.gz'

print("=" * 70)
print("SOCRATES Checkpoint Download (Final - Header-Only Trim)")
print("=" * 70)
print(f"Downloading {REMOTE_FILE} from instance {INSTANCE}...")
print("This may take 2-5 minutes for 1.8GB file...")
print()

# Create backup directory
os.makedirs('./backups/test-checkpoints', exist_ok=True)

# Get expected file size
print("Step 1: Getting remote file size...")
proc = subprocess.Popen(
    ['/usr/local/bin/python3.11', '-m', 'thunder.thunder', 'connect', INSTANCE],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)
stdout, stderr = proc.communicate(f"stat -f%z {REMOTE_FILE} 2>/dev/null || stat -c%s {REMOTE_FILE}\n")

file_size = None
for line in stdout.strip().split('\n'):
    line = line.strip()
    if line.isdigit():
        file_size = int(line)

if not file_size or file_size < 1000:
    print(f"ERROR: Could not get valid file size")
    sys.exit(1)

print(f"Remote file size: {file_size:,} bytes ({file_size / (1024*1024):.1f} MB)")
print()

# Download in binary mode
print("Step 2: Downloading via Thunder connect (binary mode)...")
proc = subprocess.Popen(
    ['/usr/local/bin/python3.11', '-m', 'thunder.thunder', 'connect', INSTANCE],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=False
)

stdout_bytes, stderr_bytes = proc.communicate(b"cat " + REMOTE_FILE.encode() + b"\n")

if proc.returncode != 0:
    print(f"ERROR: Thunder connect failed: {stderr_bytes.decode()}")
    sys.exit(1)

print(f"Received {len(stdout_bytes):,} bytes from Thunder CLI")
print()

# Step 3: Find gzip header and extract from there to EOF
print("Step 3: Finding gzip header (trimming banner from beginning only)...")

gzip_start = -1
for i in range(min(10000, len(stdout_bytes) - 1)):  # Search first 10KB for header
    if stdout_bytes[i] == 0x1f and stdout_bytes[i+1] == 0x8b:
        gzip_start = i
        print(f"Found gzip magic number at byte {i}")
        break

if gzip_start == -1:
    print("ERROR: Could not find gzip magic number in first 10KB")
    sys.exit(1)

# Extract everything from gzip start to end
# DO NOT trim the end - gzip files have their own EOF markers
tarball_data = stdout_bytes[gzip_start:]

print(f"Extracted {len(tarball_data):,} bytes starting from gzip header")
print(f"Expected {file_size:,} bytes")
print(f"Difference: {len(tarball_data) - file_size:,} bytes")
print()

# Write to file
print("Step 4: Writing to file...")
with open(LOCAL_FILE, 'wb') as f:
    f.write(tarball_data)

local_size = os.path.getsize(LOCAL_FILE)
print(f"Wrote {local_size:,} bytes ({local_size / (1024*1024):.1f} MB)")
print()

# Verify tarball integrity
print("Step 5: Verifying tarball integrity...")
result = subprocess.run(['tar', '-tzf', LOCAL_FILE], capture_output=True)

if result.returncode != 0:
    print(f"ERROR: Tarball verification failed!")
    print(f"tar stderr: {result.stderr.decode()}")
    print()
    print("Diagnostics:")
    print(f"  File size: {local_size:,} bytes")
    print(f"  Expected: {file_size:,} bytes")
    print(f"  Difference: {local_size - file_size:,} bytes")
    print()

    # Show last 200 bytes to see what's at the end
    with open(LOCAL_FILE, 'rb') as f:
        f.seek(-min(200, local_size), 2)  # Seek to 200 bytes before end
        tail_bytes = f.read()
    print(f"Last 200 bytes (hex): {tail_bytes.hex()[:400]}")
    print(f"Last 200 bytes (text attempt): {tail_bytes}")
    sys.exit(1)

file_count = len(result.stdout.decode().strip().split('\n'))
print(f"✓ Tarball is VALID! Contains {file_count} files")
print()

print("=" * 70)
print("SUCCESS: Checkpoint downloaded and verified!")
print("=" * 70)
print(f"File: {LOCAL_FILE}")
print(f"Size: {local_size:,} bytes ({local_size / (1024*1024):.1f} MB)")
print()
print("Next steps:")
print(f"  1. Extract: tar -xzf {LOCAL_FILE} -C ./backups/test-checkpoints/")
print(f"  2. Verify: ls -lh ./backups/test-checkpoints/checkpoint-10/")
print(f"  3. Use for model evaluation or training resume")
