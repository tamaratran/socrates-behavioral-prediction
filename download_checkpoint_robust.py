#!/usr/bin/env python3
"""
Robust checkpoint download with Thunder CLI banner filtering
Uses binary mode with intelligent header/footer stripping
"""
import os
import subprocess
import sys
import re

os.environ['TNR_API_TOKEN'] = 'af22ef23aca94b95661ed2a3f5d8dbe94e78bc39d8c2a38c4c34ac25a77f3759'

INSTANCE = '1'
REMOTE_FILE = '/tmp/checkpoint-10.tar.gz'
LOCAL_FILE = './backups/test-checkpoints/checkpoint-10.tar.gz'

print("=" * 60)
print("SOCRATES Checkpoint Download (Robust Binary Transfer)")
print("=" * 60)
print(f"Downloading {REMOTE_FILE} from instance {INSTANCE}...")
print("This may take 2-5 minutes for 1.8GB file...")
print()

# Create backup directory
os.makedirs('./backups/test-checkpoints', exist_ok=True)

# First, get exact file size for verification
print("Step 1: Getting file size...")
proc = subprocess.Popen(
    ['/usr/local/bin/python3.11', '-m', 'thunder.thunder', 'connect', INSTANCE],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)
stdout, stderr = proc.communicate(f"stat -f%z {REMOTE_FILE} 2>/dev/null || stat -c%s {REMOTE_FILE}\n")

# Extract file size (last numeric line before exit message)
file_size = None
for line in stdout.strip().split('\n'):
    line = line.strip()
    if line.isdigit():
        file_size = int(line)

if not file_size or file_size < 1000:
    print(f"ERROR: Could not get valid file size")
    print(f"Output: {stdout}")
    sys.exit(1)

print(f"Remote file size: {file_size / (1024*1024):.1f} MB")
print()

# Download file in binary mode
print("Step 2: Downloading via Thunder connect (binary mode)...")
proc = subprocess.Popen(
    ['/usr/local/bin/python3.11', '-m', 'thunder.thunder', 'connect', INSTANCE],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=False  # Binary mode
)

# Send cat command
stdout_bytes, stderr_bytes = proc.communicate(b"cat " + REMOTE_FILE.encode() + b"\n")

if proc.returncode != 0:
    print(f"ERROR: Thunder connect failed: {stderr_bytes.decode()}")
    sys.exit(1)

print(f"Received {len(stdout_bytes)} bytes from Thunder CLI")
print()

# Step 3: Intelligent banner stripping
print("Step 3: Filtering Thunder CLI banner...")

# Find the start of gzip data (gzip magic number: 0x1f 0x8b)
gzip_start = -1
for i in range(len(stdout_bytes) - 1):
    if stdout_bytes[i] == 0x1f and stdout_bytes[i+1] == 0x8b:
        gzip_start = i
        print(f"Found gzip header at byte {i}")
        break

if gzip_start == -1:
    print("ERROR: Could not find gzip magic number in output")
    print(f"First 200 bytes: {stdout_bytes[:200]}")
    sys.exit(1)

# Find the end marker (Thunder exit message at the end)
# The exit message is always at the end, so we need to trim from the end
# Look for the last occurrence of the exit banner pattern
exit_pattern = b'\xe2\x9a\xa1'  # Lightning bolt emoji in "Exiting thunder instance"
gzip_end = len(stdout_bytes)

# Search backwards for exit marker
for i in range(len(stdout_bytes) - 1, gzip_start, -1):
    if stdout_bytes[i:i+3] == exit_pattern:
        # Found exit marker, trim everything from here
        gzip_end = i
        # But we need to go back to the last newline before the exit message
        for j in range(i - 1, gzip_start, -1):
            if stdout_bytes[j] == 0x0a:  # newline
                gzip_end = j
                break
        print(f"Found Thunder exit marker, trimming from byte {gzip_end}")
        break

# Extract the clean tarball data
tarball_data = stdout_bytes[gzip_start:gzip_end]

print(f"Extracted {len(tarball_data)} bytes of tarball data")
print(f"Expected {file_size} bytes")
print()

# Write to file
print("Step 4: Writing to file...")
with open(LOCAL_FILE, 'wb') as f:
    f.write(tarball_data)

local_size = os.path.getsize(LOCAL_FILE)
print(f"Wrote {local_size / (1024*1024):.1f} MB to {LOCAL_FILE}")
print()

# Verify file size
if abs(local_size - file_size) > 1024:  # Allow 1KB difference
    print(f"WARNING: Size mismatch!")
    print(f"  Expected: {file_size / (1024*1024):.1f} MB")
    print(f"  Got:      {local_size / (1024*1024):.1f} MB")
    print(f"  Diff:     {abs(local_size - file_size) / (1024*1024):.2f} MB")
else:
    print(f"✓ File size matches!")

# Verify it's a valid tarball
print()
print("Step 5: Verifying tarball integrity...")
result = subprocess.run(['tar', '-tzf', LOCAL_FILE], capture_output=True)
if result.returncode != 0:
    print(f"ERROR: Tarball verification failed!")
    print(f"tar output: {result.stderr.decode()}")
    sys.exit(1)

file_count = len(result.stdout.decode().strip().split('\n'))
print(f"✓ Tarball is valid! Contains {file_count} files")
print()

print("=" * 60)
print("SUCCESS: Checkpoint downloaded and verified!")
print("=" * 60)
print(f"File: {LOCAL_FILE}")
print(f"Size: {local_size / (1024*1024):.1f} MB")
print()
print("Next steps:")
print(f"  Extract: tar -xzf {LOCAL_FILE} -C ./backups/test-checkpoints/")
print(f"  Verify: ls -lh ./backups/test-checkpoints/checkpoint-10/")
