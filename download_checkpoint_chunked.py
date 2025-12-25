#!/usr/bin/env python3
"""
Download large checkpoint from Thunder using chunked base64 transfer
"""
import os
import subprocess
import sys
import base64

os.environ['TNR_API_TOKEN'] = 'af22ef23aca94b95661ed2a3f5d8dbe94e78bc39d8c2a38c4c34ac25a77f3759'

INSTANCE = '1'
REMOTE_FILE = '/tmp/checkpoint-10.tar.gz'
LOCAL_FILE = './backups/test-checkpoints/checkpoint-10.tar.gz'
CHUNK_SIZE = 10 * 1024 * 1024  # 10MB chunks

print(f"Downloading {REMOTE_FILE} from instance {INSTANCE}...")
print("This will take 2-5 minutes for 1.8GB file...")

# First, get file size
print("\nGetting file size...")
proc = subprocess.Popen(
    ['/usr/local/bin/python3.11', '-m', 'thunder.thunder', 'connect', INSTANCE],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)
stdout, stderr = proc.communicate(f"stat -f%z {REMOTE_FILE}\n")
if proc.returncode != 0:
    print(f"Error getting file size: {stderr}")
    sys.exit(1)

try:
    file_size = int(stdout.strip().split('\n')[-2])  # Get second to last line (before exit message)
    print(f"File size: {file_size / (1024*1024):.1f} MB")
except:
    print(f"Could not parse file size from: {stdout}")
    sys.exit(1)

# Download via dd and base64 in chunks
os.makedirs(os.path.dirname(LOCAL_FILE), exist_ok=True)

print("\nDownloading in chunks...")
offset = 0
chunk_num = 0

with open(LOCAL_FILE, 'wb') as out_file:
    while offset < file_size:
        chunk_num += 1
        remaining = file_size - offset
        current_chunk = min(CHUNK_SIZE, remaining)

        progress_pct = (offset / file_size) * 100
        print(f"Chunk {chunk_num}: {progress_pct:.1f}% ({offset/(1024*1024):.1f}/{file_size/(1024*1024):.1f} MB)...", end='', flush=True)

        # Use dd to extract chunk and base64 encode
        cmd = f"dd if={REMOTE_FILE} bs=1M skip={offset//(1024*1024)} count={current_chunk//(1024*1024)+1} 2>/dev/null | base64\n"

        proc = subprocess.Popen(
            ['/usr/local/bin/python3.11', '-m', 'thunder.thunder', 'connect', INSTANCE],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False
        )
        stdout_bytes, stderr_bytes = proc.communicate(cmd.encode())

        if proc.returncode != 0:
            print(f"\nError downloading chunk: {stderr_bytes.decode()}")
            sys.exit(1)

        # Decode base64 and write
        try:
            # Extract base64 data (skip Thunder prompt/footer)
            lines = stdout_bytes.split(b'\n')
            base64_data = b''.join(l for l in lines if l and not l.startswith(b'\x1b') and b'Thunder' not in l)
            chunk_data = base64.b64decode(base64_data)

            # Only write the exact bytes we need
            bytes_to_write = min(len(chunk_data), current_chunk)
            out_file.write(chunk_data[:bytes_to_write])
            print(" done")
        except Exception as e:
            print(f"\nError decoding chunk: {e}")
            sys.exit(1)

        offset += current_chunk

print(f"\n✓ Download complete!")
print(f"Saved to: {LOCAL_FILE}")

# Verify file size
local_size = os.path.getsize(LOCAL_FILE)
print(f"Local file size: {local_size / (1024*1024):.1f} MB")
if local_size > 0:
    print("✓ Download successful!")
else:
    print("⚠ Warning: File is empty")
    sys.exit(1)
