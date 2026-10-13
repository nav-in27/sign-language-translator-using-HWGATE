"""
Quick probe: Download ONE H5 file from ISL500 and inspect its internal structure.
"""
import urllib.request
import urllib.parse
import json
import sys
import io
import numpy as np

# Install h5py if needed
try:
    import h5py
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "h5py"])
    import h5py

# Download a single sample
filepath = "Landmarks/MediaPipe/ISL_DATA_USER001/Hello__session106__clip014.h5"
encoded = urllib.parse.quote(filepath, safe="/")
url = f"https://huggingface.co/datasets/ISL500/ISL-DATA/resolve/main/{encoded}"

print(f"Downloading: {url}")
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req, timeout=30) as resp:
    h5_bytes = resp.read()

print(f"Downloaded {len(h5_bytes)} bytes")

# Inspect H5 structure
def print_h5_tree(group, indent=0):
    prefix = "  " * indent
    for key in list(group.keys())[:20]:
        item = group[key]
        if isinstance(item, h5py.Dataset):
            print(f"{prefix}{key}: Dataset, shape={item.shape}, dtype={item.dtype}")
            if item.shape[0] <= 5:
                print(f"{prefix}  data sample: {np.array(item)[:3]}")
        elif isinstance(item, h5py.Group):
            print(f"{prefix}{key}: Group ({len(item)} children)")
            if indent < 3:
                print_h5_tree(item, indent + 1)
    total = len(list(group.keys()))
    if total > 20:
        print(f"{prefix}... ({total} total keys)")

with h5py.File(io.BytesIO(h5_bytes), 'r') as f:
    print(f"\nH5 root keys: {list(f.keys())}")
    print(f"H5 root attrs: {dict(f.attrs)}")
    print_h5_tree(f)
    
    # Try to understand the actual landmark format
    for key in f.keys():
        item = f[key]
        if isinstance(item, h5py.Dataset):
            data = np.array(item)
            print(f"\nDataset '{key}': shape={data.shape}, dtype={data.dtype}")
            print(f"  min={data.min():.4f}, max={data.max():.4f}, mean={data.mean():.4f}")
            if data.ndim == 3:
                print(f"  Looks like: (T={data.shape[0]}, V={data.shape[1]}, C={data.shape[2]})")
            elif data.ndim == 2:
                print(f"  Looks like: (T={data.shape[0]}, features={data.shape[1]})")
