import numpy as np
import json

tr = np.load('data/unified_136class/train_split.npz')
va = np.load('data/unified_136class/val_split.npz')
te = np.load('data/unified_136class/test_split.npz')

with open('configs/authoritative_136_classes_manifest.json') as f:
    classes = json.load(f)['classes']
    
ty_idx = classes.index('thank_you')
c_tr = int(np.sum(tr['labels'] == ty_idx))
c_va = int(np.sum(va['labels'] == ty_idx))
c_te = int(np.sum(te['labels'] == ty_idx))

print(f"thank_you in train: {c_tr}")
print(f"thank_you in val:   {c_va}")
print(f"thank_you in test:  {c_te}")
print(f"thank_you total:    {c_tr + c_va + c_te}")

# Let's inspect data/additional_landmarks/thank_you/*.npz files
import glob
ty_files = glob.glob(r"data/additional_landmarks/thank_you/*.npz")
has_feat = 0
has_kps_only = 0
for f in ty_files:
    d = np.load(f, allow_pickle=True)
    if 'features' in d:
        has_feat += 1
    else:
        has_kps_only += 1
print(f"\nIn data/additional_landmarks/thank_you ({len(ty_files)} total files):")
print(f"  Files with 'features': {has_feat}")
print(f"  Files with 'keypoints' only: {has_kps_only}")
