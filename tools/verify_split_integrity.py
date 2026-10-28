import numpy as np
import json
from pathlib import Path

def verify_splits():
    print("="*70)
    print("VERIFYING SPLIT INTEGRITY AND LABEL MAPPINGS")
    print("="*70)
    
    with open('configs/authoritative_136_classes_manifest.json') as f:
        classes = json.load(f)['classes']
        
    split_data = {}
    for split in ['train_split', 'val_split', 'test_split']:
        path = f"data/unified_136class/{split}.npz"
        d = np.load(path, allow_pickle=True)
        feats = d['features']
        labels = d['labels']
        c_names = d['class_names']
        assert 'recording_group_ids' in d.files, f"{split} missing recording_group_ids"
        split_data[split] = d
        
        assert len(feats) == len(labels), f"{split} length mismatch"
        assert feats.shape[1:] == (32, 133, 6), f"{split} feature shape mismatch: {feats.shape}"
        assert not np.isnan(feats).any(), f"{split} contains NaNs"
        assert not np.isinf(feats).any(), f"{split} contains Infs"
        
        # Check that each sample's integer label matches its class name in authoritative manifest
        for idx, (lbl, s_cls) in enumerate(zip(labels, c_names)):
            m_cls = classes[lbl]
            assert m_cls == s_cls, f"{split} class mismatch at sample {idx}: manifest[{lbl}]={m_cls} vs sample={s_cls}"
            
        print(f"[+] {split}: {len(feats)} samples, {len(set(labels))} classes represented. ALL VALID.")

    groups = {name: set(data['recording_group_ids'].astype(str)) for name, data in split_data.items()}
    assert not groups['train_split'] & groups['val_split']
    assert not groups['train_split'] & groups['test_split']
    assert not groups['val_split'] & groups['test_split']

    signers = {name: set(data['signer_ids'].astype(str)) - {'UNKNOWN_SIGNER'} for name, data in split_data.items()}
    signer_independent = bool(signers['train_split'] and not (
        signers['train_split'] & signers['val_split'] or
        signers['train_split'] & signers['test_split'] or
        signers['val_split'] & signers['test_split']))
    print(f"[+] Recording-group overlap: train/val=0, train/test=0, val/test=0")
    print(f"[+] Signer-independent split available: {signer_independent}")
        
    print("\n[SUCCESS] Split integrity, label alignment, and feature shapes are 100% verified.")

if __name__ == "__main__":
    verify_splits()
