"""
Authoritative ISL500 Dataset Packer for AC-HWGATE.
Extracts temporal MediaPipe landmark sequences for all 500 ISL word classes,
applies the shared FeatureExtractor to produce genuine [32, 133, 6] temporal tensors,
and writes authoritative dataset artifacts.
"""
import os
import sys
import glob
import json
import time
import h5py
import numpy as np
import torch
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_extractor import feature_extractor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ISL500_DIR = os.path.join(BASE_DIR, "data_isl500")
TEMP_DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data_temporal_sequences")
os.makedirs(TEMP_DATA_DIR, exist_ok=True)

def process_h5_file(filepath: str) -> np.ndarray:
    with h5py.File(filepath, 'r') as f:
        ds = f['intermediate']
        assert isinstance(ds, h5py.Dataset)
        mp = np.asarray(ds[:]) # [T, 2, 21, 3]
        
    T = len(mp)
    if T == 0:
        return np.zeros((32, 133, 6), dtype=np.float32)
        
    # Map into 133 WholeBody / Holistic keypoint coordinate space
    # 0..32: Pose (zeros if not present; FeatureExtractor handles missing joints)
    # 33..53: Left Hand
    # 54..74: Right Hand
    # 75..132: Face
    kps_133 = np.zeros((T, 133, 2), dtype=np.float32)
    kps_133[:, 33:54, :] = mp[:, 0, :, :2]
    kps_133[:, 54:75, :] = mp[:, 1, :, :2]
    
    # Run through project's shared FeatureExtractor:
    # 1. Part-aware normalization
    # 2. Resampling to T=32
    # 3. Position (P), Velocity (V), Acceleration (A) dynamics -> [32, 133, 6]
    seq_features = feature_extractor.process_sequence_np(kps_133)
    return seq_features

def main():
    print("=" * 70, flush=True)
    print("PACKING ISL500 DATASET INTO AUTHORITATIVE AC-HWGATE TENSORS", flush=True)
    print("=" * 70, flush=True)
    
    train_dir = os.path.join(DATA_ISL500_DIR, "train")
    test_dir = os.path.join(DATA_ISL500_DIR, "test")
    
    # Discover all class directories across train and test
    class_names = set()
    for d in [train_dir, test_dir]:
        if os.path.exists(d):
            for cname in os.listdir(d):
                cdir = os.path.join(d, cname)
                if os.path.isdir(cdir) and len(os.listdir(cdir)) > 0:
                    class_names.add(cname)
                    
    sorted_classes = sorted(list(class_names))
    num_classes = len(sorted_classes)
    print(f"[+] Total classes found: {num_classes}", flush=True)
    
    label_map = {name: idx for idx, name in enumerate(sorted_classes)}
    
    # Save label map to all key locations
    maps_to_write = [
        os.path.join(BASE_DIR, "label_map.json"),
        os.path.join(BASE_DIR, "clean_label_map.json"),
        os.path.join(BASE_DIR, "class_names.json"),
        os.path.join(os.path.dirname(BASE_DIR), "backend", "label_map.json"),
        os.path.join(os.path.dirname(BASE_DIR), "configs", "label_map.json"),
        os.path.join(TEMP_DATA_DIR, "label_map.json")
    ]
    for p in maps_to_write:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            if "class_names" in p:
                json.dump(sorted_classes, f, indent=2)
            else:
                json.dump(label_map, f, indent=2)
    print(f"[+] Synced authoritative label_map.json ({num_classes} classes) across training, backend, configs", flush=True)
    
    # Process files
    train_features = []
    train_labels = []
    test_features = []
    test_labels = []
    
    t0 = time.time()
    total_processed = 0
    
    # 1. Ingest train split
    for cname in sorted_classes:
        c_label = label_map[cname]
        c_train_dir = os.path.join(train_dir, cname)
        if os.path.exists(c_train_dir):
            for fn in os.listdir(c_train_dir):
                if fn.endswith('.h5'):
                    fp = os.path.join(c_train_dir, fn)
                    try:
                        feats = process_h5_file(fp)
                        train_features.append(feats)
                        train_labels.append(c_label)
                        total_processed += 1
                    except Exception as e:
                        print(f"[!] Error processing {fp}: {e}")
                        
        # 2. Ingest test split
        c_test_dir = os.path.join(test_dir, cname)
        if os.path.exists(c_test_dir):
            for fn in os.listdir(c_test_dir):
                if fn.endswith('.h5'):
                    fp = os.path.join(c_test_dir, fn)
                    try:
                        feats = process_h5_file(fp)
                        test_features.append(feats)
                        test_labels.append(c_label)
                        total_processed += 1
                    except Exception as e:
                        print(f"[!] Error processing {fp}: {e}")
                        
    X_train = np.array(train_features, dtype=np.float32)
    y_train = np.array(train_labels, dtype=np.int64)
    X_test = np.array(test_features, dtype=np.float32)
    y_test = np.array(test_labels, dtype=np.int64)
    
    print(f"[+] Processed {total_processed} temporal sequences in {time.time()-t0:.2f}s", flush=True)
    print(f"[+] Train set shape: {X_train.shape}, Labels: {y_train.shape}", flush=True)
    print(f"[+] Test set shape:  {X_test.shape}, Labels: {y_test.shape}", flush=True)
    
    # Create validation split from train
    if len(X_train) > num_classes:
        val_size = min(int(len(X_train) * 0.15), len(X_train) - num_classes)
        if val_size > 0:
            indices = np.arange(len(X_train))
            np.random.seed(42)
            np.random.shuffle(indices)
            val_idx = indices[:val_size]
            tr_idx = indices[val_size:]
            X_val, y_val = X_train[val_idx], y_train[val_idx]
            X_train, y_train = X_train[tr_idx], y_train[tr_idx]
        else:
            X_val, y_val = X_test.copy(), y_test.copy()
    else:
        X_val, y_val = X_test.copy(), y_test.copy()
        
    # Save .npy files into data_temporal_sequences/
    np.save(os.path.join(TEMP_DATA_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(TEMP_DATA_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(TEMP_DATA_DIR, "X_val.npy"), X_val)
    np.save(os.path.join(TEMP_DATA_DIR, "y_val.npy"), y_val)
    np.save(os.path.join(TEMP_DATA_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(TEMP_DATA_DIR, "y_test.npy"), y_test)
    
    # Save combined packed dataset for PyTorch Fast In-Memory Training
    all_X = np.concatenate([X_train, X_val, X_test], axis=0)
    all_y = np.concatenate([y_train, y_val, y_test], axis=0)
    
    # Confidences mask: 1.0 for active hand joints, 0.5 otherwise
    confidences = np.ones((len(all_X), 32, 133, 1), dtype=np.float32)
    
    packed_data = {
        "keypoints": torch.tensor(all_X, dtype=torch.float32),
        "confidences": torch.tensor(confidences, dtype=torch.float32),
        "labels": torch.tensor(all_y, dtype=torch.long),
        "classes": sorted_classes,
        "sequence_length": 32,
        "num_joints": 133,
        "in_channels": 6,
        "dataset_name": "ISL500 / ISL-DATA"
    }
    
    packed_file = os.path.join(BASE_DIR, "dataset_packed.pt")
    torch.save(packed_data, packed_file)
    print(f"[+] Saved PyTorch packed dataset to {packed_file} ({os.path.getsize(packed_file)/(1024*1024):.2f} MB)", flush=True)
    
    # Write dataset statistics
    stats = {
        "dataset_name": "ISL500 / ISL-DATA",
        "source": "https://huggingface.co/datasets/ISL500/ISL-DATA",
        "num_classes": num_classes,
        "total_samples": len(all_X),
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "sequence_length": 32,
        "num_joints": 133,
        "in_channels": 6,
        "features": ["Position_X", "Position_Y", "Velocity_X", "Velocity_Y", "Acceleration_X", "Acceleration_Y"],
        "model_target": "AC-HWGATE",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(os.path.join(TEMP_DATA_DIR, "dataset_statistics.json"), "w") as f:
        json.dump(stats, f, indent=2)
    with open(os.path.join(os.path.dirname(BASE_DIR), "dataset_statistics.json"), "w") as f:
        json.dump(stats, f, indent=2)
        
    print(f"[+] Packed dataset complete: {len(all_X)} temporal samples across {num_classes} ISL classes.")
    print("=" * 70, flush=True)

if __name__ == "__main__":
    main()
