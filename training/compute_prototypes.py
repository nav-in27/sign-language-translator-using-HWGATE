"""
Computes and exports class prototypes for OOD / Unknown sign rejection.
Calculates 500 class centroid prototypes in the 128-dim embedding space.
"""
import os
import sys
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, BASE_DIR)

from models.ac_hwgate import ACHWGATE

def compute_prototypes():
    device = torch.device("cpu")
    model_path = os.path.join(BASE_DIR, "checkpoints", "ac_hwgate_best.pth")
    dataset_path = os.path.join(BASE_DIR, "dataset_packed.pt")
    
    print("[*] Loading AC-HWGATE model...")
    model = ACHWGATE(num_classes=500, in_channels=6).to(device)
    if os.path.exists(model_path):
        chk = torch.load(model_path, map_location=device)
        model.load_state_dict(chk['model_state_dict'], strict=False)
        print(f"[+] Loaded weights from {model_path}")
    model.eval()
    
    # Initialize prototypes array [500, 128]
    embed_dim = 128
    num_classes = 500
    prototypes = np.zeros((num_classes, embed_dim), dtype=np.float32)
    counts = np.zeros(num_classes, dtype=np.int32)
    
    if os.path.exists(dataset_path):
        print(f"[*] Extracting embeddings from {dataset_path}...")
        data = torch.load(dataset_path, map_location=device)
        X = data['keypoints'] # [1500, 32, 133, 6]
        y = data['labels']    # [1500]
        
        batch_size = 64
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch_x = X[i:i+batch_size].to(device)
                batch_y = y[i:i+batch_size].numpy()
                
                _, feat_tensor = model(batch_x, return_embeddings=True)
                feat = feat_tensor.cpu().numpy()
                
                # Normalize features
                norms = np.linalg.norm(feat, axis=1, keepdims=True) + 1e-8
                feat_norm = feat / norms
                
                for emb, label in zip(feat_norm, batch_y):
                    prototypes[label] += emb
                    counts[label] += 1
                    
        print(f"[+] Computed embeddings for {len(X)} samples across {np.sum(counts > 0)} classes.")
        
    # For any class without samples, fallback to normalized classifier weights
    classifier_weights = model.classifier.weight.detach().cpu().numpy() # [500, 128]
    w_norms = np.linalg.norm(classifier_weights, axis=1, keepdims=True) + 1e-8
    classifier_weights_norm = classifier_weights / w_norms
    
    for c in range(num_classes):
        if counts[c] > 0:
            prototypes[c] /= counts[c]
            p_norm = np.linalg.norm(prototypes[c]) + 1e-8
            prototypes[c] /= p_norm
        else:
            prototypes[c] = classifier_weights_norm[c]
            
    # Save to both training and backend
    out_paths = [
        os.path.join(BASE_DIR, "class_prototypes.npy"),
        os.path.join(ROOT_DIR, "backend", "models", "class_prototypes.npy")
    ]
    for outp in out_paths:
        os.makedirs(os.path.dirname(outp), exist_ok=True)
        np.save(outp, prototypes)
        print(f"[+] Saved class prototypes to {outp} (shape: {prototypes.shape})")

if __name__ == "__main__":
    compute_prototypes()
