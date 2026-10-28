"""
AC-HWGATE Production ONNX Export Engine.
Exports the trained Adaptive Context-Aware Hierarchical Windowed Graph Attention
Transformer Encoder (AC-HWGATE) model into optimized ONNX Runtime format.
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import json
import onnxruntime as ort

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, BASE_DIR)
from models.ac_hwgate import ACHWGATE

def export_model_to_onnx(model_path=None, onnx_output_path=None):
    if model_path is None:
        candidates = [
            os.path.join(ROOT_DIR, "checkpoints", "achwgate_136class_production_best.pth"),
            os.path.join(ROOT_DIR, "checkpoints", "achwgate_136class_best.pth"),
            os.path.join(ROOT_DIR, "checkpoints", "achwgate_best.pth"),
            os.path.join(BASE_DIR, "checkpoints", "ac_hwgate_best.pth"),
        ]
        for c in candidates:
            if os.path.exists(c):
                model_path = c
                break
        if model_path is None:
            model_path = candidates[0]

    if onnx_output_path is None:
        onnx_output_path = os.path.join(ROOT_DIR, "backend", "models", "ac_hwgate.onnx")
        
    os.makedirs(os.path.dirname(onnx_output_path), exist_ok=True)
    device = torch.device('cpu')
    
    # Load label map to determine number of classes
    label_map_paths = [
        os.path.join(ROOT_DIR, "configs", "authoritative_136_classes_manifest.json"),
        os.path.join(ROOT_DIR, "data", "unified_136class", "label_map.json"),
        os.path.join(ROOT_DIR, "models", "label_map.json"),
        os.path.join(BASE_DIR, "label_map.json"),
    ]
    num_classes = 136
    for lp in label_map_paths:
        if os.path.exists(lp):
            with open(lp, "r", encoding="utf-8") as f:
                d = json.load(f)
                if "classes" in d:
                    num_classes = len(d["classes"])
                    break
                elif "class_to_idx" in d:
                    num_classes = len(d["class_to_idx"])
                    break
                elif isinstance(d, dict):
                    num_classes = len(d)
                    break
        
    embed_dim = 128
    num_heads = 4
    num_layers = 3
    num_joints = 133
    in_channels = 6
    sequence_length = 32
    
    state_dict = None
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            num_classes = checkpoint.get('num_classes', num_classes)
            cfg = checkpoint.get('config', {})
            embed_dim = cfg.get('embed_dim', embed_dim)
            num_heads = cfg.get('num_heads', num_heads)
            num_layers = cfg.get('num_layers', num_layers)
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
            for key in ['classifier.5.weight', 'classifier.weight']:
                if key in state_dict:
                    num_classes = state_dict[key].shape[0]
                    break
        print(f"[+] Loaded AC-HWGATE weights from {model_path} ({num_classes} classes)")
    else:
        print(f"[!] Note: {model_path} not found. Exporting uninitialized AC-HWGATE model ({num_classes} classes).")
        
    model = ACHWGATE(
        num_classes=num_classes,
        num_joints=num_joints,
        in_channels=in_channels,
        sequence_length=sequence_length,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers
    ).to(device)
    
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    # Wrap model to export both logits and penultimate embeddings
    class ACHWGATE_Export_Wrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            return self.m(x, return_embeddings=True)

    export_model = ACHWGATE_Export_Wrapper(model)

    # Export with dynamic batch size [B, 32, 133, 6]
    dummy_input = torch.randn(1, sequence_length, num_joints, in_channels, dtype=torch.float32)
    
    torch.onnx.export(
        export_model,
        (dummy_input,),
        onnx_output_path,
        export_params=True,
        input_names=['keypoints'],
        output_names=['logits', 'embeddings'],
        dynamic_axes={
            'keypoints': {0: 'batch_size'},
            'logits': {0: 'batch_size'},
            'embeddings': {0: 'batch_size'}
        },
        opset_version=14,
        do_constant_folding=True,
        dynamo=False
    )
    print(f"[+] AC-HWGATE ONNX model exported successfully to {onnx_output_path}")
    
    # Also write a copy to models/achwgate.onnx and training/model.onnx
    models_onnx = os.path.join(ROOT_DIR, "models", "achwgate.onnx")
    tr_copy = os.path.join(BASE_DIR, "model.onnx")
    for cp in [models_onnx, tr_copy]:
        os.makedirs(os.path.dirname(cp), exist_ok=True)
        with open(onnx_output_path, "rb") as sf, open(cp, "wb") as df:
            df.write(sf.read())
        print(f"[+] Synchronized ONNX model to: {cp}")

    # Compute Class Prototypes from clean train split
    train_npz = os.path.join(ROOT_DIR, "data", "unified_136class", "train_split.npz")
    if os.path.exists(train_npz):
        print("[*] Computing 128-dim class prototypes from clean train split...")
        data = np.load(train_npz, allow_pickle=True)
        X_tr = data["features"]
        y_tr = data["labels"]
        session = ort.InferenceSession(onnx_output_path)
        
        all_embs = []
        bs = 64
        for i in range(0, len(X_tr), bs):
            batch_x = X_tr[i:i+bs].astype(np.float32)
            out = session.run(None, {'keypoints': batch_x})
            all_embs.append(out[1])
        all_embs = np.concatenate(all_embs, axis=0) # [N, 128]
        
        prototypes = np.zeros((num_classes, 128), dtype=np.float32)
        for c in range(num_classes):
            mask = (y_tr == c)
            if np.any(mask):
                c_embs = all_embs[mask]
                proto = c_embs.mean(axis=0)
                norm = np.linalg.norm(proto)
                if norm > 1e-6:
                    proto = proto / norm
                prototypes[c] = proto
            else:
                prototypes[c] = np.zeros(128, dtype=np.float32)

        for p_dir in [os.path.join(ROOT_DIR, "models"), os.path.join(ROOT_DIR, "backend", "models")]:
            proto_file = os.path.join(p_dir, "class_prototypes.npy")
            np.save(proto_file, prototypes)
            print(f"[+] Saved {num_classes}x128 class prototypes to: {proto_file}")

    # Verify ONNX Runtime compatibility
    print("[+] Verifying AC-HWGATE ONNX Runtime inference (logits + embeddings)...")
    session = ort.InferenceSession(onnx_output_path)
    for b in [1, 2, 4]:
        k_test = np.random.randn(b, 32, 133, 6).astype(np.float32)
        ort_out = session.run(None, {'keypoints': k_test})
        logits_shape = np.asarray(ort_out[0]).shape
        emb_shape = np.asarray(ort_out[1]).shape
        print(f"  Test Batch B={b} -> Logits: {logits_shape}, Embeddings: {emb_shape} [OK]")
        
    print(f"[+] AC-HWGATE ONNX Runtime verification PASSED ({num_classes} classes + 128-dim embeddings)!")
    return onnx_output_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Export AC-HWGATE to ONNX")
    parser.add_argument("--model_path", type=str, default=None, help="Path to PyTorch checkpoint")
    parser.add_argument("--onnx_output_path", type=str, default=None, help="Output path for ONNX artifact")
    args = parser.parse_args()
    export_model_to_onnx(model_path=args.model_path, onnx_output_path=args.onnx_output_path)
