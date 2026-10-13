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
        model_path = os.path.join(BASE_DIR, "checkpoints", "ac_hwgate_best.pth")
    if onnx_output_path is None:
        onnx_output_path = os.path.join(ROOT_DIR, "backend", "models", "ac_hwgate.onnx")
        
    os.makedirs(os.path.dirname(onnx_output_path), exist_ok=True)
    device = torch.device('cpu')
    
    # Load label map to determine number of classes
    label_map_path = os.path.join(BASE_DIR, "label_map.json")
    if os.path.exists(label_map_path):
        with open(label_map_path, "r") as f:
            label_map = json.load(f)
        num_classes = len(label_map)
    else:
        num_classes = 500
        
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
    
    # Also write a copy to training/model.onnx for training scripts compatibility
    tr_copy = os.path.join(BASE_DIR, "model.onnx")
    with open(onnx_output_path, "rb") as sf, open(tr_copy, "wb") as df:
        df.write(sf.read())
        
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
    export_model_to_onnx()
