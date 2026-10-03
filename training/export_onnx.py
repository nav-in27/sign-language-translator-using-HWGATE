import os
import sys
import torch
import numpy as np
import json
import onnxruntime as ort

# Add parent dir to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.cwa_hwgate import CWA_HWGATE

def export_model_to_onnx(model_path=None, onnx_output_path=None):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if model_path is None:
        model_path = os.path.join(base_dir, "checkpoints", "best_model.pth")
        if not os.path.exists(model_path):
            model_path = os.path.join(base_dir, "cwa_hwgate_pretrained.pth")
    if onnx_output_path is None:
        onnx_output_path = os.path.join(base_dir, "model.onnx")
        
    device = torch.device('cpu')
    label_map_path = os.path.join(base_dir, "label_map.json")
    with open(label_map_path, "r") as f:
        label_map = json.load(f)
    num_classes = len(label_map)
    
    embed_dim = 256
    num_heads = 8
    num_layers = 4
    num_joints = 133
    in_channels = 2
    
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
            for key in ['classifier.6.weight', 'classifier.weight']:
                if key in state_dict:
                    num_classes = state_dict[key].shape[0]
                    break
        print(f"[+] Loaded weights from {model_path} ({num_classes} classes)")
    else:
        print(f"[!] Warning: {model_path} not found. Exporting uninitialized model.")
        
    model = CWA_HWGATE(
        num_classes=num_classes,
        num_joints=num_joints,
        in_channels=in_channels,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers
    ).to(device)
    
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    # Export with dynamic sequence length and batch size
    dummy_keypoints = torch.randn(1, 16, num_joints, in_channels, dtype=torch.float32)
    dummy_confidences = torch.ones(1, 16, num_joints, 1, dtype=torch.float32)
    
    torch.onnx.export(
        model,
        (dummy_keypoints, dummy_confidences),
        onnx_output_path,
        export_params=True,
        input_names=['keypoints', 'confidences'],
        output_names=['logits'],
        dynamic_axes={
            'keypoints': {0: 'batch_size', 1: 'sequence_length'},
            'confidences': {0: 'batch_size', 1: 'sequence_length'},
            'logits': {1: 'batch_size'}
        },
        opset_version=14,
        do_constant_folding=True,
        dynamo=False
    )
    print(f"[+] ONNX model exported successfully to {onnx_output_path}")
    
    # Verify ONNX Runtime compatibility across multiple sequence lengths
    print("[+] Verifying ONNX model across dynamic sequence lengths (1, 8, 16, 64)...")
    session = ort.InferenceSession(onnx_output_path)
    for T in [1, 8, 16, 64]:
        k_test = np.random.randn(1, T, 133, 2).astype(np.float32)
        c_test = np.ones((1, T, 133, 1), dtype=np.float32)
        ort_out = session.run(None, {'keypoints': k_test, 'confidences': c_test})
        out_shape = ort_out[0].shape
        print(f"  Test Sequence Length T={T:02d} -> Logits Output Shape: {out_shape} [OK]")
        
    print("[+] ONNX Runtime dynamic sequence verification PASSED!")
    return onnx_output_path

if __name__ == "__main__":
    export_model_to_onnx()
