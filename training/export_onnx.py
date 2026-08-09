import os
import sys
import torch
import numpy as np
import json
import onnxruntime as ort

# Add parent dir to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.baseline_models import SimpleLSTM

def export_model_to_onnx(model_path, onnx_output_path, default_num_classes=264):
    device = torch.device('cpu')
    num_classes = default_num_classes
    state_dict = None
    
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        if 'fc.weight' in state_dict:
            num_classes = state_dict['fc.weight'].shape[0]
        elif 'classifier.weight' in state_dict:
            num_classes = state_dict['classifier.weight'].shape[0]
        elif 'classifier.3.weight' in state_dict:
            num_classes = state_dict['classifier.3.weight'].shape[0]
        print(f"Loaded PyTorch model from: {model_path} (Detected num_classes={num_classes})")
    
    model = SimpleLSTM(num_classes=num_classes, num_joints=80, in_channels=3).to(device)
    if state_dict is not None:
        try:
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Warning loading state dict: {e}")
        
    model.eval()
    
    # Dummy inputs: keypoints [1, 64, 80, 3] & confidences [1, 64, 80, 1]
    dummy_keypoints = torch.randn(1, 64, 80, 3, dtype=torch.float32)
    dummy_confidences = torch.ones(1, 64, 80, 1, dtype=torch.float32)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        (dummy_keypoints, dummy_confidences),
        onnx_output_path,
        export_params=True,
        input_names=['input_keypoints', 'confidences'],
        output_names=['logits'],
        dynamic_axes={
            'input_keypoints': {0: 'batch_size', 1: 'sequence_length'},
            'confidences': {0: 'batch_size', 1: 'sequence_length'},
            'logits': {0: 'batch_size'}
        },
        opset_version=14,
        dynamo=False
    )
    print(f"ONNX model exported successfully to: {onnx_output_path}")
    
    # Export Preprocessing Config
    prep_config = {
        "num_classes": num_classes,
        "input_shape": [1, 64, 80, 3],
        "keypoint_format": "pose_33 + lh_21 + rh_21 + pad_5",
        "normalization": {
            "origin": "shoulder_midpoint",
            "scale": "inter_shoulder_distance"
        },
        "rejection_threshold": 0.65,
        "min_frames": 15,
        "window_size": 64
    }
    config_path = os.path.join(os.path.dirname(onnx_output_path), "preprocessing_config.json")
    with open(config_path, "w") as f:
        json.dump(prep_config, f, indent=4)
    print(f"Preprocessing config saved to: {config_path}")
    
    # Validate Parity
    ort_session = ort.InferenceSession(onnx_output_path)
    inputs = ort_session.get_inputs()
    if len(inputs) >= 2:
        ort_inputs = {inputs[0].name: dummy_keypoints.numpy(), inputs[1].name: dummy_confidences.numpy()}
    else:
        ort_inputs = {inputs[0].name: dummy_keypoints.numpy()}
        
    ort_outs = ort_session.run(None, ort_inputs)
    
    with torch.no_grad():
        pt_out = model(dummy_keypoints, dummy_confidences).numpy()
        
    max_diff = np.max(np.abs(pt_out - ort_outs[0]))
    print(f"Max absolute difference PyTorch vs ONNX: {max_diff:.6e}")
    if max_diff < 1e-4:
        print("SUCCESS: PyTorch and ONNX predictions are numerically equivalent (< 1e-4)!")
    else:
        print("WARNING: ONNX numerical discrepancy detected!")
        
    return max_diff < 1e-4

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "LSTM_weights.pth")
    onnx_path = os.path.join(base_dir, "model.onnx")
    export_model_to_onnx(model_path, onnx_path)
