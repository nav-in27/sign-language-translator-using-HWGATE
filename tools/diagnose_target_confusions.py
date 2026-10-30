import os
import json
import numpy as np
import torch
import onnxruntime as ort

data_dir = 'data/unified_136class'
with open('backend/label_map.json', 'r') as f:
    label_map = json.load(f)
c2i = label_map['class_to_idx']
i2c = label_map['idx_to_class']

target_classes = ['beautiful', 'good', 'hello', 'thank_you', 'give', 'onion']

print("=== 1. SAMPLE COUNTS IN 136-CLASS SPLITS ===")
for split in ['train', 'val', 'test']:
    p = os.path.join(data_dir, f'{split}_split.npz')
    if not os.path.exists(p):
        print(f"Missing {p}")
        continue
    data = np.load(p, allow_pickle=True)
    features = data['features']
    labels = data['labels']
    sources = data['sources'] if 'sources' in data else []
    print(f"\n--- Split: {split} (Total samples: {len(labels)}) ---")
    for cls_name in target_classes:
        idx = c2i[cls_name]
        mask = (labels == idx)
        cnt = int(np.sum(mask))
        pct = (cnt / len(labels)) * 100
        srcs = list(set(sources[mask])) if len(sources) > 0 and cnt > 0 else []
        print(f"  {cls_name:12s} (idx {idx:3d}): {cnt:3d} samples ({pct:5.2f}%) | sources: {srcs}")

print("\n=== 2. RUNNING ONNX PRODUCTION MODEL INFERENCE ON TARGET CLASSES ===")
onnx_path = 'backend/models/ac_hwgate.onnx'
if not os.path.exists(onnx_path):
    onnx_path = 'models/ac_hwgate.onnx'

session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
output_names = [o.name for o in session.get_outputs()]
print(f"Loaded ONNX model: {onnx_path}")
print(f"Input name: {input_name}, shape: {session.get_inputs()[0].shape}")
print(f"Output names: {output_names}")

for split in ['val', 'test']:
    p = os.path.join(data_dir, f'{split}_split.npz')
    data = np.load(p, allow_pickle=True)
    features = data['features'] # (N, 32, 133, 6)
    labels = data['labels']
    print(f"\n=======================================================")
    print(f"=== SPLIT: {split.upper()} EVALUATION ON TARGET CLASSES ===")
    print(f"=======================================================")
    for cls_name in target_classes:
        target_idx = c2i[cls_name]
        indices = np.where(labels == target_idx)[0]
        if len(indices) == 0:
            print(f"No samples for {cls_name} in {split}")
            continue
        print(f"\n--- Target: {cls_name.upper()} (idx {target_idx}, {len(indices)} samples in {split}) ---")
        correct_count = 0
        for i, idx in enumerate(indices):
            feat = features[idx:idx+1] # (1, 32, 133, 6)
            # AC-HWGATE expects [B, 32, 133, 6] or [B, 32, 798]?
            # Let's check input shape required by ONNX:
            expected_shape = session.get_inputs()[0].shape
            if len(expected_shape) == 4:
                inp = feat.astype(np.float32)
            else:
                inp = feat.reshape(1, 32, -1).astype(np.float32)
            
            outputs = session.run(None, {input_name: inp})
            logits = outputs[0][0] # (136,)
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)
            
            top5_indices = np.argsort(probs)[::-1][:5]
            top1_idx = top5_indices[0]
            top1_name = i2c[str(top1_idx)]
            top1_prob = probs[top1_idx]
            margin = top1_prob - probs[top5_indices[1]] if len(top5_indices) > 1 else top1_prob
            
            is_correct = (top1_idx == target_idx)
            if is_correct:
                correct_count += 1
            status_str = "MATCH" if is_correct else "CONFUSED"
            
            top5_str = ", ".join([f"{i2c[str(ti)]}:{probs[ti]:.3f}" for ti in top5_indices])
            print(f"  Sample #{i+1} [{status_str}]: Top-1={top1_name} ({top1_prob:.3f}, margin={margin:.3f}) | Top-5: [{top5_str}]")
        acc = (correct_count / len(indices)) * 100
        print(f"  --> Accuracy for {cls_name}: {correct_count}/{len(indices)} ({acc:.1f}%)")
