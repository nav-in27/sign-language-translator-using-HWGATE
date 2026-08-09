import sys, os, json
import numpy as np
import torch

SCRATCH_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ISL_REC_DIR = os.path.join(SCRATCH_DIR, "isl_recognition")
sys.path.append(ISL_REC_DIR)

from model import ISLModel as HWGATEModel

# Load label map
with open(os.path.join(ISL_REC_DIR, "label_map.json"), "r") as f:
    label_map = json.load(f)
reverse_map = {v: k for k, v in label_map.items()}
num_classes = len(label_map)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HWGATEModel(num_classes=num_classes).to(device)
weights_path = os.path.join(ISL_REC_DIR, "best_model.pth")
model.load_state_dict(torch.load(weights_path, map_location=device))
model.eval()
print(f"Model loaded. Num classes: {num_classes}")

# Load a REAL training sample and run it through the model
import glob
dataset_dir = r"C:\Users\navee\Downloads\archive (9)"
npy_files = glob.glob(os.path.join(dataset_dir, '**', '*.npy'), recursive=True)[:5]

for npy_path in npy_files:
    label = os.path.basename(os.path.dirname(npy_path))
    sequence = np.load(npy_path)
    num_frames = sequence.shape[0]
    
    # Exact same preprocessing as dataset.py
    pose = sequence[:, 0:132].reshape(num_frames, 33, 4)[:, :, :3]
    lh = sequence[:, 132:195].reshape(num_frames, 21, 3)
    rh = sequence[:, 195:258].reshape(num_frames, 21, 3)
    combined = np.concatenate([pose, lh, rh], axis=1)
    pad_kps = np.zeros((num_frames, 5, 3))
    seq = np.concatenate([combined, pad_kps], axis=1)  # (num_frames, 80, 3)
    
    # Pad/truncate to 64 frames
    if seq.shape[0] > 64:
        seq = seq[:64]
    elif seq.shape[0] < 64:
        pad_len = 64 - seq.shape[0]
        padding = np.zeros((pad_len, 80, 3), dtype=np.float32)
        seq = np.vstack((seq, padding))
    
    seq = seq.astype(np.float32)
    seq_tensor = torch.tensor(seq).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(seq_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        pred_class = reverse_map[predicted_idx.item()]
    
    print(f"True: {label:30s} | Predicted: {pred_class:30s} | Confidence: {confidence.item():.4f}")
    
    # Also show top-5 predictions
    top5_conf, top5_idx = torch.topk(probabilities, 5, dim=1)
    top5_labels = [(reverse_map[idx.item()], conf.item()) for idx, conf in zip(top5_idx[0], top5_conf[0])]
    for lbl, conf in top5_labels:
        print(f"    {lbl:30s}: {conf:.4f}")
    print()
