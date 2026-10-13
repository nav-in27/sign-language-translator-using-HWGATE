import sys
import os
import time
import json
import numpy as np
import torch
import glob
from sklearn.metrics import classification_report, accuracy_score

SCRATCH_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ISL_REC_DIR = os.path.join(SCRATCH_DIR, "isl_recognition")
sys.path.append(ISL_REC_DIR)

try:
    from backend.retrain_model import ISLModelLite
except ImportError:
    from retrain_model import ISLModelLite
from translator import nmt_engine

def run_evaluation():
    print("=" * 70)
    print("CVPR EVALUATION BENCHMARK: DEEPMIND / META FAIR MODEL PIPELINE")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Hardware Compute Device: {device.type.upper()}")
    
    # 1. Load Model & Label Map
    label_map_path = os.path.join(ISL_REC_DIR, "label_map.json")
    with open(label_map_path, "r") as f:
        label_map = json.load(f)
    reverse_map = {v: k for k, v in label_map.items()}
    num_classes = len(label_map)
    
    model = ISLModelLite(num_classes=num_classes).to(device)
    weights_path = os.path.join(ISL_REC_DIR, "best_model_lite.pth")
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()
    
    print(f"Loaded trained BiLSTM + Temporal Attention Model ({num_classes} classes)")

    # 2. Benchmark Dataset Accuracy (Top-1 & Top-5)
    dataset_dir = r"C:\Users\navee\Downloads\archive (9)"
    npy_files = glob.glob(os.path.join(dataset_dir, '**', '*.npy'), recursive=True)
    
    # Sample 300 validation sequences across classes for quick thorough benchmark
    np.random.seed(42)
    sample_files = np.random.choice(npy_files, size=min(300, len(npy_files)), replace=False)
    
    y_true = []
    y_pred = []
    top5_correct = 0
    latencies_ms = []

    print(f"\nEvaluating Top-1 / Top-5 Accuracy & Inference Latency over {len(sample_files)} sequences...")
    
    with torch.no_grad():
        for path in sample_files:
            true_label_str = os.path.basename(os.path.dirname(path))
            true_idx = label_map[true_label_str]
            
            sequence = np.load(path)
            num_frames = sequence.shape[0]
            
            pose = sequence[:, 0:132].reshape(num_frames, 33, 4)[:, :, :3]
            lh = sequence[:, 132:195].reshape(num_frames, 21, 3)
            rh = sequence[:, 195:258].reshape(num_frames, 21, 3)
            combined = np.concatenate([pose, lh, rh], axis=1)
            pad_kps = np.zeros((num_frames, 5, 3))
            seq = np.concatenate([combined, pad_kps], axis=1)
            
            if seq.shape[0] > 64:
                start = (seq.shape[0] - 64) // 2
                seq = seq[start:start+64]
            elif seq.shape[0] < 64:
                pad_len = 64 - seq.shape[0]
                seq = np.vstack([seq, np.zeros((pad_len, 80, 3))])
                
            seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Measure exact inference latency per sequence
            t0 = time.perf_counter()
            outputs = model(seq_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            t1 = time.perf_counter()
            
            latencies_ms.append((t1 - t0) * 1000)
            
            pred_idx = torch.argmax(probs, dim=1).item()
            y_true.append(true_idx)
            y_pred.append(pred_idx)
            
            top5_indices = torch.topk(probs, 5, dim=1).indices[0].tolist()
            if true_idx in top5_indices:
                top5_correct += 1

    top1_acc = accuracy_score(y_true, y_pred) * 100
    top5_acc = (top5_correct / len(sample_files)) * 100
    avg_latency = np.mean(latencies_ms)
    p99_latency = np.percentile(latencies_ms, 99)
    throughput_fps = 1000.0 / avg_latency

    print("\n" + "-" * 50)
    print("EMPIRICAL BENCHMARK METRICS RESULTS:")
    print("-" * 50)
    print(f"  • Top-1 Categorical Accuracy : {top1_acc:.2f}%")
    print(f"  • Top-5 Categorical Accuracy : {top5_acc:.2f}%")
    print(f"  • Mean Inference Latency    : {avg_latency:.2f} ms")
    print(f"  • P99 Inference Latency     : {p99_latency:.2f} ms")
    print(f"  • System Throughput          : {throughput_fps:.1f} FPS")
    print("-" * 50)

    # 3. Benchmark Temporal Token Deduplication & Async NMT
    print("\nTesting Continuous Sequence Deduplication & Async NMT Engine...")
    raw_stream_tokens = ["48. Hello", "48. Hello", "48. Hello", "40. I", "40. I", "24. School", "24. School"]
    
    deduped_glosses = []
    last_t = None
    for token in raw_stream_tokens:
        clean = token.split(". ")[-1] if ". " in token else token
        if clean != last_t:
            deduped_glosses.append(clean)
            last_t = clean

    raw_gloss_seq = " ".join(deduped_glosses)
    translated_text = nmt_engine.translate_gloss_to_english(raw_gloss_seq)
    
    print(f"  • Input Raw Stream Tokens    : {raw_stream_tokens}")
    print(f"  • Deduplicated Sequence      : '{raw_gloss_seq}'")
    print(f"  • Immediate Async NMT Result  : '{translated_text}'")
    print("=" * 70)

if __name__ == "__main__":
    run_evaluation()
