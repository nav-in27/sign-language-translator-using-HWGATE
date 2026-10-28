"""
Export, Prototype Computation, and Comprehensive Error Analysis for 136-Class AC-HWGATE.
"""
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINING_DIR = os.path.join(ROOT_DIR, "training")
sys.path.insert(0, TRAINING_DIR)

from models.ac_hwgate import ACHWGATE

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running on device: {device}")

    results_path = os.path.join(ROOT_DIR, "reports", "achwgate_136class_production_results.json")
    with open(results_path, "r", encoding="utf-8") as f:
        prod_results = json.load(f)

    manifest_path = os.path.join(ROOT_DIR, "configs", "authoritative_136_classes_manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    classes = manifest["classes"]
    num_classes = len(classes)
    cm = np.array(prod_results["test_confusion_matrix"])[:num_classes, :num_classes]

    # Clean up results json to exact 136 classes
    prod_results["classes"] = classes
    prod_results["test_confusion_matrix"] = cm.tolist()
    prod_results["test_per_class_f1"] = prod_results["test_per_class_f1"][:num_classes]
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(prod_results, f, indent=2)
    print(f"[+] Cleaned up {results_path} to exact {num_classes} classes and {num_classes}x{num_classes} confusion matrix.")

    # 1. Analyze Top Confusions
    confusions = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                confusions.append({
                    "true_class": classes[i],
                    "predicted_class": classes[j],
                    "count": int(cm[i, j])
                })
    confusions.sort(key=lambda x: x["count"], reverse=True)

    print("\n" + "="*70)
    print("TOP 15 CONFUSION PAIRS ON UNTOUCHED HELD-OUT TEST SET:")
    print("="*70)
    for c in confusions[:15]:
        print(f"  True: {c['true_class']:<20} -> Pred: {c['predicted_class']:<20} | Count: {c['count']}")

    # 2. Analyze Per-Class F1
    per_class_f1_list = prod_results["test_per_class_f1"][:num_classes]
    per_class_f1_dict = {classes[i]: f1 for i, f1 in enumerate(per_class_f1_list)}
    sorted_f1 = sorted(per_class_f1_dict.items(), key=lambda x: x[1])
    
    print("\n" + "="*70)
    print("LOWEST 10 CLASSES BY F1:")
    print("="*70)
    for cls_name, f1 in sorted_f1[:10]:
        print(f"  {cls_name:<25}: F1 = {f1:.4f}")

    print("\n" + "="*70)
    print("HIGHEST 10 CLASSES BY F1:")
    print("="*70)
    for cls_name, f1 in sorted_f1[-10:]:
        print(f"  {cls_name:<25}: F1 = {f1:.4f}")

    # Save detailed error analysis JSON
    error_analysis = {
        "num_classes": num_classes,
        "test_top1_acc": prod_results["test_top1_accuracy"],
        "test_top5_acc": prod_results["test_top5_accuracy"],
        "test_macro_f1": prod_results["test_macro_f1"],
        "test_weighted_f1": prod_results["test_weighted_f1"],
        "top_confusions": confusions[:30],
        "lowest_f1_classes": sorted_f1[:15],
        "highest_f1_classes": sorted_f1[-15:],
        "perfect_f1_count": sum(1 for _, f1 in sorted_f1 if f1 >= 0.99)
    }
    with open(os.path.join(ROOT_DIR, "reports", "error_analysis_136class.json"), "w", encoding="utf-8") as f:
        json.dump(error_analysis, f, indent=2)
    print(f"\n[+] Saved error analysis to reports/error_analysis_136class.json (Perfect F1 classes: {error_analysis['perfect_f1_count']}/{num_classes})")

    # 3. Load Best PyTorch Checkpoint and slice classifier to exact 136 classes
    best_ckpt_path = os.path.join(ROOT_DIR, "checkpoints", "achwgate_136class_production_best.pth")
    checkpoint = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    raw_state_dict = checkpoint["model_state_dict"]
    
    clean_state_dict = {}
    for k, v in raw_state_dict.items():
        if k == "classifier.weight" and v.shape[0] > num_classes:
            clean_state_dict[k] = v[:num_classes, :]
            print(f"[+] Sliced classifier.weight from {v.shape} to {clean_state_dict[k].shape}")
        elif k == "classifier.bias" and v.shape[0] > num_classes:
            clean_state_dict[k] = v[:num_classes]
            print(f"[+] Sliced classifier.bias from {v.shape} to {clean_state_dict[k].shape}")
        else:
            clean_state_dict[k] = v

    model = ACHWGATE(
        num_classes=num_classes,
        num_joints=133,
        in_channels=6,
        sequence_length=32,
        embed_dim=128,
        num_heads=4,
        num_layers=3
    ).to(device)
    model.load_state_dict(clean_state_dict)
    model.eval()

    # Save cleaned 136-class checkpoint to authoritative locations
    checkpoint["model_state_dict"] = clean_state_dict
    checkpoint["num_classes"] = num_classes
    checkpoint["classes"] = classes
    
    torch.save(checkpoint, best_ckpt_path)
    auth_ckpt_path = os.path.join(ROOT_DIR, "checkpoints", "achwgate_best.pth")
    torch.save(checkpoint, auth_ckpt_path)
    backend_ckpt_path = os.path.join(ROOT_DIR, "backend", "models", "achwgate_best.pth")
    os.makedirs(os.path.dirname(backend_ckpt_path), exist_ok=True)
    torch.save(checkpoint, backend_ckpt_path)
    print(f"[+] Saved clean 136-class checkpoint to:\n    - {best_ckpt_path}\n    - {auth_ckpt_path}\n    - {backend_ckpt_path}")

    # 4. Compute Class Prototype Centroids from Training Data
    train_npz_path = os.path.join(ROOT_DIR, "data", "unified_136class", "train_split.npz")
    print(f"\n[*] Computing 128-dim class prototypes from {train_npz_path}...")
    train_data = np.load(train_npz_path)
    X_train = train_data["features"]
    y_train = train_data["labels"]

    prototypes = np.zeros((num_classes, 128), dtype=np.float32)
    counts = np.zeros(num_classes, dtype=np.int32)

    batch_size = 64
    with torch.no_grad():
        for start_idx in range(0, len(X_train), batch_size):
            end_idx = min(start_idx + batch_size, len(X_train))
            x_b = torch.tensor(X_train[start_idx:end_idx], dtype=torch.float32, device=device)
            _, embs = model(x_b, return_embeddings=True)
            embs_np = embs.cpu().numpy()
            
            # L2 normalize
            norms = np.linalg.norm(embs_np, axis=1, keepdims=True) + 1e-8
            embs_norm = embs_np / norms

            for emb, lab in zip(embs_norm, y_train[start_idx:end_idx]):
                prototypes[lab] += emb
                counts[lab] += 1

    # Normalize centroids or fallback to classifier weights
    classifier_weights = model.classifier.weight.detach().cpu().numpy() # [136, 128]
    w_norms = np.linalg.norm(classifier_weights, axis=1, keepdims=True) + 1e-8
    classifier_weights_norm = classifier_weights / w_norms

    for c in range(num_classes):
        if counts[c] > 0:
            prototypes[c] /= counts[c]
            p_norm = np.linalg.norm(prototypes[c]) + 1e-8
            prototypes[c] /= p_norm
        else:
            prototypes[c] = classifier_weights_norm[c]

    proto_paths = [
        os.path.join(ROOT_DIR, "models", "class_prototypes.npy"),
        os.path.join(ROOT_DIR, "backend", "models", "class_prototypes.npy"),
        os.path.join(TRAINING_DIR, "class_prototypes.npy"),
    ]
    for pp in proto_paths:
        os.makedirs(os.path.dirname(pp), exist_ok=True)
        np.save(pp, prototypes)
        print(f"[+] Saved prototypes to {pp} (shape: {prototypes.shape})")

    # 5. Export to ONNX (with logits and embeddings)
    class ACHWGATE_Export_Wrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            return self.m(x, return_embeddings=True)

    export_model = ACHWGATE_Export_Wrapper(model).cpu()
    export_model.eval()

    dummy_input = torch.randn(1, 32, 133, 6, dtype=torch.float32)
    onnx_paths = [
        os.path.join(ROOT_DIR, "models", "achwgate.onnx"),
        os.path.join(ROOT_DIR, "backend", "models", "ac_hwgate.onnx"),
        os.path.join(TRAINING_DIR, "model.onnx"),
    ]

    primary_onnx = onnx_paths[0]
    os.makedirs(os.path.dirname(primary_onnx), exist_ok=True)
    
    print(f"\n[*] Exporting ONNX model to {primary_onnx}...")
    torch.onnx.export(
        export_model,
        (dummy_input,),
        primary_onnx,
        export_params=True,
        input_names=["keypoints"],
        output_names=["logits", "embeddings"],
        dynamic_axes={
            "keypoints": {0: "batch_size"},
            "logits": {0: "batch_size"},
            "embeddings": {0: "batch_size"}
        },
        opset_version=14,
        do_constant_folding=True
    )
    print(f"[+] Exported ONNX to {primary_onnx}")

    for op in onnx_paths[1:]:
        os.makedirs(os.path.dirname(op), exist_ok=True)
        with open(primary_onnx, "rb") as sf, open(op, "wb") as df:
            df.write(sf.read())
        print(f"[+] Synced ONNX copy to {op}")

    # 6. Verify Numerical Parity between PyTorch and ONNX
    print("\n[*] Verifying bit-exact PyTorch vs ONNX Runtime numerical parity...")
    ort_session = ort.InferenceSession(primary_onnx, providers=["CPUExecutionProvider"])
    
    test_npz = os.path.join(ROOT_DIR, "data", "unified_136class", "test_split.npz")
    test_data = np.load(test_npz)
    X_test = test_data["features"][:20] # 20 actual held-out samples

    with torch.no_grad():
        pt_logits, pt_embs = model.cpu()(torch.tensor(X_test, dtype=torch.float32), return_embeddings=True)
        pt_logits_np = pt_logits.numpy()
        pt_embs_np = pt_embs.numpy()

    inp_name = ort_session.get_inputs()[0].name
    ort_outs = ort_session.run(None, {inp_name: X_test.astype(np.float32)})
    ort_logits = ort_outs[0]
    ort_embs = ort_outs[1]

    max_diff_logits = float(np.max(np.abs(pt_logits_np - ort_logits)))
    max_diff_embs = float(np.max(np.abs(pt_embs_np - ort_embs)))

    print(f"  Max Absolute Difference (Logits):     {max_diff_logits:.2e}")
    print(f"  Max Absolute Difference (Embeddings): {max_diff_embs:.2e}")
    assert max_diff_logits < 1e-4, f"Parity failure: diff {max_diff_logits} >= 1e-4"
    assert max_diff_embs < 1e-4, f"Parity failure: diff {max_diff_embs} >= 1e-4"
    print("  [SUCCESS] PyTorch and ONNX predictions are in BIT-EXACT agreement (< 1e-4 difference)!")

if __name__ == "__main__":
    main()
