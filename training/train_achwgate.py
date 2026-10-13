import os
import sys
import json
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add training dir to path
TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TRAINING_DIR)
from models.ac_hwgate import ACHWGATE

class ISLSequenceDataset(Dataset):
    """
    Dataset for AC-HWGATE temporal sequences [T=32, J=133, C=6].
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index: int):
        idx = index
        x = self.X[idx] # [32, 133, 6]
        if self.augment:
            # Subtle data augmentation: slight spatial jitter and scaling
            jitter = torch.randn_like(x) * 0.003
            scale = 1.0 + (torch.rand(1).item() - 0.5) * 0.02
            x = (x + jitter) * scale
        return x, self.y[idx]


def evaluate_model(model, dataloader, device, criterion=None):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            if criterion:
                loss = criterion(outputs, targets)
                total_loss += loss.item() * len(targets)
            preds = torch.argmax(outputs, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset) if criterion else 0.0
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)  # type: ignore
    weighted_f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)  # type: ignore
    precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)  # type: ignore
    recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)  # type: ignore

    return {
        "loss": avg_loss,
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "precision": float(precision),
        "recall": float(recall),
        "predictions": all_preds,
        "targets": all_targets
    }


def train_achwgate(
    data_dir: str = "data_temporal_sequences",
    checkpoint_dir: str = "training/checkpoints",
    epochs: int = 40,
    batch_size: int = 16,
    lr: float = 0.002,
    weight_decay: float = 1e-4,
    embed_dim: int = 128,
    num_heads: int = 4,
    num_layers: int = 3,
    sanity_test: bool = False
):
    print("=" * 70, flush=True)
    print(f"AC-HWGATE TRAINING ENGINE {'[STEP 14 SANITY OVERFIT TEST]' if sanity_test else '[FULL PRODUCTION]'}", flush=True)
    print("=" * 70, flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[+] Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

    # 1. Load authoritative datasets
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_val = np.load(os.path.join(data_dir, "X_val.npy"))
    y_val = np.load(os.path.join(data_dir, "y_val.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))

    with open(os.path.join(data_dir, "label_map.json"), "r") as f:
        label_map = json.load(f)

    class_names = [k for k, v in sorted(label_map.items(), key=lambda item: item[1])]
    num_classes = len(class_names)

    print(f"[+] Loaded Dataset: {num_classes} Classes", flush=True)
    print(f"    Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}", flush=True)

    # Class weights for imbalance handling
    counts = np.bincount(y_train, minlength=num_classes)
    total_samples = len(y_train)
    weights = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        weights[c] = total_samples / (num_classes * max(counts[c], 1))
    class_weights_t = torch.tensor(weights, dtype=torch.float32).to(device)

    # DataLoaders
    train_dataset = ISLSequenceDataset(X_train, y_train, augment=not sanity_test)
    val_dataset = ISLSequenceDataset(X_val, y_val, augment=False)
    test_dataset = ISLSequenceDataset(X_test, y_test, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 2. Instantiate AC-HWGATE
    model = ACHWGATE(
        num_classes=num_classes,
        num_joints=133,
        in_channels=6,
        sequence_length=32,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        window_size=8,
        dropout=0.0 if sanity_test else 0.1
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[+] AC-HWGATE Trainable Parameters: {total_params:,}", flush=True)

    criterion = nn.CrossEntropyLoss(weight=class_weights_t)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)
    best_checkpoint_path = os.path.join(checkpoint_dir, "ac_hwgate_best.pth")

    best_val_acc = 0.0
    patience = 12
    patience_counter = 0

    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    # 3. Training Loop
    t_start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item() * len(targets)
            preds = torch.argmax(outputs, dim=-1)
            correct += (preds == targets).sum().item()
            total += len(targets)

        scheduler.step()

        train_loss = epoch_loss / total
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation
        val_res = evaluate_model(model, val_loader, device, criterion)
        val_losses.append(val_res["loss"])
        val_accs.append(val_res["accuracy"])

        if val_res["accuracy"] > best_val_acc:
            best_val_acc = val_res["accuracy"]
            patience_counter = 0
            torch.save({
                "model_name": "AC-HWGATE",
                "model_state_dict": model.state_dict(),
                "num_classes": num_classes,
                "class_names": class_names,
                "label_map": label_map,
                "val_accuracy": best_val_acc,
                "epoch": epoch,
                "config": {
                    "num_joints": 133,
                    "in_channels": 6,
                    "sequence_length": 32,
                    "embed_dim": embed_dim,
                    "num_heads": num_heads,
                    "num_layers": num_layers
                }
            }, best_checkpoint_path)
            improved_mark = " [* BEST SAVED]"
        else:
            patience_counter += 1
            improved_mark = ""

        if epoch % 2 == 0 or epoch == 1 or epoch == epochs or improved_mark or train_acc >= 0.95:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:5.1f}% | "
                  f"Val Loss: {val_res['loss']:.4f} | Val Acc: {val_res['accuracy']*100:5.1f}%{improved_mark}", flush=True)

        if sanity_test and train_acc >= 0.95:
            print("\n" + "=" * 70, flush=True)
            print(f"[+] STEP 14 SANITY TEST PASSED! Reached {train_acc*100:.1f}% training accuracy on epoch {epoch}!", flush=True)
            print("=" * 70, flush=True)
            break

        if patience_counter >= patience and not sanity_test and epoch > 20:
            print(f"[!] Early stopping triggered at epoch {epoch}", flush=True)
            break

    elapsed_time = time.time() - t_start
    print(f"\n[+] Training completed in {elapsed_time:.1f}s. Best Val Accuracy: {best_val_acc*100:.2f}%", flush=True)

    # 4. Evaluation on Untouched Test Set
    print("\n" + "=" * 70, flush=True)
    print("EVALUATING AC-HWGATE ON UNTOUCHED TEST SET", flush=True)
    print("=" * 70, flush=True)
    
    # Load best checkpoint if saved, else current model
    if os.path.exists(best_checkpoint_path):
        chk = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(chk["model_state_dict"])
    
    test_metrics = evaluate_model(model, test_loader, device, criterion)
    print(f"Test Accuracy:  {test_metrics['accuracy']*100:.2f}%", flush=True)
    print(f"Macro F1-Score: {test_metrics['macro_f1']*100:.2f}%", flush=True)
    print(f"Weighted F1:    {test_metrics['weighted_f1']*100:.2f}%", flush=True)
    print(f"Precision:      {test_metrics['precision']*100:.2f}%", flush=True)
    print(f"Recall:         {test_metrics['recall']*100:.2f}%", flush=True)

    # 5. Save results/final_results.json
    final_results = {
        "model_name": "AC-HWGATE",
        "target_accuracy": 0.95,
        "ideal_target_accuracy": 0.97,
        "validation_accuracy": float(best_val_acc),
        "test_accuracy": float(test_metrics["accuracy"]),
        "macro_f1": float(test_metrics["macro_f1"]),
        "weighted_f1": float(test_metrics["weighted_f1"]),
        "precision": float(test_metrics["precision"]),
        "recall": float(test_metrics["recall"]),
        "number_of_classes": num_classes,
        "number_of_train_samples": len(y_train),
        "number_of_val_samples": len(y_val),
        "number_of_test_samples": len(y_test),
        "sequence_length": 32,
        "num_joints": 133,
        "in_channels": 6,
        "parameters": total_params,
        "training_time_seconds": elapsed_time
    }
    with open("results/final_results.json", "w") as f:
        json.dump(final_results, f, indent=2)
    print("[+] Saved results/final_results.json", flush=True)

    # 6. Save Confusion Matrix and Per-Class Metrics
    cm = confusion_matrix(test_metrics["targets"], test_metrics["predictions"])
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(f"AC-HWGATE Confusion Matrix (Test Acc: {test_metrics['accuracy']*100:.1f}%)")
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, class_names, rotation=45, ha='right', fontsize=8)
    plt.yticks(tick_marks, class_names, fontsize=8)
    plt.tight_layout()
    plt.ylabel('True ISL Sign')
    plt.xlabel('Predicted ISL Sign')
    plt.savefig("results/confusion_matrix.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("[+] Saved results/confusion_matrix.png", flush=True)

    raw_report = classification_report(
        test_metrics["targets"], 
        test_metrics["predictions"], 
        target_names=class_names, 
        output_dict=True, 
        zero_division=0  # type: ignore
    )
    report_dict: dict = raw_report if isinstance(raw_report, dict) else {}
    with open("results/per_class_metrics.csv", "w") as f:
        f.write("Class,Precision,Recall,F1-Score,Support\n")
        for cname in class_names:
            if cname in report_dict:
                m = report_dict[cname]
                if isinstance(m, dict):
                    f.write(f"{cname},{m['precision']:.4f},{m['recall']:.4f},{m['f1-score']:.4f},{m['support']}\n")
    print("[+] Saved results/per_class_metrics.csv", flush=True)

    # 7. Step 25: Export to ONNX
    print("\n" + "=" * 70, flush=True)
    print("STEP 25: EXPORTING AC-HWGATE TO ONNX", flush=True)
    print("=" * 70, flush=True)
    os.makedirs("backend/models", exist_ok=True)
    onnx_path = "backend/models/ac_hwgate.onnx"
    dummy_input = torch.randn(1, 32, 133, 6, device=device)
    
    model.eval()
    torch.onnx.export(
        model,
        (dummy_input,),
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input_sequence'],
        output_names=['logits'],
        dynamic_axes={'input_sequence': {0: 'batch_size'}, 'logits': {0: 'batch_size'}},
        dynamo=False
    )
    print(f"[+] Exported ONNX model to {onnx_path} ({os.path.getsize(onnx_path)/1e6:.2f} MB)", flush=True)

    try:
        import onnxruntime as ort
        ort_sess = ort.InferenceSession(onnx_path)
        pt_out = model(dummy_input).detach().cpu().numpy()
        ort_out = ort_sess.run(None, {'input_sequence': dummy_input.cpu().numpy()})[0]
        max_diff = float(np.max(np.abs(pt_out - ort_out)))
        print(f"[+] ONNX Verification: Max absolute diff vs PyTorch = {max_diff:.2e}", flush=True)
        if max_diff < 1e-4:
            print("[+] ONNX export passed verification (< 1e-4)!", flush=True)
        else:
            print(f"[!] Warning: ONNX diff is {max_diff}", flush=True)
    except Exception as e:
        print(f"[!] ONNX verification skipped: {e}", flush=True)

    # Copy authoritative label map to backend
    with open("backend/label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity_test", action="store_true", help="Run tiny sanity overfit test")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--data_dir", type=str, default="data_temporal_sequences")
    args = parser.parse_args()

    train_achwgate(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        sanity_test=args.sanity_test
    )
