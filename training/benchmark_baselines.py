import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Paths
TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(TRAINING_DIR)
sys.path.insert(0, TRAINING_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "models", "baseline_cwa_hwgate"))

from models.ac_hwgate import ACHWGATE
from cwa_hwgate import CWA_HWGATE
from train_achwgate import ISLSequenceDataset, evaluate_model

# 1. Baseline: CNN 1D
class BaselineCNN(nn.Module):
    def __init__(self, num_classes, in_channels=798, sequence_length=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # x: [B, T, 133, 6] -> flatten to [B, T, 798] -> transpose to [B, 798, T]
        B, T = x.size(0), x.size(1)
        x_flat = x.view(B, T, -1).transpose(1, 2)
        feat = self.conv(x_flat).squeeze(-1)
        return self.fc(feat)

# 2. Baseline: LSTM
class BaselineLSTM(nn.Module):
    def __init__(self, num_classes, in_channels=798, hidden_dim=256, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(in_channels, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        B, T = x.size(0), x.size(1)
        x_flat = x.view(B, T, -1)
        out, (hn, cn) = self.lstm(x_flat)
        feat = torch.cat([hn[-2], hn[-1]], dim=-1)
        return self.fc(feat)

# 3. Baseline: GRU
class BaselineGRU(nn.Module):
    def __init__(self, num_classes, in_channels=798, hidden_dim=256, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(in_channels, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        B, T = x.size(0), x.size(1)
        x_flat = x.view(B, T, -1)
        out, hn = self.gru(x_flat)
        feat = torch.cat([hn[-2], hn[-1]], dim=-1)
        return self.fc(feat)

# 4. Baseline: Spatial-Temporal GCN (ST-GCN)
class BaselineSTGCN(nn.Module):
    def __init__(self, num_classes, num_joints=133, in_channels=6):
        super().__init__()
        self.joint_embed = nn.Linear(in_channels, 64)
        self.spatial_gcn = nn.Linear(64, 128)
        self.temporal_conv = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        B, T, J, C = x.shape
        h = torch.relu(self.joint_embed(x)) # [B, T, J, 64]
        h = torch.relu(self.spatial_gcn(h.mean(dim=2))) # [B, T, 128]
        h_t = h.transpose(1, 2) # [B, 128, T]
        feat = self.pool(torch.relu(self.temporal_conv(h_t))).squeeze(-1)
        return self.fc(feat)

# 5. Baseline: CWA-HWGATE Wrapper
class BaselineCWAWrapper(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = CWA_HWGATE(num_classes=num_classes, embed_dim=256, num_heads=8, num_layers=4)

    def forward(self, x):
        # CWA expects [B, T, 133, 2] and [B, T, 133, 1] confidences
        x_pos = x[..., :2]
        confs = torch.ones(x.size(0), x.size(1), 133, 1, device=x.device, dtype=x.dtype)
        out = self.model(x_pos, confs)
        if out.dim() == 3 and out.size(0) == 1:
            out = out.squeeze(0)
        return out


def benchmark_all_models(data_dir="data_temporal_sequences", epochs=15):
    print("=" * 75)
    print("ISL COMPREHENSIVE BENCHMARK: BASELINES VS AC-HWGATE")
    print("=" * 75)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[+] Device: {device}")

    # Load datasets
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))

    with open(os.path.join(data_dir, "label_map.json"), "r") as f:
        label_map = json.load(f)
    num_classes = len(label_map)

    train_loader = DataLoader(ISLSequenceDataset(X_train, y_train, augment=True), batch_size=32, shuffle=True)
    test_loader = DataLoader(ISLSequenceDataset(X_test, y_test, augment=False), batch_size=32, shuffle=False)

    models_dict = {
        "CNN 1D": BaselineCNN(num_classes=num_classes),
        "LSTM (Bi-directional)": BaselineLSTM(num_classes=num_classes),
        "GRU (Bi-directional)": BaselineGRU(num_classes=num_classes),
        "ST-GCN": BaselineSTGCN(num_classes=num_classes),
        "CWA-HWGATE (Baseline)": BaselineCWAWrapper(num_classes=num_classes),
        "AC-HWGATE (Proposed)": ACHWGATE(num_classes=num_classes, num_joints=133, in_channels=6, sequence_length=32)
    }

    results_table = []
    dummy_single = torch.randn(1, 32, 133, 6).to(device)

    for name, model in models_dict.items():
        print(f"\n[+] Evaluating: {name}")
        model = model.to(device)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Quick train for fair comparison
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

        for ep in range(epochs):
            model.train()
            for inp, tgt in train_loader:
                inp, tgt = inp.to(device), tgt.to(device)
                optimizer.zero_grad()
                out = model(inp)
                loss = criterion(out, tgt)
                loss.backward()
                optimizer.step()

        # Measure latency & FPS over 100 runs
        model.eval()
        with torch.no_grad():
            for _ in range(10): # warm-up
                _ = model(dummy_single)
            
            t0 = time.time()
            n_runs = 60
            for _ in range(n_runs):
                _ = model(dummy_single)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_total = time.time() - t0
            lat_ms = (t_total / n_runs) * 1000.0
            fps = 1000.0 / lat_ms

        # Evaluate on test set
        metrics = evaluate_model(model, test_loader, device, criterion)
        acc_pct = metrics["accuracy"] * 100.0
        macro_f1 = metrics["macro_f1"] * 100.0
        weighted_f1 = metrics["weighted_f1"] * 100.0
        prec_pct = metrics["precision"] * 100.0
        rec_pct = metrics["recall"] * 100.0
        model_size_mb = (total_params * 4) / (1024 * 1024)

        print(f"    Accuracy:    {acc_pct:.2f}% | Macro F1: {macro_f1:.2f}%")
        print(f"    Latency:     {lat_ms:.2f} ms | FPS: {fps:.1f} | Params: {total_params:,}")

        results_table.append({
            "Model": name,
            "Accuracy": f"{acc_pct:.2f}%",
            "Macro F1": f"{macro_f1:.2f}%",
            "Weighted F1": f"{weighted_f1:.2f}%",
            "Precision": f"{prec_pct:.2f}%",
            "Recall": f"{rec_pct:.2f}%",
            "Latency": f"{lat_ms:.2f} ms",
            "FPS": f"{fps:.1f}",
            "Parameters": f"{total_params:,}",
            "Model Size": f"{model_size_mb:.2f} MB"
        })

    # Save to results.csv
    with open("results.csv", "w") as f:
        headers = ["Model", "Accuracy", "Macro F1", "Weighted F1", "Precision", "Recall", "Latency", "FPS", "Parameters", "Model Size"]
        f.write(",".join(headers) + "\n")
        for r in results_table:
            row = [r[h] for h in headers]
            f.write(",".join(row) + "\n")

    print("\n" + "=" * 75)
    print("[+] Saved comprehensive baseline comparisons to results.csv")
    print("=" * 75)

if __name__ == "__main__":
    benchmark_all_models()
