"""Evaluate one AC-HWGATE checkpoint on a specified split."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from training.models.ac_hwgate import ACHWGATE

ROOT = Path(__file__).resolve().parents[1]


def evaluate(checkpoint_path, split_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(split_path, allow_pickle=True)
    classes = json.loads((ROOT / "configs/authoritative_136_classes_manifest.json").read_text(encoding="utf-8"))["classes"]
    model = ACHWGATE(num_classes=len(classes), num_joints=133, in_channels=6, sequence_length=32, embed_dim=128, num_heads=4, num_layers=3).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(data["features"]), 64):
            batch = torch.from_numpy(data["features"][start:start + 64]).float().to(device)
            predictions.extend(model(batch).argmax(dim=1).cpu().numpy().tolist())
    targets = data["labels"].astype(int)
    predictions = np.asarray(predictions)
    return {
        "checkpoint": str(checkpoint_path),
        "split": str(split_path),
        "sample_count": int(len(targets)),
        "top1_accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, average="macro", labels=list(range(len(classes))), zero_division=0)),
        "weighted_f1": float(f1_score(targets, predictions, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(targets, predictions, average="macro", labels=list(range(len(classes))), zero_division=0)),
        "macro_recall": float(recall_score(targets, predictions, average="macro", labels=list(range(len(classes))), zero_division=0)),
        "per_class": [
            {"class": name, "count": int((targets == index).sum()), "precision": float(precision_score(targets, predictions, labels=[index], average="macro", zero_division=0)), "recall": float(recall_score(targets, predictions, labels=[index], average="macro", zero_division=0)), "f1": float(f1_score(targets, predictions, labels=[index], average="macro", zero_division=0))}
            for index, name in enumerate(classes)
        ],
        "confusion_matrix": confusion_matrix(targets, predictions, labels=list(range(len(classes)))).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("split", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.checkpoint, args.split)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("sample_count", "top1_accuracy", "macro_f1", "weighted_f1", "macro_recall")}))


if __name__ == "__main__":
    main()
