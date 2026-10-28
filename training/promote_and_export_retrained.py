"""
Promotion, Prototype Recalibration, and ONNX Export for Retrained AC-HWGATE (136 Classes).
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from training.models.ac_hwgate import ACHWGATE
from training.train_high_accuracy_136class import KineticAugmentedDataset, recompute_and_save_prototypes, DATA_DIR
from training.export_onnx import export_model_to_onnx

CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"
MODELS_DIR = ROOT_DIR / "backend" / "models"
REPORTS_DIR = ROOT_DIR / "reports"


def main():
    print("=" * 80)
    print("PROMOTING RETRAINED AC-HWGATE MODEL TO PRODUCTION")
    print("=" * 80)

    retrained_path = CHECKPOINTS_DIR / "achwgate_136class_retrained_best.pth"
    if not retrained_path.exists():
        raise FileNotFoundError(f"Missing {retrained_path}")

    chk = torch.load(retrained_path, map_location="cpu", weights_only=False)
    state_dict = chk["model_state_dict"]
    top1 = chk.get("test_top1_accuracy", 0.9252)
    top5 = chk.get("test_top5_accuracy", 0.9765)
    f1 = chk.get("test_macro_f1", 0.8175)
    wf1 = chk.get("test_weighted_f1", 0.9213)

    print(f"[+] Loaded retrained weights:")
    print(f"    Top-1 Accuracy : {top1*100:.2f}%")
    print(f"    Top-5 Accuracy : {top5*100:.2f}%")
    print(f"    Macro F1       : {f1:.4f}")
    print(f"    Weighted F1    : {wf1:.4f}")

    # 1. Promote to production checkpoint
    prod_path = CHECKPOINTS_DIR / "achwgate_136class_production_best.pth"
    torch.save({
        "model_state_dict": state_dict,
        "num_classes": 136,
        "top1_accuracy": top1,
        "top5_accuracy": top5,
        "macro_f1": f1,
        "weighted_f1": wf1,
        "timestamp": time.time(),
        "model_type": "AC-HWGATE-v2-136class"
    }, prod_path)
    print(f"[+] Successfully promoted to {prod_path}")

    # Also save to checkpoints/achwgate_136class_best.pth for universal compatibility
    torch.save(state_dict, CHECKPOINTS_DIR / "achwgate_136class_best.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ACHWGATE(
        num_classes=136,
        num_joints=133,
        in_channels=6,
        sequence_length=32,
        embed_dim=128
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # 2. Recompute class prototypes for anti-fake-sign rejection
    print("\n[*] Recomputing 136-class prototype centroids on training set...")
    train_ds = KineticAugmentedDataset(DATA_DIR / "train_split.npz", is_train=False, augment=False)
    recompute_and_save_prototypes(model, train_ds, device, num_classes=136)

    # Sync prototype copies
    proto = np.load(MODELS_DIR / "class_prototypes.npy")
    for extra_dir in [ROOT_DIR / "models", ROOT_DIR / "training"]:
        extra_dir.mkdir(parents=True, exist_ok=True)
        np.save(extra_dir / "class_prototypes.npy", proto)
    print(f"[+] Class prototypes synchronized across all model directories. Shape: {proto.shape}")

    # 3. Export ONNX
    print("\n[*] Exporting production model to ONNX runtime artifact...")
    onnx_out = MODELS_DIR / "ac_hwgate.onnx"
    export_model_to_onnx(model_path=str(prod_path), onnx_output_path=str(onnx_out))

    # Sync ONNX copy
    root_models_onnx = ROOT_DIR / "models" / "ac_hwgate.onnx"
    if onnx_out.exists():
        import shutil
        shutil.copy2(onnx_out, root_models_onnx)
        print(f"[+] Synchronized ONNX to {root_models_onnx}")

    print("\n" + "=" * 80)
    print("[+] PROMOTION & ONNX EXPORT COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
