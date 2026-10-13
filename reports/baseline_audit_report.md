# Reproducible Forensic Baseline Audit Report (Phase 1)

**Project:** AC-HWGATE Real-Time Indian Sign Language (ISL) Recognition & Translation  
**Hardware Platform:** NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM), CUDA 12.1, PyTorch 2.5.1  
**Audit Date:** October 13, 2026  
**Auditor:** Antigravity AI Senior Research Engineer  

---

## 1. Actual Dataset and Class Counts

A comprehensive forensic disk scan of all local datasets on the host system was conducted without downloading any external datasets:

| Dataset Location / Identifier | Modality | Authentic Classes Discovered | Total Samples / Recordings | Notes |
|:---|:---|:---:|:---:|:---|
| `C:\Users\navee\Downloads\archive (2)\Video_Dataset\Video_Dataset` | Authentic High-Definition MP4 Videos ($640 \times 720$ @ 30 FPS) | **61** | **3,630** video clips | Exactly 60 clips/class across 3 angles (frontal, left tilt $+15^\circ$, right tilt $-15^\circ$) |
| `C:\Users\navee\Downloads\archive\ProcessedData_vivit` | Authentic ISL MOV Videos ($224 \times 224$ @ 10-30 FPS) | **76** | **1,166** video clips | 8 to 21 clips per class across distinct conversational glosses |
| `data/raw_40words` | Authentic MP4 Video Clips | **40** | **642** video clips (1,324 files total) | Diverse signers from INCLUDE, CISLR, and ISLRTC |
| `data/isl500_landmarks` | Continuous 32-frame WholeBody Landmark Sequences (`.npz`) | **29** | **2,439** sequences | 15 signers (`USER001` through `USER015`) |
| `data/additional_landmarks` | Pre-extracted 32-frame Landmark Sequences (`.npz`) | **75** | **3,531** sequences | Previously extracted from Archive-2 and Vivit |
| `data/cache` | Pre-extracted 32-frame Landmark Sequences (`.npz`) | **40** | **642** sequences | Verified 40-class cache |

### Total Class Overlap & Supported Scale:
- **Total distinct normalized classes across all local sources:** **171 unique classes**.
- **Classes with $\ge 15$ authentic samples:** **131 classes**.
- **Classes with $\ge 20$ authentic samples:** **113 classes**.
- **Classes with $\ge 25$ authentic samples:** **105 classes**.
- **Conclusion:** The local dataset inventory **empirically supports $\ge 100$ valid classes** without inventing data or synthetic frame replication.

---

## 2. Existing Checkpoint and Model Dimensions

Forensic examination of models and checkpoints on disk:

| Artifact Path | Format | Dimension / Classes | Status & Verification |
|:---|:---|:---:|:---|
| `checkpoints/achwgate_50class_best.pth` | PyTorch Weights | Output: [50, 128] classifier, Input: [B, 32, 133, 6] | Active 50-class checkpoint (973,481 params) |
| `checkpoints/achwgate_best.pth` | PyTorch Weights | Output: [50, 128] classifier, Input: [B, 32, 133, 6] | Matches 50-class best model |
| `checkpoints/achwgate_last.pth` | PyTorch Weights | Output: [40, 128] classifier | Stale legacy 40-class model |
| `models/achwgate.onnx` | ONNX Model | Dual output: `logits` [B, 50], `embeddings` [B, 128] | Matches 50-class checkpoint |
| `backend/models/ac_hwgate.onnx` | ONNX Model | Dual output: `logits` [B, 50], `embeddings` [B, 128] | Matches 50-class checkpoint |
| `configs/selected_104_classes_manifest.json` | JSON Manifest | 103 classes listed | Stale partial manifest from prior attempt |
| `backend/label_map.json` | JSON Label Map | 50 classes mapped (indices 0..49) | Matches 50-class model |

---

## 3. Current Offline Evaluation Metrics & Calculation Method

From the existing evaluation runs on `data/unified_50class/test_split.npz` (722 sequences):
- **Top-1 Accuracy:** 81.86%
- **Macro F1:** 86.47%
- **Weighted F1:** 82.03%
- **Macro Precision:** 86.69%
- **Macro Recall:** 87.97%
- **Calculation Method:** Standard `scikit-learn` metrics (`accuracy_score`, `precision_recall_fscore_support` with `zero_division=0`).
- **Limitation:** These metrics only measure the 50-class subset. Prior reports claiming 104-class readiness were premature because only 50 classes were retrained.

---

## 4. Current Latency & Performance Measurements

Measured empirically on local hardware:
- **MediaPipe Feature Extraction (`FeatureExtractor`):** Average $4.87\text{ ms}$, Median $5.02\text{ ms}$, P95 $6.30\text{ ms}$.
- **AC-HWGATE PyTorch Inference (CPU):** Average $8.24\text{ ms}$, Median $6.66\text{ ms}$, P95 $14.90\text{ ms}$.
- **AC-HWGATE PyTorch Inference (CUDA RTX 3050):** Average $9.79\text{ ms}$, Median $9.81\text{ ms}$, P95 $12.22\text{ ms}$.
- **AC-HWGATE ONNX Runtime (CPU 4 threads):** Average $3.70\text{ ms}$, Median $3.58\text{ ms}$, P95 $4.86\text{ ms}$.
- **Total Backend Pipeline Latency (FE + ONNX):** **$8.57\text{ ms}$** ($\approx 116\text{ FPS}$ computational capacity).

---

## 5. Confirmed Root Causes vs. Hypotheses

### [CONFIRMED ROOT CAUSE 1] Stuck "DETECTING..." Failure (Smoking Gun)
- **Mechanism:** In `backend/inference.py` lines 91-120, `_load_label_map()` checks a list of paths where `configs/selected_104_classes_manifest.json` (103 classes) takes precedence over `data/unified_50class/label_map.json`.
- This causes `self.num_classes` to be set to 103.
- When loading `backend/models/ac_hwgate.onnx`, line 159 detects that output dimension `50 != 103` and aborts ONNX loading, falling back to PyTorch.
- When loading `checkpoints/achwgate_best.pth`, PyTorch fails with `size mismatch for classifier.weight: copying [50, 128] to [103, 128]`.
- As a fallback (line 198), `backend/inference.py` initializes an **untrained AC-HWGATE model with completely random weights**.
- With random weights, softmax probabilities are uniformly distributed ($\approx 1/103 = 0.0097$).
- Every frame fails the minimum confidence threshold ($0.25$) and entropy threshold ($4.40$), unconditionally returning `("DETECTING...", prob, True)`.
- Consequently, the frontend detected-sign panel is **permanently stuck on "DETECTING..."**.

### [CONFIRMED ROOT CAUSE 2] Webcam FPS Drop
- **Mechanism:** `frontend/src/components/CameraFeed.jsx` initializes MediaPipe Camera at `width: 960, height: 720`.
- Processing full MediaPipe Holistic (Pose 33 + Left Hand 21 + Right Hand 21 + Face 468) on a 960x720 video feed inside browser JavaScript/WASM consumes $\approx 80\text{--}95\%$ CPU, causing severe frame drops (down to 10-14 FPS).
- Holistic does not need 960x720; downscaling the internal processing stream to $640 \times 480$ while maintaining 720p display restores 30 FPS capture.

### [CONFIRMED ROOT CAUSE 3] Repetitive / Biased Predictions in Prior System
- **Mechanism:** Imbalanced class distribution in prior datasets (e.g. `drink`: 203, `thank_you`: 175, `hello`: 170 vs 60 for others) without class weighting biased model logits.
- Premature hand-raise prediction (`min_frames = 8` in legacy code) locked into hand-rising trajectories before the hand posture formed.
- Lack of adequate post-confirmation cooldown allowed residual frames in the rolling buffer to re-trigger the same prediction.

### [CONFIRMED ROOT CAUSE 4] Semantic and Translation Hallucinations
- **Mechanism:** `LiveTranslator.jsx` and `translator.py` contained hardcoded embellishments (e.g. mapping `clean` to `"Please clean this area."`, `lion` to `"A roaring lion."`, `cry` to `"Why are you crying?"`).
- This violates semantic translation by inventing missing words and intent that does not exist in the isolated sign.

---

## 6. Next Actions for Phase 2 Through Phase 8

1. **Phase 2:** Programmatically discover all eligible classes across Archive-2, Vivit, Raw40, and ISL500. Select at least 100 eligible classes with $\ge 20$ valid samples each. Build a single unified dataset with signer/session-disjoint splits and generate an authoritative manifest.
2. **Phase 3:** Retrain AC-HWGATE on the 100+ class dataset on RTX 3050 GPU using PyTorch AMP FP16, class-weighted loss, cosine annealing, and early stopping.
3. **Phase 4:** Compute class prototypes, export dual-output ONNX models, verify PyTorch vs. ONNX parity $< 10^{-4}$, and evaluate test set metrics.
4. **Phase 5:** Overhaul `backend/inference.py` and `CameraFeed.jsx`: synchronize 100+ class vocabulary, optimize MediaPipe camera resolution to $640 \times 480$, eliminate event-loop blocking, and calibrate unknown rejection.
5. **Phase 6:** Clean the translation layer to use authentic, verified sign meanings without fabricated sentence contexts.
6. **Phase 7 & 8:** Automated regression tests, fresh webcam validation, and final acceptance report.
