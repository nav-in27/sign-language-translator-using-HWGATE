# Pre-Optimization Comprehensive Audit Report

## 1. Executive Summary
This audit documents the state of the Indian Sign Language (ISL) Recognition and Translation System prior to commencing the maximal optimization, dataset expansion, and personalization phase.

- **Primary Architecture**: **AC-HWGATE** (*Adaptive Context-Aware Hierarchical Windowed Graph Attention Transformer Encoder*)
- **Baseline Architecture**: **CWA-HWGATE** (Retained strictly as a reference baseline in `training/models/cwa_hwgate.py`)
- **Active Production Vocabulary**: 40 canonical ISL classes
- **Hardware Profile**: NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM) with PyTorch AMP FP16 acceleration
- **Target**: Genuine webcam recognition, robust unknown gesture rejection, honest benchmark tracking, and personalization.

---

## 2. Component Inventory & Audit Results

### A. Model & Preprocessing Pipeline
- **Production Architecture**: [`training/models/ac_hwgate.py`](file:///c:/Users/navee/.gemini/antigravity-ide/scratch/isl-translator/training/models/ac_hwgate.py)
  - Parameter Count: **972,191 parameters** (3.71 MB)
  - Key Modules: `AdaptiveGraphLayer`, `HierarchicalPartEncoder`, `LocalWindowAttention`, Pre-LN `ContextAwareTransformerBlock`, `AttentionPooling`.
  - Status: Fully verified via tiny overfit sanity test (95.8% accuracy in 1.87s on RTX 3050).
- **Authoritative Feature Extractor**: [`training/feature_extractor.py`](file:///c:/Users/navee/.gemini/antigravity-ide/scratch/isl-translator/training/feature_extractor.py)
  - Keypoint Topology: 133 joints (33 Pose + 21 Left Hand + 21 Right Hand + 58 Face).
  - Channels: 6 ($P_x, P_y, V_x, V_y, A_x, A_y$).
  - Temporal Window: Exact $T=32$ frames.
  - Total Dimension: $[32, 133, 6]$ = 798 features/frame.
  - Zero-drift guarantee across training, validation, ONNX export, and live webcam inference.

### B. Current Baseline Dataset & Splits
- **Primary Corpus**: `vidit031/isl-isolated-40words` (642 video clips, 40 classes, 172 unique signers).
- **Current Video Splits**:
  - Training: 430 clips (~70%)
  - Validation: 105 clips (~15%)
  - Untouched Test: 107 clips (~15%)
- **Data Integrity**: 0 corrupt files, 0 missing files, verified via [`reports/dataset_report.json`](file:///c:/Users/navee/.gemini/antigravity-ide/scratch/isl-translator/reports/dataset_report.json).

### C. Honest Baseline Metrics (Untouched Test Split)
| Metric | Measured Baseline |
| :--- | :--- |
| **Top-1 Test Accuracy** | **63.55%** |
| **Top-3 Test Accuracy** | **85.98%** |
| **Best Validation Accuracy** | **77.14%** |
| **Macro F1 Score** | **56.61%** |
| **Weighted F1 Score** | **62.07%** |
| **Macro Precision** | **59.32%** |
| **Macro Recall** | **58.80%** |
| **Inference Latency** | **0.58 ms** per sequence |
| **Throughput** | **1,719.6 FPS** |
| **100% Test Accuracy Classes** | `father`, `teacher`, `school`, `okay`, `hospital`, `help`, `hello`, `tea` (8 classes) |

### D. Production Inference & Unknown Rejection
- **Inference Engine**: [`backend/inference.py`](file:///c:/Users/navee/.gemini/antigravity-ide/scratch/isl-translator/backend/inference.py) & [`backend/live_inference.py`](file:///c:/Users/navee/.gemini/antigravity-ide/scratch/isl-translator/backend/live_inference.py)
- **Multi-Signal Unknown Gating**:
  1. Hand Quality Gate (minimum 4 hand joints, mean confidence $\ge 0.10$).
  2. Temporal Motion Energy Gate (standard deviation threshold $\ge 0.0005$ to suppress idle static gestures).
  3. Calibrated Confidence & Margin Gate (confidence $\ge 0.45$, margin $\ge 0.05$, Shannon entropy $\le 3.60$).
  4. Class Prototype Cosine Distance Gate (similarity $\ge 0.10$ against 40 class prototypes in [`models/class_prototypes.npy`](file:///c:/Users/navee/.gemini/antigravity-ide/scratch/isl-translator/models/class_prototypes.npy)).
  5. Temporal Consensus Stability Window (at least 3 agreeing predictions over recent frames).
- **Translation & TTS**:
  - Deterministic translations defined for all 40 vocabulary classes.
  - Unknown movements yield `UNKNOWN_SIGN` / `NOT RECOGNIZED` with **100% suppression of TTS and sentence assembly**.

---

## 3. Optimization Strategy
1. **Canonical Label Normalization**: Enforce exact uniform string format (`thank_you` canonical) across all manifests, configs, and label maps.
2. **Unified Dataset Expansion**: Mine matching 40-word clips from ISL500 and compatible research sources while enforcing strict SHA256 and perceptual deduplication.
3. **Signer-Independent Disjoint Splitting**: Ensure complete isolation where Train signers $\cap$ Val signers $\cap$ Test signers = $\emptyset$.
4. **Controlled Experimentation Grid**: Systematically benchmark baseline, augmentations, balanced sampling, and hyperparameter variants.
5. **Webcam Personalization**: Provide full recording and 2-stage transfer fine-tuning tools (`tools/record_personal_samples.py`, `tools/fine_tune_personal.py`) with disjoint personal test evaluation.
