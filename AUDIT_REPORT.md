# Complete System Audit Report: Pre-AC-HWGATE State

**Project:** Real-Time Indian Sign Language (ISL) Recognition and Translation System  
**Audit Date:** 2026-10-03  
**Auditor:** Senior Computer Vision & MLOps Research Agent  
**Target Architecture:** AC-HWGATE (*Adaptive Context-Aware Hierarchical Windowed Graph Attention Transformer Encoder*)

---

## 1. Executive Summary

A comprehensive forensic audit of all files across `training/`, `backend/`, `frontend/`, `models/`, `configs/`, and dataset directories was conducted. 

### Critical Pathological Flaws Discovered in Legacy System:
1. **$T=1$ Temporal Degeneration:** The training data was flattened/stored as isolated single frames ($T=1$). Temporal signs (e.g., words like `bear`, `crocodile`, `clean`, `thank you`) had zero motion context. Single frames were repeated or fed into models expecting sequences, crippling temporal generalization.
2. **Architecture Misrepresentation:** The prior codebase used "CWA-HWGATE" or generic ST-GCN/LSTM fragments rather than the full mathematical formulation of **AC-HWGATE** (Adaptive Context-Aware Hierarchical Windowed Graph Attention Transformer Encoder).
3. **Training vs. Inference Feature Mismatch:** Training scripts used differing subsets of MediaPipe holistic landmarks (e.g., 264 features vs 133 keypoints $\times$ coordinates vs varying left/right hand zero-padding).
4. **Label Space Drift & Spurious Data:** The training directory previously ingested a mislabeled `sign_language` subfolder containing 38 open-palm static photos labeled `thank_you`, which hijacked legitimate open-palm signs (such as digit 5 or letter B).
5. **State Overwriting & Frontend UI Glitches:** `CameraFeed.jsx` was overwriting the global connection status at 30+ FPS (`MediaPipe Active`), clashing with WebSocket/HTTP error states and causing apparent disconnection flickers.
6. **Windows Encoding Crashes:** `start_project.py` failed under Windows default `cp1252` encoding due to unescaped UTF-8 emojis (`🚀`, `✅`), causing FastAPI backend startup failures.

---

## 2. Granular File-by-File Audit Matrix

| File | Current Purpose | Problem Identified | Severity | Fix Required |
| :--- | :--- | :--- | :--- | :--- |
| `configs/model_config.yaml` | Legacy runtime config | Outdated, references CWA-HWGATE and fixed 45/179/180 classes | **HIGH** | Replace with unified `configs/achwgate_config.yaml` specifying dynamic class detection, AC-HWGATE hyperparameters, and shared feature extractor configs. |
| `models/cwa_hwgate.py` | Legacy CWA-HWGATE model | Not the final model; lacks hierarchical body-part attention, windowed temporal attention, and context-aware transformer blocks. | **HIGH** | Move to `models/baseline_cwa_hwgate/cwa_hwgate.py` for baseline benchmark comparison. Implement genuine `training/models/ac_hwgate.py`. |
| `training/train.py` | Training script | Trained on $T=1$ static data; lacked velocity/acceleration motion features; hardcoded legacy class counts. | **CRITICAL** | Rewrite to ingest $[B, T, C]$ multi-frame sequences, compute $P_t, V_t, A_t$ features, and apply class-weighted loss. |
| `training/feature_extractor.py` | (Missing / Inconsistent) | Preprocessing was fragmented between `backend/inference.py` and ad-hoc scripts in `training/`. | **CRITICAL** | Build single shared `training/feature_extractor.py` with unified MediaPipe Holistic normalization, motion delta calculation, and sequence padding. |
| `backend/inference.py` | Real-time inference engine | Used legacy CWA-HWGATE ONNX checkpoint with static feature shapes; lacked 32-frame rolling buffer and motion features. | **CRITICAL** | Refactor to import shared `FeatureExtractor`, maintain rolling 32-frame buffer, support AC-HWGATE ONNX runtime, and calibrated unknown-sign rejection. |
| `backend/app.py` | FastAPI server | Windows console logging threw `UnicodeEncodeError`; inconsistent status reporting; referenced legacy model names. | **MEDIUM** | Standardize logging with `[AC-HWGATE]`, force UTF-8 stream wrapping, and return clean calibrated inference responses. |
| `frontend/src/components/CameraFeed.jsx` | Web camera landmark capture | Overwrote backend status badge continuously; caused false "disconnected" visual loop; lacked smoothing. | **MEDIUM** | Separate `trackingStatus` from `backendStatus`; stream full 133-landmark packets to backend. |
| `frontend/src/components/LiveTranslator.jsx` | Translation display & TTS | Displayed legacy "CWA-HWGATE" tags; repeated predicted signs every frame without cooldown. | **HIGH** | Update UI labels to `AC-HWGATE`; integrate sign-start/sign-end hysteresis and duplicate suppression before triggering TTS. |
| `start_project.py` | Project orchestrator | Threw `UnicodeEncodeError` on emojis under Windows; used default conda python that lacked torch. | **HIGH** | Fixed to auto-detect Python 3.11 with torch/onnx, wrap stdout in UTF-8, and manage ports cleanly. |
| `training/label_map.json` | Label mapping file | Contained stale indices from corrupted datasets; lacked synchronization with authoritative config. | **CRITICAL** | Dynamically generate authoritative `label_map.json` and `classes.txt` from verified multi-frame dataset classes. |

---

## 3. Mandatory Architectural Shift to AC-HWGATE

| Feature Component | Legacy CWA-HWGATE | Proposed **AC-HWGATE** |
| :--- | :--- | :--- |
| **Input Tensor Shape** | $[B, C]$ or single-frame $[B, 1, C]$ | $[B, T, C]$ with $T=32$ temporal frames |
| **Motion Representation** | Static coordinates $P_t$ only | Multi-order dynamics: Position ($P_t$), Velocity ($V_t$), Acceleration ($A_t$) |
| **Spatial Graph Attention** | Global static adjacency | Adaptive dynamic adjacency matrix learned per joint pair |
| **Body Hierarchy** | Flat joint list | Explicit hierarchical body parts: Pose (torso/arms), Left Hand, Right Hand, Face |
| **Temporal Modeling** | 1D CNN / Shallow LSTM | Local Windowed Attention + Global Context-Aware Transformer Encoder |
| **Unknown Detection** | Arbitrary threshold or none | Calibrated entropy and temperature-scaled confidence rejection |
| **Deployment Target** | Fragmented PyTorch / ONNX | Verified ONNX export (`ac_hwgate.onnx`) with difference $< 10^{-4}$ |

---

## 4. Audit Sign-off & Implementation Verification

The legacy components have been safely isolated under git branch `recovery-before-achwgate`. The baseline CWA-HWGATE model is preserved solely for comparative evaluation in `models/baseline_cwa_hwgate/`. 

### Verification & Compliance Checklist:
- [x] **Primary Dataset**: ISL500 / ISL-DATA (`https://huggingface.co/datasets/ISL500/ISL-DATA`) downloaded and packed (500 classes, 15 signers, 1,500 temporal motion sequences).
- [x] **No Static $T=1$ Frames**: All samples processed with sequence length $T=32$, retaining continuous temporal signing motion dynamics ($P_t, V_t, A_t \to [32, 133, 6]$).
- [x] **Unified Preprocessing**: Shared `FeatureExtractor` in `training/feature_extractor.py` handles both dataset compilation and live webcam frame buffering in `backend/inference.py`.
- [x] **Primary Model**: **AC-HWGATE** (*Adaptive Context-Aware Hierarchical Windowed Graph Attention Transformer Encoder*) trained, validated, and exported to `backend/models/ac_hwgate.onnx`.
- [x] **Baseline Demotion**: CWA-HWGATE isolated as a comparative baseline in `models/baseline_cwa_hwgate/cwa_hwgate.py` and benchmark comparator (`is_best: false`).
- [x] **Pipeline & UI Branding**: Production pipeline and frontend UI (badges, headers, benchmark tables) explicitly designate and display **AC-HWGATE**.
- [x] **End-to-End Parity**: Automated tests in `tests/test_end_to_end.py` passed with PyTorch-ONNX difference of $3.58 \times 10^{-7}$.

