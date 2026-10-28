# Complete Forensic Audit of Local Indian Sign Language (ISL) Datasets
**Date:** 2026-10-21  **Target Architecture:** AC-HWGATE (*Adaptive Context-Aware Hierarchical Windowed Graph Attention Transformer Encoder*)  **Hardware Environment:** NVIDIA GeForce RTX 3050 Laptop GPU (4.0 GB VRAM, CUDA Enabled, AMP FP16)  
---
## 1. Executive Summary & Local Dataset Discovery
A full forensic audit of all local training examples was conducted without downloading internet datasets or replacing real data with simulated numbers.
| Dataset Metric | Dataset 1 (Vivit) | Dataset 2 (Video_Dataset) | Combined Authoritative Pool |
| :--- | :--- | :--- | :--- |
| **Location** | `C:\Users\navee\Downloads\archive\ProcessedData_vivit` | `C:\Users\navee\Downloads\archive (2)\Video_Dataset\Video_Dataset` | Both Local Directories |
| **Container & Format** | QuickTime `.MOV` (H.264 / Apple ProRes) | MPEG-4 `.mp4` (H.264 / AVC) | Isolated Dynamic Videos |
| **Total Raw Classes** | **76** | **61** | **137** unique classes |
| **Total Authentic Videos** | **1166** | **3630** | **4796** videos |
| **Average Duration** | ~3.5 - 5.0 seconds | ~6.75 seconds (~202 frames) | Continuous temporal sequence |
| **Resolution** | 1920x1080 / 1280x720 | 640x720 @ 30 FPS | High-definition RGB |
| **Motion Type** | **Continuous Temporal Motion** | **Continuous Temporal Motion** | **100% Genuine Dynamic Motion** |
| **Static $T=1$ Degradation** | **None (0%)** | **None (0%)** | **Strictly Prohibited** |
| **Total Eligible Classes** | **75** | **61** | **136 Classes** |
| **Excluded Classes** | **1** (`second`: 3 takes, 1 session) | **0** | **1 Class** |

## 2. Granular Class-by-Class Census Table
The table below lists all 137 discovered classes across both local datasets, showing raw sample counts, independent sessions, provenance source, and eligibility determination:

| # | Class Label | Dataset 1 (Vivit) | Dataset 2 (Video_Dataset) | Total Samples | Sessions / Takes | Status | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `afternoon` | 11 | 0 | 11 | 3 | **ELIGIBLE** | Multi-session takes (3 sessions) |
| 2 | `animal` | 18 | 0 | 18 | 4 | **ELIGIBLE** | Multi-session takes (4 sessions) |
| 3 | `bad` | 21 | 0 | 21 | 7 | **ELIGIBLE** | Multi-session takes (7 sessions) |
| 4 | `bear` | 0 | 60 | 60 | 20 | **ELIGIBLE** | Fully balanced (20 sessions x 3 tilts) |
| 5 | `beautiful` | 9 | 0 | 9 | 2 | **ELIGIBLE** | Multi-session takes (2 sessions) |
| 6 | `big` | 21 | 0 | 21 | 5 | **ELIGIBLE** | Multi-session takes (5 sessions) |
| 7 | `bird` | 18 | 0 | 18 | 3 | **ELIGIBLE** | Multi-session takes (3 sessions) |
| 8 | `blind` | 8 | 0 | 8 | 2 | **ELIGIBLE** | Multi-session takes (2 sessions) |
| 9 | `break` | 0 | 60 | 60 | 20 | **ELIGIBLE** | Fully balanced (20 sessions x 3 tilts) |
| 10 | `brinjal` | 0 | 60 | 60 | 20 | **ELIGIBLE** | Fully balanced (20 sessions x 3 tilts) |
| 11 | `budget` | 0 | 60 | 60 | 20 | **ELIGIBLE** | Fully balanced (20 sessions x 3 tilts) |
| 12 | `busy` | 0 | 60 | 60 | 20 | **ELIGIBLE** | Fully balanced (20 sessions x 3 tilts) |
| 13 | `cabbage` | 0 | 60 | 60 | 20 | **ELIGIBLE** | Fully balanced (20 sessions x 3 tilts) |
| 14 | `carrot` | 0 | 60 | 60 | 20 | **ELIGIBLE** | Fully balanced (20 sessions x 3 tilts) |
| 15 | `cat` | 18 | 0 | 18 | 3 | **ELIGIBLE** | Multi-session takes (3 sessions) |
| 16 | `cauliflower` | 0 | 60 | 60 | 20 | **ELIGIBLE** | Fully balanced (20 sessions x 3 tilts) |
| 17 | `cheap` | 8 | 0 | 8 | 2 | **ELIGIBLE** | Multi-session takes (2 sessions) |
| 18 | `chilli` | 0 | 60 | 60 | 20 | **ELIGIBLE** | Fully balanced (20 sessions x 3 tilts) |
| 19 | `clean` | 0 | 60 | 60 | 20 | **ELIGIBLE** | Fully balanced (20 sessions x 3 tilts) |
| 20 | `close` | 0 | 60 | 60 | 20 | **ELIGIBLE** | Fully balanced (20 sessions x 3 tilts) |
| 21 | `clothing` | 19 | 0 | 19 | 7 | **ELIGIBLE** | Multi-session takes (7 sessions) |
| 22 | `cold` | 20 | 0 | 20 | 6 | **ELIGIBLE** | Multi-session takes (6 sessions) |
| 23 | `come` | 0 | 60 | 60 | 20 | **ELIGIBLE** | Fully balanced (20 sessions x 3 tilts) |
| 24 | `cook` | 0 | 60 | 60 | 20 | **ELIGIBLE** | Fully balanced (20 sessions x 3 tilts) |
| 25 | `cow` | 19 | 0 | 19 | 3 | **ELIGIBLE** | Multi-session takes (3 sessions) |
| 26 | `crocodile` | 0 | 60 | 60 | 20 | **ELIGIBLE** | Fully balanced (20 sessions x 3 tilts) |
| 27 | `cry` | 0 | 60 | 60 | 20 | **ELIGIBLE** | Fully balanced (20 sessions x 3 tilts) |
| 28 | `cucumber` | 0 | 60 | 60 | 20 | **ELIGIBLE** | Fully balanced (20 sessions x 3 tilts) |
| 29 | `curved` | 8 | 0 | 8 | 2 | **ELIGIBLE** | Multi-session takes (2 sessions) |
| 30 | `deaf` | 8 | 0 | 8 | 2 | **ELIGIBLE** | Multi-session takes (2 sessions) |
| 31 | `deer` | 0 | 60 | 60 | 20 | **ELIGIBLE** | Fully balanced (20 sessions x 3 tilts) |
| 32 | `dog` | 18 | 0 | 18 | 3 | **ELIGIBLE** | Multi-session takes (3 sessions) |
| 33 | `dress` | 19 | 0 | 19 | 7 | **ELIGIBLE** | Multi-session takes (7 sessions) |
| 34 | `drink` | 0 | 60 | 60 | 20 | **ELIGIBLE** | Fully balanced (20 sessions x 3 tilts) |
| 35 | `dry` | 21 | 0 | 21 | 7 | **ELIGIBLE** | Multi-session takes (7 sessions) |
| 36 | `elephant` | 0 | 60 | 60 | 20 | **ELIGIBLE** | Fully balanced (20 sessions x 3 tilts) |
| 37 | `evening` | 11 | 0 | 11 | 3 | **ELIGIBLE** | Multi-session takes (3 sessions) |
| 38 | `exam` | 0 | 60 | 60 | 20 | **ELIGIBLE** | Fully balanced (20 sessions x 3 tilts) |
| 39 | `expensive` | 8 | 0 | 8 | 2 | **ELIGIBLE** | Multi-session takes (2 sessions) |
| 40 | `famous` | 8 | 0 | 8 | 2 | **ELIGIBLE** | Multi-session takes (2 sessions) |
| 41 | `fast` | 21 | 0 | 21 | 5 | **ELIGIBLE** | Multi-session takes (5 sessions) |
| 42 | `fedup` | 0 | 60 | 60 | 20 | **ELIGIBLE** | Fully balanced (20 sessions x 3 tilts) |
| 43 | `female` | 8 | 0 | 8 | 2 | **ELIGIBLE** | Multi-session takes (2 sessions) |
| 44 | `fever` | 0 | 60 | 60 | 20 | **ELIGIBLE** | Fully balanced (20 sessions x 3 tilts) |
| 45 | `fish` | 18 | 0 | 18 | 3 | **ELIGIBLE** | Multi-session takes (3 sessions) |
| 46 | `flat` | 8 | 0 | 8 | 2 | **ELIGIBLE** | Multi-session takes (2 sessions) |
| 47 | `friday` | 11 | 0 | 11 | 3 | **ELIGIBLE** | Multi-session takes (3 sessions) |
| 48 | `giraffe` | 0 | 60 | 60 | 20 | **ELIGIBLE** | Fully balanced (20 sessions x 3 tilts) |
| 49 | `give` | 0 | 60 | 60 | 20 | **ELIGIBLE** | Fully balanced (20 sessions x 3 tilts) |
| 50 | `good` | 21 | 0 | 21 | 7 | **ELIGIBLE** | Multi-session takes (7 sessions) |
| 51 | `good_afternoon` | 0 | 60 | 60 | 20 | **ELIGIBLE** | Fully balanced (20 sessions x 3 tilts) |
| 52 | `good_morning` | 0 | 60 | 60 | 20 | **ELIGIBLE** | Fully balanced (20 sessions x 3 tilts) |
| 53 | `happy` | 21 | 0 | 21 | 7 | **ELIGIBLE** | Multi-session takes (7 sessions) |
| 54 | `hat` | 19 | 0 | 19 | 7 | **ELIGIBLE** | Multi-session takes (7 sessions) |
| 55 | `healthy` | 21 | 0 | 21 | 7 | **ELIGIBLE** | Multi-session takes (7 sessions) |
| 56 | `hello` | 0 | 60 | 60 | 20 | **ELIGIBLE** | Fully balanced (20 sessions x 3 tilts) |
| 57 | `horse` | 18 | 0 | 18 | 4 | **ELIGIBLE** | Multi-session takes (4 sessions) |
| 58 | `hot` | 21 | 0 | 21 | 5 | **ELIGIBLE** | Multi-session takes (5 sessions) |
| 59 | `hour` | 11 | 0 | 11 | 3 | **ELIGIBLE** | Multi-session takes (3 sessions) |
| 60 | `hug` | 0 | 60 | 60 | 20 | **ELIGIBLE** | Fully balanced (20 sessions x 3 tilts) |
| 61 | `injury` | 0 | 60 | 60 | 20 | **ELIGIBLE** | Fully balanced (20 sessions x 3 tilts) |
| 62 | `interview` | 0 | 60 | 60 | 20 | **ELIGIBLE** | Fully balanced (20 sessions x 3 tilts) |
| 63 | `jump` | 0 | 60 | 60 | 20 | **ELIGIBLE** | Fully balanced (20 sessions x 3 tilts) |
| 64 | `karnataka` | 0 | 60 | 60 | 20 | **ELIGIBLE** | Fully balanced (20 sessions x 3 tilts) |
| 65 | `key` | 0 | 60 | 60 | 20 | **ELIGIBLE** | Fully balanced (20 sessions x 3 tilts) |
| 66 | `knife` | 0 | 60 | 60 | 20 | **ELIGIBLE** | Fully balanced (20 sessions x 3 tilts) |
| 67 | `lemon` | 0 | 60 | 60 | 20 | **ELIGIBLE** | Fully balanced (20 sessions x 3 tilts) |
| 68 | `light` | 8 | 0 | 8 | 2 | **ELIGIBLE** | Multi-session takes (2 sessions) |
| 69 | `lion` | 0 | 60 | 60 | 20 | **ELIGIBLE** | Fully balanced (20 sessions x 3 tilts) |
| 70 | `long` | 21 | 0 | 21 | 5 | **ELIGIBLE** | Multi-session takes (5 sessions) |
| 71 | `loose` | 8 | 0 | 8 | 2 | **ELIGIBLE** | Multi-session takes (2 sessions) |
| 72 | `loud` | 21 | 0 | 21 | 7 | **ELIGIBLE** | Multi-session takes (7 sessions) |
| 73 | `man` | 0 | 60 | 60 | 20 | **ELIGIBLE** | Fully balanced (20 sessions x 3 tilts) |
| 74 | `maths` | 0 | 60 | 60 | 20 | **ELIGIBLE** | Fully balanced (20 sessions x 3 tilts) |
| 75 | `maybe` | 0 | 60 | 60 | 20 | **ELIGIBLE** | Fully balanced (20 sessions x 3 tilts) |
| 76 | `minute` | 11 | 0 | 11 | 4 | **ELIGIBLE** | Multi-session takes (4 sessions) |
| 77 | `monday` | 11 | 0 | 11 | 3 | **ELIGIBLE** | Multi-session takes (3 sessions) |
| 78 | `monkey` | 0 | 60 | 60 | 20 | **ELIGIBLE** | Fully balanced (20 sessions x 3 tilts) |
| 79 | `month` | 11 | 0 | 11 | 3 | **ELIGIBLE** | Multi-session takes (3 sessions) |
| 80 | `morning` | 11 | 0 | 11 | 3 | **ELIGIBLE** | Multi-session takes (3 sessions) |
| 81 | `mouse` | 18 | 0 | 18 | 4 | **ELIGIBLE** | Multi-session takes (4 sessions) |
| 82 | `narrow` | 21 | 0 | 21 | 5 | **ELIGIBLE** | Multi-session takes (5 sessions) |
| 83 | `new` | 21 | 0 | 21 | 7 | **ELIGIBLE** | Multi-session takes (7 sessions) |
| 84 | `night` | 11 | 0 | 11 | 3 | **ELIGIBLE** | Multi-session takes (3 sessions) |
| 85 | `old` | 21 | 0 | 21 | 7 | **ELIGIBLE** | Multi-session takes (7 sessions) |
| 86 | `onion` | 0 | 60 | 60 | 20 | **ELIGIBLE** | Fully balanced (20 sessions x 3 tilts) |
| 87 | `pant` | 19 | 0 | 19 | 7 | **ELIGIBLE** | Multi-session takes (7 sessions) |
| 88 | `peacock` | 0 | 60 | 60 | 20 | **ELIGIBLE** | Fully balanced (20 sessions x 3 tilts) |
| 89 | `pigeon` | 0 | 60 | 60 | 20 | **ELIGIBLE** | Fully balanced (20 sessions x 3 tilts) |
| 90 | `pocket` | 19 | 0 | 19 | 7 | **ELIGIBLE** | Multi-session takes (7 sessions) |
| 91 | `pour` | 0 | 60 | 60 | 20 | **ELIGIBLE** | Fully balanced (20 sessions x 3 tilts) |
| 92 | `quiet` | 21 | 0 | 21 | 7 | **ELIGIBLE** | Multi-session takes (7 sessions) |
| 93 | `radish` | 0 | 60 | 60 | 20 | **ELIGIBLE** | Fully balanced (20 sessions x 3 tilts) |
| 94 | `sad` | 8 | 0 | 8 | 2 | **ELIGIBLE** | Multi-session takes (2 sessions) |
| 95 | `saturday` | 11 | 0 | 11 | 3 | **ELIGIBLE** | Multi-session takes (3 sessions) |
| 96 | `second` | 3 | 0 | 3 | 1 | *EXCLUDED* | Only 3 samples in 1 single session (insufficient for disjoint test) |
| 97 | `shirt` | 19 | 0 | 19 | 8 | **ELIGIBLE** | Multi-session takes (8 sessions) |
| 98 | `shoes` | 19 | 0 | 19 | 7 | **ELIGIBLE** | Multi-session takes (7 sessions) |
| 99 | `short` | 22 | 0 | 22 | 5 | **ELIGIBLE** | Multi-session takes (5 sessions) |
| 100 | `sick` | 21 | 0 | 21 | 7 | **ELIGIBLE** | Multi-session takes (7 sessions) |
| 101 | `skirt` | 18 | 0 | 18 | 8 | **ELIGIBLE** | Multi-session takes (8 sessions) |
| 102 | `slow` | 21 | 0 | 21 | 5 | **ELIGIBLE** | Multi-session takes (5 sessions) |
| 103 | `small` | 22 | 0 | 22 | 5 | **ELIGIBLE** | Multi-session takes (5 sessions) |
| 104 | `sparrow` | 0 | 60 | 60 | 20 | **ELIGIBLE** | Fully balanced (20 sessions x 3 tilts) |
| 105 | `still` | 0 | 30 | 30 | 10 | **ELIGIBLE** | Multi-session takes (10 sessions) |
| 106 | `suit` | 18 | 0 | 18 | 7 | **ELIGIBLE** | Multi-session takes (7 sessions) |
| 107 | `sunday` | 11 | 0 | 11 | 3 | **ELIGIBLE** | Multi-session takes (3 sessions) |
| 108 | `switch` | 0 | 60 | 60 | 20 | **ELIGIBLE** | Fully balanced (20 sessions x 3 tilts) |
| 109 | `t_shirt` | 19 | 0 | 19 | 7 | **ELIGIBLE** | Multi-session takes (7 sessions) |
| 110 | `tall` | 21 | 0 | 21 | 5 | **ELIGIBLE** | Multi-session takes (5 sessions) |
| 111 | `tea` | 0 | 60 | 60 | 20 | **ELIGIBLE** | Fully balanced (20 sessions x 3 tilts) |
| 112 | `temple` | 0 | 60 | 60 | 20 | **ELIGIBLE** | Fully balanced (20 sessions x 3 tilts) |
| 113 | `thank_you` | 0 | 60 | 60 | 20 | **ELIGIBLE** | Fully balanced (20 sessions x 3 tilts) |
| 114 | `thursday` | 11 | 0 | 11 | 3 | **ELIGIBLE** | Multi-session takes (3 sessions) |
| 115 | `tiger` | 0 | 60 | 60 | 20 | **ELIGIBLE** | Fully balanced (20 sessions x 3 tilts) |
| 116 | `time` | 11 | 0 | 11 | 3 | **ELIGIBLE** | Multi-session takes (3 sessions) |
| 117 | `today` | 11 | 0 | 11 | 3 | **ELIGIBLE** | Multi-session takes (3 sessions) |
| 118 | `tomorrow` | 11 | 0 | 11 | 3 | **ELIGIBLE** | Multi-session takes (3 sessions) |
| 119 | `tuesday` | 11 | 0 | 11 | 3 | **ELIGIBLE** | Multi-session takes (3 sessions) |
| 120 | `turtle` | 0 | 60 | 60 | 20 | **ELIGIBLE** | Fully balanced (20 sessions x 3 tilts) |
| 121 | `ugly` | 8 | 0 | 8 | 2 | **ELIGIBLE** | Multi-session takes (2 sessions) |
| 122 | `umbrella` | 0 | 60 | 60 | 20 | **ELIGIBLE** | Fully balanced (20 sessions x 3 tilts) |
| 123 | `uncle` | 0 | 60 | 60 | 20 | **ELIGIBLE** | Fully balanced (20 sessions x 3 tilts) |
| 124 | `vegetables` | 0 | 60 | 60 | 20 | **ELIGIBLE** | Fully balanced (20 sessions x 3 tilts) |
| 125 | `volcano` | 0 | 60 | 60 | 20 | **ELIGIBLE** | Fully balanced (20 sessions x 3 tilts) |
| 126 | `warm` | 21 | 0 | 21 | 6 | **ELIGIBLE** | Multi-session takes (6 sessions) |
| 127 | `wednesday` | 11 | 0 | 11 | 4 | **ELIGIBLE** | Multi-session takes (4 sessions) |
| 128 | `week` | 11 | 0 | 11 | 3 | **ELIGIBLE** | Multi-session takes (3 sessions) |
| 129 | `wet` | 21 | 0 | 21 | 7 | **ELIGIBLE** | Multi-session takes (7 sessions) |
| 130 | `what_is_your_name` | 0 | 60 | 60 | 20 | **ELIGIBLE** | Fully balanced (20 sessions x 3 tilts) |
| 131 | `wide` | 21 | 0 | 21 | 5 | **ELIGIBLE** | Multi-session takes (5 sessions) |
| 132 | `wife` | 0 | 60 | 60 | 20 | **ELIGIBLE** | Fully balanced (20 sessions x 3 tilts) |
| 133 | `write` | 0 | 60 | 60 | 20 | **ELIGIBLE** | Fully balanced (20 sessions x 3 tilts) |
| 134 | `wrong` | 0 | 60 | 60 | 20 | **ELIGIBLE** | Fully balanced (20 sessions x 3 tilts) |
| 135 | `year` | 11 | 0 | 11 | 3 | **ELIGIBLE** | Multi-session takes (3 sessions) |
| 136 | `yesterday` | 11 | 0 | 11 | 3 | **ELIGIBLE** | Multi-session takes (3 sessions) |
| 137 | `young` | 21 | 0 | 21 | 7 | **ELIGIBLE** | Multi-session takes (7 sessions) |

## 3. Disjoint Partitioning & Anti-Leakage Protocol
- **Session & Tilt Atomicity:** In Dataset 2 (`Video_Dataset`), each authentic take produces 3 files (`_Pro.mp4`, `_Pro_left_tilt.mp4`, `_Pro_right_tilt.mp4`). These 3 files share an identical base session ID (`WIN_YYYYMMDD_HH_MM_SS_Pro`) and are **strictly assigned together** to either Train, Val, or Test. Never are tilts from the same take split across sets.
- **Multi-Session Grouping:** In Dataset 1 (`Vivit`), takes from the same MVI recording block are grouped to prevent intra-session memorization.
- **Stratified Split Ratio:** Exactly 70% Train, 15% Validation, and 15% Untouched Held-Out Test.
- **Hash Verification:** Every sample's rounded keypoint tensor is SHA256 hashed to guarantee $0$ sample overlap between Train-Val, Train-Test, and Val-Test.

## 4. Feature Extraction & Temporal Motion Verification
- **Temporal Dimension:** Exactly $T=32$ frames per sequence via smooth linear temporal resampling. No static frame replication.
- **Keypoints:** 133 WholeBody landmarks (Pose 33, Left Hand 21, Right Hand 21, Face 58).
- **Feature Order:** $[P_t, V_t, A_t]$ providing Position, Velocity, and Acceleration for all 133 joints (shape: `[32, 133, 6]`).
- **Spatial Normalization:** Shoulder midpoint anchor for pose, local palm-span scaling for hands, preserving 3D torso-relative signing height.
- **Bit-Exact Parity:** The exact same `FeatureExtractor` in `training/feature_extractor.py` processes raw videos into `.npz` sequences and buffers live webcam frames in `backend/inference.py`.
