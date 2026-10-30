# COMPREHENSIVE DATASET & VOCABULARY AUDIT REPORT

## 1. System Specifications & Input Representation
- **Model Architecture:** AC-HWGATE (Adaptive Context-Aware Hierarchical Windowed Graph Attention Transformer Encoder)
- **Input Representation:** `[Batch, T=32, V=133, C=6]` (Kinematic 6-channel: $x, y, dx, dy, ddx, ddy$)
- **Landmark Extraction:** MediaPipe Holistic 133 keypoints (Pose: 33, Left Hand: 21, Right Hand: 21, Face: 58)
- **Total Dataset Samples:** 4518 (Train: 3121, Val: 675, Test: 722)
- **Split Independence:** Strict Zero-Leakage (Disjoint Signers USER001-USER015 and Disjoint Recording Sessions)

## 2. Dataset Sources & Inventory

| Dataset Source | Path | Modality | Status |
| :--- | :--- | :--- | :---: |
| **RAW_40WORDS (INCLUDE/CISLR)** | `c:\Users\navee\.gemini\antigravity-ide\scratch\isl-translator\data\raw_40words` | Video MP4 & MediaPipe JSON | ✅ ACTIVE |
| **ISL500_LANDMARKS** | `c:\Users\navee\.gemini\antigravity-ide\scratch\isl-translator\data\isl500_landmarks` | Keypoint NPZ (15 signers USER001-USER015) | ✅ ACTIVE |
| **ADDITIONAL_LANDMARKS (Archive2/ViViT)** | `c:\Users\navee\.gemini\antigravity-ide\scratch\isl-translator\data\additional_landmarks` | Extracted NPZ Sequences | ✅ ACTIVE |
| **CACHE_NPZ (Vidit031 / INCLUDE Benchmark)** | `c:\Users\navee\.gemini\antigravity-ide\scratch\isl-translator\data\cache` | Processed 32-frame NPZ | ✅ ACTIVE |
| **ARCHIVE2_RAW_VIDEOS** | `C:\Users\navee\Downloads\archive (2)\Video_Dataset\Video_Dataset` | Raw MP4 Videos (tilt sessions) | ✅ ACTIVE |
| **VIVIT_RAW_VIDEOS** | `C:\Users\navee\Downloads\archive\ProcessedData_vivit` | Raw MOV Videos | ✅ ACTIVE |

## 3. Five-Category Vocabulary Accounting

- **1. Words available in dataset:** 50 words
- **2. Words used during training:** 50 words
- **3. Words supported by final checkpoint:** 50 words
- **4. Words that passed independent testing:** 44 words
- **5. Words approved for live demo:** 44 words

## 4. Per-Class Dataset Audit (All 50 Classes)

| Class Name | Train | Val | Test | Total | Recording Groups | Quality & Prior Performance | Suitable for Live Demo |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| `bear` | 42 | 9 | 9 | 60 | 5 | High quality, verified clean | ✅ APPROVED |
| `break` | 42 | 9 | 9 | 60 | 5 | High quality, verified clean | ✅ APPROVED |
| `clean` | 42 | 9 | 9 | 60 | 5 | High quality, verified clean | ✅ APPROVED |
| `close` | 42 | 9 | 9 | 60 | 5 | High quality, verified clean | ✅ APPROVED |
| `come` | 44 | 10 | 10 | 64 | 5 | High quality, verified clean | ✅ APPROVED |
| `cook` | 42 | 9 | 9 | 60 | 5 | High quality, verified clean | ✅ APPROVED |
| `cry` | 42 | 9 | 9 | 60 | 5 | High quality, verified clean | ✅ APPROVED |
| `drink` | 121 | 32 | 56 | 209 | 15 | Confusion observed (f1=0.68) | ⚠️ NEEDS TUNING |
| `eat` | 80 | 25 | 46 | 151 | 15 | Confusion observed (f1=0.44) | ⚠️ NEEDS TUNING |
| `elephant` | 42 | 9 | 9 | 60 | 5 | High quality, verified clean | ✅ APPROVED |
| `father` | 66 | 18 | 13 | 97 | 15 | High quality, verified clean | ✅ APPROVED |
| `food` | 60 | 18 | 13 | 91 | 5 | Confusion observed (f1=0.29) | ⚠️ NEEDS TUNING |
| `friend` | 81 | 18 | 18 | 117 | 5 | High quality, verified clean | ✅ APPROVED |
| `give` | 42 | 9 | 9 | 60 | 5 | High quality, verified clean | ✅ APPROVED |
| `go` | 65 | 18 | 12 | 95 | 5 | High quality, verified clean | ✅ APPROVED |
| `he` | 67 | 15 | 14 | 96 | 5 | High quality, verified clean | ✅ APPROVED |
| `hello` | 122 | 26 | 23 | 171 | 15 | High quality, verified clean | ✅ APPROVED |
| `help` | 69 | 16 | 12 | 97 | 15 | High quality, verified clean | ✅ APPROVED |
| `home` | 56 | 14 | 9 | 79 | 15 | Confusion observed (f1=0.46) | ⚠️ NEEDS TUNING |
| `jump` | 42 | 9 | 9 | 60 | 5 | High quality, verified clean | ✅ APPROVED |
| `key` | 42 | 9 | 9 | 60 | 5 | High quality, verified clean | ✅ APPROVED |
| `knife` | 42 | 9 | 9 | 60 | 5 | High quality, verified clean | ✅ APPROVED |
| `lion` | 42 | 9 | 9 | 60 | 5 | High quality, verified clean | ✅ APPROVED |
| `market` | 81 | 24 | 14 | 119 | 15 | High quality, verified clean | ✅ APPROVED |
| `me` | 111 | 28 | 19 | 158 | 5 | High quality, verified clean | ✅ APPROVED |
| `monkey` | 42 | 9 | 9 | 60 | 5 | High quality, verified clean | ✅ APPROVED |
| `mother` | 71 | 18 | 15 | 104 | 15 | High quality, verified clean | ✅ APPROVED |
| `no` | 67 | 15 | 14 | 96 | 15 | High quality, verified clean | ✅ APPROVED |
| `please` | 69 | 18 | 14 | 101 | 15 | Confusion observed (f1=0.67) | ⚠️ NEEDS TUNING |
| `school` | 81 | 21 | 18 | 120 | 15 | High quality, verified clean | ✅ APPROVED |
| `she` | 66 | 16 | 16 | 98 | 5 | High quality, verified clean | ✅ APPROVED |
| `sister` | 67 | 17 | 15 | 99 | 5 | High quality, verified clean | ✅ APPROVED |
| `sit` | 69 | 14 | 15 | 98 | 5 | High quality, verified clean | ✅ APPROVED |
| `student` | 69 | 10 | 16 | 95 | 5 | High quality, verified clean | ✅ APPROVED |
| `tea` | 108 | 18 | 23 | 149 | 15 | High quality, verified clean | ✅ APPROVED |
| `teacher` | 65 | 9 | 14 | 88 | 5 | High quality, verified clean | ✅ APPROVED |
| `temple` | 42 | 9 | 9 | 60 | 5 | High quality, verified clean | ✅ APPROVED |
| `thank_you` | 124 | 22 | 25 | 171 | 15 | High quality, verified clean | ✅ APPROVED |
| `tiger` | 42 | 9 | 9 | 60 | 5 | High quality, verified clean | ✅ APPROVED |
| `today` | 67 | 8 | 18 | 93 | 5 | High quality, verified clean | ✅ APPROVED |
| `umbrella` | 42 | 9 | 9 | 60 | 5 | High quality, verified clean | ✅ APPROVED |
| `uncle` | 42 | 9 | 9 | 60 | 5 | High quality, verified clean | ✅ APPROVED |
| `water` | 67 | 10 | 15 | 92 | 15 | Confusion observed (f1=0.55) | ⚠️ NEEDS TUNING |
| `what` | 64 | 8 | 17 | 89 | 5 | High quality, verified clean | ✅ APPROVED |
| `where` | 65 | 9 | 15 | 89 | 5 | High quality, verified clean | ✅ APPROVED |
| `wife` | 42 | 9 | 9 | 60 | 5 | High quality, verified clean | ✅ APPROVED |
| `write` | 43 | 10 | 10 | 63 | 5 | High quality, verified clean | ✅ APPROVED |
| `wrong` | 42 | 9 | 9 | 60 | 5 | High quality, verified clean | ✅ APPROVED |
| `yes` | 68 | 9 | 14 | 91 | 15 | High quality, verified clean | ✅ APPROVED |
| `you` | 70 | 10 | 18 | 98 | 5 | High quality, verified clean | ✅ APPROVED |

## 5. Summary of Diagnostic Findings
1. **Hello/Greeting Root Cause:** The previous 136-class production checkpoint relied solely on single-session tilted camera videos for `hello`, creating extreme feature distribution mismatch with real signers and causing confusions with `fish` and `morning`. In contrast, the curated 50-class corpus contains 171 multi-signer samples of `hello`.
2. **Missing Essential Words in 136-class:** `please`, `yes`, and `no` were entirely omitted from the 136-class vocabulary manifest, guaranteeing false predictions when signed. The 50-class dataset fully integrates all of them with ~100+ samples each.
3. **Physics Kinematic Inconsistency:** Prior baseline augmentation jittered positions without recomputing velocity and acceleration channels, injecting contradictory physical dynamics into training. This will be fixed in Experiment C.
