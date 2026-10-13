# Comprehensive Dataset Inventory & Audit Report (ISL 50-Class System)

**Project:** Real-Time Indian Sign Language (ISL) Recognition & Translation System  
**Architecture:** AC-HWGATE (*Adaptive Context-Aware Hierarchical Windowed Graph Attention Transformer Encoder*)  
**Target Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM)  
**Date:** 2026-10-12  

---

## 1. Executive Summary & Dataset Discovery

A complete forensic inspection and inventory of all local datasets, annotations, label mappings, preprocessing pipelines, model checkpoints, and inference engines was conducted.

### 1.1 Located Local Datasets
Two primary local dataset sources were discovered and verified across local storage without requiring external internet downloads:

1. **Dataset A: Authentic Multi-Source ISL Benchmark Corpus**
   - **Locations:**
     - `data/raw_40words/`: 642 authentic MP4 video clips with rich per-clip metadata in `metadata.csv`.
     - `data/isl500_landmarks/`: 2,439 pre-extracted continuous temporal landmark sequences (`.npz`) across 29 classes.
     - `data/cache/`: 645 verified pre-extracted sequence tensors (`.npz`) covering 40 classes.
     - `training/isl500_meta/MappingFiles/`: Person-dependent and person-independent signer mapping files (`PersonDependentTrain.csv`: 69,507 mappings; `PersonDependentTest.csv`: 17,256 mappings).
   - **Modalities & Structure:** 133 WholeBody landmarks (33 Pose, 21 Left Hand, 21 Right Hand, 58 Face) with 6 motion feature channels ($P_x, P_y, V_x, V_y, A_x, A_y$) over $T=32$ temporal frames ($[32, 133, 6]$ tensor format).
   - **Signer Diversity:** 15 distinct signers (`USER001` through `USER015`) in ISL500 plus multiple authentic signers from CISLR, INCLUDE, and ISLRTC in `raw_40words`.

2. **Dataset B: Local High-Definition ISL Video Corpus (Archive-2)**
   - **Location:** `C:\Users\navee\Downloads\archive (2)\Video_Dataset\Video_Dataset\`
   - **Total Content:** 61 verified ISL classes, 3,630 authentic high-resolution MP4 video recordings ($640 \times 720$ @ 30 FPS).
   - **Structure:** Exactly 60 video clips per class (except `Still`: 30 clips).
   - **Recording Diversity:** Each class contains multi-angle recording sessions: frontal view (`WIN_*_Pro.mp4`), left tilt ($+15^\circ$, `WIN_*_left_tilt.mp4`), and right tilt ($-15^\circ$, `WIN_*_right_tilt.mp4`).

---

## 2. Granular Class Inventory (Discovered Classes & Sample Counts)

Across Dataset A and Dataset B, **95 unique ISL classes** were discovered. Below is the authoritative sample count matrix:

### Top Discovered Classes by Usable Sample Count

| # | Normalized Class Label | ISL500 Landmarks (`.npz`) | Raw 40-Words (`.mp4`) | Archive-2 Videos (`.mp4`) | Total Available Samples | Usable Samples (Quality-Verified) |
|---|---|---|---|---|---|---|
| 1 | `drink` | 133 | 16 | 60 | 209 | 209 |
| 2 | `hello` | 75 | 36 | 60 | 171 | 171 |
| 3 | `thank_you` | 72 | 39 | 60 | 171 | 171 |
| 4 | `tea` | 72 | 17 | 60 | 149 | 149 |
| 5 | `me` | 156 | 3 | 0 | 159 | 159 |
| 6 | `eat` | 133 | 18 | 0 | 151 | 151 |
| 7 | `school` | 83 | 37 | 0 | 120 | 120 |
| 8 | `market` | 83 | 36 | 0 | 119 | 119 |
| 9 | `friend` | 79 | 38 | 0 | 117 | 117 |
| 10 | `mother` | 86 | 18 | 0 | 104 | 104 |
| 11 | `please` | 83 | 18 | 0 | 101 | 101 |
| 12 | `sister` | 82 | 17 | 0 | 99 | 99 |
| 13 | `she` | 81 | 17 | 0 | 98 | 98 |
| 14 | `sit` | 79 | 19 | 0 | 98 | 98 |
| 15 | `you` | 80 | 18 | 0 | 98 | 98 |
| 16 | `father` | 81 | 16 | 0 | 97 | 97 |
| 17 | `help` | 81 | 16 | 0 | 97 | 97 |
| 18 | `he` | 79 | 17 | 0 | 96 | 96 |
| 19 | `no` | 79 | 17 | 0 | 96 | 96 |
| 20 | `go` | 78 | 17 | 0 | 95 | 95 |
| 21 | `student` | 78 | 17 | 0 | 95 | 95 |
| 22 | `water` | 74 | 18 | 0 | 92 | 92 |
| 23 | `food` | 74 | 17 | 0 | 91 | 91 |
| 24 | `yes` | 74 | 17 | 0 | 91 | 91 |
| 25 | `what` | 72 | 17 | 0 | 89 | 89 |
| 26 | `where` | 72 | 17 | 0 | 89 | 89 |
| 27 | `teacher` | 71 | 17 | 0 | 88 | 88 |
| 28 | `today` | 71 | 17 | 5 | 93 | 93 |
| 29 | `home` | 78 | 1 | 0 | 79 | 79 |
| 30 | `come` | 0 | 4 | 60 | 64 | 64 |
| 31 | `write` | 0 | 3 | 60 | 63 | 63 |
| 32 | `bear` | 0 | 0 | 60 | 60 | 60 |
| 33 | `break` | 0 | 0 | 60 | 60 | 60 |
| 34 | `clean` | 0 | 0 | 60 | 60 | 60 |
| 35 | `close` | 0 | 0 | 60 | 60 | 60 |
| 36 | `cook` | 0 | 0 | 60 | 60 | 60 |
| 37 | `cry` | 0 | 0 | 60 | 60 | 60 |
| 38 | `elephant` | 0 | 0 | 60 | 60 | 60 |
| 39 | `give` | 0 | 0 | 60 | 60 | 60 |
| 40 | `jump` | 0 | 0 | 60 | 60 | 60 |
| 41 | `key` | 0 | 0 | 60 | 60 | 60 |
| 42 | `knife` | 0 | 0 | 60 | 60 | 60 |
| 43 | `lion` | 0 | 0 | 60 | 60 | 60 |
| 44 | `monkey` | 0 | 0 | 60 | 60 | 60 |
| 45 | `temple` | 0 | 0 | 60 | 60 | 60 |
| 46 | `tiger` | 0 | 0 | 60 | 60 | 60 |
| 47 | `umbrella` | 0 | 0 | 60 | 60 | 60 |
| 48 | `uncle` | 0 | 0 | 60 | 60 | 60 |
| 49 | `wife` | 0 | 0 | 60 | 60 | 60 |
| 50 | `wrong` | 0 | 0 | 60 | 60 | 60 |

---

## 3. Data-Quality, Anomalies & Conflict Resolution

### 3.1 Label Discrepancies and Semantic Conflicts Resolved
1. **`thankyou` vs `thank_you` vs `Thank you`:**
   - Dataset A contained both `thank_you` and `thank you`.
   - Dataset B labeled the folder as `Thank you`.
   - **Resolution:** Canonicalized strictly to `thank_you`.

2. **`house` vs `home`:**
   - In Dataset A, ISL500 provided 78 sequences for `home`, while INCLUDE metadata contained 1 entry for `house`.
   - **Resolution:** Mapped `house` alias to `home` (canonical ISL gloss for residential dwelling).

3. **`myself` / `i` vs `me`:**
   - In ISL500, `me` was represented across 156 clips; Vidit031 metadata aliased `i` and `myself`.
   - **Resolution:** Canonicalized to `me`.

4. **`writer` vs `write`:**
   - Dataset B contained folder `Writer` with 60 video clips.
   - Dataset A had 3 clips under `write`.
   - **Resolution:** Both refer to the ISL sign for writing/scribe action. Mapped to canonical gloss `write`.

5. **Exclusion of Compound Phrases and Multi-Word Sentences:**
   - Dataset B included conversational compound phrases: `Good Morning` (60), `Good afternoon` (60), and the full sentence `What is your Name` (60).
   - **Decision:** Excluded from the 50-class isolated word vocabulary to prevent structural mismatch with single-sign lexical classification.

### 3.2 Data Leakage Prevention Across Multi-Tilt Video Sessions
In Dataset B, each recording session contains 3 files per sign:
- Frontal perspective (`WIN_*_Pro.mp4`)
- Left tilt ($+15^\circ$, `WIN_*_left_tilt.mp4`)
- Right tilt ($-15^\circ$, `WIN_*_right_tilt.mp4`)

**CRITICAL LEAKAGE RULE:** Tilt augmentations of the same recording session share identical signer clothing, physical appearance, and background. Splitting `left_tilt` into train and `right_tilt` into test causes pseudo-overfitting and inflated offline test metrics.  
**Enforced Solution:** All tilt variations for a given recording session (`session_id`) are grouped atomically into the same partition (Session-Independent Split).

---

## 4. Selection of the 50 Classes

1. **Empirical Threshold:** Minimum sample threshold set to $\ge 60$ valid, non-corrupt sequences per class.
2. **Balanced Representation:**
   - Every selected class has between 60 and 209 authentic samples (mean: 90.3 samples/class).
   - Weighted loss balancing and `WeightedRandomSampler` ensure equal gradient updates per class during training.
3. **No Synthetic Fabrication:** Zero synthetic frame repetition, zero static photo duplication, zero placeholder injection.
4. **Machine-Readable Manifest:** Synchronized at `configs/selected_50_classes_manifest.json` and loaded identically across training, ONNX export, FastAPI backend, and React frontend.
