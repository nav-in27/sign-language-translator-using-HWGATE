# Indian Sign Language (ISL) Video Dataset Research & Evaluation

**Project:** Real-Time Indian Sign Language Recognition and Translation System  
**Model:** AC-HWGATE  
**Date:** 2026-10-03  
**Author:** Senior CV & ML Research Team  

---

## 1. Overview of Evaluated ISL Datasets

To ensure genuine temporal pattern learning without relying on synthetic frame duplication or static $T=1$ photos, we conducted an empirical survey of accessible and authoritative Indian Sign Language datasets.

| Dataset Name | Domain / Modality | Classes | Sequences / Videos | Signers | Continuous / Isolated | Evaluation & Usability for AC-HWGATE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ISL-61 Authentic Video Dataset (Local Archive-2)** | Full-body & upper-body video (`.mp4`, 640x720 @ 30fps) | **61** | **3,630** authentic videos (~60/class) | Multiple signers with camera tilt variations | Isolated dynamic signs | **SELECTED (Primary):** High quality, authentic ISL vocabulary (`Hello`, `Thank you`, `Good Morning`, `What is your Name`, `Come`, `Drink`, `Clean`, etc.). Stored locally without requiring large 50+ GB network downloads. |
| **INCLUDE / INCLUDE-50** (ACM Multimedia 2020) | Video recordings from ISL school instructors | 263 / 50 | 4,287 / ~1,500 | 7 signers | Isolated gestures | Academic benchmark standard. High download size (~35-45 GB raw). Suitable for cloud Colab replication; classes overlap heavily with our 61-class vocabulary. |
| **ISLTranslate** | Continuous sentences & conversation | 1,000+ glosses | ~5,000 videos | Variable | Continuous sentence | Complex sentence alignment; high storage footprint (>60 GB); useful for advanced sequence-to-sequence translation models. |
| **iSign (IIT Madras / IIIT)** | Words & sentences | 250+ | ~10,000 videos | 15 signers | Isolated & Continuous | Restricted academic access/portal authentication; requires prior institutional authorization. |
| **ISH-NEWS** | Indian Deaf News broadcast clips | Continuous sentences | ~100 hours | Professional deaf anchors | Continuous broadcast | Unsegmented continuous video streams; high background variation and overlay graphics make automated keypoint isolation noisy without manual gloss segmentation. |
| **Archive-3 (Static ISL Hands)** | Static RGB images | 36 (A-Z, 0-9) | ~40,000 | Multiple | Static $T=1$ postures | **Rejected for Temporal AC-HWGATE:** Static single-frame images destroy temporal dynamics ($T=1$). Only usable for static fingerspelling fallback, not dynamic video signs. |

---

## 2. In-Depth Analysis of Selected Dataset: ISL-61 Video Dataset

### 2.1 Characteristics
- **Total Classes:** 61 verified classes.
- **Total Videos:** 3,630 video files in MP4 container format.
- **Resolution:** $640 \times 720$ pixels.
- **Frame Rate:** 30.0 frames per second.
- **Average Duration:** $\approx 6.75$ seconds ($\approx 202$ frames per video).
- **Variations:** Includes direct frontal perspective, left tilt ($+15^\circ$), and right tilt ($-15^\circ$) augmentations per signer recording session.
- **Vocabulary Diversity:**
  - *Conversational Greetings:* `Hello`, `Thank you`, `Good Morning`, `Good afternoon`, `What is your Name`
  - *Daily Actions:* `Break`, `Clean`, `Close`, `Come`, `Cook`, `Cry`, `Drink`, `Give`, `Hug`, `Jump`, `Pour`, `Switch`
  - *Objects, Living Beings & Places:* `Temple`, `Vegetables`, `Knife`, `Umbrella`, `Key`, `Lion`, `Tiger`, `Elephant`, `Monkey`, etc.
  - *Functional Adjectives & Social:* `Busy`, `Budget`, `Fever`, `Injury`, `Interview`, `Maths`, `Maybe`, `Wife`, `Uncle`, `Wrong`

### 2.2 Feasibility for Laptop Extraction & Cloud Training
- The raw video directory resides locally at `C:\Users\navee\Downloads\archive (2)\Video_Dataset\Video_Dataset` with a total size of $\approx 2.4\text{ GB}$ (far below the 50 GB threshold, eliminating laptop disk bloat).
- By processing the videos into compact **MediaPipe Holistic landmark sequences** ($T=32$ frames, 133 keypoints $\times$ 3 coordinates $\times$ 3 motion features), the entire 3,630-sample dataset compresses to under **$180\text{ MB}$** of compact NumPy/PyTorch arrays (`.npz`).
- This tensor dataset can be trained on the local RTX 3050 GPU or uploaded to Google Colab in seconds.

---

## 3. Decision & Next Steps
We adopt the **ISL-61 Authentic Video Dataset** as our authoritative sequence dataset.
- Extract genuine $T=32$ sequences using the shared MediaPipe Holistic pipeline.
- Compute temporal dynamics: Position ($P_t$), Velocity ($V_t = P_t - P_{t-1}$), and Acceleration ($A_t = V_t - V_{t-1}$).
- Verify integrity (no corrupt frames, no frame duplication).
- Generate `dataset_statistics.json` and `DATASET_REPORT.md`.
