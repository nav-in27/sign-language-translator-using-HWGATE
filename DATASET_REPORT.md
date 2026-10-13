# Authoritative Dataset Report: ISL500 / ISL-DATA Benchmark

**Date:** 2026-10-04  
**Primary Model:** AC-HWGATE (*Adaptive Context-Aware Hierarchical Windowed Graph Attention Transformer Encoder*)  
**Baseline Model:** CWA-HWGATE (Comparative Baseline Only)  
**Dataset Source:** [ISL500 / ISL-DATA on Hugging Face](https://huggingface.co/datasets/ISL500/ISL-DATA)

---

## 1. Summary Statistics

- **Primary Dataset:** ISL500 / ISL-DATA
- **Total ISL Word Classes:** 500
- **Total Signers / Users:** 15 distinct signers (`USER001` through `USER015`)
- **Total Video Clips in Dataset:** Approximately 7,500 authentic videos
- **Modalities Provided:** Raw MP4 Videos, Google MediaPipe Holistic Landmarks, MMPose Pose Landmarks, Person-Dependent Train/Test Mapping Files
- **Temporal Modeling:** Strictly continuous temporal sequences ($T=32$ frames)
- **Joints per Frame:** 133 WholeBody/Holistic landmarks (33 Pose, 21 Left Hand, 21 Right Hand, 58 Face)
- **Channels per Joint:** 6 features (Position $P_x, P_y$, Velocity $V_x, V_y$, Acceleration $A_x, A_y$)
- **Input Tensor Shape:** $[B, 32, 133, 6]$
- **Static Frame ($T=1$) Flattening:** **Strictly Prohibited & Eliminated**
- **Preprocessing Pipeline:** Shared project `FeatureExtractor` guaranteeing exact bit-level parity between training tensors and webcam real-time inference.

---

## 2. Multi-Signer Person-Dependent & Independent Splits

The dataset provides structured mapping files:
- `PersonDependentTrain.csv`: 69,507 sequence mappings across 15 signers
- `PersonDependentTest.csv`: 17,256 sequence mappings across 15 signers

Every single one of the 500 ISL vocabulary words (from `Absent`, `Accept`, `Accident`, `Actor` to `Yesterday`, `You`, `Young`, `Zebra`, `Zipper`, `Zoo`) is represented across multiple signers to ensure cross-user invariant representation learning.

---

## 3. Motion Dynamics Feature Formulation

For joint $j \in \{0, \dots, 132\}$ at frame $t \in \{0, \dots, 31\}$:
1. **Normalized Position ($P_t$):**
   $$P_t = \text{PartNormalize}(\text{Keypoint}_{t, j})$$
2. **Velocity ($V_t$):**
   $$V_t = P_t - P_{t-1} \quad (V_0 = 0)$$
3. **Acceleration ($A_t$):**
   $$A_t = V_t - V_{t-1} \quad (A_0 = A_1 = 0)$$

The resulting $[32, 133, 6]$ tensor encapsulates both spatial skeletal posture and temporal signing motion velocity and acceleration.
