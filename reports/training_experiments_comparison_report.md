# AC-HWGATE 50-CLASS CONTROLLED TRAINING EXPERIMENTS COMPARISON

- **Evaluation Protocol:** Strictly validation-driven model selection on disjoint recording sessions.
- **Hardware Acceleration:** NVIDIA GeForce RTX 3050 Laptop GPU (PyTorch AMP FP16).
- **Untouched Test Split:** 594 sequences across independent signers.

| Experiment ID | Architecture & Enhancements | Val Accuracy | Val Macro F1 | GPU Latency | FPS | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **EXP_A** | Baseline (Standard CrossEntropy, Basic Jitter) | 88.85% | 91.66% | 5.36 ms | 186.7 | **SELECTED SOTA** |
| **EXP_B** | Class-Balanced Training (Weighted Focal Loss, Smoothing 0.03) | 85.74% | 89.92% | 4.49 ms | 222.7 | Evaluated |
| **EXP_C** | Physics-Consistent Kinematic Augmentation (Continuous Pos+Vel+Acc) | 76.23% | 82.65% | 3.76 ms | 265.7 | Evaluated |
| **EXP_D** | Warmup Cosine Scheduler + Regularization (Weight Decay 3e-4) | 80.99% | 86.62% | 3.46 ms | 288.8 | Evaluated |
| **EXP_E** | Proposed Production SOTA (Best Combined Formulation) | 84.28% | 88.64% | 6.75 ms | 148.0 | Evaluated |

## Winning Model Untouched Test Evaluation

- **Selected Checkpoint:** `c:\Users\navee\.gemini\antigravity-ide\scratch\isl-translator\checkpoints\exp_50class_exp_a_best.pth`
- **Top-1 Test Accuracy:** **77.61%**
- **Test Macro F1 Score:** **85.90%**
- **Test Weighted F1:** **77.76%**
- **Test Macro Precision:** **86.81%**
- **Test Macro Recall:** **87.23%**
- **Inference Speed:** **5.36 ms / sequence** (186.7 FPS)
