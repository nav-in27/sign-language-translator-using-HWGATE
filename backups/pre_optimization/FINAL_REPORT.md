# FINAL PRODUCTION REPORT — AC-HWGATE 40-CLASS ISL SYSTEM

## 1. Project & Architecture Overview
- **Dataset**: `vidit031/isl-isolated-40words` on Hugging Face (40 classes)
- **Number of Videos**: 642 total valid clips
- **Number of Signers**: 172 signers
- **Train Videos**: 430 clips (~70%)
- **Validation Videos**: 105 clips (~15%)
- **Test Videos**: 107 clips (~15%, completely untouched)
- **Model**: **AC-HWGATE** (Adaptive Context-Aware Hierarchical Windowed Graph Attention Transformer Encoder)
- **Parameters**: 972,191 parameters (3.71 MB)

## 2. Hardware & Training Efficiency
- **GPU Used**: NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM)
- **VRAM Used**: 27.5 MB allocated / ~1.4 GB reserved peak (Well under 3.5 GB safety threshold)
- **Training Time**: 2.3 minutes (45 epochs with early stopping & cosine annealing)
- **Precision**: PyTorch AMP (FP16 automatic mixed precision)
- **Effective Batch Size**: 32 (Physical batch size 8 × Gradient accumulation 4)

## 3. Honest Measured Accuracy & Benchmark Results
| Metric | Result |
| :--- | :--- |
| **Final Test Top-1 Accuracy** | **63.55%** |
| **Final Test Top-3 Accuracy** | **85.98%** |
| **Best Validation Accuracy** | **77.14%** |
| **Macro F1 Score** | **56.61%** |
| **Weighted F1 Score** | **62.07%** |
| **Macro Precision** | **59.32%** |
| **Macro Recall** | **58.80%** |
| **Unknown Rejection Accuracy** | **100.0%** (0 false positives on idle, zero-hand & jitter tests) |
| **Average Inference Latency** | **0.58 ms** per sequence |
| **FPS Throughput** | **1,719.6 FPS** (GPU Tensor Core accelerated) |

## 4. Top Performing Classes
| Class | Test Accuracy |
| :--- | :--- |
| father | 100.0% |
| teacher | 100.0% |
| school | 100.0% |
| okay | 100.0% |
| hospital | 100.0% |
| help | 100.0% |
| hello | 100.0% |
| tea | 100.0% |

## 5. Artifacts & Deliverables
- Best PyTorch Checkpoint: `checkpoints/achwgate_best.pth`
- Production ONNX Models: `models/achwgate.onnx` & `backend/models/ac_hwgate.onnx`
- Class Prototype Vectors: `models/class_prototypes.npy`
- Confusion Matrix Chart: `reports/confusion_matrix.png`
- Per-Class Metrics CSV: `reports/per_class_metrics.csv`
- Acceptance Test Suite: `tests/test_end_to_end.py` (10/10 Passed)
- Live Demo Application: `python demo/live_demo.py`
- Personal Recording Tool: `python tools/record_personal_samples.py`
- Personal Fine-Tuning: `python tools/fine_tune_personal.py`
