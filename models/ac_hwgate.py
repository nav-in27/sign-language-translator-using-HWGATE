"""
Adaptive Context-Aware Hierarchical Windowed Graph Attention Transformer Encoder (AC-HWGATE).
Production Model Architecture for Indian Sign Language (ISL) Translation.
"""
import os
import sys

# Ensure training/models can be imported directly or re-exported
TRAINING_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "training", "models")
if TRAINING_MODELS_DIR not in sys.path:
    sys.path.insert(0, TRAINING_MODELS_DIR)

try:
    from training.models.ac_hwgate import ACHWGATE, PreLNWindowedAttention, PreLNTransformerBlock, AC_HWGATE
except ImportError:
    from ac_hwgate import ACHWGATE, PreLNWindowedAttention, PreLNTransformerBlock, AC_HWGATE

__all__ = ["ACHWGATE", "PreLNWindowedAttention", "PreLNTransformerBlock", "AC_HWGATE"]
