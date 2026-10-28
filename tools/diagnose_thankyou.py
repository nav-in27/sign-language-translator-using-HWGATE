import numpy as np
import json
import torch
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from backend.inference import isl_model
from backend.translator import nmt_engine

def diagnose():
    print("="*70)
    print("DIAGNOSTIC TRACE FOR THANK_YOU & GOOD_MORNING")
    print("="*70)
    
    with open('configs/authoritative_136_classes_manifest.json') as f:
        manifest = json.load(f)
    classes = manifest['classes']
    
    ty_idx = classes.index('thank_you')
    gm_idx = classes.index('good_morning')
    print(f"[+] 'thank_you' class index: {ty_idx}")
    print(f"[+] 'good_morning' class index: {gm_idx}")
    
    data = np.load('data/unified_136class/test_split.npz')
    labels = data['labels']
    feats = data['features']
    
    ty_indices = np.where(labels == ty_idx)[0]
    gm_indices = np.where(labels == gm_idx)[0]
    print(f"[+] Test split count for 'thank_you': {len(ty_indices)}")
    print(f"[+] Test split count for 'good_morning': {len(gm_indices)}")
    
    print("\n--- Evaluating each 'thank_you' held-out test sample on ONNX ---")
    correct_ty = 0
    for i, idx in enumerate(ty_indices):
        feat = feats[idx] # (32, 133, 6)
        out = isl_model.ort_session.run(None, {'keypoints': feat[None, ...].astype(np.float32)})[0]
        logits = out[0]
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        top5_idx = np.argsort(logits)[-5:][::-1]
        top1_idx = top5_idx[0]
        pred_label = classes[top1_idx]
        is_corr = (top1_idx == ty_idx)
        if is_corr:
            correct_ty += 1
            
        print(f"\nSample {i+1} (idx {idx}):")
        print(f"  Top-1: '{pred_label}' (conf: {probs[top1_idx]:.4f}, logit: {logits[top1_idx]:.2f}) -> {'CORRECT' if is_corr else 'WRONG'}")
        print("  Top-5:")
        for rank, c_idx in enumerate(top5_idx):
            print(f"    #{rank+1}: {classes[c_idx]} (idx {c_idx}, prob {probs[c_idx]:.4f}, logit {logits[c_idx]:.2f})")
            
        # Also test translation
        trans = nmt_engine.translate_gloss_to_english(pred_label)
        print(f"  Translation: '{trans}'")
        
    print(f"\nOverall 'thank_you' Accuracy on test split: {correct_ty}/{len(ty_indices)} ({correct_ty/max(1, len(ty_indices))*100:.1f}%)")

    print("\n--- Evaluating each 'good_morning' held-out test sample on ONNX ---")
    correct_gm = 0
    for i, idx in enumerate(gm_indices):
        feat = feats[idx]
        out = isl_model.ort_session.run(None, {'keypoints': feat[None, ...].astype(np.float32)})[0]
        logits = out[0]
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        top5_idx = np.argsort(logits)[-5:][::-1]
        top1_idx = top5_idx[0]
        pred_label = classes[top1_idx]
        is_corr = (top1_idx == gm_idx)
        if is_corr:
            correct_gm += 1
            
        print(f"\nSample {i+1} (idx {idx}):")
        print(f"  Top-1: '{pred_label}' (conf: {probs[top1_idx]:.4f}, logit: {logits[top1_idx]:.2f}) -> {'CORRECT' if is_corr else 'WRONG'}")
        trans = nmt_engine.translate_gloss_to_english(pred_label)
        print(f"  Translation: '{trans}'")

    print(f"\nOverall 'good_morning' Accuracy on test split: {correct_gm}/{len(gm_indices)} ({correct_gm/max(1, len(gm_indices))*100:.1f}%)")

if __name__ == "__main__":
    diagnose()
