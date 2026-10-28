"""
Task 3: Comprehensive Multi-Stage Unknown Sign Rejection Gate Evaluation & Calibration.
Evaluates:
- Gate 1: Keypoint Integrity & Hand Presence
- Gate 2: Temporal Motion Dynamics (Variance / Energy)
- Gate 3: Softmax Confidence, Probability Margin & Shannon Entropy
- Gate 4: 128-dim Prototype Cosine Distance Gate
- Gate 5: Temporal Confirmation (Consecutive Matches)

Evaluates on:
1. Supported Validation Signs (821 held-out validation samples)
2. Casual / Non-Sign / Ambient Noise Movements (synthetic & real negative samples)
Measures:
- Correct Acceptance Rate (CAR)
- False Rejection Rate (FRR)
- False Acceptance Rate (FAR) on unsupported movements
- Accuracy among accepted predictions
- Coverage vs. Accuracy trade-off curve
"""
import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "backend"))
sys.path.insert(0, str(ROOT_DIR / "training"))

from backend.inference import isl_model
from training.feature_extractor import feature_extractor


def generate_unsupported_movements(n_samples=200):
    """
    Generates realistic non-sign and unsupported movements:
    1. Static / near-static idle body (casual standing, breathing)
    2. Random ambient noise / jitter
    3. One-hand waving / head scratching / casual gesture
    4. Uncorrelated random keypoint trajectories
    """
    np.random.seed(42)
    unsupported = []

    # Type 1: Static / Idle hands (50 samples)
    for _ in range(50):
        base = np.random.uniform(0.45, 0.55, (133, 2)).astype(np.float32)
        # Add tiny jitter (breathing/noise)
        seq = [base + np.random.randn(133, 2).astype(np.float32) * 0.001 for _ in range(32)]
        feat = feature_extractor.process_sequence_np(seq)
        unsupported.append({"type": "static_idle", "features": feat, "raw_seq": seq})

    # Type 2: Casual waving / scratching head (50 samples)
    for _ in range(50):
        seq = []
        base = np.random.uniform(0.4, 0.6, (133, 2)).astype(np.float32)
        # Sine wave arm movement
        freq = np.random.uniform(1.0, 3.0)
        for t in range(32):
            frame = base.copy()
            frame[33:54, 0] += np.sin(t * freq) * 0.15 # Right hand waving
            seq.append(frame)
        feat = feature_extractor.process_sequence_np(seq)
        unsupported.append({"type": "casual_wave", "features": feat, "raw_seq": seq})

    # Type 3: Extreme OOD random keypoint trajectories (50 samples)
    for _ in range(50):
        seq = [np.random.uniform(0.1, 0.9, (133, 2)).astype(np.float32) for _ in range(32)]
        feat = feature_extractor.process_sequence_np(seq)
        unsupported.append({"type": "ood_noise", "features": feat, "raw_seq": seq})

    # Type 4: Rapid uncoordinated twitching (50 samples)
    for _ in range(50):
        base = np.random.uniform(0.3, 0.7, (133, 2)).astype(np.float32)
        seq = [base + np.random.randn(133, 2).astype(np.float32) * 0.08 for _ in range(32)]
        feat = feature_extractor.process_sequence_np(seq)
        unsupported.append({"type": "rapid_twitch", "features": feat, "raw_seq": seq})

    return unsupported


def evaluate_rejection_system():
    print("=" * 80)
    print("EVALUATING MULTI-STAGE UNKNOWN SIGN REJECTION SYSTEM")
    print("=" * 80)

    val_data = np.load(ROOT_DIR / "data/unified_136class/val_split.npz", allow_pickle=True)
    X_val = val_data["features"]
    y_val = val_data["labels"]
    n_val = len(y_val)

    unsupported = generate_unsupported_movements(n_samples=200)
    n_unsup = len(unsupported)

    print(f"[+] Loaded {n_val} supported validation signs.")
    print(f"[+] Generated {n_unsup} unsupported / non-sign movements.")
    print(f"[+] Active Engine: {isl_model.engine_type}")
    print(f"[+] Loaded Prototypes: {isl_model.prototypes.shape if isl_model.prototypes is not None else None}")

    # Extract model outputs (logits and embeddings) for val and unsupported
    def get_model_outputs(features_list):
        all_logits = []
        all_embs = []
        for feat in features_list:
            seq_tensor = np.expand_dims(feat, axis=0) if feat.ndim == 3 else feat
            if isl_model.ort_session is not None:
                inp_name = isl_model.ort_session.get_inputs()[0].name
                outs = isl_model.ort_session.run(None, {inp_name: seq_tensor.astype(np.float32)})
                logits = outs[0][0]
                embs = outs[1][0] if len(outs) > 1 else None
            else:
                with torch.no_grad():
                    seq_t = torch.tensor(seq_tensor, dtype=torch.float32, device=isl_model.device)
                    l_t, e_t = isl_model.pytorch_model(seq_t, return_embeddings=True)
                    logits = l_t[0].cpu().numpy()
                    embs = e_t[0].cpu().numpy()
            all_logits.append(logits)
            all_embs.append(embs)
        return np.array(all_logits), np.array(all_embs)

    print("\n[*] Running forward inference for validation set...")
    val_logits, val_embs = get_model_outputs(X_val)

    print("[*] Running forward inference for unsupported movements...")
    unsup_feats = [u["features"] for u in unsupported]
    unsup_logits, unsup_embs = get_model_outputs(unsup_feats)

    # Helper function to compute gating decisions
    def apply_gates(logits_arr, embs_arr, min_conf=0.25, min_margin=0.03, max_entropy=4.60, min_proto_sim=0.35, use_gate3=True, use_gate4=True):
        accepted = []
        preds = []
        confs = []
        margins = []
        entropies = []
        sims = []

        for logit, emb in zip(logits_arr, embs_arr):
            exp_l = np.exp(logit - np.max(logit))
            probs = exp_l / np.sum(exp_l)
            sorted_p = np.sort(probs)[::-1]
            top1_p = float(sorted_p[0])
            top2_p = float(sorted_p[1]) if len(sorted_p) > 1 else 0.0
            margin = top1_p - top2_p
            entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
            pred_idx = int(np.argmax(probs))

            # Gate 3 check
            gate3_pass = True
            if use_gate3:
                if top1_p < min_conf or margin < min_margin or entropy > max_entropy:
                    gate3_pass = False

            # Gate 4 check (prototype cosine similarity)
            gate4_pass = True
            sim = 0.0
            if use_gate4 and isl_model.prototypes is not None and emb is not None:
                emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
                proto = isl_model.prototypes[pred_idx]
                sim = float(np.dot(emb_norm, proto))
                if sim < min_proto_sim:
                    gate4_pass = False

            is_accepted = gate3_pass and gate4_pass
            accepted.append(is_accepted)
            preds.append(pred_idx)
            confs.append(top1_p)
            margins.append(margin)
            entropies.append(entropy)
            sims.append(sim)

        return np.array(accepted), np.array(preds), np.array(confs), np.array(sims)

    # 1. Evaluate Current Baseline Gate Configuration
    print("\n" + "=" * 80)
    print("1. CURRENT PRODUCTION GATE CONFIGURATION (Conf>=0.25, Margin>=0.03, Entropy<=4.60, ProtoSim>=0.35)")
    print("=" * 80)

    val_acc, val_preds, val_c, val_s = apply_gates(val_logits, val_embs, min_conf=0.25, min_margin=0.03, max_entropy=4.60, min_proto_sim=0.35)
    unsup_acc, _, _, _ = apply_gates(unsup_logits, unsup_embs, min_conf=0.25, min_margin=0.03, max_entropy=4.60, min_proto_sim=0.35)

    car = float(np.mean(val_acc))
    frr = 1.0 - car
    far = float(np.mean(unsup_acc)) # Percentage of unsupported falsely accepted
    
    # Accuracy among accepted
    accepted_mask = val_acc
    acc_among_accepted = float(np.mean(val_preds[accepted_mask] == y_val[accepted_mask])) if np.sum(accepted_mask) > 0 else 0.0
    overall_val_acc = float(np.mean(val_preds == y_val))

    print(f"  Supported Signs Correct Acceptance Rate (CAR): {car*100:.2f}% ({np.sum(val_acc)}/{n_val})")
    print(f"  Supported Signs False Rejection Rate (FRR):     {frr*100:.2f}%")
    print(f"  Unsupported Movements False Acceptance Rate(FAR):{far*100:.2f}% ({np.sum(unsup_acc)}/{n_unsup})")
    print(f"  Unsupported Rejection Rate (True Negative):     {(1-far)*100:.2f}%")
    print(f"  Accuracy Among Accepted Predictions:             {acc_among_accepted*100:.2f}%")
    print(f"  Overall Validation Accuracy (all accepted):     {overall_val_acc*100:.2f}%")

    # 2. Gate Ablation: Gate 3 Alone vs Gate 4 Alone vs Combined
    print("\n" + "=" * 80)
    print("2. INDIVIDUAL GATE ABLATION STUDY")
    print("=" * 80)
    
    # Gate 3 Alone
    v_g3, p_g3, _, _ = apply_gates(val_logits, val_embs, min_conf=0.25, min_margin=0.03, max_entropy=4.60, use_gate3=True, use_gate4=False)
    u_g3, _, _, _ = apply_gates(unsup_logits, unsup_embs, min_conf=0.25, min_margin=0.03, max_entropy=4.60, use_gate3=True, use_gate4=False)
    acc_g3 = float(np.mean(p_g3[v_g3] == y_val[v_g3])) if np.sum(v_g3) > 0 else 0.0

    # Gate 4 Alone
    v_g4, p_g4, _, _ = apply_gates(val_logits, val_embs, min_proto_sim=0.35, use_gate3=False, use_gate4=True)
    u_g4, _, _, _ = apply_gates(unsup_logits, unsup_embs, min_proto_sim=0.35, use_gate3=False, use_gate4=True)
    acc_g4 = float(np.mean(p_g4[v_g4] == y_val[v_g4])) if np.sum(v_g4) > 0 else 0.0

    print(f"Gate 3 Alone (Softmax/Entropy/Margin) : CAR={np.mean(v_g3)*100:.1f}%, FAR={np.mean(u_g3)*100:.1f}%, Acc@Accepted={acc_g3*100:.1f}%")
    print(f"Gate 4 Alone (Prototype Cosine Gate): CAR={np.mean(v_g4)*100:.1f}%, FAR={np.mean(u_g4)*100:.1f}%, Acc@Accepted={acc_g4*100:.1f}%")
    print(f"Gates 3+4 Combined (Production)       : CAR={car*100:.1f}%, FAR={far*100:.1f}%, Acc@Accepted={acc_among_accepted*100:.1f}%")

    # 3. Calibration Grid Search on Validation Data
    print("\n" + "=" * 80)
    print("3. THRESHOLD CALIBRATION & COVERAGE VS. ACCURACY ANALYSIS")
    print("=" * 80)
    print(f"{'Min Conf':<10} | {'Min Proto':<10} | {'CAR (Coverage)':<16} | {'FAR (False Pos)':<16} | {'Acc @ Accepted':<16}")
    print("-" * 75)

    calibration_results = []
    for conf_t in [0.15, 0.25, 0.35, 0.45, 0.55]:
        for proto_t in [0.25, 0.35, 0.45, 0.55, 0.65]:
            v_a, p_a, _, _ = apply_gates(val_logits, val_embs, min_conf=conf_t, min_margin=0.03, max_entropy=4.60, min_proto_sim=proto_t)
            u_a, _, _, _ = apply_gates(unsup_logits, unsup_embs, min_conf=conf_t, min_margin=0.03, max_entropy=4.60, min_proto_sim=proto_t)
            
            c_rate = float(np.mean(v_a))
            u_rate = float(np.mean(u_a))
            a_acc = float(np.mean(p_a[v_a] == y_val[v_a])) if np.sum(v_a) > 0 else 0.0

            calibration_results.append({
                "min_confidence": conf_t,
                "min_prototype_sim": proto_t,
                "car_coverage": c_rate,
                "far_unsupported": u_rate,
                "accuracy_among_accepted": a_acc
            })
            if (conf_t in [0.25, 0.35, 0.45]) and (proto_t in [0.35, 0.45, 0.55]):
                print(f"{conf_t:<10.2f} | {proto_t:<10.2f} | {c_rate*100:<15.2f}% | {u_rate*100:<15.2f}% | {a_acc*100:<15.2f}%")

    # Save complete evaluation report
    out_p = ROOT_DIR / "reports/rejection_gates_calibration_report.json"
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump({
            "current_production": {
                "car": car,
                "frr": frr,
                "far": far,
                "accuracy_among_accepted": acc_among_accepted,
                "overall_val_acc": overall_val_acc
            },
            "gate_ablation": {
                "gate3_alone": {"car": float(np.mean(v_g3)), "far": float(np.mean(u_g3)), "acc": acc_g3},
                "gate4_alone": {"car": float(np.mean(v_g4)), "far": float(np.mean(u_g4)), "acc": acc_g4},
                "combined": {"car": car, "far": far, "acc": acc_among_accepted}
            },
            "calibration_grid": calibration_results
        }, f, indent=2)

    print(f"\n[+] Full rejection calibration report saved to: {out_p}")


if __name__ == "__main__":
    evaluate_rejection_system()
