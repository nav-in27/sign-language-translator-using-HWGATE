import numpy as np
import json
import sys
sys.path.insert(0, "backend")
from inference import isl_model

data = np.load("data/unified/test_split.npz")
feats = data["features"] # [563, 32, 133, 6]
labels = data["labels"]

confs = []
sims = []
correct = 0

for i in range(len(feats)):
    f = feats[i]
    pred_label, c, unk = isl_model.predict_sequence_features(f)
    pred_idx = list(isl_model.label_map.keys())[list(isl_model.label_map.values()).index(labels[i])] if labels[i] in isl_model.label_map.values() else -1
    
    # Raw ONNX run
    assert isl_model.ort_session is not None, "ort_session is not loaded"
    inp_name = isl_model.ort_session.get_inputs()[0].name
    outs = isl_model.ort_session.run(None, {inp_name: np.expand_dims(f, 0)})
    logits = np.asarray(outs[0])[0]
    exp_l = np.exp(logits - np.max(logits))
    probs = exp_l / np.sum(exp_l)
    top_p = float(np.max(probs))
    pred_k = int(np.argmax(probs))
    confs.append(top_p)
    
    if len(outs) > 1 and isl_model.prototypes is not None:
        emb = np.asarray(outs[1])[0]
        emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
        sim = float(np.dot(emb_norm, isl_model.prototypes[pred_k]))
        sims.append(sim)

print("Confidence distribution on held-out test set:")
print(f"  Min: {np.min(confs):.4f} | 10th percentile: {np.percentile(confs, 10):.4f} | Median: {np.median(confs):.4f} | Mean: {np.mean(confs):.4f}")
print("Prototype similarity distribution on held-out test set:")
print(f"  Min: {np.min(sims):.4f} | 10th percentile: {np.percentile(sims, 10):.4f} | Median: {np.median(sims):.4f} | Mean: {np.mean(sims):.4f}")
