"""Regression checks for the authoritative AC-HWGATE production contract."""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from backend.inference import isl_model


ROOT = Path(__file__).resolve().parents[1]


def test_authoritative_manifest_is_the_runtime_mapping():
    manifest = json.loads((ROOT / "configs" / "authoritative_136_classes_manifest.json").read_text())
    classes = manifest["classes"]
    assert isl_model.num_classes == len(classes) == 136
    assert [isl_model.reverse_map[i] for i in range(len(classes))] == classes


def test_onnx_contract_matches_runtime_mapping():
    session = ort.InferenceSession(
        str(ROOT / "backend" / "models" / "ac_hwgate.onnx"),
        providers=["CPUExecutionProvider"],
    )
    assert session.get_inputs()[0].shape[1:] == [32, 133, 6]
    assert session.get_outputs()[0].shape == ["batch_size", 136]
    assert session.get_outputs()[1].shape == ["batch_size", 128]


def test_critical_held_out_classes_do_not_collapse_to_common_words():
    split = np.load(ROOT / "data" / "unified_136class" / "test_split.npz", allow_pickle=True)
    names = split["class_names"].astype(str)
    results = {}
    for class_name in ("thank_you", "good_morning", "good", "morning"):
        indices = np.flatnonzero(names == class_name)
        assert len(indices) > 0, f"missing held-out examples for {class_name}"
        predictions = [isl_model.predict_sequence_features(split["features"][i]).label for i in indices]
        results[class_name] = predictions
        if class_name in {"thank_you", "good_morning"}:
            assert sum(pred == class_name.upper() for pred in predictions) == len(predictions)
    assert len(set(results["thank_you"] + results["good_morning"])) == 2
    assert any(pred != "GOOD" for pred in results["morning"] + results["good"])
