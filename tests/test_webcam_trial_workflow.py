import importlib.util
import json


def load_trial_tool():
    spec = importlib.util.spec_from_file_location("capture_webcam_trials", "tools/capture_webcam_trials.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_empty_trial_evaluation_is_explicit(tmp_path):
    tool = load_trial_tool()
    csv_report = tmp_path / "trials.csv"
    json_report = tmp_path / "trials.json"
    tool.evaluate(tmp_path, csv_report, json_report)
    summary = json.loads(json_report.read_text(encoding="utf-8"))
    assert summary["sample_count"] == 0
    assert summary["top1_accuracy_all_trials"] is None
    assert summary["confirmation_latency_p50_ms"] is None
    assert summary["per_class"] == {}
    assert csv_report.exists()
