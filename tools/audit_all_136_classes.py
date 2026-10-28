"""Produce one reproducible class-by-class audit for the clean 136-class data."""

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "unified_136class"
MANIFEST = ROOT / "configs" / "authoritative_136_classes_manifest.json"
BASELINE = ROOT / "reports" / "clean_unweighted_aug_test_final.json"
JSON_REPORT = ROOT / "reports" / "all_136_class_audit.json"
CSV_REPORT = ROOT / "reports" / "all_136_class_audit.csv"


def split_rows(split):
    data = np.load(DATA / f"{split}_split.npz", allow_pickle=True)
    names = np.asarray(data["class_names"]).astype(str)
    rows = {}
    for label, name in enumerate(json.loads(MANIFEST.read_text(encoding="utf-8"))["classes"]):
        mask = names == name
        features = data["features"][mask].astype(np.float32)
        motion = np.linalg.norm(features[:, :, 2:4], axis=3).mean(axis=(1, 2)) if len(features) else np.array([])
        rows[name] = {
            "count": int(mask.sum()),
            "recording_groups": int(len(set(map(str, data["recording_group_ids"][mask])))),
            "finite_count": int(np.isfinite(features).all(axis=(1, 2, 3)).sum()) if len(features) else 0,
            "zero_feature_count": int((np.abs(features).sum(axis=(1, 2, 3)) == 0).sum()) if len(features) else 0,
            "motion_min": float(motion.min()) if len(motion) else None,
            "motion_median": float(np.median(motion)) if len(motion) else None,
            "motion_max": float(motion.max()) if len(motion) else None,
        }
    return rows


def main():
    classes = json.loads(MANIFEST.read_text(encoding="utf-8"))["classes"]
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    per_class = {item["class"]: item for item in baseline["per_class"]}
    confusion = np.asarray(baseline["confusion_matrix"])
    rows = []
    split_data = {split: split_rows(split) for split in ("train", "val", "test")}
    for index, name in enumerate(classes):
        row = {"class_index": index, "class": name}
        for split in ("train", "val", "test"):
            for key, value in split_data[split][name].items():
                row[f"{split}_{key}"] = value
        row.update({f"test_{key}": value for key, value in per_class[name].items() if key != "class"})
        mistakes = []
        for target, count in enumerate(confusion[index]):
            if target != index and count:
                mistakes.append({"predicted_class": classes[target], "count": int(count)})
        row["test_confusions"] = sorted(mistakes, key=lambda item: -item["count"])
        rows.append(row)
    report = {
        "manifest_class_count": len(classes),
        "classes_present_in_manifest": len(rows),
        "baseline_report": str(BASELINE),
        "baseline_metrics": {key: baseline.get(key) for key in ("top1_accuracy", "top5_accuracy", "macro_f1", "weighted_f1", "macro_recall")},
        "classes_with_less_than_three_test_groups": [row["class"] for row in rows if row["test_recording_groups"] < 3],
        "classes_with_invalid_landmarks": [row["class"] for row in rows if row["test_finite_count"] != row["test_count"] or row["test_zero_feature_count"]],
        "rows": rows,
    }
    JSON_REPORT.parent.mkdir(parents=True, exist_ok=True)
    JSON_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    fieldnames = [key for key, value in rows[0].items() if not isinstance(value, list)]
    with CSV_REPORT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: value for key, value in row.items() if key in fieldnames} for row in rows)
    print(json.dumps({"classes": len(rows), "weak_test_group_classes": len(report["classes_with_less_than_three_test_groups"]), "invalid_classes": len(report["classes_with_invalid_landmarks"]), "json": str(JSON_REPORT), "csv": str(CSV_REPORT)}))


if __name__ == "__main__":
    main()
