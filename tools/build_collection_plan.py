"""Rank additional independent recordings needed for 136-class generalization."""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "reports" / "all_136_class_audit.json"
VALIDATION = ROOT / "reports" / "reproduce_lr5e4_val.json"
JSON_REPORT = ROOT / "reports" / "data_collection_plan_136.json"
CSV_REPORT = ROOT / "reports" / "data_collection_plan_136.csv"


def main():
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    validation = json.loads(VALIDATION.read_text(encoding="utf-8"))
    metrics = {row["class"]: row for row in validation["per_class"]}
    ranked = []
    for row in audit["rows"]:
        name = row["class"]
        metric = metrics[name]
        groups = row["train_recording_groups"]
        support = row["val_count"]
        confusion_count = sum(item["count"] for item in row["test_confusions"])
        uncertainty = 1.0 if support <= 2 else 0.5 if support <= 3 else 0.0
        score = (1.0 - metric["recall"]) + (1.0 - metric["f1"]) + min(1.0, 3.0 / max(groups, 1)) + uncertainty + min(1.0, confusion_count / 3.0)
        reasons = []
        if metric["recall"] < 0.5:
            reasons.append("low validation recall")
        if groups < 10:
            reasons.append("fewer than 10 independent training groups")
        if confusion_count:
            reasons.append("observed confusion errors")
        if support <= 2:
            reasons.append("small validation sample")
        ranked.append({
            "class": name,
            "priority_score": round(score, 4),
            "train_samples": row["train_count"],
            "train_groups": groups,
            "validation_samples": support,
            "validation_recall": metric["recall"],
            "validation_f1": metric["f1"],
            "test_groups": row["test_recording_groups"],
            "known_test_confusions": row["test_confusions"],
            "reasons": reasons,
            "recommended_new_groups": max(0, 10 - groups),
        })
    ranked.sort(key=lambda item: (-item["priority_score"], item["class"]))
    report = {
        "protocol": "Prioritize independently recorded sessions; never count tilt/crop augmentations as independent groups.",
        "recommended_trial_design": {
            "minimum_repetitions_per_new_group": 3,
            "recommended_signers": "At least 3 distinct signers where available",
            "capture_rules": ["one intended label entered before each trial", "new camera session per group", "preserve failed and rejected trials", "keep evaluation recordings outside training until protocol is finalized"],
        },
        "rows": ranked,
    }
    JSON_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    fields = [key for key, value in ranked[0].items() if not isinstance(value, list)]
    with CSV_REPORT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: value for key, value in row.items() if key in fields} for row in ranked)
    print(json.dumps({"classes": len(ranked), "top_priorities": ranked[:15], "json": str(JSON_REPORT), "csv": str(CSV_REPORT)}))


if __name__ == "__main__":
    main()
