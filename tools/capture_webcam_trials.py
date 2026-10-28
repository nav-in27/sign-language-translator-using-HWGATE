"""Capture and evaluate independently labelled physical-webcam trials.

Capture mode asks for the intended label before recording and stores raw
landmarks, confidences, timestamps, and trial metadata. Evaluation mode feeds
only landmarks/confidences to the live engine; the intended label is read after
prediction and is never passed to the model.
"""

import argparse
import csv
import sys
import json
import re
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TRIAL_DIR = ROOT / "data" / "webcam_trials"
REPORT = ROOT / "reports" / "webcam_trials_evaluation.csv"
SUMMARY_REPORT = ROOT / "reports" / "webcam_trials_evaluation.json"

from backend.inference import ISLInferenceEngine


def extract_frame(results):
    points = []
    confidences = []

    def append_landmarks(items, default_confidence):
        if items:
            for landmark in items.landmark:
                points.append([landmark.x, landmark.y, landmark.z])
                confidences.append([getattr(landmark, "visibility", default_confidence)])
        else:
            count = 33 if len(points) == 0 else 21 if len(points) in (33, 54) else 58
            points.extend([[0.0, 0.0, 0.0]] * count)
            confidences.extend([[0.0]] * count)

    append_landmarks(results.pose_landmarks, 1.0)
    append_landmarks(results.left_hand_landmarks, 1.0)
    append_landmarks(results.right_hand_landmarks, 1.0)
    append_landmarks(results.face_landmarks, 1.0)
    return np.asarray(points[:133], dtype=np.float32), np.asarray(confidences[:133], dtype=np.float32)


def next_trial_id():
    existing = []
    for path in TRIAL_DIR.glob("trial_*.npz"):
        match = re.match(r"trial_(\d+)_", path.name)
        if match:
            existing.append(int(match.group(1)))
    return max(existing, default=0) + 1


def capture(labels, trials_per_label, camera_index):
    TRIAL_DIR.mkdir(parents=True, exist_ok=True)
    video_dir = TRIAL_DIR / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open webcam {camera_index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    try:
        trial_id = next_trial_id() - 1
        for label in labels:
            for repetition in range(1, trials_per_label + 1):
                input(f"Label recorded as '{label}'. Press Enter, then perform trial {repetition}...")
                keypoints, confidences, timestamps = [], [], []
                trial_id += 1
                normalized_label = label.lower().replace(" ", "_")
                video_path = video_dir / f"trial_{trial_id:04d}_{normalized_label}.mp4"
                writer = cv2.VideoWriter(
                    str(video_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    30.0,
                    (640, 480),
                )
                start = time.time()
                try:
                    while time.time() - start < 2.0:
                        ok, frame = cap.read()
                        if not ok:
                            continue
                        writer.write(frame)
                        result = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        kp, conf = extract_frame(result)
                        keypoints.append(kp)
                        confidences.append(conf)
                        timestamps.append(time.time())
                        cv2.imshow("AC-HWGATE labelled trial (press q to abort)", frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            return
                finally:
                    writer.release()
                out = TRIAL_DIR / f"trial_{trial_id:04d}_{normalized_label}.npz"
                np.savez_compressed(
                    out,
                    intended_label=normalized_label,
                    trial_id=f"trial_{trial_id:04d}",
                    capture_timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    keypoints=np.asarray(keypoints, dtype=np.float32),
                    confidences=np.asarray(confidences, dtype=np.float32),
                    timestamps=np.asarray(timestamps, dtype=np.float64),
                    video_path=str(video_path),
                )
                print(f"Saved {out} ({len(keypoints)} frames, video={video_path})")
    finally:
        cap.release()
        holistic.close()
        cv2.destroyAllWindows()


def _metric_report(rows):
    labels = sorted({row["intended_label"] for row in rows})
    per_class = {}
    for label in labels:
        tp = sum(row["predicted_label"] == label and row["intended_label"] == label for row in rows)
        fp = sum(row["predicted_label"] == label and row["intended_label"] != label for row in rows)
        fn = sum(row["intended_label"] == label and row["predicted_label"] != label for row in rows)
        support = sum(row["intended_label"] == label for row in rows)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    latencies = [row["confirmation_latency_ms"] for row in rows if row["confirmation_latency_ms"] != ""]
    accepted = sum(row["accepted"] for row in rows)
    rejected = sum(not row["accepted"] for row in rows)
    correct = sum(row["correct"] for row in rows)
    macro_f1 = sum(item["f1"] for item in per_class.values()) / len(per_class) if per_class else 0.0
    return {
        "sample_count": len(rows),
        "accepted_count": accepted,
        "rejected_or_no_confirmation_count": rejected,
        "correct_count": correct,
        "top1_accuracy_all_trials": correct / len(rows) if rows else None,
        "accepted_accuracy": correct / accepted if accepted else None,
        "macro_f1": macro_f1 if rows else None,
        "rejection_rate": rejected / len(rows) if rows else None,
        "missed_detection_rate": sum(row["status"] == "no_confirmation" for row in rows) / len(rows) if rows else None,
        "latency_sample_count": len(latencies),
        "confirmation_latency_p50_ms": float(np.percentile(latencies, 50)) if latencies else None,
        "confirmation_latency_p95_ms": float(np.percentile(latencies, 95)) if latencies else None,
        "per_class": per_class,
        "insufficient_trial_classes": [label for label, item in per_class.items() if item["support"] < 3],
    }


def evaluate(trial_dir=TRIAL_DIR, report=REPORT, summary_report=SUMMARY_REPORT):
    rows = []
    for path in sorted(Path(trial_dir).glob("trial_*.npz")):
        data = np.load(path, allow_pickle=True)
        engine = ISLInferenceEngine()
        intended = str(data["intended_label"].item())
        keypoints = data["keypoints"]
        confidences = data["confidences"]
        timestamps = data["timestamps"]
        confirmed = None
        top_predictions = []
        for index, (kp, conf) in enumerate(zip(keypoints, confidences), start=1):
            result = engine.predict(kp.tolist(), conf.tolist(), seq_id=index, timestamp=float(timestamps[index - 1]))
            top_predictions = result.top_predictions or top_predictions
            if result.state == "CONFIRMED" and confirmed is None:
                confirmed = result
                confirmation_timestamp = float(timestamps[index - 1])
        predicted = confirmed.confirmed_sign.lower() if confirmed else ""
        rows.append({
            "trial_id": str(data["trial_id"].item()),
            "intended_label": intended,
            "predicted_label": predicted,
            "accepted": bool(confirmed),
            "correct": bool(confirmed and predicted == intended),
            "confidence": float(confirmed.confidence) if confirmed else 0.0,
            "top_predictions": json.dumps(top_predictions),
            "translation": confirmed.translation if confirmed else "",
            "confirmation_latency_ms": (confirmation_timestamp - float(timestamps[0])) * 1000.0 if confirmed and len(timestamps) else "",
            "status": "accepted" if confirmed else "no_confirmation",
        })
    report = Path(report)
    summary_report = Path(summary_report)
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["trial_id"])
        writer.writeheader()
        writer.writerows(rows)
    summary = _metric_report(rows)
    summary["csv_report"] = str(report)
    summary_report.parent.mkdir(parents=True, exist_ok=True)
    summary_report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({**summary, "report": str(report), "summary_report": str(summary_report)}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--labels", nargs="+", default=[])
    parser.add_argument("--trials-per-label", type=int, default=3)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    if args.capture:
        capture(args.labels, args.trials_per_label, args.camera)
    if args.evaluate:
        evaluate()
    if not args.capture and not args.evaluate:
        parser.error("choose --capture and/or --evaluate")


if __name__ == "__main__":
    main()
