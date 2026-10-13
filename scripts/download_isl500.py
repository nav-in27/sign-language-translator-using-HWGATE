"""
ISL500 Batch Downloader + Converter (High-Speed Multi-Threaded version)
Downloads ~4,062 H5 files, converts (T,2,21,3) hand landmarks to (32,133,6).
Uses ThreadPoolExecutor for 10x-20x faster downloads.
"""
import urllib.request
import urllib.parse
import json
import os
import sys
import csv
import time
import hashlib
import io
import numpy as np
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Force unbuffered output
if hasattr(sys.stdout, 'reconfigure'):
    getattr(sys.stdout, 'reconfigure')(line_buffering=True)

BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"
DATA_DIR = BASE_DIR / "data" / "isl500_landmarks"
DATA_DIR.mkdir(parents=True, exist_ok=True)

import h5py

ISL500_TO_CANONICAL = {
    "Drink": "drink", "Eat": "eat", "Father": "father", "Food": "food",
    "Friend": "friend", "Go": "go", "He": "he", "Hello": "hello",
    "Help": "help", "House": "home", "Market": "market", "I": "me",
    "Myself": "me", "Mother": "mother", "No": "no", "Please": "please",
    "School": "school", "She": "she", "Sister": "sister", "Sit": "sit",
    "Student": "student", "Tea": "tea", "Teacher": "teacher",
    "Thankyou": "thank_you", "Today": "today", "Water": "water",
    "What": "what", "Where": "where", "Yes": "yes", "You": "you",
}


def download_h5(filepath, max_retries=3):
    url = f"https://huggingface.co/datasets/ISL500/ISL-DATA/resolve/main/{filepath}"
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=25) as resp:
                return resp.read()
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(1.0 * (attempt + 1))
            else:
                raise


def convert_h5_to_features(h5_bytes):
    """Convert ISL500 (T, 2, 21, 3) -> our (32, 133, 6)."""
    with h5py.File(io.BytesIO(h5_bytes), "r") as f:
        data = np.array(f["intermediate"])  # (T, 2, 21, 3)

    T = data.shape[0]
    if T < 5:
        return None

    # Build 133-landmark: pose(33) + left(21) + right(21) + face(58)
    landmarks = np.zeros((T, 133, 3), dtype=np.float32)
    landmarks[:, 33:54, :] = data[:, 0, :, :3]  # Left hand
    landmarks[:, 54:75, :] = data[:, 1, :, :3]  # Right hand

    # Resample to 32 frames
    indices = np.linspace(0, T - 1, 32).astype(int)
    resampled = landmarks[indices]  # (32, 133, 3)

    # Velocity
    velocity = np.zeros_like(resampled)
    velocity[1:] = resampled[1:] - resampled[:-1]

    return np.concatenate([resampled, velocity], axis=-1).astype(np.float32)


def process_single_item(rec):
    canonical = ISL500_TO_CANONICAL.get(rec["word_raw"])
    if not canonical:
        return None

    cls_dir = DATA_DIR / canonical
    cls_dir.mkdir(parents=True, exist_ok=True)

    fp = rec["filepath"]
    user = rec["user"]
    sess = rec.get("session", "0")
    clip = rec.get("clip", "0")
    out_name = f"{canonical}__{user}__s{sess}__c{clip}.npz"
    out_path = cls_dir / out_name

    if out_path.exists():
        return {
            "sample_id": out_name,
            "class": canonical,
            "source": "ISL500",
            "signer": user,
            "status": "cached"
        }

    try:
        h5_bytes = download_h5(fp)
        features = convert_h5_to_features(h5_bytes)

        if features is not None and features.shape == (32, 133, 6):
            sha = hashlib.sha256(features.tobytes()).hexdigest()[:16]
            np.savez_compressed(
                out_path,
                features=features,
                word=canonical,
                source="ISL500",
                signer=user,
                session=sess,
                hash=sha
            )
            return {
                "sample_id": out_name,
                "class": canonical,
                "source": "ISL500",
                "signer": user,
                "hash": sha,
                "status": "ok"
            }
        else:
            return {
                "sample_id": out_name,
                "class": canonical,
                "source": "ISL500",
                "signer": user,
                "status": "bad_shape"
            }
    except Exception as e:
        return {
            "sample_id": out_name,
            "class": canonical,
            "source": "ISL500",
            "signer": user,
            "status": f"err:{str(e)[:40]}"
        }


def main():
    print("=" * 65, flush=True)
    print("  ISL500 MULTI-THREADED DOWNLOAD & CONVERSION (16 WORKERS)", flush=True)
    print("=" * 65, flush=True)

    target_csv = REPORTS_DIR / "isl500_target_files.csv"
    with open(target_csv, "r", encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    valid_records = [r for r in records if r["word_raw"] in ISL500_TO_CANONICAL]
    total_count = len(valid_records)
    print(f"[+] Total target records to process: {total_count}", flush=True)

    ok = cached = rejected = fail = 0
    manifest = []
    t0 = time.time()

    MAX_WORKERS = 16
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {executor.submit(process_single_item, r): r for r in valid_records}
        completed_count = 0
        last_log_time = t0

        for fut in as_completed(future_map):
            completed_count += 1
            res = fut.result()
            if res:
                manifest.append(res)
                st = res["status"]
                if st == "ok":
                    ok += 1
                elif st == "cached":
                    cached += 1
                elif st == "bad_shape":
                    rejected += 1
                else:
                    fail += 1

            now = time.time()
            if completed_count % 100 == 0 or completed_count == total_count or (now - last_log_time >= 5.0):
                last_log_time = now
                elapsed = now - t0
                pct = completed_count / total_count * 100.0
                rate = completed_count / max(1e-3, elapsed)
                rem_sec = (total_count - completed_count) / max(1e-3, rate)
                print(f"[{completed_count:4d}/{total_count}] ({pct:5.1f}%) | "
                      f"ok: {ok} | cached: {cached} | fail: {fail} | "
                      f"rate: {rate:.1f} it/s | ETA: {rem_sec:.0f}s", flush=True)

    manifest_path = REPORTS_DIR / "isl500_download_manifest.csv"
    if manifest:
        fieldnames = ["sample_id", "class", "source", "signer", "hash", "status"]
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(manifest)

    elapsed = time.time() - t0
    print(f"\n{'=' * 65}", flush=True)
    print(f"  COMPLETE in {elapsed:.1f}s", flush=True)
    print(f"  Downloaded: {ok} | Cached: {cached} | Rejected: {rejected} | Failed: {fail}", flush=True)
    print(f"  Valid total: {ok + cached}", flush=True)
    print(f"  Manifest: {manifest_path}", flush=True)
    print(f"{'=' * 65}", flush=True)


if __name__ == "__main__":
    main()
