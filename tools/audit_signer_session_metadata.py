"""Audit available dataset metadata without inventing signer identities."""

import json
import re
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "signer_session_metadata_audit.json"
DATASET_ROOTS = [
    Path(r"C:\Users\navee\Downloads\archive (2)\Video_Dataset"),
    Path(r"C:\Users\navee\Downloads\archive\ProcessedData_vivit"),
]
ARCHIVES = [
    Path(r"C:\Users\navee\Downloads\archive (2).zip"),
    Path(r"C:\Users\navee\Downloads\archive.zip"),
]


def nonempty(value):
    if value is None:
        return False
    if isinstance(value, np.ndarray):
        return value.size > 0 and any(nonempty(item) for item in value.reshape(-1)[:10])
    text = str(value).strip()
    return bool(text) and text.upper() not in {"UNKNOWN", "UNKNOWN_SIGNER", "NONE", "NAN"}


def inspect_npz_files():
    files = list((ROOT / "data").rglob("*.npz"))
    field_counts = Counter()
    signer_values = set()
    session_values = set()
    source_examples = []
    for path in files:
        try:
            with np.load(path, allow_pickle=True) as data:
                fields = set(data.files)
                field_counts.update(fields)
                for field in ("signer_id", "signer", "subject_id", "subject", "participant_id", "participant", "person_id"):
                    if field in data and nonempty(data[field]):
                        signer_values.update(str(item) for item in np.asarray(data[field]).reshape(-1)[:100])
                if "session_id" in data:
                    session_values.update(str(item) for item in np.asarray(data["session_id"]).reshape(-1)[:100])
                if "source_video" in data and len(source_examples) < 10:
                    source_examples.extend(str(item) for item in np.asarray(data["source_video"]).reshape(-1)[:10])
        except Exception:
            field_counts["CORRUPT_OR_UNREADABLE"] += 1
    return {
        "npz_file_count": len(files),
        "metadata_field_counts": dict(field_counts),
        "explicit_signer_values_found": sorted(signer_values),
        "explicit_signer_metadata_available": bool(signer_values),
        "session_value_count": len(session_values),
        "source_video_examples": source_examples[:10],
    }


def inspect_clean_manifests():
    result = {}
    for split in ("train", "val", "test"):
        path = ROOT / "data" / "unified_136class" / f"{split}_split.npz"
        if not path.exists():
            result[split] = {"exists": False}
            continue
        with np.load(path, allow_pickle=True) as data:
            signers = sorted({str(item) for item in data["signer_ids"].reshape(-1)})
            groups = {str(item) for item in data["recording_group_ids"].reshape(-1)}
            result[split] = {
                "exists": True,
                "sample_count": int(len(data["labels"])),
                "signer_ids": signers,
                "recording_group_count": len(groups),
            }
    available = all(item.get("exists") and item.get("signer_ids") not in (["UNKNOWN_SIGNER"], []) for item in result.values())
    result["signer_independent_split_verified"] = False if not available else None
    return result


def inspect_sources():
    result = []
    for root in DATASET_ROOTS:
        exists = root.exists()
        files = [p for p in root.rglob("*") if p.is_file()] if exists else []
        classes = sorted({p.parent.name for p in files})
        result.append({
            "path": str(root),
            "exists": exists,
            "file_count": len(files),
            "class_directory_count": len(classes),
            "class_examples": classes[:20],
            "filename_patterns": sorted({re.sub(r"\d+", "<N>", p.name) for p in files})[:20],
        })
    return result


def inspect_archives():
    result = []
    for archive in ARCHIVES:
        item = {"path": str(archive), "exists": archive.exists()}
        if archive.exists():
            with zipfile.ZipFile(archive) as handle:
                names = handle.namelist()
                item["member_count"] = len(names)
                item["member_examples"] = names[:10]
                item["signer_named_members"] = [name for name in names if re.search(r"signer|subject|participant|person", name, re.I)][:20]
                item["session_named_members"] = [name for name in names if re.search(r"session|take", name, re.I)][:20]
        result.append(item)
    return result


def main():
    npz = inspect_npz_files()
    clean_manifests = inspect_clean_manifests()
    report = {
        "dataset_sources": inspect_sources(),
        "archives": inspect_archives(),
        "local_npz_metadata": npz,
        "clean_production_manifests": clean_manifests,
        "signer_independent_evaluation": {
            "status": "unverified",
            "reason": "The authoritative clean manifests contain only UNKNOWN_SIGNER. Historical intermediate NPZ fields are not trusted without provenance verification.",
        } if clean_manifests.get("signer_independent_split_verified") is False else {
            "status": "metadata_requires_manual_verification",
            "reason": "Explicit signer-like fields exist and require provenance verification before constructing a signer split.",
        },
        "session_protocol": "Use verified recording_group IDs from the clean manifests; raw MVI basename is not globally unique.",
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
