"""
ISL500 Dataset Deep Inspection:
- Extract all unique word labels from 100K+ file listing
- Map against our 40 target classes
- Count samples per class per signer
- Generate coverage report
"""
import urllib.request
import urllib.parse
import json
import re
import csv
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

TARGET_40 = [
    "hello", "goodbye", "thank_you", "sorry", "please", "yes", "no", "help",
    "stop", "okay", "me", "you", "he", "she", "mother", "father", "brother",
    "sister", "friend", "teacher", "student", "home", "school", "hospital",
    "market", "eat", "drink", "water", "food", "tea", "come", "go", "sit",
    "stand", "read", "write", "what", "where", "when", "today"
]

# Alias mapping: ISL500 labels -> our canonical labels
ALIAS_MAP = {
    "hello": "hello",
    "hi": "hello",
    "goodbye": "goodbye",
    "bye": "goodbye",
    "bye bye": "goodbye",
    "byebye": "goodbye",
    "thankyou": "thank_you",
    "thank you": "thank_you",
    "thank_you": "thank_you",
    "thanks": "thank_you",
    "sorry": "sorry",
    "please": "please",
    "yes": "yes",
    "no": "no",
    "help": "help",
    "stop": "stop",
    "okay": "okay",
    "ok": "okay",
    "me": "me",
    "i": "me",
    "you": "you",
    "he": "he",
    "she": "she",
    "mother": "mother",
    "mom": "mother",
    "mum": "mother",
    "mama": "mother",
    "father": "father",
    "dad": "father",
    "papa": "father",
    "brother": "brother",
    "sister": "sister",
    "friend": "friend",
    "teacher": "teacher",
    "student": "student",
    "home": "home",
    "house": "home",
    "school": "school",
    "hospital": "hospital",
    "market": "market",
    "eat": "eat",
    "food": "food",
    "drink": "drink",
    "water": "water",
    "tea": "tea",
    "come": "come",
    "go": "go",
    "sit": "sit",
    "stand": "stand",
    "read": "read",
    "write": "write",
    "what": "what",
    "where": "where",
    "when": "when",
    "today": "today",
}

def fetch_isl500_file_list():
    """Fetch the full file listing from ISL500/ISL-DATA using the HF API with pagination."""
    all_files = []
    # The HF API sibling list is capped; use the tree endpoint instead
    page = 0
    cursor = ""
    while True:
        url = f"https://huggingface.co/api/datasets/ISL500/ISL-DATA/tree/main?recursive=true"
        if cursor:
            url += f"&cursor={cursor}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if not data:
                    break
                for item in data:
                    rpath = item.get("path", "")
                    if rpath.endswith(".h5"):
                        all_files.append(rpath)
                # Check for pagination
                if len(data) < 1000:
                    break
                # Use last item path as cursor
                cursor = urllib.parse.quote(data[-1].get("path", ""), safe="")
                page += 1
                if page > 200:  # Safety limit
                    break
        except Exception as e:
            print(f"Error fetching page {page}: {e}")
            break
    return all_files

def parse_filename(filepath):
    """Parse ISL500 filename: Landmarks/MediaPipe/USER/Word__sessionXXX__clipXXX.h5"""
    parts = filepath.split("/")
    if len(parts) < 4:
        return None, None, None, None
    user = parts[2]  # e.g., ISL_DATA_USER001
    filename = parts[-1]  # e.g., Hello__session106__clip014.h5
    # Extract word from filename
    match = re.match(r"^(.+?)__session(\d+)__clip(\d+)\.h5$", filename)
    if match:
        word_raw = match.group(1)
        session = int(match.group(2))
        clip = int(match.group(3))
        return user, word_raw, session, clip
    return user, None, None, None

def main():
    print("=" * 70)
    print("  ISL500 DATASET DEEP INSPECTION & 40-CLASS COVERAGE ANALYSIS")
    print("=" * 70)
    
    print("\n[1/4] Fetching ISL500 file listing from HuggingFace...")
    files = fetch_isl500_file_list()
    print(f"  Found {len(files)} .h5 landmark files")
    
    if len(files) == 0:
        print("[ERROR] No files found. Trying alternative approach...")
        # Use the siblings API instead
        url = "https://huggingface.co/api/datasets/ISL500/ISL-DATA"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            siblings = data.get("siblings", [])
            files = [s["rfilename"] for s in siblings if s["rfilename"].endswith(".h5")]
            print(f"  Found {len(files)} .h5 files from siblings API")
    
    print("\n[2/4] Parsing file metadata...")
    word_counts = defaultdict(int)
    user_word_counts = defaultdict(lambda: defaultdict(int))
    users = set()
    all_records = []
    
    for fp in files:
        user, word_raw, session, clip = parse_filename(fp)
        if word_raw is None:
            continue
        word_lower = word_raw.lower().replace(" ", "_").strip()
        word_counts[word_raw] += 1
        users.add(user)
        user_word_counts[user][word_raw] += 1
        all_records.append({
            "filepath": fp,
            "user": user,
            "word_raw": word_raw,
            "word_lower": word_lower,
            "session": session,
            "clip": clip
        })
    
    print(f"  Total parsed records: {len(all_records)}")
    print(f"  Unique words in ISL500: {len(word_counts)}")
    print(f"  Unique signers: {len(users)}")
    
    # Sort words by count
    sorted_words = sorted(word_counts.items(), key=lambda x: -x[1])
    print(f"\n  Top 20 words by count:")
    for w, c in sorted_words[:20]:
        print(f"    {w}: {c} samples")
    
    print(f"\n  Bottom 10 words by count:")
    for w, c in sorted_words[-10:]:
        print(f"    {w}: {c} samples")
    
    print("\n[3/4] Mapping ISL500 words to our 40 target classes...")
    # Build mapping
    target_matches = {}  # canonical -> list of ISL500 raw words
    for raw_word in word_counts:
        normalized = raw_word.lower().replace(" ", "_").strip()
        # Check alias map
        canonical = ALIAS_MAP.get(normalized)
        if canonical is None:
            # Try direct match
            canonical = ALIAS_MAP.get(raw_word.lower().strip())
        if canonical and canonical in TARGET_40:
            if canonical not in target_matches:
                target_matches[canonical] = []
            if raw_word not in target_matches[canonical]:
                target_matches[canonical].append(raw_word)
    
    print(f"\n  Matched {len(target_matches)} of 40 target classes in ISL500:")
    total_matching_samples = 0
    coverage_data = []
    for canonical in sorted(TARGET_40):
        if canonical in target_matches:
            raw_words = target_matches[canonical]
            count = sum(word_counts[rw] for rw in raw_words)
            total_matching_samples += count
            signers = set()
            for rec in all_records:
                if rec["word_raw"] in raw_words:
                    signers.add(rec["user"])
            print(f"    {canonical:15s} -> ISL500 labels: {raw_words}, {count} samples, {len(signers)} signers")
            coverage_data.append({
                "canonical_class": canonical,
                "isl500_labels": "|".join(raw_words),
                "isl500_count": count,
                "isl500_signers": len(signers),
                "match_type": "EXACT" if len(raw_words) == 1 and raw_words[0].lower() == canonical.replace("_", " ") else "ALIAS"
            })
        else:
            print(f"    {canonical:15s} -> NOT FOUND in ISL500")
            coverage_data.append({
                "canonical_class": canonical,
                "isl500_labels": "",
                "isl500_count": 0,
                "isl500_signers": 0,
                "match_type": "MISSING"
            })
    
    print(f"\n  Total matching samples in ISL500: {total_matching_samples}")
    missing = [c for c in TARGET_40 if c not in target_matches]
    print(f"  Missing classes ({len(missing)}): {missing}")
    
    # Save coverage report
    coverage_path = REPORTS_DIR / "isl500_40class_coverage.csv"
    with open(coverage_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["canonical_class", "isl500_labels", "isl500_count", "isl500_signers", "match_type"])
        writer.writeheader()
        writer.writerows(coverage_data)
    print(f"\n  Saved coverage report: {coverage_path}")
    
    # Save full word list
    wordlist_path = REPORTS_DIR / "isl500_full_wordlist.csv"
    with open(wordlist_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["word_raw", "count", "num_signers"])
        for w, c in sorted_words:
            signers = set(user for user, wc in user_word_counts.items() if w in wc)
            writer.writerow([w, c, len(signers)])
    print(f"  Saved full wordlist: {wordlist_path}")
    
    # Save matching file list for download
    matching_files_path = REPORTS_DIR / "isl500_target_files.csv"
    target_raw_words = set()
    for rws in target_matches.values():
        target_raw_words.update(rws)
    
    matching_records = [rec for rec in all_records if rec["word_raw"] in target_raw_words]
    with open(matching_files_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "user", "word_raw", "word_lower", "session", "clip"])
        writer.writeheader()
        writer.writerows(matching_records)
    print(f"  Saved {len(matching_records)} target file paths: {matching_files_path}")
    
    print("\n[4/4] Summary")
    print(f"  ISL500 total files: {len(files)}")
    print(f"  ISL500 unique words: {len(word_counts)}")
    print(f"  ISL500 unique signers: {len(users)}")
    print(f"  Our 40-class matches: {len(target_matches)}/40")
    print(f"  Total matching samples: {total_matching_samples}")
    print(f"  Missing classes: {len(missing)}")
    print(f"  Signers: {sorted(users)}")

if __name__ == "__main__":
    main()
