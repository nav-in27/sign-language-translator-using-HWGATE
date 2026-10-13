"""
Comprehensive ISL500 inspection using the siblings API (returns all 100K files).
Parse ALL files to find our 40 target class matches.
"""
import urllib.request
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

# Case-insensitive alias map
ALIAS_MAP = {}
for w in TARGET_40:
    ALIAS_MAP[w] = w
# Add common aliases
extra = {
    "hi": "hello", "bye": "goodbye", "bye bye": "goodbye", "byebye": "goodbye",
    "bye_bye": "goodbye", "thankyou": "thank_you", "thank you": "thank_you",
    "thanks": "thank_you", "ok": "okay", "i": "me", "myself": "me",
    "mom": "mother", "mum": "mother", "mama": "mother", "mummy": "mother",
    "dad": "father", "papa": "father", "daddy": "father",
    "house": "home", "eating": "eat", "drinking": "drink",
    "reading": "read", "writing": "write", "sitting": "sit",
    "standing": "stand", "coming": "come", "going": "go",
    "stopping": "stop", "helping": "help",
}
ALIAS_MAP.update(extra)

def fetch_all_siblings():
    """Fetch full sibling list from ISL500/ISL-DATA."""
    url = "https://huggingface.co/api/datasets/ISL500/ISL-DATA"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        siblings = data.get("siblings", [])
        return [s["rfilename"] for s in siblings if s["rfilename"].endswith(".h5")]

def parse_filename(filepath):
    parts = filepath.split("/")
    if len(parts) < 4:
        return None, None, None, None
    user = parts[2]
    filename = parts[-1]
    match = re.match(r"^(.+?)__session(\d+)__clip(\d+)\.h5$", filename)
    if match:
        return user, match.group(1), int(match.group(2)), int(match.group(3))
    return user, None, None, None

def main():
    print("=" * 70)
    print("  ISL500 FULL INSPECTION (100K+ FILES)")
    print("=" * 70)
    
    print("\n[1/3] Fetching ALL ISL500 file listing...")
    files = fetch_all_siblings()
    print(f"  Total .h5 files: {len(files)}")
    
    print("\n[2/3] Parsing all records...")
    word_counts = defaultdict(int)
    user_set = set()
    all_records = []
    
    for fp in files:
        user, word_raw, session, clip = parse_filename(fp)
        if word_raw is None:
            continue
        word_counts[word_raw] += 1
        user_set.add(user)
        all_records.append({
            "filepath": fp,
            "user": user,
            "word_raw": word_raw,
            "session": session,
            "clip": clip
        })
    
    print(f"  Parsed {len(all_records)} records")
    print(f"  Unique words: {len(word_counts)}")
    print(f"  Unique signers: {len(user_set)}")
    print(f"  Signers: {sorted(user_set)}")
    
    # Print ALL unique words
    sorted_words = sorted(word_counts.items(), key=lambda x: -x[1])
    print(f"\n  All {len(sorted_words)} unique words:")
    for w, c in sorted_words:
        print(f"    {w}: {c}")
    
    print("\n[3/3] Matching against 40 target classes...")
    target_matches = defaultdict(list)
    for raw_word in word_counts:
        norm = raw_word.lower().replace(" ", "_").strip()
        canonical = ALIAS_MAP.get(norm)
        if canonical is None:
            canonical = ALIAS_MAP.get(raw_word.lower().strip())
        if canonical and canonical in TARGET_40:
            if raw_word not in target_matches[canonical]:
                target_matches[canonical].append(raw_word)
    
    total_match = 0
    print(f"\n  Target class coverage:")
    for c in sorted(TARGET_40):
        if c in target_matches:
            rws = target_matches[c]
            count = sum(word_counts[rw] for rw in rws)
            total_match += count
            signers = len(set(rec["user"] for rec in all_records if rec["word_raw"] in rws))
            print(f"    [MATCH] {c:15s} <- {rws}, {count} samples, {signers} signers")
        else:
            print(f"    [MISS]  {c:15s}")
    
    matched = len(target_matches)
    missing = [c for c in TARGET_40 if c not in target_matches]
    print(f"\n  Matched: {matched}/40, Total samples: {total_match}")
    print(f"  Missing ({len(missing)}): {missing}")
    
    # Save matching file list
    target_raw_words = set()
    for rws in target_matches.values():
        target_raw_words.update(rws)
    
    matching_records = [rec for rec in all_records if rec["word_raw"] in target_raw_words]
    outpath = REPORTS_DIR / "isl500_target_files.csv"
    with open(outpath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "user", "word_raw", "session", "clip"])
        writer.writeheader()
        writer.writerows(matching_records)
    print(f"\n  Saved {len(matching_records)} matching file paths to: {outpath}")

if __name__ == "__main__":
    main()
