"""
Inspect INCLUDE dataset (ai4bharat/INCLUDE) and Data.gov ISL dataset
for the 11 classes missing from ISL500.

Missing: goodbye, sorry, stop, okay, brother, hospital, come, stand, read, write, when
"""
import urllib.request
import json
import re
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"

MISSING_11 = ["goodbye", "sorry", "stop", "okay", "brother", "hospital", "come", "stand", "read", "write", "when"]

ALIAS_MAP_MISSING = {
    "goodbye": ["goodbye", "bye", "bye bye", "bye_bye", "byebye"],
    "sorry": ["sorry", "apologize", "apology"],
    "stop": ["stop", "halt"],
    "okay": ["okay", "ok", "alright"],
    "brother": ["brother"],
    "hospital": ["hospital"],
    "come": ["come", "coming"],
    "stand": ["stand", "standing", "standup", "stand up", "stand_up"],
    "read": ["read", "reading"],
    "write": ["write", "writing"],
    "when": ["when"],
}

def fetch_repo_files(repo_id):
    url = f"https://huggingface.co/api/datasets/{repo_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        siblings = data.get("siblings", [])
        return [s["rfilename"] for s in siblings]

print("=" * 70)
print("  INCLUDE DATASET INSPECTION (ai4bharat/INCLUDE)")
print("=" * 70)

try:
    include_files = fetch_repo_files("ai4bharat/INCLUDE")
    print(f"Total files: {len(include_files)}")
    
    # INCLUDE is organized like: category/word/video.mp4
    include_words = set()
    for f in include_files:
        parts = f.split("/")
        if len(parts) >= 3 and f.endswith(".mp4"):
            word = parts[1].lower().strip()
            include_words.add(word)
        elif len(parts) >= 2 and f.endswith(".mp4"):
            word = parts[0].lower().strip()
            include_words.add(word)
    
    print(f"Unique words/categories: {len(include_words)}")
    print(f"Sample words: {sorted(list(include_words))[:30]}")
    
    # Check overlap
    for cls in MISSING_11:
        aliases = ALIAS_MAP_MISSING[cls]
        found = [a for a in aliases if a in include_words]
        if found:
            print(f"  [FOUND] {cls} -> {found}")
        else:
            # Fuzzy check
            partial = [w for w in include_words if cls in w or w in cls]
            if partial:
                print(f"  [MAYBE] {cls} -> partial matches: {partial}")
            else:
                print(f"  [MISS]  {cls}")
except Exception as e:
    print(f"Error inspecting INCLUDE: {e}")

print("\n" + "=" * 70)
print("  DATA.GOV ISL DATASET INSPECTION")
print("=" * 70)

try:
    datagov_files = fetch_repo_files("silentone0725/Indian_Sign_Language_Data.gov_Rencoded")
    print(f"Total files: {len(datagov_files)}")
    
    # Data.gov is organized: letter/word.mp4
    datagov_words = set()
    for f in datagov_files:
        if f.endswith(".mp4"):
            basename = f.split("/")[-1].replace(".mp4", "")
            word_clean = basename.lower().replace("_", " ").replace("-", " ").strip()
            datagov_words.add(word_clean)
    
    print(f"Unique video files: {len(datagov_words)}")
    
    # Check overlap with missing classes
    for cls in MISSING_11:
        aliases = ALIAS_MAP_MISSING[cls]
        found = [a for a in aliases if a in datagov_words]
        if found:
            print(f"  [FOUND] {cls} -> {found}")
        else:
            partial = [w for w in datagov_words if cls in w.split() or w.startswith(cls)]
            if partial:
                print(f"  [MAYBE] {cls} -> {partial[:5]}")
            else:
                print(f"  [MISS]  {cls}")
except Exception as e:
    print(f"Error inspecting Data.gov: {e}")

# Check current vidit031 dataset for the missing classes
print("\n" + "=" * 70)
print("  CURRENT DATASET (vidit031) COVERAGE OF MISSING 11")
print("=" * 70)

try:
    vidit_files = fetch_repo_files("vidit031/isl-isolated-40words")
    vidit_words = set()
    for f in vidit_files:
        parts = f.split("/")
        if len(parts) >= 2 and f.endswith(".mp4"):
            vidit_words.add(parts[0].lower())
    
    print(f"Vidit031 classes: {sorted(vidit_words)}")
    for cls in MISSING_11:
        if cls in vidit_words:
            count = sum(1 for f in vidit_files if f.startswith(cls + "/") and f.endswith(".mp4"))
            print(f"  [FOUND] {cls}: {count} videos in vidit031")
        else:
            print(f"  [MISS]  {cls}")
except Exception as e:
    print(f"Error: {e}")
