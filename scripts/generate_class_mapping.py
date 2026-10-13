import csv
import json
import re
from pathlib import Path

CURRENT_CLASSES = [
    'brother', 'come', 'drink', 'eat', 'father', 'food', 'friend', 'go', 'goodbye', 
    'he', 'hello', 'help', 'home', 'hospital', 'market', 'me', 'mother', 'no', 'okay', 
    'please', 'read', 'school', 'she', 'sister', 'sit', 'sorry', 'stand', 'stop', 
    'student', 'tea', 'teacher', 'thank_you', 'today', 'water', 'what', 'when', 'where', 
    'write', 'yes', 'you'
]

ADD1_CLASSES = [
    'afternoon', 'animal', 'bad', 'beautiful', 'big', 'bird', 'blind', 'cat', 'cheap', 
    'clothing', 'cold', 'cow', 'curved', 'deaf', 'dog', 'dress', 'dry', 'evening', 
    'expensive', 'famous', 'fast', 'female', 'fish', 'flat', 'friday', 'good', 'happy', 
    'hat', 'healthy', 'horse', 'hot', 'hour', 'light', 'long', 'loose', 'loud', 'minute', 
    'monday', 'month', 'morning', 'mouse', 'narrow', 'new', 'night', 'old', 'pant', 
    'pocket', 'quiet', 'sad', 'saturday', 'second', 'shirt', 'shoes', 'short', 'sick', 
    'skirt', 'slow', 'small', 'suit', 'sunday', 't_shirt', 'tall', 'thursday', 'time', 
    'today', 'tomorrow', 'tuesday', 'ugly', 'warm', 'wednesday', 'week', 'wet', 'wide', 
    'year', 'yesterday', 'young'
]

ADD2_CLASSES = [
    'Bear', 'Break', 'Brinjal', 'Budget', 'Busy', 'Cabbage', 'Carrot', 'Cauliflower', 
    'Chilli', 'Clean', 'Close', 'Come', 'Cook', 'Crocodile', 'Cry', 'Cucumber', 'Deer', 
    'Drink', 'Elephant', 'Exam', 'Fedup', 'Fever', 'Giraffe', 'Give', 'Good Morning', 
    'Good afternoon', 'Hello', 'Hug', 'Injury', 'Interview', 'Jump', 'Karnataka', 'Key', 
    'Knife', 'Lemon', 'Lion', 'Man', 'Maths', 'Maybe', 'Monkey', 'Onion', 'Peacock', 
    'Pigeon', 'Pour', 'Radish', 'Sparrow', 'Still', 'Switch', 'Tea', 'Temple', 'Thank you', 
    'Tiger', 'Turtle', 'Umbrella', 'Uncle', 'Vegetables', 'Volcano', 'What is your Name', 
    'Wife', 'Writer', 'Wrong'
]

def normalize_label(s):
    # Remove underscores, hyphens, extra whitespace, lowercase
    clean = re.sub(r"[_\-\s]+", " ", s.lower()).strip()
    return clean

curr_norm = {normalize_label(c): c for c in CURRENT_CLASSES}

rows = []

# Map Additional 1
for a in sorted(ADD1_CLASSES):
    norm_a = normalize_label(a)
    if norm_a in curr_norm:
        status = "MATCH"
        matched_curr = curr_norm[norm_a]
        note = "Exact lexical match"
    elif norm_a in ["good", "morning", "evening", "night", "food"]:
        status = "POTENTIAL_OVERLAP_REVIEW"
        matched_curr = "NONE"
        note = "Semantic concept relates to greetings or daily needs; keep separate"
    else:
        status = "NEW_ISOLATED_WORD_CLASS"
        matched_curr = "NONE"
        note = "Distinct ISL vocabulary word"
    rows.append({
        "dataset_source": "Additional_1 (ProcessedData_vivit)",
        "additional_class": a,
        "current_class": matched_curr,
        "match_status": status,
        "notes": note
    })

# Map Additional 2
for a in sorted(ADD2_CLASSES):
    norm_a = normalize_label(a)
    if norm_a in curr_norm:
        status = "MATCH"
        matched_curr = curr_norm[norm_a]
        note = "Exact lexical match"
    elif norm_a in ["thank you"]:
        status = "MATCH"
        matched_curr = "thank_you"
        note = "Exact lexical match with whitespace vs underscore"
    elif norm_a in ["writer"]:
        status = "POTENTIAL_OVERLAP_REVIEW"
        matched_curr = "write"
        note = "Agentive noun 'writer' vs verb 'write'; distinct morphological signs in ISL"
    elif norm_a in ["what is your name"]:
        status = "PHRASE_SENTENCE_EXCLUDE_FROM_ISOLATED"
        matched_curr = "NONE"
        note = "Complete conversational sentence; exclude from isolated single-word classifier"
    elif norm_a in ["good morning", "good afternoon"]:
        status = "COMPOUND_PHRASE_CLASS"
        matched_curr = "NONE"
        note = "Compound two-sign phrase"
    else:
        status = "NEW_ISOLATED_WORD_CLASS"
        matched_curr = "NONE"
        note = "Distinct ISL vocabulary word"
    rows.append({
        "dataset_source": "Additional_2 (Video_Dataset)",
        "additional_class": a,
        "current_class": matched_curr,
        "match_status": status,
        "notes": note
    })

reports_dir = Path(r"c:\Users\navee\.gemini\antigravity-ide\scratch\isl-translator\reports")
out_csv = reports_dir / "class_mapping_review.csv"

with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["dataset_source", "additional_class", "current_class", "match_status", "notes"])
    writer.writeheader()
    writer.writerows(rows)

print(f"[+] Successfully generated: {out_csv} with {len(rows)} entries")

# Summary print
matches = [r for r in rows if r["match_status"] == "MATCH"]
print(f"Total Direct Matches with Current 40 Classes: {len(matches)}")
for m in matches:
    print(f"  * {m['dataset_source']} -> '{m['additional_class']}' maps to Current '{m['current_class']}'")
