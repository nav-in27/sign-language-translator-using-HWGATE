import os
import sys
import json
from pathlib import Path

# Let's inspect the exact candidates
archive2_classes = [
    'bear', 'break', 'brinjal', 'budget', 'busy', 'cabbage', 'carrot', 'cauliflower', 
    'chilli', 'clean', 'close', 'come', 'cook', 'crocodile', 'cry', 'cucumber', 
    'deer', 'drink', 'elephant', 'exam', 'fedup', 'fever', 'giraffe', 'give', 
    'good_afternoon', 'good_morning', 'hug', 'injury', 'interview', 'jump', 
    'karnataka', 'key', 'knife', 'lemon', 'lion', 'man', 'maths', 'maybe', 
    'monkey', 'onion', 'peacock', 'pigeon', 'pour', 'radish', 'sparrow', 
    'switch', 'tea', 'temple', 'thank_you', 'tiger', 'turtle', 'umbrella', 
    'uncle', 'vegetables', 'volcano', 'wife', 'write', 'wrong'
] # 58 strong isolated signs from Archive-2 (excluding Still and full phrase What is your Name)

raw40_new_classes = [
    'brother', 'eat', 'father', 'food', 'friend', 'go', 'goodbye', 'he', 
    'help', 'home', 'hospital', 'market', 'me', 'mother', 'no', 'okay', 
    'please', 'read', 'school', 'she', 'sister', 'sit', 'sorry', 'stand', 
    'stop', 'student', 'teacher', 'today', 'water', 'what', 'when', 'where', 
    'yes', 'you'
] # 34 core ISL signs from Raw40 / ISL500 / Cache

# 58 + 34 = 92 classes so far!

# To reach 104 classes, we need 12 classes from Vivit.
# Let's choose 12 of the cleanest, highest-count Vivit classes:
vivit_selected_classes = [
    'animal', 'bad', 'big', 'bird', 'cat', 'cold', 'cow', 'dog', 
    'fast', 'fish', 'good', 'happy', 'hat', 'healthy', 'horse', 'hot'
] # 16 classes from Vivit

total_candidates = sorted(list(set(archive2_classes + raw40_new_classes + vivit_selected_classes)))
print(f"Total Selected Vocabulary: {len(total_candidates)} classes")
print(f"Archive-2: {len(archive2_classes)} classes")
print(f"Raw40/ISL500: {len(raw40_new_classes)} classes")
print(f"Vivit: {len(vivit_selected_classes)} classes")
print("\nVocabulary List:")
for i, c in enumerate(total_candidates):
    print(f"{i+1:>3}. {c}")

with open("reports/candidate_108_classes.json", "w") as f:
    json.dump({
        "num_classes": len(total_candidates),
        "classes": total_candidates,
        "archive2_classes": archive2_classes,
        "raw40_classes": raw40_new_classes,
        "vivit_classes": vivit_selected_classes
    }, f, indent=2)

print("\nSaved candidate manifest to reports/candidate_108_classes.json")
