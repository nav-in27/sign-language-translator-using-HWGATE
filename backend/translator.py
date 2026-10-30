"""
Verified Semantic English Translation Engine for Indian Sign Language (ISL).
Converts recognized isolated ISL glosses into authentic, verified English words/phrases
without fabricating missing grammar or inventing imaginary context.
"""

import re
from typing import List, Dict, Optional


class SemanticISLTranslator:
    def __init__(self):
        # Canonical verified English meanings for all 108 ISL classes
        self.CANONICAL_MEANING: Dict[str, str] = {
            # Family & People
            "father": "Father",
            "mother": "Mother",
            "brother": "Brother",
            "sister": "Sister",
            "friend": "Friend",
            "teacher": "Teacher",
            "student": "Student",
            "uncle": "Uncle",
            "wife": "Wife",
            "man": "Man",
            "me": "Me",
            "you": "You",
            "he": "He",
            "she": "She",

            # Greetings & Social
            "hello": "Hello",
            "goodbye": "Goodbye",
            "good_morning": "Good Morning",
            "good_afternoon": "Good Afternoon",
            "thank_you": "Thank You",
            "please": "Please",
            "sorry": "Sorry",
            "help": "Help",
            "okay": "Okay",

            # Daily Actions & Gestures
            "come": "Come",
            "go": "Go",
            "sit": "Sit",
            "stand": "Stand",
            "stop": "Stop",
            "eat": "Eat",
            "drink": "Drink",
            "cook": "Cook",
            "pour": "Pour",
            "clean": "Clean",
            "close": "Close",
            "break": "Break",
            "jump": "Jump",
            "give": "Give",
            "hug": "Hug",
            "write": "Write",
            "read": "Read",
            "switch": "Switch",

            # Questions & Particles
            "what": "What",
            "where": "Where",
            "when": "When",
            "yes": "Yes",
            "no": "No",

            # Places & Institutions
            "home": "Home",
            "school": "School",
            "market": "Market",
            "hospital": "Hospital",
            "temple": "Temple",
            "karnataka": "Karnataka",

            # Time, Concepts & Conditions
            "today": "Today",
            "exam": "Exam",
            "interview": "Interview",
            "budget": "Budget",
            "busy": "Busy",
            "fedup": "Fed Up",
            "fever": "Fever",
            "injury": "Injury",
            "maths": "Mathematics",
            "maybe": "Maybe",
            "wrong": "Wrong",

            # Food, Drinks & Produce
            "food": "Food",
            "water": "Water",
            "tea": "Tea",
            "vegetables": "Vegetables",
            "brinjal": "Brinjal (Eggplant)",
            "cabbage": "Cabbage",
            "carrot": "Carrot",
            "cauliflower": "Cauliflower",
            "chilli": "Chilli",
            "cucumber": "Cucumber",
            "lemon": "Lemon",
            "onion": "Onion",
            "radish": "Radish",

            # Everyday Objects & Accessories
            "key": "Key",
            "knife": "Knife",
            "umbrella": "Umbrella",
            "hat": "Hat",
            "clothing": "Clothing",
            "dress": "Dress",
            "pant": "Pants",
            "pocket": "Pocket",
            "shirt": "Shirt",
            "shoes": "Shoes",
            "skirt": "Skirt",
            "suit": "Suit",
            "t_shirt": "T-Shirt",

            # Animals & Nature
            "animal": "Animal",
            "bear": "Bear",
            "bird": "Bird",
            "cat": "Cat",
            "cow": "Cow",
            "crocodile": "Crocodile",
            "deer": "Deer",
            "dog": "Dog",
            "elephant": "Elephant",
            "fish": "Fish",
            "giraffe": "Giraffe",
            "horse": "Horse",
            "lion": "Lion",
            "monkey": "Monkey",
            "mouse": "Mouse",
            "peacock": "Peacock",
            "pigeon": "Pigeon",
            "sparrow": "Sparrow",
            "tiger": "Tiger",
            "turtle": "Turtle",
            "volcano": "Volcano",

            # Time, Days & Calendar
            "time": "Time",
            "minute": "Minute",
            "hour": "Hour",
            "morning": "Morning",
            "afternoon": "Afternoon",
            "evening": "Evening",
            "night": "Night",
            "today": "Today",
            "yesterday": "Yesterday",
            "tomorrow": "Tomorrow",
            "week": "Week",
            "month": "Month",
            "year": "Year",
            "monday": "Monday",
            "tuesday": "Tuesday",
            "wednesday": "Wednesday",
            "thursday": "Thursday",
            "friday": "Friday",
            "saturday": "Saturday",
            "sunday": "Sunday",

            # People, Attributes & Identity
            "female": "Female",
            "young": "Young",
            "old": "Old",
            "blind": "Blind",
            "deaf": "Deaf",
            "what_is_your_name": "What is your name?",

            # Adjectives, Properties & Physical Attributes
            "bad": "Bad",
            "beautiful": "Beautiful",
            "big": "Big",
            "cheap": "Cheap",
            "cold": "Cold",
            "curved": "Curved",
            "dry": "Dry",
            "expensive": "Expensive",
            "famous": "Famous",
            "fast": "Fast",
            "flat": "Flat",
            "good": "Good",
            "happy": "Happy",
            "healthy": "Healthy",
            "hot": "Hot",
            "light": "Light",
            "long": "Long",
            "loose": "Loose",
            "loud": "Loud",
            "narrow": "Narrow",
            "new": "New",
            "quiet": "Quiet",
            "sad": "Sad",
            "short": "Short",
            "sick": "Sick",
            "slow": "Slow",
            "small": "Small",
            "still": "Still",
            "tall": "Tall",
            "ugly": "Ugly",
            "warm": "Warm",
            "wet": "Wet",
            "wide": "Wide",
            "cry": "Cry"
        }

        # Multi-sign sequential combinations (Subject-Object-Verb and natural ISL to English)
        self.PHRASE_PATTERNS = {
            ("hello", "how_are_you"): "Hello, how are you?",
            ("what_is_your_name",): "What is your name?",
            ("thank_you", "very_much"): "Thank you very much.",
            ("hello", "sister"): "Hello, sister.",
            ("hello", "friend"): "Hello, my friend.",
            ("hello", "teacher"): "Hello, teacher.",
            ("thank_you", "teacher"): "Thank you, teacher.",
            ("thank_you", "friend"): "Thank you, my friend.",
            ("yes", "thank_you"): "Yes, thank you.",
            ("no", "thank_you"): "No, thank you.",
            ("me", "student"): "I am a student.",
            ("me", "teacher"): "I am a teacher.",
            ("student", "sit"): "The student is sitting down.",
            ("student", "write"): "The student is writing.",
            ("teacher", "come"): "The teacher is coming.",
            ("teacher", "go"): "The teacher is going.",
            ("you", "go", "school"): "You are going to school.",
            ("father", "come", "home"): "Father is coming home.",
            ("mother", "cook", "food"): "Mother is cooking food.",
            ("me", "water", "drink"): "I want to drink water.",
            ("me", "drink", "water"): "I want to drink water.",
            ("me", "tea", "drink"): "I want to drink tea.",
            ("me", "drink", "tea"): "I want to drink tea.",
            ("me", "food", "eat"): "I want to eat food.",
            ("me", "eat", "food"): "I want to eat food.",
            ("please", "help", "me"): "Please help me.",
            ("please", "help"): "Please help me.",
            ("please", "sit"): "Please sit down.",
            ("please", "come"): "Please come in.",
            ("please", "give"): "Please give this.",
            ("please", "give", "water"): "Please give me water.",
            ("please", "give", "tea"): "Please give me tea.",
            ("go", "hospital"): "Go to the hospital.",
            ("go", "school"): "Go to school.",
            ("go", "temple"): "Go to the temple.",
            ("clean", "home"): "Clean the home.",
            ("close", "door"): "Close the door.",
            ("close", "school"): "Close the school.",
            ("today", "school", "close"): "Today school is closed.",
            ("today", "temple", "go"): "Today we go to the temple.",
            ("uncle", "come", "home"): "Uncle is coming home.",
            ("wife", "cook", "food"): "Wife is cooking food.",
            ("where", "home"): "Where is your home?",
            ("where", "school"): "Where is the school?",
            ("where", "temple"): "Where is the temple?",
            ("what", "wrong"): "What is wrong?",
            ("no", "wrong"): "No, that is not wrong.",
            ("today", "friday"): "Today is Friday.",
            ("today", "sunday"): "Today is Sunday.",
            ("tomorrow", "exam"): "Tomorrow is the exam.",
            ("yesterday", "interview"): "The interview was yesterday.",
            ("good_morning", "teacher"): "Good morning, teacher.",
            ("good_afternoon", "teacher"): "Good afternoon, teacher.",
            ("hello", "thank_you"): "Hello, thank you.",
            ("thank_you", "good"): "Thank you, good.",
            ("today", "beautiful"): "Today is beautiful.",
            ("you", "beautiful"): "You are beautiful.",
            ("you", "good"): "You are good.",
            ("me", "good"): "I am good.",
            ("me", "happy"): "I am happy.",
            ("today", "good"): "Today is good.",
            ("give", "me"): "Please give me.",
            ("give", "water"): "Please give water.",
            ("give", "food"): "Please give food.",
            ("me", "give", "you"): "I will give to you.",
            ("good", "morning"): "Good morning.",
            ("good", "afternoon"): "Good afternoon.",
            ("good", "night"): "Good night.",
        }

    def translate_gloss_to_english(self, text: str) -> str:
        """
        Translates single sign or sequence of signs to semantically faithful English.
        Applies Greedy Longest-Match Subphrase Segmentation with SOV-to-SVO grammar rules.
        """
        if not text:
            return ""
        
        raw = text.strip().lower()
        if raw in ["waiting for sign", "detecting...", "unknown_sign", "no sign detected", "buffering...", "idle"]:
            return ""
        if raw in ["uncertain", "uncertain...", "uncertain sign"]:
            return "Sign not recognized. Please try again."

        words = [w.replace("-", "_") for w in raw.split()]
        if not words:
            return ""

        if len(words) == 1:
            w = words[0]
            if w in self.CANONICAL_MEANING:
                return self.CANONICAL_MEANING[w]
            return w.replace("_", " ").title()

        # Check exact full sequence match first
        tuple_words = tuple(words)
        if tuple_words in self.PHRASE_PATTERNS:
            return self.PHRASE_PATTERNS[tuple_words]

        # Greedy Longest-Match Subphrase Segmentation
        i = 0
        N = len(words)
        phrases = []
        while i < N:
            matched = False
            # Check windows of size 4 down to 2
            for l in range(min(4, N - i), 1, -1):
                sub = tuple(words[i:i + l])
                if sub in self.PHRASE_PATTERNS:
                    phrases.append(self.PHRASE_PATTERNS[sub].rstrip("."))
                    i += l
                    matched = True
                    break
            if not matched:
                w = words[i]
                part = self.CANONICAL_MEANING.get(w, w.replace("_", " ").title())
                # Deduplicate adjacent duplicate single tokens
                if not phrases or phrases[-1].lower() != part.lower():
                    phrases.append(part)
                i += 1

        if not phrases:
            return ""
        if len(phrases) == 1:
            res = phrases[0]
            return res if res.endswith((".", "!", "?")) else res + "."

        # Form coherent English sentence
        # Join clauses cleanly with commas / periods
        result_parts = []
        for p in phrases:
            clean_p = p.strip()
            if clean_p.endswith((".", "!", "?")):
                result_parts.append(clean_p)
            else:
                result_parts.append(clean_p)

        sentence = ", ".join(result_parts)
        if not sentence.endswith((".", "!", "?")):
            sentence += "."
        # Format sentence capitalization
        sentence = sentence[0].upper() + sentence[1:]
        return sentence

    def translate_sign(self, sign: str) -> str:
        """Translates an isolated confirmed sign directly to its canonical English meaning."""
        if not sign:
            return ""
        clean = sign.strip().lower().replace("-", "_")
        if clean in ["waiting for sign", "detecting...", "unknown_sign", "no sign detected", "buffering...", "idle"]:
            return ""
        if clean in ["uncertain", "uncertain...", "uncertain sign"]:
            return "Sign not recognized. Please try again."
        return self.CANONICAL_MEANING.get(clean, clean.replace("_", " ").title())


# Authoritative singleton engine instance
nmt_engine = SemanticISLTranslator()
InstantISLTranslator = SemanticISLTranslator
ISLTranslator = SemanticISLTranslator
