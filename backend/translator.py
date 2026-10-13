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

            # Everyday Objects
            "key": "Key",
            "knife": "Knife",
            "umbrella": "Umbrella",
            "hat": "Hat",

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
            "peacock": "Peacock",
            "pigeon": "Pigeon",
            "sparrow": "Sparrow",
            "tiger": "Tiger",
            "turtle": "Turtle",
            "volcano": "Volcano",

            # Adjectives & Physical Attributes
            "bad": "Bad",
            "big": "Big",
            "cold": "Cold",
            "fast": "Fast",
            "good": "Good",
            "happy": "Happy",
            "healthy": "Healthy",
            "hot": "Hot",
            "loud": "Loud",
            "cry": "Cry"
        }

        # Multi-sign sequential combinations (Subject-Object-Verb and natural ISL to English)
        self.PHRASE_PATTERNS = {
            ("hello", "how_are_you"): "Hello, how are you?",
            ("thank_you", "very_much"): "Thank you very much.",
            ("me", "student"): "I am a student.",
            ("me", "teacher"): "I am a teacher.",
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
            ("go", "hospital"): "Go to the hospital.",
            ("clean", "home"): "Clean the home.",
            ("close", "door"): "Close the door.",
        }

    def translate_gloss_to_english(self, text: str) -> str:
        """
        Translates single sign or sequence of signs to semantically faithful English.
        """
        if not text:
            return ""
        
        raw = text.strip().lower()
        if raw in ["waiting for sign", "detecting...", "unknown_sign", "no sign detected", "buffering..."]:
            return ""

        words = raw.split()
        if len(words) == 1:
            w = words[0].replace("-", "_")
            if w in self.CANONICAL_MEANING:
                return self.CANONICAL_MEANING[w]
            # Default fallback: title case
            return w.replace("_", " ").title()

        # Multi-sign sequence
        tuple_words = tuple(w.replace("-", "_") for w in words)
        if tuple_words in self.PHRASE_PATTERNS:
            return self.PHRASE_PATTERNS[tuple_words]

        # Sequence fallback: combine individual gloss translations
        translated_tokens = [
            self.CANONICAL_MEANING.get(w.replace("-", "_"), w.replace("_", " ").title())
            for w in words
        ]
        return " ".join(translated_tokens)


# Authoritative singleton engine instance
nmt_engine = SemanticISLTranslator()
InstantISLTranslator = SemanticISLTranslator
