# Reproduction and Diagnostic Report: 86-Second Screen Recording

**Reference Recording:** `Screen Recording 2026-10-30 193834.mp4` (86.4s, 30.0 FPS, 2591 frames)

## 1. Frame-by-Frame Gesture Timeline & Failure Reproduction

| Event | Time Range | Ground Truth Sign | Model Observed | Confirmed Sign | Displayed Sentence | Primary Failure Mode |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| G01 | 00:00 - 00:10 | **TEA** | TEA (80.6%), TEA (94.9%) | TEA, DRINK | `Drink, Tea.` | Repeated Duplicate Triggering & Synonym Flip |
| G02 | 00:11 - 00:17 | **GO** | No (33.1%), Go (28.0%) | GO | `Go` | Low Confidence & Margin Conflict |
| G03 | 00:18 - 00:34 | **HELLO** | Tea (81.3%), No (31.4%) | *None (Uncertain)* | `Sign not recognized. Please try again. (Tea lingering in sentence)` | UNRECOGNIZED GESTURE (Severe Misclassification) |
| G04 | 00:35 - 00:43 | **SISTER** | SISTER (65.6%), SISTER (42.3%) | SISTER | `Jump, Sister: (False Jump prefix)` | False Positive Insertion & Duplicate Confirmation |
| G05 | 00:44 - 00:60 | **THANK_YOU** | Thank You (72.8%), Thank You (49.2%) | THANK_YOU | `Sister, Thank You: (lingering Sister)` | Lingering Stale Sentence Retention |
| G06 | 00:61 - 00:70 | **LION** | Lion (31.1%), Home (9.3%) | LION | `Lion, Hello, Tea. (Unstable accumulation)` | Borderline Confidence & Zombie Token Accumulation |
| G07 | 00:71 - 00:78 | **NO** | NO (65.2%), Temple (27.0%) | NO | `No` | Competition with TEA |
| G08 | 00:79 - 00:86 | **HOME** | Home (49.5%), School (23.4%) | HOME | `Hello, Home. (Zombie Hello retained)` | Zombie Token Retention |

## 2. Root Cause Analysis

### A. Dataset Tilt Poisoning (Why HELLO & LION Failed)
- **Finding:** In `data/additional_landmarks/hello/`, 120 out of 156 samples (77%) were synthesized from tilted camera takes (`WIN_20231103_..._tilt.mp4`) with an average torso tilt of **-45.7 degrees**.
- **Consequence:** Upright frontal signing caused the waving hand coordinates to project outside the learned tilted cluster. The model predicted `HELLO 30.6%` and `NO 31.4%`, failing the 35% margin gate and producing repeated `"Sign not recognized. Please try again."`.

### B. Missing Bout Lockout & 133ms Cooldown (Why TEA & DRINK Repeated)
- **Finding:** In `backend/inference.py`, `cooldown_frames` was set to only 4 frames (133ms), and `self.bout_confirmed_sign` was never assigned when a sign confirmed in Gate 5.
- **Consequence:** When a user held a gesture for 2 to 4 seconds, the engine fired a confirmation, waited 133ms, and fired again on the same hand motion, alternating between near-neighbors `TEA` and `DRINK`.

### C. Zombie Sentence Accumulation (Why Sentences Were Unstable)
- **Finding:** In `backend/app.py`, the 3.5s pause timeout cleared `session_state['gloss_buffer'] = []` but omitted `session_state['last_translated'] = ''`.
- **Consequence:** Old translations lingered across pause boundaries and prepended to new signs, producing nonsensical sequences like `'Lion, Hello, Tea.'` and `'Hello, Home.'`.
