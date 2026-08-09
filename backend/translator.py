import os
import threading
import queue
import re

class NMTTranslator:
    def __init__(self):
        self.is_ready = False
        self.pipeline = None
        self.request_queue = queue.Queue()
        self.latest_translation = ""
        self.last_translated_glosses = ""
        
        # Load model in background thread
        threading.Thread(target=self._load_model, daemon=True).start()
        # Worker thread for non-blocking background translation
        threading.Thread(target=self._translation_worker, daemon=True).start()
        
    def _load_model(self):
        try:
            from transformers import pipeline
            print("Loading Qwen1.5-0.5B-Chat for NMT Translation...")
            self.pipeline = pipeline("text-generation", model="Qwen/Qwen1.5-0.5B-Chat", device="cpu")
            self.is_ready = True
            print("NMT Translation Engine Ready.")
        except Exception as e:
            print(f"NMT LLM Load Error: {e}")
            self.is_ready = False

    def _clean_glosses(self, gloss_sequence: str) -> str:
        # Strip numbers like "48. Hello" -> "Hello"
        cleaned = re.sub(r'\b\d+\.\s*', '', gloss_sequence)
        return cleaned.strip()

    def _fast_fallback(self, gloss_sequence: str) -> str:
        cleaned = self._clean_glosses(gloss_sequence)
        if not cleaned:
            return ""
        # Simple grammatical rule-based formatting for instant UX
        words = cleaned.split()
        if len(words) == 1:
            return words[0].capitalize() + "."
        elif len(words) == 2:
            return f"{words[0].capitalize()} {words[1].lower()}."
        else:
            return f"{' '.join(words).capitalize()}."

    def _translation_worker(self):
        while True:
            gloss_sequence = self.request_queue.get()
            if not gloss_sequence:
                self.request_queue.task_done()
                continue
                
            cleaned = self._clean_glosses(gloss_sequence)
            
            if not self.is_ready or not self.pipeline:
                self.latest_translation = self._fast_fallback(cleaned)
                self.request_queue.task_done()
                continue
                
            prompt = (
                f"<|im_start|>system\n"
                f"You are a sign language translator. Convert sign glosses into 1 short natural English sentence. Output ONLY the sentence.<|im_end|>\n"
                f"<|im_start|>user\nGlosses: {cleaned}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            
            try:
                result = self.pipeline(prompt, max_new_tokens=20, do_sample=False)
                generated_text = result[0]['generated_text']
                translation = generated_text.split("<|im_start|>assistant\n")[-1].strip()
                # Clean any quotes or stray markdown
                translation = re.sub(r'^["\']|["\']$', '', translation)
                
                # Check for AI refusal strings
                if "AI language model" in translation or "cannot translate" in translation.lower():
                    translation = self._fast_fallback(cleaned)
                    
                self.latest_translation = translation
            except Exception as e:
                print(f"NMT Error: {e}")
                self.latest_translation = self._fast_fallback(cleaned)
                
            self.request_queue.task_done()

    def translate_gloss_to_english(self, gloss_sequence: str) -> str:
        cleaned = self._clean_glosses(gloss_sequence)
        if not cleaned:
            return ""
            
        # Immediately return fast fallback, and queue background LLM job if new glosses arrived
        if gloss_sequence != self.last_translated_glosses:
            self.last_translated_glosses = gloss_sequence
            self.latest_translation = self._fast_fallback(cleaned)
            # Push to background worker queue (overwrite old pending requests)
            while not self.request_queue.empty():
                try:
                    self.request_queue.get_nowait()
                except queue.Empty:
                    break
            self.request_queue.put(gloss_sequence)
            
        return self.latest_translation

# Singleton instance
nmt_engine = NMTTranslator()
