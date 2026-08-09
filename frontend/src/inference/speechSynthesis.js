/**
 * Text-to-Speech via Web Speech API
 * 
 * This module converts the final translated natural language sentence into audio.
 * It will target Indian English (en-IN) or regional languages (e.g. hi-IN) based on user preference.
 */

export const speakText = (text, lang = 'en-IN') => {
  if (!window.speechSynthesis) {
    console.warn("Web Speech API not supported in this browser.");
    return;
  }
  
  // Cancel any ongoing speech to avoid overlapping audio
  window.speechSynthesis.cancel();
  
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = lang;
  utterance.rate = 1.0;
  utterance.pitch = 1.0;
  
  window.speechSynthesis.speak(utterance);
};
