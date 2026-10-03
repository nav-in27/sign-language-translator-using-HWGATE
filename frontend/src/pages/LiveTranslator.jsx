import React, { useState, useCallback, useRef } from 'react';
import CameraFeed from '../components/CameraFeed';
import '../index.css';

function LiveTranslator() {
  const [gloss, setGloss] = useState("Waiting for sign...");
  const [confidence, setConfidence] = useState(0);
  const [rawGlosses, setRawGlosses] = useState("");
  const [translation, setTranslation] = useState("");
  const [trackingStatus, setTrackingStatus] = useState("MediaPipe Initializing...");
  const [backendStatus, setBackendStatus] = useState("Connecting to AI Engine...");
  const [fps, setFps] = useState(0);
  const [soundEnabled, setSoundEnabled] = useState(false);
  
  const pipelineReady = true;

  // Refs for tracking mutable state without causing re-renders
  const lastSpokenWord = useRef("");
  const sentenceTimeoutRef = useRef(null);
  const lastFrameTime = useRef(Date.now());
  const isPredicting = useRef(false);

  const speakText = (text) => {
    if (soundEnabled && 'speechSynthesis' in window) {
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 1.0;
      window.speechSynthesis.speak(utterance);
    }
  };

  const handleFrame = useCallback(async (keypoints, confidences) => {
    // Calculate FPS
    const now = Date.now();
    const currentFps = Math.round(1000 / (now - lastFrameTime.current));
    lastFrameTime.current = now;
    if (Math.random() < 0.15) setFps(currentFps);

    if (!pipelineReady || isPredicting.current) return;

    isPredicting.current = true;
    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ keypoints, confidences })
      });
      
      const data = await response.json();
      
      if (data.error) {
        console.error("Backend Predict Error:", data.error);
        setBackendStatus("AI Engine Error");
        isPredicting.current = false;
        return;
      }
      
      setBackendStatus("AI Engine Active (ONNX)");
      
      const specialLabels = ["Buffering...", "No Sign Detected", "UNKNOWN SIGN"];
      if (specialLabels.includes(data.label)) {
        setGloss(data.label);
        setConfidence(data.confidence || 0);
      } else {
        setGloss(data.label);
        setConfidence(data.confidence || 0);
      }
      
      // Update UI with Backend Translation Output
      if (data.raw_glosses !== undefined) {
        setRawGlosses(data.raw_glosses);
      }
      if (data.translation !== undefined) {
        setTranslation(data.translation);
        
        // Audio Speech synthesis for full translated sentences
        if (data.translation && data.translation !== lastSpokenWord.current) {
           lastSpokenWord.current = data.translation;
           
           if (sentenceTimeoutRef.current) {
             clearTimeout(sentenceTimeoutRef.current);
           }
           
           sentenceTimeoutRef.current = setTimeout(() => {
               speakText(data.translation);
               lastSpokenWord.current = "";
           }, 800);
        }
      }
      
    } catch (err) {
      console.error("Backend Connection Error:", err);
      setBackendStatus("AI Engine Offline (Run start_project.py)");
    } finally {
      isPredicting.current = false;
    }
  }, [pipelineReady]);

  return (
    <div className="translator-layout">
      {/* Header Bar */}
      <div className="header-bar">
        <h1 className="header-title">
          ISL Real-Time Sign Language Translator
        </h1>
        <div className="header-controls">
          <button className={`btn-neon ${soundEnabled ? 'active' : ''}`} onClick={() => setSoundEnabled(!soundEnabled)}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path></svg>
            {soundEnabled ? 'AUDIO ON' : 'ENABLE AUDIO'}
          </button>
          
          <div className="status-badges">
            <span className="badge">FPS: {fps}</span>
            <span className={`badge ${trackingStatus.includes('Active') ? 'online' : 'offline'}`}>{trackingStatus}</span>
            <span className={`badge ${backendStatus.includes('Active') ? 'online' : 'offline'}`}>{backendStatus}</span>
          </div>
        </div>
      </div>
      
      {/* Main Video View */}
      <div className="camera-glass-container">
        <div className="glass-panel">
          <CameraFeed onFrameCaptured={handleFrame} onModelStatus={setTrackingStatus} />
        </div>
      </div>

      {/* Translation Results Cards */}
      <div className="translation-results-grid">
        <div className="result-box primary-sign">
          <div className="result-label">
            <span>Detected Sign</span>
            <span className="conf-tag">{(confidence * 100).toFixed(1)}% Match</span>
          </div>
          <div className="result-value highlighted">{gloss}</div>
        </div>

        <div className="result-box translation-box">
          <div className="result-label">English Sentence Translation</div>
          <div className="result-value translation-text">
            {translation.length > 0 ? translation : "Perform signs to translate..."}
          </div>
        </div>

        <div className="result-box gloss-box">
          <div className="result-label">Sign Gloss Sequence</div>
          <div className="result-value gloss-text">
            {rawGlosses.length > 0 ? rawGlosses : "---"}
          </div>
        </div>
      </div>
    </div>
  );
}

export default LiveTranslator;
