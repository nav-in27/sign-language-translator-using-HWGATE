import React, { useState, useCallback, useRef } from 'react';
import CameraFeed from '../components/CameraFeed';
import '../index.css';

function LiveTranslator() {
  const [gloss, setGloss] = useState("Waiting for sign...");
  const [confidence, setConfidence] = useState(0);
  const [rawGlosses, setRawGlosses] = useState("");
  const [translation, setTranslation] = useState("");
  const [modelStatus, setModelStatus] = useState("Initializing...");
  const [fps, setFps] = useState(0);
  const [soundEnabled, setSoundEnabled] = useState(false);
  
  const pipelineReady = true;

  // Refs for tracking mutable state without causing re-renders
  const lastPredictions = useRef([]);
  const lastSpokenWord = useRef("");
  const sentenceTimeoutRef = useRef(null);
  const lastFrameTime = useRef(Date.now());

  const speakText = (text) => {
    if (soundEnabled && 'speechSynthesis' in window) {
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 1.0;
      window.speechSynthesis.speak(utterance);
    }
  };

  const isPredicting = useRef(false);

  const handleFrame = useCallback(async (keypoints, confidences) => {
    // Calculate FPS on every frame arrival
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
        isPredicting.current = false;
        return;
      }
      
      setModelStatus("Backend Connected");
      
      const specialLabels = ["Buffering...", "No Sign Detected"];
      if (specialLabels.includes(data.label)) {
        setGloss(data.label);
        setConfidence(0);
      } else {
        // Direct real-time prediction display for low latency
        setGloss(data.label);
        setConfidence(data.confidence);
      }
      
      // Update UI with Backend NMT Engine Output
      if (data.raw_glosses !== undefined) {
        setRawGlosses(data.raw_glosses);
      }
      if (data.translation !== undefined) {
        setTranslation(data.translation);
        
        // Handle Speech for full translated sentences
        if (data.translation && data.translation !== lastSpokenWord.current) {
           lastSpokenWord.current = data.translation;
           
           // Clear any existing timeout
           if (sentenceTimeoutRef.current) {
             clearTimeout(sentenceTimeoutRef.current);
           }
           
           sentenceTimeoutRef.current = setTimeout(() => {
               speakText(data.translation);
               lastSpokenWord.current = "";
           }, 1000); // speak if stable for 1s
        }
      }
      
    } catch (err) {
      console.error("Backend Error:", err);
      setModelStatus("Backend Disconnected");
    } finally {
      isPredicting.current = false;
    }
  }, [pipelineReady]);

  return (
    <div className="translator-layout">
      <div className="hero-text-section">
        <h1 className="hero-title">
          <span className="highlight">COMMUNICATE</span>
          <span className="secondary">BEYOND</span>
        </h1>
        <p className="hero-subtitle">
          The next evolution of human-computer interaction. Raw neural translation powering real-time expression. Elite. Unfiltered.
        </p>
        
        <div style={{ display: 'flex', gap: '1rem', alignItems: 'center', flexWrap: 'wrap' }}>
          <button className="btn-neon" onClick={() => {
            setSoundEnabled(!soundEnabled);
          }}>
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polygon points="23 7 16 12 23 17 23 7"></polygon><rect x="1" y="5" width="15" height="14" rx="2" ry="2"></rect></svg>
            {soundEnabled ? 'SOUND ACTIVE' : 'CONNECT AUDIO'}
          </button>
          
          <div style={{ display: 'flex', gap: '1rem', color: 'var(--text-secondary)', fontSize: '0.9rem', padding: '0 1rem' }}>
            <span>FPS: {fps}</span>
            <span>|</span>
            <span style={{ color: modelStatus.includes('Connected') ? 'var(--accent-neon)' : 'inherit' }}>{modelStatus}</span>
          </div>
        </div>
        
        <div className="translation-results" style={{ marginTop: '1.5rem' }}>
          <div className="result-box">
            <div className="result-label">Current Sign <span style={{ float: 'right', color: 'var(--accent-neon)' }}>{(confidence * 100).toFixed(1)}%</span></div>
            <div className="result-value highlighted">{gloss}</div>
          </div>
          <div className="result-box">
            <div className="result-label">Natural English Translation (Qwen NMT)</div>
            <div className="result-value" style={{ color: 'var(--accent-neon)' }}>
              {translation.length > 0 ? translation : "..."}
            </div>
          </div>
          <div className="result-box">
            <div className="result-label">Raw Sign Glosses (CTC Decoder)</div>
            <div className="result-value" style={{ fontSize: '1rem', color: 'var(--text-secondary)' }}>
              {rawGlosses.length > 0 ? rawGlosses.toUpperCase() : "..."}
            </div>
          </div>
        </div>
      </div>
      
      <div className="camera-glass-container">
        <div className="glass-panel">
          <CameraFeed onFrameCaptured={handleFrame} onModelStatus={setModelStatus} />
        </div>
      </div>
    </div>
  );
}

export default LiveTranslator;
