import React, { useState, useCallback, useRef, useEffect } from 'react';
import CameraFeed from '../components/CameraFeed';
import '../index.css';

// Minimal fallback formatter: does NOT fabricate sentences.
// The backend translator.py provides the authoritative English translation via data.translation.
function formatGlossToEnglish(label) {
  if (!label || label === "WAITING FOR SIGN" || label === "DETECTING..." || label === "UNKNOWN_SIGN") {
    return "";
  }
  // Simple title case — no sentence fabrication
  const clean = label.toLowerCase().replace(/_/g, " ").trim();
  return clean.charAt(0).toUpperCase() + clean.slice(1);
}

function LiveTranslator() {
  const [detectedSign, setDetectedSign] = useState("WAITING FOR SIGN");
  const [uiState, setUiState] = useState("WAITING"); // WAITING, DETECTING, CONFIRMED
  const [confidence, setConfidence] = useState(0);
  const [translation, setTranslation] = useState("");
  const [rawGlosses, setRawGlosses] = useState([]);
  const [trackingStatus, setTrackingStatus] = useState("MediaPipe Active");
  const [backendStatus, setBackendStatus] = useState("Connecting...");
  const [fps, setFps] = useState(30);
  const [soundEnabled, setSoundEnabled] = useState(true);
  const [audioState, setAudioState] = useState("READY"); // READY, SPEAKING, MUTED

  // Refs for stateful debounce, speech prevention, and WebSocket streaming
  const wsRef = useRef(null);
  const isWsConnectedRef = useRef(false);
  const lastConfirmedSignRef = useRef(null);
  const isHttpPendingRef = useRef(false);
  const lastFrameTimeRef = useRef(Date.now());
  const soundEnabledRef = useRef(soundEnabled);
  soundEnabledRef.current = soundEnabled;

  // Robust Non-blocking Speech Synthesis (handles Chromium queue state)
  const speakText = useCallback((text) => {
    if (!soundEnabledRef.current || !('speechSynthesis' in window) || !text) {
      return;
    }
    try {
      // Unfreeze synthesis queue if paused
      if (window.speechSynthesis.paused) {
        window.speechSynthesis.resume();
      }
      window.speechSynthesis.cancel(); // Prevent audio overlap

      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 1.05;
      utterance.pitch = 1.0;
      utterance.lang = 'en-US';

      setAudioState(`SPEAKING: "${text}"`);
      utterance.onend = () => {
        setAudioState(soundEnabledRef.current ? "READY" : "MUTED");
      };
      utterance.onerror = () => {
        setAudioState(soundEnabledRef.current ? "READY" : "MUTED");
      };

      // Asynchronous speak (never blocks render or inference thread)
      setTimeout(() => {
        window.speechSynthesis.speak(utterance);
      }, 10);
    } catch (err) {
      console.warn("TTS Error:", err);
      setAudioState("READY");
    }
  }, []);

  // Process prediction payload from backend (shared by WebSocket and HTTP)
  const handlePredictionPayload = useCallback((data) => {
    if (!data || data.error) return;

    if (data.engine) {
      setBackendStatus(data.engine);
    }

    const label = data.label || "";
    const status = data.status || (data.is_unknown ? "DETECTING" : "CONFIRMED");
    const conf = Number(data.confidence) || 0.0;

    if (status === "CONFIRMED" && !data.is_unknown && label !== "UNKNOWN_SIGN") {
      const formattedLabel = label.toUpperCase();
      setUiState("CONFIRMED");
      setDetectedSign(formattedLabel);
      setConfidence(conf);

      // Instant English Translation from backend or dynamic formatter
      const englishSentence = data.translation || formatGlossToEnglish(label);
      setTranslation(englishSentence);

      // Update confirmed sign history
      setRawGlosses((prev) => {
        if (prev.length === 0 || prev[prev.length - 1] !== formattedLabel) {
          const nextList = [...prev, formattedLabel];
          return nextList.slice(-6);
        }
        return prev;
      });

      // Speak once per confirmed sign event
      if (formattedLabel !== lastConfirmedSignRef.current) {
        lastConfirmedSignRef.current = formattedLabel;
        speakText(englishSentence);
      }
    } else if (status === "DETECTING" || label === "DETECTING...") {
      setUiState("DETECTING");
      setDetectedSign("DETECTING...");
      // Do not clear translation; allow user to read previous confirmed sign
    } else {
      // WAITING FOR SIGN / Hands resting
      setUiState("WAITING");
      setDetectedSign("WAITING FOR SIGN");
      setConfidence(0);
      lastConfirmedSignRef.current = null; // Reset so next gesture triggers speech
    }
  }, [speakText]);

  // Initialize High-Speed WebSocket Connection with Automatic HTTP Fallback
  useEffect(() => {
    let reconnectTimer = null;
    let ws = null;

    const connectWebSocket = () => {
      try {
        ws = new WebSocket("ws://127.0.0.1:8000/ws/predict");
        wsRef.current = ws;

        ws.onopen = () => {
          isWsConnectedRef.current = true;
          setBackendStatus("AC-HWGATE Real-Time (WebSocket)");
        };

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            handlePredictionPayload(data);
          } catch (e) {
            // Ignore parse errors
          }
        };

        ws.onerror = () => {
          isWsConnectedRef.current = false;
          setBackendStatus("AC-HWGATE Active (HTTP Fallback)");
        };

        ws.onclose = () => {
          isWsConnectedRef.current = false;
          reconnectTimer = setTimeout(connectWebSocket, 2000);
        };
      } catch (err) {
        isWsConnectedRef.current = false;
        setBackendStatus("AC-HWGATE Active (HTTP)");
      }
    };

    connectWebSocket();

    return () => {
      if (reconnectTimer) clearTimeout(reconnectTimer);
      if (ws) {
        ws.onclose = null;
        ws.close();
      }
    };
  }, [handlePredictionPayload]);

  // Frame captured by MediaPipe
  const handleFrame = useCallback(async (keypoints, confidences) => {
    const now = Date.now();
    const currentFps = Math.round(1000 / Math.max(1, (now - lastFrameTimeRef.current)));
    lastFrameTimeRef.current = now;
    if (Math.random() < 0.12) setFps(currentFps);

    // If WebSocket is connected and ready, stream over WebSocket (< 2 ms latency)
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ keypoints, confidences }));
      return;
    }

    // Fallback: fast HTTP fetch
    if (isHttpPendingRef.current) return;
    isHttpPendingRef.current = true;

    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ keypoints, confidences })
      });
      const data = await response.json();
      handlePredictionPayload(data);
    } catch (err) {
      setBackendStatus("Backend Reconnecting...");
    } finally {
      isHttpPendingRef.current = false;
    }
  }, [handlePredictionPayload]);

  return (
    <div className="live-translator-container">
      {/* Top Header Bar */}
      <header className="translator-header">
        <div className="title-group">
          <h1 className="main-title">REAL-TIME INDIAN SIGN LANGUAGE TRANSLATOR</h1>
          <span className="architecture-tag">AC-HWGATE • Graph Attention Transformer (Production SOTA)</span>
        </div>

        <div className="controls-group">
          <button 
            className={`audio-toggle-btn ${soundEnabled ? 'active' : ''}`}
            onClick={() => {
              const next = !soundEnabled;
              setSoundEnabled(next);
              if (!next && 'speechSynthesis' in window) {
                window.speechSynthesis.cancel();
              }
              setAudioState(next ? "READY" : "MUTED");
            }}
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
              <path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path>
            </svg>
            {soundEnabled ? 'AUDIO ON' : 'AUDIO MUTED'}
          </button>

          <div className="status-chips">
            <span className="chip online">AC-HWGATE 104-CLASS</span>
            <span className="chip">{fps} FPS</span>
            <span className="chip online">{trackingStatus}</span>
          </div>
        </div>
      </header>

      {/* Two-Column Responsive Layout */}
      <main className="two-column-workspace">
        {/* Left Column: Live Camera Feed */}
        <section className="column-left camera-section">
          <div className="card-box camera-card">
            <div className="card-header">
              <span className="header-label">LIVE CAMERA FEED</span>
              <span className="live-indicator"><span className="pulse-dot"></span>LIVE</span>
            </div>
            <div className="camera-wrapper">
              <CameraFeed onFrameCaptured={handleFrame} onModelStatus={setTrackingStatus} />
            </div>
          </div>
        </section>

        {/* Right Column: Real-Time Detection, Translation, and Audio */}
        <section className="column-right info-section">
          {/* Card 1: Detected Sign */}
          <div className={`card-box sign-card ${uiState.toLowerCase()}`}>
            <div className="card-header">
              <span className="header-label">DETECTED SIGN</span>
              {uiState === "CONFIRMED" && (
                <span className="confidence-badge">{(confidence * 100).toFixed(1)}% Match</span>
              )}
            </div>
            <div className={`sign-display-text ${uiState.toLowerCase()}`}>
              {detectedSign}
            </div>
          </div>

          {/* Card 2: English Translation */}
          <div className="card-box translation-card">
            <div className="card-header">
              <span className="header-label">ENGLISH TRANSLATION</span>
            </div>
            <div className="translation-display-text">
              {translation ? translation : "Perform a sign naturally to see English translation..."}
            </div>
          </div>

          {/* Card 3: Audio Status */}
          <div className="card-box audio-card">
            <div className="card-header">
              <span className="header-label">AUDIO / SPEECH SYNTHESIS</span>
            </div>
            <div className="audio-status-row">
              <div className={`audio-dot ${audioState.startsWith("SPEAKING") ? 'speaking' : soundEnabled ? 'ready' : 'muted'}`}></div>
              <span className="audio-status-text">
                {soundEnabled ? audioState : "AUDIO MUTED (Click button above to enable)"}
              </span>
            </div>
          </div>

          {/* Card 4: Confirmed Gloss History */}
          <div className="card-box gloss-history-card">
            <div className="card-header">
              <span className="header-label">CONFIRMED SIGN SEQUENCE</span>
            </div>
            <div className="gloss-chips-container">
              {rawGlosses.length > 0 ? (
                rawGlosses.map((g, i) => (
                  <span key={i} className="gloss-chip">{g}</span>
                ))
              ) : (
                <span className="gloss-placeholder">---</span>
              )}
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

export default LiveTranslator;
