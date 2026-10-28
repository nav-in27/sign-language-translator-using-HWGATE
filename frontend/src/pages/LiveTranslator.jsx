import React, { useState, useCallback, useRef, useEffect } from 'react';
import CameraFeed from '../components/CameraFeed';
import { API_BASE_URL, WS_BASE_URL } from '../config';
import '../index.css';

// Minimal fallback formatter: does NOT fabricate sentences.
// The backend translator.py provides the authoritative English translation via data.translation.
function formatGlossToEnglish(label) {
  if (!label || label === "WAITING FOR SIGN" || label === "DETECTING..." || label === "UNKNOWN_SIGN") {
    return "";
  }
  const clean = label.toLowerCase().replace(/_/g, " ").trim();
  return clean.charAt(0).toUpperCase() + clean.slice(1);
}

function normalizeRecognitionState(state) {
  const normalized = String(state || "WAITING").toUpperCase();
  if (normalized === "COLLECTING_FRAMES" || normalized === "BUFFERING") return "LISTENING";
  if (normalized === "RECOGNIZING" || normalized === "PROVISIONAL") return "RECOGNIZING";
  if (normalized === "WAITING" || normalized === "NO_SIGN") return "READY";
  return normalized;
}

function LiveTranslator() {
  // 1. Current recognition state (READY, LISTENING, RECOGNIZING, CONFIRMED, UNCERTAIN)
  const [recognitionState, setRecognitionState] = useState("READY");

  // 2. Current buffer progress
  const [bufferProgress, setBufferProgress] = useState({ length: 0, required: 16, percentage: 0 });

  // 3. Current provisional prediction
  const [currentProvisional, setCurrentProvisional] = useState({ sign: null, confidence: 0 });

  // 4. Current confirmed sign
  const [currentConfirmed, setCurrentConfirmed] = useState({
    sign: null,
    confidence: 0,
    eventId: null,
    timestamp: null
  });

  // 5. Confirmed-sign history (list of unique confirmed sign tokens)
  const [confirmedHistory, setConfirmedHistory] = useState([]);

  // 6. Current translation (only set for the current confirmed sign window)
  const [currentTranslation, setCurrentTranslation] = useState("");

  // 7. Historical translations (list of { eventId, sign, translation, timestamp })
  const [historicalTranslations, setHistoricalTranslations] = useState([]);

  // 8. Speech status (READY, SPEAKING: "...", MUTED, ERROR)
  const [speechStatus, setSpeechStatus] = useState("READY");

  // Diagnostic and telemetry state
  const [trackingStatus, setTrackingStatus] = useState("MediaPipe Active");
  const [backendStatus, setBackendStatus] = useState("Connecting...");
  const [fps, setFps] = useState(30);
  const [soundEnabled, setSoundEnabled] = useState(true);
  const [modelStatus, setModelStatus] = useState({ ready: null });
  const [topPredictions, setTopPredictions] = useState([]);
  const [rejectionReason, setRejectionReason] = useState('');
  const [cameraDiagnostics, setCameraDiagnostics] = useState({ fps: 0, handsDetected: false, poseDetected: false });

  // Refs for stateful debounce, sequence ID tracking, and speech prevention
  const wsRef = useRef(null);
  const isWsConnectedRef = useRef(false);
  const sessionIdRef = useRef(
    typeof crypto !== 'undefined' && crypto.randomUUID
      ? crypto.randomUUID()
      : `sess_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`
  );
  const lastProcessedSessionIdRef = useRef(null);
  const lastProcessedSeqIdRef = useRef(0);
  const lastProcessedWindowIdRef = useRef(null);
  const lastSpokenEventIdRef = useRef(null);
  const lastConfirmedEventIdRef = useRef(null);
  const isHttpPendingRef = useRef(false);
  const lastFrameTimeRef = useRef(Date.now());
  const socketGenerationRef = useRef(0);
  const soundEnabledRef = useRef(soundEnabled);
  soundEnabledRef.current = soundEnabled;

  useEffect(() => {
    fetch(`${API_BASE_URL}/model/status`, { cache: 'no-store' })
      .then((response) => response.json())
      .then((data) => {
        setModelStatus(data);
        setBackendStatus(data.ready ? `${data.engine} • ${data.num_classes} classes` : "Model unavailable");
      })
      .catch(() => {
        setModelStatus({ ready: false, load_error: "Model status endpoint unavailable" });
        setBackendStatus("Backend unavailable");
      });
  }, []);

  // Documented state reset helper
  const clearConfirmedState = useCallback((reason) => {
    console.log(`[LiveTranslator] Cleared confirmed state. Reason: ${reason}`);
    setCurrentConfirmed({ sign: null, confidence: 0, eventId: null, timestamp: null });
    setCurrentTranslation("");
  }, []);

  // Robust Non-blocking Speech Synthesis (strictly tied to unique event_id)
  const speakText = useCallback((text, eventId) => {
    if (!soundEnabledRef.current || !('speechSynthesis' in window) || !text) {
      return;
    }
    try {
      if (window.speechSynthesis.paused) {
        window.speechSynthesis.resume();
      }
      window.speechSynthesis.cancel(); // Cancel obsolete or previous audio

      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 1.0;
      utterance.pitch = 1.0;
      utterance.lang = 'en-US';

      setSpeechStatus(`SPEAKING: "${text}"`);
      utterance.onend = () => {
        setSpeechStatus(soundEnabledRef.current ? "READY" : "MUTED");
      };
      utterance.onerror = (e) => {
        console.warn("[TTS Error]", e);
        setSpeechStatus(soundEnabledRef.current ? "READY" : "MUTED");
      };

      // Asynchronous speak (never blocks render or inference capture thread)
      setTimeout(() => {
        window.speechSynthesis.speak(utterance);
      }, 0);
    } catch (err) {
      console.warn("TTS Exception:", err);
      setSpeechStatus("READY");
    }
  }, []);

  // Process prediction payload from backend (shared by WebSocket and HTTP)
  const handlePredictionPayload = useCallback((data) => {
    if (!data) return;
    if (data.error) {
      setRecognitionState("ERROR");
      setBackendStatus(`Backend error: ${data.error}`);
      return;
    }

    // Associate responses with session and window IDs; ignore stale responses that arrive out of order
    if (data.session_id) {
      if (lastProcessedSessionIdRef.current && data.session_id !== lastProcessedSessionIdRef.current) {
        console.log(`[LiveTranslator] Backend session changed from ${lastProcessedSessionIdRef.current} to ${data.session_id}; resetting seq tracker`);
        lastProcessedSeqIdRef.current = 0;
      }
      lastProcessedSessionIdRef.current = data.session_id;
    }

    // Discard stale out-of-order sequence responses
    if (data.seq_id && data.seq_id <= lastProcessedSeqIdRef.current) {
      return;
    }
    if (data.seq_id) {
      lastProcessedSeqIdRef.current = data.seq_id;
    }
    if (data.window_id !== undefined && data.window_id !== null) {
      lastProcessedWindowIdRef.current = data.window_id;
    }

    if (data.engine) {
      setBackendStatus(data.engine);
    }

    // Update temporal buffer progress
    const bufLen = Number(data.buffer_length ?? data.live_diagnostics?.buffer_length ?? 0);
    const bufReq = Number(data.buffer_required ?? data.live_diagnostics?.required_frames ?? 16);
    const bufPct = Math.min(100, Math.round((bufLen / Math.max(1, bufReq)) * 100));
    setBufferProgress({ length: bufLen, required: bufReq, percentage: bufPct });

    setTopPredictions(Array.isArray(data.top_predictions) ? data.top_predictions : []);
    setRejectionReason(data.rejection_reason || '');

    const rawState = data.state || data.status || "WAITING";
    const state = normalizeRecognitionState(rawState);
    const conf = Number(data.confidence) || 0.0;
    setRecognitionState(state);

    if (state === "CONFIRMED" && !data.is_unknown && data.confirmed_sign) {
      const activeSign = data.confirmed_sign.toUpperCase();
      const translationText = typeof data.translation === "string" ? data.translation.trim() : formatGlossToEnglish(activeSign);
      const eventId = data.event_id || `evt_${activeSign}_${Date.now()}`;

      // Update current confirmed sign and translation
      setCurrentConfirmed({
        sign: activeSign,
        confidence: conf,
        eventId: eventId,
        timestamp: Date.now()
      });
      setCurrentTranslation(translationText);
      setCurrentProvisional({ sign: null, confidence: 0 });

      // Update confirmed history and historical translations strictly on unique event_id
      if (eventId !== lastConfirmedEventIdRef.current) {
        lastConfirmedEventIdRef.current = eventId;

        setConfirmedHistory((prev) => {
          if (prev.length > 0 && prev[prev.length - 1] === activeSign) {
            return prev;
          }
          return [...prev, activeSign].slice(-8);
        });

        if (translationText) {
          setHistoricalTranslations((prev) => {
            if (prev.length > 0 && prev[prev.length - 1].eventId === eventId) {
              return prev;
            }
            return [...prev, {
              eventId,
              sign: activeSign,
              translation: translationText,
              timestamp: Date.now()
            }].slice(-6);
          });
        }

        // Non-blocking TTS speech: strictly once per unique confirmed event_id
        if (translationText && eventId !== lastSpokenEventIdRef.current) {
          lastSpokenEventIdRef.current = eventId;
          speakText(translationText, eventId);
        }
      }
    } else if (state === "RECOGNIZING") {
      // Only set provisional if backend returned a calibrated provisional_sign (never fall back to random argmax noise)
      if (data.provisional_sign) {
        setCurrentProvisional({
          sign: data.provisional_sign.toUpperCase(),
          confidence: conf
        });
      } else {
        setCurrentProvisional({ sign: null, confidence: 0 });
      }
    } else {
      // LISTENING, READY, UNCERTAIN
      setCurrentProvisional({ sign: null, confidence: 0 });
    }
  }, [speakText]);

  // Initialize High-Speed WebSocket Connection with Automatic HTTP Fallback
  useEffect(() => {
    let reconnectTimer = null;
    let ws = null;

    const connectWebSocket = () => {
      const generation = socketGenerationRef.current + 1;
      socketGenerationRef.current = generation;
      try {
        ws = new WebSocket(`${WS_BASE_URL}/ws/predict`);
        wsRef.current = ws;

        ws.onopen = () => {
          if (generation !== socketGenerationRef.current) return;
          lastProcessedSeqIdRef.current = 0;
          setRecognitionState("NO_SIGN");
          setCurrentProvisional({ sign: null, confidence: 0 });
          clearConfirmedState("WebSocket connection initialized / new session started");
          setConfirmedHistory([]);
          setHistoricalTranslations([]);
          setBufferProgress({ length: 0, required: 16, percentage: 0 });
          setTopPredictions([]);
          setRejectionReason("");
          lastSpokenEventIdRef.current = null;
          lastConfirmedEventIdRef.current = null;
          isWsConnectedRef.current = true;
          setBackendStatus("AC-HWGATE Real-Time (WebSocket)");
        };

        ws.onmessage = (event) => {
          if (generation !== socketGenerationRef.current || ws !== wsRef.current) return;
          try {
            const data = JSON.parse(event.data);
            handlePredictionPayload(data);
          } catch (e) {
            console.error("[LiveTranslator] Invalid backend message", e);
            setRecognitionState("ERROR");
            setBackendStatus("Backend message error");
          }
        };

        ws.onerror = () => {
          isWsConnectedRef.current = false;
          setRecognitionState("ERROR");
          setBackendStatus("WebSocket error; HTTP fallback");
        };

        ws.onclose = () => {
          isWsConnectedRef.current = false;
          setRecognitionState("ERROR");
          setBackendStatus("Backend disconnected; reconnecting");
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
  }, [handlePredictionPayload, clearConfirmedState]);

  // Frame captured by MediaPipe
  const handleFrame = useCallback(async (keypoints, confidences, seqId, timestamp) => {
    const now = Date.now();
    const currentFps = Math.round(1000 / Math.max(1, (now - lastFrameTimeRef.current)));
    lastFrameTimeRef.current = now;
    if (Math.random() < 0.12) setFps(currentFps);

    const payload = {
      keypoints,
      confidences,
      seq_id: seqId || 0,
      timestamp: timestamp || (now / 1000.0),
      session_id: sessionIdRef.current,
      trace_id: window.crypto?.randomUUID ? window.crypto.randomUUID() : `${Date.now()}-${seqId}`
    };

    // If WebSocket is connected and ready, stream over WebSocket (< 2 ms latency)
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(payload));
      return;
    }

    // Fallback: fast HTTP fetch
    if (isHttpPendingRef.current) return;
    isHttpPendingRef.current = true;

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await response.json();
      if (data.error) setBackendStatus(`Backend error: ${data.error}`);
      handlePredictionPayload(data);
    } catch (err) {
      setRecognitionState("ERROR");
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
              setSpeechStatus(next ? "READY" : "MUTED");
            }}
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
              <path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path>
            </svg>
            {soundEnabled ? 'AUDIO ON' : 'AUDIO MUTED'}
          </button>

          <div className="status-chips">
            <span className="chip online">AC-HWGATE 136-CLASS</span>
            <span className={`chip ${modelStatus?.ready === true ? 'online' : ''}`}>
              {modelStatus?.ready === true
                ? `MODEL READY • ${modelStatus.artifact_kind?.toUpperCase()}`
                : modelStatus?.ready === false ? "MODEL FAILED" : "MODEL LOADING"}
            </span>
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
              <CameraFeed onFrameCaptured={handleFrame} onModelStatus={setTrackingStatus} onFrameTelemetry={setCameraDiagnostics} />
            </div>
          </div>
        </section>

        {/* Right Column: Real-Time Detection, Translation, and Audio */}
        <section className="column-right info-section">
          {/* Card 1: Detected Sign */}
          <div className={`card-box sign-card ${recognitionState.toLowerCase()}`}>
            <div className="card-header">
              <span className="header-label">DETECTED SIGN</span>
              <div className="header-badges">
                <span className={`state-badge ${recognitionState.toLowerCase()}`}>
                  {recognitionState}
                </span>
                {recognitionState === "CONFIRMED" && currentConfirmed.confidence > 0 && (
                  <span className="confidence-badge">{(currentConfirmed.confidence * 100).toFixed(1)}% Match</span>
                )}
                {recognitionState === "RECOGNIZING" && currentProvisional.confidence > 0 && (
                  <span className="eval-badge">Recognizing {(currentProvisional.confidence * 100).toFixed(0)}%</span>
                )}
                {recognitionState === "LISTENING" && (
                  <span className="eval-badge">Listening</span>
                )}
              </div>
            </div>
            <div className={`sign-display-text ${recognitionState.toLowerCase()}`}>
              {recognitionState === "CONFIRMED" && (currentConfirmed.sign || "CONFIRMED")}
              {recognitionState === "RECOGNIZING" && (currentProvisional.sign ? currentProvisional.sign : "Recognizing...")}
              {recognitionState === "LISTENING" && "Listening..."}
              {recognitionState === "UNCERTAIN" && "Uncertain"}
              {recognitionState === "READY" && "Ready — perform sign"}
              {recognitionState === "ERROR" && "BACKEND ERROR"}
              {recognitionState === "INITIALIZING" && "INITIALIZING..."}
            </div>
          </div>

          {topPredictions.length > 0 && recognitionState !== "CONFIRMED" && (
            <div className="prediction-diagnostics" aria-live="polite">
              <div className="diagnostic-title">LIVE MODEL CANDIDATES</div>
              <div className="prediction-list">
                {topPredictions.slice(0, 3).map((prediction) => (
                  <span className="prediction-pill" key={prediction.class}>
                    {prediction.class.replace(/_/g, ' ')} {(Number(prediction.prob) * 100).toFixed(1)}%
                  </span>
                ))}
              </div>
              {rejectionReason && <div className="rejection-reason">{rejectionReason}</div>}
              <div className="rejection-reason">
                Camera {cameraDiagnostics.fps || "…"} FPS · pose {cameraDiagnostics.poseDetected ? "detected" : "missing"} · hands {cameraDiagnostics.handsDetected ? "detected" : "missing"} · status {recognitionState.toLowerCase()}
              </div>
            </div>
          )}

          {/* Card 2: English Translation */}
          <div className={`card-box translation-card ${recognitionState.toLowerCase()}`}>
            <div className="card-header">
              <span className="header-label">ENGLISH TRANSLATION</span>
              {recognitionState === "CONFIRMED" && (
                <span className="active-tag verified">CURRENT CONFIRMED</span>
              )}
            </div>
            <div className="translation-content-area">
              {recognitionState === "CONFIRMED" && currentTranslation ? (
                <div className="current-confirmed-translation">
                  <div className="confirmed-translation-text">{currentTranslation}</div>
                </div>
              ) : recognitionState === "UNCERTAIN" ? (
                <div className="uncertain-translation-prompt">
                  <span className="uncertain-icon">⚠️</span>
                  <span className="uncertain-text">Uncertain — please repeat sign clearly</span>
                </div>
              ) : recognitionState === "LISTENING" ? (
                <div className="ongoing-state-prompt">
                  Listening for sign gesture...
                </div>
              ) : recognitionState === "RECOGNIZING" ? (
                <div className="ongoing-state-prompt">
                  Recognizing active sign gesture...
                </div>
              ) : (
                <div className="ongoing-state-prompt">
                  Perform a sign naturally to see English translation...
                </div>
              )}
            </div>
          </div>

          {/* Card 3: Audio Status */}
          <div className="card-box audio-card">
            <div className="card-header">
              <span className="header-label">AUDIO / SPEECH SYNTHESIS</span>
            </div>
            <div className="audio-status-row">
              <div className={`audio-dot ${speechStatus.startsWith("SPEAKING") ? 'speaking' : soundEnabled ? 'ready' : 'muted'}`}></div>
              <span className="audio-status-text">
                {soundEnabled ? speechStatus : "AUDIO MUTED (Click button above to enable)"}
              </span>
            </div>
          </div>

          {/* Card 4: Confirmed Gloss History */}
          <div className="card-box gloss-history-card">
            <div className="card-header">
              <span className="header-label">CONFIRMED SIGN SEQUENCE</span>
              {confirmedHistory.length > 0 && (
                <span className="history-count-badge">{confirmedHistory.length} signs</span>
              )}
            </div>
            <div className="gloss-chips-container">
              {confirmedHistory.length > 0 ? (
                confirmedHistory.map((g, i) => (
                  <span key={`${g}-${i}`} className="gloss-chip">{g}</span>
                ))
              ) : (
                <span className="gloss-placeholder">---</span>
              )}
            </div>
            {historicalTranslations.length > 0 && (
              <div className="previous-confirmed-banner">
                <span className="previous-tag">LATEST CONFIRMED TRANSLATION</span>
                <span className="previous-value">
                  "{historicalTranslations[historicalTranslations.length - 1].translation}"
                </span>
              </div>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}

export default LiveTranslator;
