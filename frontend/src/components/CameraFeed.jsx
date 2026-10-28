import React, { useRef, useEffect, useState } from 'react';

const CameraFeed = ({ onFrameCaptured, onModelStatus, onFrameTelemetry }) => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const frameSeqIdRef = useRef(0);
  const lastFrameTimestampRef = useRef(0);
  const [isStreaming, setIsStreaming] = useState(false);

  const get133Keypoints = (results) => {
    let keypoints = [];
    let confidences = [];
    
    // 1. Pose (33)
    if (results.poseLandmarks) {
      for (let i=0; i<33; i++) {
        const lm = results.poseLandmarks[i];
        keypoints.push([lm.x, lm.y, lm.z || 0.0]);
        confidences.push([lm.visibility ?? 1.0]);
      }
    } else {
      for (let i=0; i<33; i++) { keypoints.push([0, 0, 0]); confidences.push([0]); }
    }
    
    // 2. Left Hand (21)
    if (results.leftHandLandmarks) {
      for (let i=0; i<21; i++) {
        const lm = results.leftHandLandmarks[i];
        keypoints.push([lm.x, lm.y, lm.z || 0.0]);
        confidences.push([1.0]);
      }
    } else {
      for (let i=0; i<21; i++) { keypoints.push([0, 0, 0]); confidences.push([0]); }
    }
    
    // 3. Right Hand (21)
    if (results.rightHandLandmarks) {
      for (let i=0; i<21; i++) {
        const lm = results.rightHandLandmarks[i];
        keypoints.push([lm.x, lm.y, lm.z || 0.0]);
        confidences.push([1.0]);
      }
    } else {
      for (let i=0; i<21; i++) { keypoints.push([0, 0, 0]); confidences.push([0]); }
    }
    
    // 4. Face (58)
    if (results.faceLandmarks) {
      for (let i=0; i<58; i++) {
        const lm = results.faceLandmarks[i];
        keypoints.push([lm.x, lm.y, lm.z || 0.0]);
        confidences.push([1.0]);
      }
    } else {
      for (let i=0; i<58; i++) { keypoints.push([0, 0, 0]); confidences.push([0]); }
    }
    
    return { keypoints, confidences };
  };

  useEffect(() => {
    let camera = null;
    let holistic = null;
    let mounted = true;
    let isProcessing = false;
    
    // Wait for globals to load from CDN
    const checkGlobals = setInterval(() => {
      if (!mounted) {
        clearInterval(checkGlobals);
        return;
      }
      if (window.Holistic && window.Camera) {
        clearInterval(checkGlobals);
        startMediaPipe();
      }
    }, 100);

    const startMediaPipe = () => {
      if (!mounted) return;
      try {
        holistic = new window.Holistic({locateFile: (file) => {
          return `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`;
        }});
        
        holistic.setOptions({
          modelComplexity: 1,
          smoothLandmarks: true,
          enableSegmentation: false,
          smoothSegmentation: false,
          refineFaceLandmarks: false,
          minDetectionConfidence: 0.5,
          minTrackingConfidence: 0.5
        });
        
        holistic.onResults((results) => {
          if (!mounted) return;
          if (onModelStatus) onModelStatus("MediaPipe Active");
          
          if (!canvasRef.current) return;
          const canvasCtx = canvasRef.current.getContext('2d');
          canvasCtx.save();
          canvasCtx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
          
          // Draw the video frame
          canvasCtx.drawImage(results.image, 0, 0, canvasRef.current.width, canvasRef.current.height);
          
          // Draw Skeleton overlays with high-contrast vibrant colors
          if (results.poseLandmarks) {
            window.drawConnectors(canvasCtx, results.poseLandmarks, window.POSE_CONNECTIONS, {color: '#ccff00', lineWidth: 4});
            window.drawLandmarks(canvasCtx, results.poseLandmarks, {color: '#ff0055', lineWidth: 3, radius: 4});
          }
          if (results.leftHandLandmarks) {
            window.drawConnectors(canvasCtx, results.leftHandLandmarks, window.HAND_CONNECTIONS, {color: '#00f3ff', lineWidth: 4});
            window.drawLandmarks(canvasCtx, results.leftHandLandmarks, {color: '#ccff00', lineWidth: 2, radius: 3});
          }
          if (results.rightHandLandmarks) {
            window.drawConnectors(canvasCtx, results.rightHandLandmarks, window.HAND_CONNECTIONS, {color: '#00f3ff', lineWidth: 4});
            window.drawLandmarks(canvasCtx, results.rightHandLandmarks, {color: '#ff0055', lineWidth: 2, radius: 3});
          }
          
          canvasCtx.restore();
          
          // Send keypoints to parent for backend inference with monotonic sequence ID & timestamp
          const { keypoints, confidences } = get133Keypoints(results);
          frameSeqIdRef.current += 1;
          const timestamp = Date.now() / 1000.0;
          const frameIntervalMs = lastFrameTimestampRef.current ? (timestamp * 1000) - lastFrameTimestampRef.current : 0;
          lastFrameTimestampRef.current = timestamp * 1000;
          if (onFrameTelemetry) {
            onFrameTelemetry({
              seqId: frameSeqIdRef.current,
              timestamp,
              fps: frameIntervalMs > 0 ? Math.round(1000 / frameIntervalMs) : 0,
              poseDetected: Boolean(results.poseLandmarks),
              handsDetected: Boolean(results.leftHandLandmarks || results.rightHandLandmarks),
            });
          }
          
          // Send keypoints whenever tracking has keypoint data
          onFrameCaptured(keypoints, confidences, frameSeqIdRef.current, timestamp);
        });

        if (videoRef.current && mounted) {
          camera = new window.Camera(videoRef.current, {
            onFrame: async () => {
              if (!mounted || !videoRef.current || isProcessing) return;
              isProcessing = true;
              try {
                await holistic.send({image: videoRef.current});
              } catch (frameErr) {
                // Ignore transient frame errors
              } finally {
                isProcessing = false;
              }
            },
            width: 640,
            height: 480
          });
          
          camera.start().then(() => {
            if (mounted) setIsStreaming(true);
          }).catch((camErr) => {
            console.error("Camera start error:", camErr);
            if (onModelStatus) onModelStatus("Camera Error");
          });
        }
      } catch (initErr) {
        console.error("MediaPipe initialization error:", initErr);
        if (onModelStatus) onModelStatus("MediaPipe Error");
      }
    };

    return () => {
      mounted = false;
      clearInterval(checkGlobals);
      if (camera) {
        try { camera.stop(); } catch (e) {}
      }
      if (holistic) {
        try { holistic.close(); } catch (e) {}
      }
    };
  }, []);

  return (
    <div className="camera-container">
      {/* Hidden original video feed */}
      <video 
        ref={videoRef} 
        style={{ display: 'none' }}
      />
      
      {/* Visible high-res canvas with skeletal drawing */}
      <canvas 
        ref={canvasRef} 
        width={640} 
        height={480} 
        className="main-camera-canvas"
      />
      {!isStreaming && (
        <div className="camera-loading-overlay">
          <div className="spinner"></div>
          <p className="loading-text">Starting High-Definition Camera Stream...</p>
        </div>
      )}
    </div>
  );
};

export default CameraFeed;

