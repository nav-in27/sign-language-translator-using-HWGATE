import React, { useRef, useEffect, useState } from 'react';

const CameraFeed = ({ onFrameCaptured, onModelStatus }) => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [isStreaming, setIsStreaming] = useState(false);

  const get133Keypoints = (results) => {
    let keypoints = [];
    let confidences = [];
    
    // 1. Pose (33)
    if (results.poseLandmarks) {
      for (let i=0; i<33; i++) {
        const lm = results.poseLandmarks[i];
        keypoints.push([lm.x, lm.y, lm.z || 0.0]);
        confidences.push([lm.visibility || 1.0]);
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
    
    // Wait for globals to load from CDN
    const checkGlobals = setInterval(() => {
      if (window.Holistic && window.Camera) {
        clearInterval(checkGlobals);
        startMediaPipe();
      }
    }, 100);

    const startMediaPipe = () => {
      const holistic = new window.Holistic({locateFile: (file) => {
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
        if (onModelStatus) onModelStatus("MediaPipe Active");
        
        const canvasCtx = canvasRef.current.getContext('2d');
        canvasCtx.save();
        canvasCtx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
        
        // Draw the video frame
        canvasCtx.drawImage(results.image, 0, 0, canvasRef.current.width, canvasRef.current.height);
        
        // Draw Skeleton
        if (results.poseLandmarks) {
          window.drawConnectors(canvasCtx, results.poseLandmarks, window.POSE_CONNECTIONS, {color: '#00FF00', lineWidth: 4});
          window.drawLandmarks(canvasCtx, results.poseLandmarks, {color: '#FF0000', lineWidth: 2});
        }
        if (results.leftHandLandmarks) {
          window.drawConnectors(canvasCtx, results.leftHandLandmarks, window.HAND_CONNECTIONS, {color: '#CC0000', lineWidth: 5});
          window.drawLandmarks(canvasCtx, results.leftHandLandmarks, {color: '#00FF00', lineWidth: 2});
        }
        if (results.rightHandLandmarks) {
          window.drawConnectors(canvasCtx, results.rightHandLandmarks, window.HAND_CONNECTIONS, {color: '#00CC00', lineWidth: 5});
          window.drawLandmarks(canvasCtx, results.rightHandLandmarks, {color: '#FF0000', lineWidth: 2});
        }
        
        canvasCtx.restore();
        
        // Send keypoints to App.jsx for backend inference
        const { keypoints, confidences } = get133Keypoints(results);
        
        // Only send if we detect a body or hands
        if (results.poseLandmarks || results.leftHandLandmarks || results.rightHandLandmarks) {
          onFrameCaptured(keypoints, confidences);
        }
      });

      if (videoRef.current) {
        camera = new window.Camera(videoRef.current, {
          onFrame: async () => {
            await holistic.send({image: videoRef.current});
          },
          width: 640,
          height: 480
        });
        
        camera.start().then(() => {
          setIsStreaming(true);
        });
      }
    };

    return () => {
      clearInterval(checkGlobals);
      if (camera) camera.stop();
    };
  }, []);

  return (
    <div className="camera-container">
      {/* Hidden original video feed */}
      <video 
        ref={videoRef} 
        style={{ display: 'none' }}
      />
      
      {/* Visible canvas with skeletal drawing */}
      <canvas 
        ref={canvasRef} 
        width={640} 
        height={480} 
        style={{
          borderRadius: '12px',
          boxShadow: '0 10px 25px rgba(0,0,0,0.2)',
          transform: 'scaleX(-1)' // Mirror effect
        }}
      />
      {!isStreaming && <p className="loading-text">Loading MediaPipe Models...</p>}
    </div>
  );
};

export default CameraFeed;
