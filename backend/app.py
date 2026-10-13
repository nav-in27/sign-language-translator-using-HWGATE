from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Union
import uvicorn
import subprocess
import os
import sys
import json
import time

backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

try:
    from backend.inference import isl_model
    from backend.translator import nmt_engine
except ImportError:
    from inference import isl_model
    from translator import nmt_engine

app = FastAPI(title="ISL Translator Edge API")

# Allow CORS for local React development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class KeypointData(BaseModel):
    keypoints: List[List[float]]
    confidences: Union[List[List[float]], List[float]]

# Session state for tracking gloss buffer and real-time translation
session_state: Dict[str, Any] = {
    "gloss_buffer": [],
    "last_sign_time": 0.0,
    "last_translated": "",
    "last_gloss": ""
}

@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            kps = data.get("keypoints")
            confs = data.get("confidences")
            if not kps or not confs or len(kps) != 133:
                continue

            label, conf, is_unknown = isl_model.predict(kps, confs)
            status = getattr(isl_model, "last_status", "CONFIRMED" if not is_unknown else "DETECTING")
            current_time = time.time()

            if not is_unknown and status == "CONFIRMED":
                if label != session_state["last_gloss"]:
                    session_state["gloss_buffer"].append(label)
                    if len(session_state["gloss_buffer"]) > 6:
                        session_state["gloss_buffer"].pop(0)
                    session_state["last_gloss"] = label
                    session_state["last_sign_time"] = current_time
                    gloss_string = " ".join(session_state["gloss_buffer"])
                    session_state["last_translated"] = nmt_engine.translate_gloss_to_english(gloss_string)

            if current_time - session_state["last_sign_time"] > 3.5 and len(session_state["gloss_buffer"]) > 0:
                session_state["gloss_buffer"] = []
                session_state["last_gloss"] = ""

            await websocket.send_json({
                "label": label,
                "confidence": round(conf, 4),
                "is_unknown": is_unknown,
                "status": status,
                "engine": isl_model.engine_type,
                "translation": session_state["last_translated"] if session_state["last_translated"] else (nmt_engine.translate_gloss_to_english(label) if status == "CONFIRMED" else ""),
                "raw_glosses": " ".join(session_state["gloss_buffer"])
            })
    except WebSocketDisconnect:
        pass
    except Exception as e:
        import traceback
        traceback.print_exc()

@app.post("/predict")
async def predict_gesture(data: KeypointData):
    try:
        if len(data.keypoints) != 133 or len(data.confidences) != 133:
            return {"error": "Expected exactly 133 keypoints"}
            
        label, conf, is_unknown = isl_model.predict(data.keypoints, data.confidences)
        status = getattr(isl_model, "last_status", "CONFIRMED" if not is_unknown else "DETECTING")
        current_time = time.time()
        
        # Sentence Accumulation Logic
        special_labels = ["WAITING FOR SIGN", "DETECTING...", "UNKNOWN_SIGN", "No Sign Detected", "Buffering...", "UNKNOWN SIGN"]
        if not is_unknown and status == "CONFIRMED" and label not in special_labels:
            if label != session_state["last_gloss"]:
                session_state["gloss_buffer"].append(label)
                if len(session_state["gloss_buffer"]) > 6:
                    session_state["gloss_buffer"].pop(0)
                session_state["last_gloss"] = label
                session_state["last_sign_time"] = current_time
                
                # Trigger Instant NMT Translation
                gloss_string = " ".join(session_state["gloss_buffer"])
                session_state["last_translated"] = nmt_engine.translate_gloss_to_english(gloss_string)
                
        # Clear buffer if user stops signing for 3.5 seconds
        if current_time - session_state["last_sign_time"] > 3.5 and len(session_state["gloss_buffer"]) > 0:
            session_state["gloss_buffer"] = []
            session_state["last_gloss"] = ""
        
        return {
            "label": label,
            "confidence": round(conf, 4),
            "is_unknown": is_unknown,
            "status": status,
            "engine": isl_model.engine_type,
            "translation": session_state["last_translated"] if session_state["last_translated"] else (nmt_engine.translate_gloss_to_english(label) if status == "CONFIRMED" else ""),
            "raw_glosses": " ".join(session_state["gloss_buffer"])
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/reload_model")
async def reload_model():
    try:
        isl_model.load_model()
        isl_model.prototypes = isl_model._load_prototypes()
        return {"status": "success", "engine": isl_model.engine_type, "classes": isl_model.num_classes}
    except Exception as e:
        return {"status": "error", "message": str(e)}

class TranslateRequest(BaseModel):
    text: str

@app.post("/translate")
async def translate_text(req: TranslateRequest):
    result = nmt_engine.translate_gloss_to_english(req.text)
    return {"input": req.text, "translation": result}

@app.post("/benchmark/run")
async def run_benchmark():
    training_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "training")
    status_path = os.path.join(training_dir, "benchmark_status.json")
    with open(status_path, "w") as f:
        json.dump({"message": "Initializing...", "progress": 0}, f)
        
    subprocess.Popen(["python", "benchmark_runner.py"], cwd=training_dir)
    return {"status": "started"}

@app.get("/benchmark/status")
async def benchmark_status():
    training_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "training")
    status_path = os.path.join(training_dir, "benchmark_status.json")
    if os.path.exists(status_path):
        with open(status_path, "r") as f:
            try:
                return json.load(f)
            except:
                return {"message": "Reading...", "progress": 0}
    return {"message": "Idle", "progress": 0}

@app.get("/benchmark/results")
async def benchmark_results():
    training_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "training")
    results_path = os.path.join(training_dir, "benchmark_results.json")
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            try:
                return json.load(f)
            except:
                return {"error": "Corrupt JSON"}
    return {"error": "Not found"}

@app.get("/health")
def health_check():
    return {
        "status": "Active",
        "engine": isl_model.engine_type,
        "num_classes": isl_model.num_classes,
        "min_confidence": isl_model.min_confidence
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
