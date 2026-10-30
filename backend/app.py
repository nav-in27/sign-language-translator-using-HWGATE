from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Union, Optional
import uvicorn
import subprocess
import os
import sys
import json
import time
import uuid
import hashlib
from datetime import datetime, timezone

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
    seq_id: Optional[int] = 0
    timestamp: Optional[float] = 0.0
    trace_id: Optional[str] = None
    session_id: Optional[str] = None

# Session state for tracking gloss buffer and real-time translation
session_state: Dict[str, Any] = {
    "gloss_buffer": [],
    "last_sign_time": 0.0,
    "last_translated": "",
    "last_gloss": "",
    "last_event_id": None
}


def reset_live_session() -> None:
    """Start a fresh browser stream without inheriting stale sequence state."""
    isl_model.reset()
    session_state["gloss_buffer"] = []
    session_state["last_gloss"] = ""
    session_state["last_translated"] = ""
    session_state["last_sign_time"] = 0.0
    session_state["last_event_id"] = None


def enrich_live_payload(payload: Dict[str, Any], trace_id: str) -> Dict[str, Any]:
    """Attach the active buffer contract to every response sent to the browser."""
    payload["trace_id"] = trace_id
    payload["live_diagnostics"] = isl_model.get_live_diagnostics()
    return payload

@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    await websocket.accept()
    reset_live_session()
    print("[WS CONNECT] live prediction session reset", flush=True)
    try:
        while True:
            data = await websocket.receive_json()
            kps = data.get("keypoints")
            confs = data.get("confidences")
            seq_id = int(data.get("seq_id", 0))
            timestamp = float(data.get("timestamp", 0.0))
            trace_id = str(data.get("trace_id") or uuid.uuid4())
            session_id = data.get("session_id")
            if not kps or not confs or len(kps) != 133 or len(confs) != 133:
                await websocket.send_json({
                    "error": "Expected exactly 133 keypoints and 133 confidence values",
                    "state": "ERROR",
                    "seq_id": seq_id,
                    "trace_id": trace_id,
                })
                continue

            result = isl_model.predict(kps, confs, seq_id=seq_id, timestamp=timestamp, session_id=session_id)
            current_time = time.time()

            # Maintain session gloss sequence when new sign confirmed (strictly on unique event_id)
            if result.state == "CONFIRMED" and result.confirmed_sign:
                if result.event_id and result.event_id != session_state.get("last_event_id"):
                    session_state["last_event_id"] = result.event_id
                    # Stage B: Preserve sign order and prevent duplicate consecutive tokens
                    if not session_state["gloss_buffer"] or session_state["gloss_buffer"][-1] != result.confirmed_sign:
                        session_state["gloss_buffer"].append(result.confirmed_sign)
                    if len(session_state["gloss_buffer"]) > 6:
                        session_state["gloss_buffer"].pop(0)
                    session_state["last_gloss"] = result.confirmed_sign
                    session_state["last_sign_time"] = current_time
                    # Stage C: Translate full multi-sign sequence
                    full_sentence = nmt_engine.translate_gloss_to_english(" ".join(session_state["gloss_buffer"]))
                    session_state["last_translated"] = full_sentence

            # Clear buffer and stale sentence when user pauses signing for > 3.0s
            if current_time - session_state["last_sign_time"] > 3.0 and len(session_state["gloss_buffer"]) > 0:
                session_state["gloss_buffer"] = []
                session_state["last_gloss"] = ""
                session_state["last_event_id"] = None
                session_state["last_translated"] = ""

            payload = enrich_live_payload(result.to_dict(), trace_id)
            payload["raw_glosses"] = " ".join(session_state["gloss_buffer"])
            if session_state.get("last_translated") and len(session_state["gloss_buffer"]) > 0:
                payload["sentence_translation"] = session_state["last_translated"]
                if len(session_state["gloss_buffer"]) > 1:
                    payload["translation"] = session_state["last_translated"]
            else:
                payload["sentence_translation"] = ""

            if result.state == "CONFIRMED":
                print(f"[WS CONFIRMED] seq_id={seq_id} sign={result.confirmed_sign} transl='{payload.get('translation', result.translation)}' evt={result.event_id} speech={result.speech_action}", flush=True)
            elif result.state in ("RECOGNIZING", "UNCERTAIN") and result.confidence > 0.1:
                print(f"[WS {result.state}] seq_id={seq_id} prov={result.provisional_sign} conf={result.confidence:.3f} rej={result.rejection_reason}", flush=True)

            await websocket.send_json(payload)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        import traceback
        traceback.print_exc()

@app.post("/predict")
async def predict_gesture(data: KeypointData):
    try:
        trace_id = data.trace_id or str(uuid.uuid4())
        # A browser page starts its monotonic stream at one. HTTP fallback must
        # apply the same reconnect semantics as the WebSocket path.
        if data.seq_id == 1 and isl_model.last_processed_seq_id > 1:
            reset_live_session()
            print("[HTTP CONNECT] live prediction session reset", flush=True)
        if len(data.keypoints) != 133 or len(data.confidences) != 133:
            return {"error": "Expected exactly 133 keypoints"}
            
        result = isl_model.predict(
            data.keypoints,
            data.confidences,
            seq_id=data.seq_id or 0,
            timestamp=data.timestamp or 0.0,
            session_id=data.session_id
        )
        current_time = time.time()
        
        if result.state == "CONFIRMED" and result.confirmed_sign:
            if result.event_id and result.event_id != session_state.get("last_event_id"):
                session_state["last_event_id"] = result.event_id
                if not session_state["gloss_buffer"] or session_state["gloss_buffer"][-1] != result.confirmed_sign:
                    session_state["gloss_buffer"].append(result.confirmed_sign)
                if len(session_state["gloss_buffer"]) > 6:
                    session_state["gloss_buffer"].pop(0)
                session_state["last_gloss"] = result.confirmed_sign
                session_state["last_sign_time"] = current_time
                full_sentence = nmt_engine.translate_gloss_to_english(" ".join(session_state["gloss_buffer"]))
                session_state["last_translated"] = full_sentence
                
        # Clear buffer and stale sentence when user pauses signing for > 3.0s
        if current_time - session_state["last_sign_time"] > 3.0 and len(session_state["gloss_buffer"]) > 0:
            session_state["gloss_buffer"] = []
            session_state["last_gloss"] = ""
            session_state["last_event_id"] = None
            session_state["last_translated"] = ""
        
        payload = enrich_live_payload(result.to_dict(), trace_id)
        payload["raw_glosses"] = " ".join(session_state["gloss_buffer"])
        if session_state.get("last_translated") and len(session_state["gloss_buffer"]) > 0:
            payload["sentence_translation"] = session_state["last_translated"]
            if len(session_state["gloss_buffer"]) > 1:
                payload["translation"] = session_state["last_translated"]
        else:
            payload["sentence_translation"] = ""
        return payload
    except Exception as e:
        return {"error": str(e)}

@app.get("/pipeline/trace")
def get_pipeline_trace():
    """Returns recent stage-by-stage event log for complete pipeline trace."""
    return {
        "count": len(isl_model.event_log),
        "events": isl_model.get_latest_event_trace()
    }

@app.post("/reload_model")
async def reload_model():
    try:
        isl_model.load_model()
        isl_model.prototypes = isl_model._load_prototypes()
        return {"status": "success", "engine": isl_model.engine_type, "classes": isl_model.num_classes}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/model/status")
def model_status():
    return isl_model.get_model_status()

@app.get("/evaluation/summary")
def evaluation_summary():
    report_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", "achwgate_136class_production_test_final.json")
    if not os.path.exists(report_path):
        report_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", "clean_unweighted_aug_lr5e4_test_final.json")
    if not os.path.exists(report_path):
        return {"status": "not_evaluated", "message": "No current evaluation report is available."}
    with open(report_path, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    checkpoint_path = report.get("checkpoint")
    if checkpoint_path and not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(report_path)), checkpoint_path))
    def file_hash(path):
        if not path or not os.path.exists(path):
            return None
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    current = isl_model.get_model_status()
    report["status"] = "current_checkpoint" if file_hash(checkpoint_path) == current.get("checkpoint_sha256") else "historical_checkpoint"
    report["report_checkpoint_path"] = checkpoint_path
    report["report_checkpoint_sha256"] = file_hash(checkpoint_path)
    report["current_checkpoint_sha256"] = current.get("checkpoint_sha256")
    report["evaluation_report_path"] = os.path.abspath(report_path)
    report["report_mtime_utc"] = datetime.fromtimestamp(os.path.getmtime(report_path), timezone.utc).isoformat()
    return report

class TranslateRequest(BaseModel):
    text: str

@app.post("/reset")
async def api_reset():
    reset_live_session()
    return {"status": "success", "message": "Live session and sentence buffer reset"}

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

@app.get("/benchmark/comparison")
async def benchmark_comparison():
    rep_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
    comp_path = os.path.join(rep_dir, "model_comparison_benchmark.json")
    if os.path.exists(comp_path):
        with open(comp_path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except Exception as e:
                return {"error": f"Failed reading benchmark json: {str(e)}"}
    return {"error": "Benchmark comparison not found. Run training/train_and_evaluate_baselines.py."}

@app.get("/benchmark/confusion_matrix")
async def benchmark_confusion_matrix(model: Optional[str] = None):
    rep_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
    cm_path = os.path.join(rep_dir, "confusion_matrix_136class.json")
    if os.path.exists(cm_path):
        with open(cm_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if model and "models" in data and model in data["models"]:
                    return {
                        "dataset_split": data.get("dataset_split"),
                        "sample_count": data.get("sample_count"),
                        "num_classes": data.get("num_classes"),
                        "class_labels": data.get("class_labels"),
                        "model_name": model,
                        "data": data["models"][model]
                    }
                return data
            except Exception as e:
                return {"error": f"Failed reading confusion matrix json: {str(e)}"}
    return {"error": "Confusion matrix bundle not found. Run training/train_and_evaluate_baselines.py."}

@app.get("/benchmark/results")
async def benchmark_results():
    rep_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
    comp_path = os.path.join(rep_dir, "model_comparison_benchmark.json")
    if os.path.exists(comp_path):
        with open(comp_path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except:
                pass
    return {"status": "pending", "message": "Benchmark results not yet generated"}

@app.get("/health")
def health_check():
    status = isl_model.get_model_status()
    return {
        "status": "Active",
        "ready": status["ready"],
        "engine": status["engine"],
        "num_classes": status["num_classes"],
        "min_confidence": isl_model.min_confidence,
        "model_sha256": status["artifact_sha256"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
