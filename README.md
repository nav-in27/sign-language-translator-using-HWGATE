# 🤟 Indian Sign Language (ISL) Real-Time Translator

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19-61DAFB.svg)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-8-646CFF.svg)](https://vitejs.dev/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-Enabled-005CED.svg)](https://onnxruntime.ai/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end, edge-accelerated Indian Sign Language (ISL) real-time translation system powered by a novel **Confidence-Weighted Adaptive Hierarchical Windowed Graph Adaptive Transformer Engine (CWA-HWGATE)**. 

The system captures 3D keypoint landmarks via webcam using **MediaPipe Holistic**, streams skeleton coordinates to a high-speed **ONNX/PyTorch inference engine**, accumulates sign glosses, and translates them into natural English sentences using an integrated **Neural Machine Translation (NMT) / Small Language Model (SLM)** pipeline based on `Qwen1.5-0.5B-Chat`.

---

## ✨ Key Features

- **⚡ Real-Time 3D Landmark Tracking**: Uses MediaPipe Holistic to track 133 3D keypoints (33 pose, 21 left hand, 21 right hand) normalized in 3D relative to shoulder width and body center.
- **🧠 CWA-HWGATE Architecture**: Injects pose-estimator confidence directly into multi-head spatial/temporal attention matrices, effectively filtering out motion blur and low-confidence hand frames.
- **🚀 Dual-Engine Inference**: ONNX Runtime acceleration for sub-2ms latency with automatic fallback to PyTorch (`ISLModelLite`).
- **🛡️ Robust Signal Processing**: Features rejection thresholding ($\ge 0.65$ confidence) for unknown gestures, 64-frame sliding window temporal accumulation, and a 5-frame majority-voting debouncer.
- **🗣️ Sign Gloss to Natural English Translation**: Asynchronous NMT translation engine using `Qwen1.5-0.5B-Chat` with background thread queuing and instant rule-based fallback (<1ms UI response).
- **📊 Interactive Model Benchmarking Suite**: Built-in benchmarking pipeline to compare **CNN, LSTM, GRU, ST-GCN, and CWA-HWGATE** models in real time with interactive charts (Recharts) and exportable reports (PDF & CSV).
- **🎨 Glassmorphic Dark UI**: Built with React 19, Tailwind CSS, Lucide icons, and live audio text-to-speech synthesis (Web Speech API).
- **📦 Single-Command Deployment**: One-line Python launcher script (`start_project.py`) to manage both backend and frontend servers with clean shutdown handling.

---

## 📐 System Architecture

```
                 +-----------------------------------+
                 |        Webcam Video Feed          |
                 +-----------------------------------+
                                   |
                                   v
                 +-----------------------------------+
                 |    MediaPipe Holistic (Browser)   |
                 |      (133 3D Keypoints @ 30FPS)   |
                 +-----------------------------------+
                                   |
                             JSON / HTTP POST
                                   v
                 +-----------------------------------+
                 |       FastAPI Backend Server      |
                 |        (http://localhost:8000)    |
                 +-----------------------------------+
                     /                           \
                    /                             \
                   v                               v
 +----------------------------------+    +----------------------------------+
 |    ISL Inference Engine          |    |     NMT Translation Engine       |
 |  - 3D Landmark Normalization     |    |  - Fast Grammatical Fallback     |
 |  - ONNX Runtime / PyTorch        |    |  - Qwen1.5-0.5B-Chat (LLM)       |
 |  - Sliding Window (64 frames)    |    |  - Async Background Worker       |
 |  - Majority Voting Debouncing    |    +----------------------------------+
 +----------------------------------+                     |
                   \                                     /
                    v                                   v
                 +-----------------------------------+
                 |     Vite + React 19 Frontend      |
                 |  - Live Gloss & English Display   |
                 |  - Speech Synthesis (Audio Output)|
                 |  - Benchmark Dashboard            |
                 +-----------------------------------+
```

### Technical Innovations in CWA-HWGATE

1. **Confidence-Weighted Attention (CWA)**:
   $$ \text{Attn}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}} \cdot \sigma(W_g \cdot c + b_g)^T\right) V $$
   where $c$ is the per-frame MediaPipe landmark confidence score, preventing motion-blurred frames from degrading sequence representations.

2. **Hierarchical Graph Embedding**:
   Fuses spatial joint topology with temporal sequence windowing (64 frames) to maintain invariant features under variable signing speeds and lightning conditions.

---

## 🛠️ Tech Stack

### Backend & AI/ML
- **Framework**: FastAPI, Uvicorn
- **Deep Learning**: PyTorch 2.0+, ONNX Runtime
- **LLM / NMT**: Hugging Face `transformers` (`Qwen/Qwen1.5-0.5B-Chat`)
- **Data & Math**: NumPy, SciPy, scikit-learn

### Frontend
- **Framework**: React 19, Vite 8
- **Styling**: Tailwind CSS, Vanilla CSS
- **Pose Detection**: `@mediapipe/holistic`, `@mediapipe/camera_utils`
- **Charts & Export**: Recharts, jsPDF, jsPDF-AutoTable, PapaParse
- **Icons**: Lucide React

---

## 📊 Benchmark Results

Evaluated on 4,985 sequence samples across 45 ISL classes:

| Model Architecture | Accuracy (%) | Macro F1 | Latency (ms) | FPS Throughput | Model Size (MB) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Simple CNN** | 25.58% | 0.154 | 20.37 ms | 49.09 FPS | 0.31 MB | Baseline |
| **Simple LSTM** | 24.37% | 0.138 | 1.27 ms | 787.76 FPS | 0.80 MB | Baseline |
| **Simple GRU** | 24.57% | 0.146 | 1.25 ms | 797.29 FPS | 0.60 MB | Baseline |
| **ST-GCN** | 26.88% | 0.184 | 1.32 ms | 755.02 FPS | 0.15 MB | Baseline |
| **CWA-HWGATE (Ours)** | **41.22%** | **0.373** | **1.44 ms** | **694.64 FPS** | **12.35 MB** | **🏆 Best** |

---

## 📁 Repository Structure

```
isl-translator/
├── backend/
│   ├── app.py                  # FastAPI server & route handlers
│   ├── inference.py            # Real-time ISLInferenceEngine (ONNX/PyTorch)
│   ├── translator.py           # NMT Engine (Qwen1.5-0.5B-Chat + fallback)
│   ├── eval_pipeline.py        # Metrics & evaluation pipeline
│   └── test_*.py               # Automated pipeline test scripts
├── frontend/
│   ├── src/
│   │   ├── components/         # CameraFeed webcam & MediaPipe overlay
│   │   ├── pages/              # LiveTranslator & Benchmark views
│   │   ├── inference/          # Speech synthesis & SLM translators
│   │   ├── App.jsx             # Main routing & state
│   │   └── index.css           # Glassmorphic CSS styling
│   ├── package.json            # Node.js dependencies
│   └── vite.config.js          # Vite server configuration
├── training/
│   ├── models/                 # Model architectures (CWA-HWGATE, ST-GCN, etc.)
│   ├── benchmark_runner.py     # Model benchmarking script
│   ├── benchmark_results.json  # Benchmarking outputs & metrics
│   ├── colab_isl_training.ipynb# Google Colab GPU training notebook
│   ├── export_onnx.py          # PyTorch -> ONNX export script
│   ├── extract_keypoints.py    # MediaPipe landmark dataset builder
│   └── train.py                # Model training entrypoint
├── start_project.py            # Orchestrated cross-platform launcher
├── .gitignore                  # Git ignore rules
└── README.md                   # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites

- **Python**: 3.9 or higher
- **Node.js**: 18.x or higher (with `npm` installed)
- **Webcam**: Required for live camera inference

---

### Quick Start (Recommended)

Run the full system using the included orchestrator script:

```bash
python start_project.py
```

This single command will:
1. Check & install frontend dependencies (`npm install`) if needed.
2. Launch the FastAPI backend on `http://localhost:8000`.
3. Launch the Vite React frontend on `http://localhost:5173`.
4. Monitor both processes with graceful `Ctrl+C` termination handling.

---

### Manual Setup (Step-by-Step)

#### 1. Clone the Repository
```bash
git clone https://github.com/nav-in27/sign-language-translator-using-HWGATE.git
cd sign-language-translator-using-HWGATE
```

#### 2. Backend Setup
```bash
cd backend
pip install -r requirements.txt # or install fastapi uvicorn torch onnxruntime transformers numpy
python app.py
```
*Backend will run at:* `http://localhost:8000`

#### 3. Frontend Setup
In a separate terminal window:
```bash
cd frontend
npm install
npm run dev
```
*Frontend will run at:* `http://localhost:5173`

---

## 🔗 API Reference

### `POST /predict`
Processes keypoint array from client and returns predicted ISL gloss and English translation.

**Request Body**:
```json
{
  "keypoints": [[x, y, z], ...], // 133 landmarks
  "confidences": [[c], ...]     // 133 confidence scores
}
```

**Response**:
```json
{
  "label": "HELLO",
  "confidence": 0.9421,
  "is_unknown": false,
  "engine": "ONNX Runtime",
  "translation": "Hello.",
  "raw_glosses": "HELLO"
}
```

### `POST /benchmark/run`
Triggers asynchronous execution of model evaluation across all baseline architectures.

### `GET /benchmark/status`
Returns current progress percentage and step details of running benchmark.

### `GET /benchmark/results`
Returns latest JSON benchmark metrics (accuracy, F1, latency, FPS, model size).

### `GET /health`
Returns backend health status, active engine, and rejection parameters.

---

## 🏋️ Training & Exporting Custom Models

1. **Extract Keypoints**:
   ```bash
   python training/extract_keypoints.py --data_dir /path/to/videos --out_dir training/data_extracted_full
   ```

2. **Train CWA-HWGATE Model**:
   ```bash
   python training/train.py --dataset training/data_extracted_full --epochs 50
   ```

3. **Export to ONNX**:
   ```bash
   python training/export_onnx.py --model_path training/cwa_hwgate_pretrained.pth --output training/model.onnx
   ```

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for details.

---

## 👥 Authors & Acknowledgments

- **Naveen R** ([@nav-in27](https://github.com/nav-in27)) - Project Architect & Developer
- Developed using **MediaPipe**, **PyTorch**, **FastAPI**, **React**, and **ONNX Runtime**.
