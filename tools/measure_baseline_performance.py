import time
import os
import sys
import numpy as np
import torch
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "training"))
sys.path.insert(0, str(ROOT_DIR / "backend"))

from training.feature_extractor import feature_extractor
from training.models.ac_hwgate import ACHWGATE
import onnxruntime as ort

def benchmark_models():
    print("=" * 60)
    print("BASELINE LATENCY & THROUGHPUT BENCHMARK")
    print("=" * 60)

    # 1. Feature Extractor benchmark
    dummy_seq = [np.random.randn(133, 2).astype(np.float32) for _ in range(32)]
    
    # Warmup
    for _ in range(10):
        _ = feature_extractor.process_sequence_np(dummy_seq)
        
    fe_times = []
    for _ in range(100):
        t0 = time.perf_counter()
        _ = feature_extractor.process_sequence_np(dummy_seq)
        fe_times.append((time.perf_counter() - t0) * 1000)
    print(f"[Feature Extractor] Avg: {np.mean(fe_times):.3f} ms | P50: {np.median(fe_times):.3f} ms | P95: {np.percentile(fe_times, 95):.3f} ms")

    # 2. PyTorch AC-HWGATE (CUDA & CPU)
    input_tensor_cuda = torch.randn(1, 32, 133, 6, device="cuda" if torch.cuda.is_available() else "cpu")
    input_tensor_cpu = torch.randn(1, 32, 133, 6, device="cpu")
    
    model_50 = ACHWGATE(num_classes=50, in_channels=6, sequence_length=32)
    ckpt_path = ROOT_DIR / "checkpoints" / "achwgate_50class_best.pth"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model_50.load_state_dict(state_dict)
    model_50.eval()

    # CPU benchmark
    model_50_cpu = model_50.to("cpu")
    for _ in range(10):
        with torch.no_grad():
            _ = model_50_cpu(input_tensor_cpu)
    cpu_times = []
    for _ in range(100):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model_50_cpu(input_tensor_cpu)
        cpu_times.append((time.perf_counter() - t0) * 1000)
    print(f"[PyTorch AC-HWGATE (CPU)] Avg: {np.mean(cpu_times):.3f} ms | P50: {np.median(cpu_times):.3f} ms | P95: {np.percentile(cpu_times, 95):.3f} ms")

    # CUDA benchmark
    if torch.cuda.is_available():
        model_50_cuda = model_50.to("cuda")
        for _ in range(10):
            with torch.no_grad():
                _ = model_50_cuda(input_tensor_cuda)
        torch.cuda.synchronize()
        cuda_times = []
        for _ in range(100):
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model_50_cuda(input_tensor_cuda)
            torch.cuda.synchronize()
            cuda_times.append((time.perf_counter() - t0) * 1000)
        print(f"[PyTorch AC-HWGATE (RTX 3050 CUDA)] Avg: {np.mean(cuda_times):.3f} ms | P50: {np.median(cuda_times):.3f} ms | P95: {np.percentile(cuda_times, 95):.3f} ms")

    # 3. ONNX Runtime benchmark
    onnx_path = ROOT_DIR / "backend" / "models" / "ac_hwgate.onnx"
    if onnx_path.exists():
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
        inp_name = sess.get_inputs()[0].name
        dummy_inp = np.random.randn(1, 32, 133, 6).astype(np.float32)
        for _ in range(10):
            _ = sess.run(None, {inp_name: dummy_inp})
        ort_times = []
        for _ in range(100):
            t0 = time.perf_counter()
            _ = sess.run(None, {inp_name: dummy_inp})
            ort_times.append((time.perf_counter() - t0) * 1000)
        print(f"[ONNX Runtime (CPU 4 threads)] Avg: {np.mean(ort_times):.3f} ms | P50: {np.median(ort_times):.3f} ms | P95: {np.percentile(ort_times, 95):.3f} ms")

    # 4. Total End-to-End single-prediction latency (FE + Inference)
    total_latency = np.mean(fe_times) + (np.mean(cuda_times) if torch.cuda.is_available() else np.mean(ort_times))
    print("-" * 60)
    print(f"Total Theoretical Pipeline Latency: {total_latency:.2f} ms (~{1000/total_latency:.0f} FPS capable)")
    print("=" * 60)

if __name__ == "__main__":
    benchmark_models()
