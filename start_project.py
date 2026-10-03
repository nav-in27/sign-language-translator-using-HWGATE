import subprocess
import os
import sys
import time
import shutil

# Ensure Windows cp1252 terminal encoding never crashes on unicode
try:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if sys.stderr.encoding and sys.stderr.encoding.lower() != 'utf-8':
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

def find_working_python(root_dir):
    """
    Finds a Python interpreter that has torch, onnxruntime, fastapi, and uvicorn installed.
    Searches system paths, known Python 3.11 installs, and virtualenvs.
    """
    candidates = [
        r"C:\Users\navee\AppData\Local\Programs\Python\Python311\python.exe",
        sys.executable,
        os.path.join(os.path.dirname(root_dir), "isl_recognition", "venv", "Scripts", "python.exe"),
        shutil.which("python")
    ]
    
    seen = set()
    for cand in candidates:
        if not cand or cand in seen:
            continue
        seen.add(cand)
        if os.path.exists(cand):
            try:
                res = subprocess.run(
                    [cand, "-c", "import torch, onnxruntime, fastapi, uvicorn; print('READY')"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if res.returncode == 0 and "READY" in res.stdout:
                    return cand
            except Exception:
                continue
                
    return None

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.join(root_dir, "backend")
    frontend_dir = os.path.join(root_dir, "frontend")
    
    print("=========================================")
    print("[*] Starting ISL Translator Project")
    print("=========================================\n")
    
    # 1. Ensure frontend dependencies exist
    if not os.path.exists(os.path.join(frontend_dir, "node_modules")):
        print("[*] Node modules not found. Installing frontend dependencies...")
        subprocess.run(["npm", "install"], cwd=frontend_dir, shell=True)
        print("[+] Frontend dependencies installed!\n")
    
    # 2. Automatically locate the valid Python environment with AI dependencies
    python_cmd = find_working_python(root_dir)
    if not python_cmd:
        print("[!] CRITICAL ERROR: Could not find a Python interpreter with 'torch', 'onnxruntime', and 'fastapi' installed!")
        print("Please run: pip install torch onnxruntime fastapi uvicorn mediapipe")
        sys.exit(1)
        
    print(f"[+] Selected AI Python Environment: {python_cmd}\n")
    
    # 3. Launch Backend Server (FastAPI on Port 8000)
    print("[*] Starting Backend Server (FastAPI on http://localhost:8000)...")
    backend_process = subprocess.Popen(
        [python_cmd, "app.py"],
        cwd=backend_dir
    )
    
    # Wait for backend to initialize
    print("    Waiting for backend AI models to load...")
    time.sleep(3)
    
    # 4. Launch Frontend Dev Server (Vite)
    print("[*] Starting Frontend Server (Vite)...")
    frontend_process = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=frontend_dir,
        shell=True
    )
    
    print("\n=======================================================")
    print("[+] All Services Running Successfully!")
    print("    - Backend API: http://localhost:8000")
    print("    - Frontend UI: http://localhost:5173")
    print("[!] Press Ctrl+C in this terminal to shut down both servers.")
    print("=======================================================\n")
    
    try:
        backend_process.wait()
        frontend_process.wait()
    except KeyboardInterrupt:
        print("\n\n[*] Shutting down servers...")
        backend_process.terminate()
        frontend_process.terminate()
        time.sleep(1)
        if backend_process.poll() is None:
            backend_process.kill()
        if frontend_process.poll() is None:
            frontend_process.kill()
        print("[+] Servers stopped successfully. Goodbye!")

if __name__ == "__main__":
    main()
