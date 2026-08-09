import subprocess
import os
import sys
import time

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.join(root_dir, "backend")
    frontend_dir = os.path.join(root_dir, "frontend")
    
    print("=========================================")
    print("🚀 Starting ISL Translator Project 🚀")
    print("=========================================\n")
    
    # Ensure npm dependencies are installed first
    if not os.path.exists(os.path.join(frontend_dir, "node_modules")):
        print("Node modules not found. Installing frontend dependencies...")
        subprocess.run(["npm", "install"], cwd=frontend_dir, shell=True)
        print("Frontend dependencies installed!\n")
    
    # We will try to use the virtual environment we created earlier if it exists
    venv_python = os.path.join(os.path.dirname(root_dir), "isl_recognition", "venv", "Scripts", "python.exe")
    if os.path.exists(venv_python):
        print("✅ Found virtual environment!")
        python_cmd = venv_python
    else:
        print("⚠️ Virtual environment not found, using default Python.")
        python_cmd = sys.executable

    print("-> Starting Backend Server (FastAPI)...")
    backend_process = subprocess.Popen(
        [python_cmd, "app.py"],
        cwd=backend_dir
    )
    
    time.sleep(2) # Give the backend a moment to start
    
    print("-> Starting Frontend Server (Vite)...")
    frontend_process = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=frontend_dir,
        shell=True
    )
    
    print("\n✅ Both servers are now running!")
    print("   - Backend API: http://localhost:8000")
    print("   - Frontend UI: http://localhost:5173 (Check Vite output)")
    print("\n🛑 Press Ctrl+C in this terminal to gracefully shut down both servers.\n")
    
    try:
        # Keep the script running to hold the processes alive
        backend_process.wait()
        frontend_process.wait()
    except KeyboardInterrupt:
        print("\n\nShutting down servers...")
        backend_process.terminate()
        frontend_process.terminate()
        
        # Give them a moment to terminate gracefully
        time.sleep(1)
        if backend_process.poll() is None:
            backend_process.kill()
        if frontend_process.poll() is None:
            frontend_process.kill()
            
        print("Servers stopped successfully. Goodbye!")

if __name__ == "__main__":
    main()
