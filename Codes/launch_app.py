import subprocess
import sys
import os
import time
import webbrowser
import socket

def wait_for_server(host="localhost", port=8501, timeout=30):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=2):
                return True
        except OSError:
            time.sleep(1)
    return False

def get_base_path():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        return os.path.dirname(os.path.abspath(__file__))

def main():
    base_path = get_base_path()
    app3_path = os.path.join(base_path, "Codes", "koi_eye.py")

    cmd = f'python -m streamlit run "{app3_path}" --server.headless true'
    print(f"Lanzando Streamlit con: {cmd}")

    proc = subprocess.Popen(cmd, shell=True)

    if wait_for_server():
        webbrowser.open("http://localhost:8501")
    else:
        print("⚠️  Streamlit no se inició correctamente. Intentá abrir: http://localhost:8501")

    try:
        # Espera activa a que termine el proceso
        while proc.poll() is None:
            time.sleep(1)
    finally:
        proc.terminate()

if __name__ == "__main__":
    main()
