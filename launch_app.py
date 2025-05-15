import subprocess
import sys
import os
import time
import webbrowser
import socket

if any("streamlit" in arg for arg in sys.argv):
    sys.exit()

def wait_for_server(host="localhost", port=8501, timeout=30):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=2):
                return True
        except OSError:
            time.sleep(1)
    return False

def main():
    current_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    app3_path = os.path.join(current_dir, "Codes", "app3.py")
    python_exe = sys.executable

    proc = subprocess.Popen([python_exe, "-m", "streamlit", "run", app3_path])

    if wait_for_server():
        webbrowser.open("http://localhost:8501")

    try:
        input("Presioná ENTER para cerrar la app...\n")
    except EOFError:
        pass
    finally:
        proc.terminate()

if __name__ == "__main__":
    main()
