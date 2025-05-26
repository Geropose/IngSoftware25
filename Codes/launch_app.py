import os
import subprocess
import sys
import time
import webbrowser
import socket

# Archivo que marca que ya se ejecutó el instalador
flag_file = "instalado.flag"

def ejecutar_instalador():
    print("Ejecutando instalador por primera vez...")
    # Ejecuta instalador.bat desde la carpeta del ejecutable o script
    base_path = get_base_path()
    instalador_path = os.path.join(base_path, "instalador.bat")
    result = subprocess.call([instalador_path])
    if result == 0:
        with open(flag_file, "w") as f:
            f.write("Instalador ejecutado")
        print("Instalador finalizado correctamente.")
    else:
        print("Error al ejecutar instalador.bat. Verifica el archivo.")

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
        # Si es un exe
        return os.path.dirname(sys.executable)
    else:
        # Si es script .py
        return os.path.dirname(os.path.abspath(__file__))

def main():
    # Ejecutar instalador solo si no está la marca
    if not os.path.exists(flag_file):
        ejecutar_instalador()
    else:
        print("Instalador ya fue ejecutado previamente.")

    # Ejecutar Streamlit
    base_path = get_base_path()
    app3_path = os.path.join(base_path, "inicio_koi.py")

    cmd = f'python -m streamlit run "{app3_path}" --server.headless true'
    print(f"Lanzando Streamlit con: {cmd}")

    proc = subprocess.Popen(cmd, shell=True)

    if wait_for_server():
        webbrowser.open("http://localhost:8501")
    else:
        print("⚠️  Streamlit no se inició correctamente. Intentá abrir: http://localhost:8501")

    try:
        while proc.poll() is None:
            time.sleep(1)
    finally:
        proc.terminate()

if __name__ == "__main__":
    main()
