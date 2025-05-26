import streamlit as st
import subprocess
import os
import webbrowser
from yt_dlp import YoutubeDL
def get_stream_url(video_url):
    ydl_opts = {
    'quiet': True,
    'format': 'best[ext=m3u8]/best',
    'skip_download': True
    }
st.set_page_config(page_title="KOI", layout="centered")

st.title("🐟 Bienvenido a KOI 🐟")
st.subheader("¡ATENCION! ") 
st.error("Al seleccionar el modo KOI-EYE CAM o KOI-EYE ONLINE debe esperar a que se levante el servidos.")
st.text(" ⏳ Espera aproximada 10 segundos. ⏳ ")

st.markdown("---")

if st.button("🚀 KOI-EYE OFFLINE"):
    st.success("Abriendo módulo KOI EYE OFFLINE")
    # Ejecuta el script koi_eye.py en un nuevo proceso de Streamlit
    python_path = "python"  # O "python" según tu sistema
    koi_eye_path = os.path.abspath("./Codes/koi_eye.py")
    subprocess.Popen([python_path, "-m", "streamlit", "run", koi_eye_path])

st.markdown("---")

if st.button("🎥 KOI-EYE CAM"):
    st.success("Iniciando KOI-EYE en CAM")
    koi_live_path = os.path.abspath("./Codes/koi_eye_cam.py")
    subprocess.Popen(["python", koi_live_path])
    webbrowser.open_new_tab("http://localhost:8001/video")

st.markdown("---")

# Botón para ejecutar transmisión YOLOv8
video_url = st.text_input("📥 Pegá aquí el link del live de YOUTUBE para ejecutarlo en modo KOI-EYE ONLINE:")
if video_url:
    with st.spinner("⏳ Obteniendo la URL del stream del live..."):
        try:
            # Extraer URL directa con yt-dlp
            result = subprocess.run(["yt-dlp", "-g", video_url], capture_output=True, text=True, check=True)
            stream_link = result.stdout.strip().split('\n')[0]
            st.success("✅ Stream encontrado.")
        except subprocess.CalledProcessError as e:
            st.error("❌ No se pudo obtener el stream. Revisá la URL.")
            st.stop()

if st.button("🎥 KOI-EYE ONLINE"):
    koi_live_path = os.path.abspath("./Codes/koi_eye_live.py")
    subprocess.Popen(["python", koi_live_path, stream_link])
    webbrowser.open_new_tab("http://localhost:8000/video")


