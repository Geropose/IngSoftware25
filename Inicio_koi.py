import streamlit as st
import subprocess
import os
import webbrowser

st.set_page_config(page_title="Inicio", layout="centered")

st.title("👁️ Bienvenido a KOI Tools")
st.write("Seleccioná una de las funcionalidades disponibles.")

st.markdown("---")

col1, col2 = st.columns(2)

# Botón para ejecutar Koi Eye
with col1:
    if st.button("🚀 KOI-EYE OFFLINE"):
        st.success("Abriendo módulo KOI EYE...")
        # Ejecuta el script koi_eye.py en un nuevo proceso de Streamlit
        python_path = "python"  # O "python" según tu sistema
        koi_eye_path = os.path.abspath("./Codes/koi_eye.py")
        subprocess.Popen([python_path, "-m", "streamlit", "run", koi_eye_path])

# Botón para ejecutar transmisión YOLOv8
with col2:
    video_url = st.text_input("📥 Pegá aquí la URL del video en vivo (YouTube, etc):")

    if video_url:
        with st.spinner("⏳ Obteniendo stream con yt-dlp..."):
            try:
                # Extraer URL directa con yt-dlp
                result = subprocess.run(["yt-dlp", "-g", video_url], capture_output=True, text=True, check=True)
                stream_link = result.stdout.strip().split('\n')[0]
                st.success("✅ Stream encontrado.")
            except subprocess.CalledProcessError as e:
                st.error("❌ No se pudo obtener el stream. Revisá la URL.")
                st.stop()
    if st.button("🎥 KOI-EYE ONLINE"):
        st.success("Iniciando stream en vivo")
        koi_live_path = os.path.abspath("./Codes/koi_eye_live.py")
        subprocess.Popen(["python", koi_live_path, stream_link])
        webbrowser.open_new_tab("http://localhost:8000/video")


