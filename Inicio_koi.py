import streamlit as st
import subprocess
import os
import webbrowser
from yt_dlp import YoutubeDL
import requests
import time

def get_stream_url(video_url):
    """Obtiene la URL directa del stream de YouTube"""
    ydl_opts = {
        'quiet': True,
        'format': 'best[ext=m3u8]/best',
        'skip_download': True
    }
    
    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            return info['url']
    except Exception as e:
        st.error(f"Error al obtener stream: {e}")
        return None

st.set_page_config(page_title="KOI", layout="centered")

st.title("🐟 Bienvenido a KOI 🐟")
st.subheader("¡ATENCIÓN! ") 
st.error("Al seleccionar el modo KOI-EYE CAM o KOI-EYE ONLINE debe esperar a que se levante el servidor.")
st.text(" ⏳ Espera aproximada 10 segundos. ⏳ ")

st.markdown("---")

# Sección OFFLINE
if st.button("🚀 KOI-EYE OFFLINE"):
    st.success("Abriendo módulo KOI EYE OFFLINE")
    python_path = "python"
    koi_eye_path = os.path.abspath("./Codes/koi_eye.py")
    subprocess.Popen([python_path, "-m", "streamlit", "run", koi_eye_path])

st.markdown("---")

# Sección CAM (Puerto 8001)
st.subheader("📹 Modo Cámara (Puerto 8001)")

if st.button("🎥 KOI-EYE CAM"):
    st.success("Iniciando KOI-EYE en CAM")
    koi_live_path = os.path.abspath("./Codes/koi_eye_cam.py")
    subprocess.Popen(["python", koi_live_path])
    time.sleep(2)
    webbrowser.open_new_tab("http://localhost:8001/video")

st.markdown("---")

# Sección ONLINE con YouTube - SIMPLIFICADA
st.subheader("🌐 Modo Online con YouTube")

video_url = st.text_input("📥 Pegá aquí el link del live de YOUTUBE:")

if video_url:
    # Verificar y obtener stream URL
    if 'stream_link' not in st.session_state or st.session_state.get('last_url') != video_url:
        with st.spinner("⏳ Obteniendo la URL del stream del live..."):
            stream_link = get_stream_url(video_url)
            if stream_link:
                st.session_state.stream_link = stream_link
                st.session_state.last_url = video_url
                st.success("✅ Stream encontrado.")
            else:
                st.error("❌ No se pudo obtener el stream. Revisá la URL.")
                st.stop()
    
    # SOLO UN BOTÓN PARA ONLINE
    if st.button("🔥 KOI-TRACKER LIVE ONLINE"):
        st.success("Iniciando KOI-TRACKER LIVE ONLINE")
        koi_tracker_path = os.path.abspath("./Codes/koi_tracker_live.py")
        subprocess.Popen(["python", koi_tracker_path, st.session_state.stream_link])
        time.sleep(3)
        
        # Abrir múltiples pestañas
        webbrowser.open_new_tab("http://localhost:8000/video")
        webbrowser.open_new_tab("http://localhost:8000/heatmap")
        webbrowser.open_new_tab("http://localhost:8000/trajectories")

st.markdown("---")

# CONTROLES SIMPLIFICADOS
st.subheader("🎛️ Controles del Sistema")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**🎥 Control CAM (Puerto 8001):**")
    
    if st.button("🛑 Detener CAM"):
        try:
            response = requests.get("http://localhost:8001/stop", timeout=5)
            if response.status_code == 200:
                st.success("✅ Captura CAM detenida.")
            else:
                st.warning("⚠️ No se pudo detener la captura CAM.")
        except Exception as e:
            st.error(f"❌ Error al detener captura CAM: {e}")
    
    if st.button("⬇️ Descargar Video CAM"):
        webbrowser.open_new_tab("http://localhost:8001/download")

with col2:
    st.markdown("**🌐 Control ONLINE (Puerto 8000):**")
    
    if st.button("🛑 Detener ONLINE"):
        try:
            # Intentar detener múltiples veces para asegurar que pare
            for _ in range(3):
                response = requests.get("http://localhost:8000/stop", timeout=3)
                time.sleep(0.5)
            st.success("✅ Stream Online detenido.")
        except Exception as e:
            st.error(f"❌ Error al detener stream: {e}")
    
    if st.button("⬇️ Descargar Video Online"):
        webbrowser.open_new_tab("http://localhost:8000/download")

st.markdown("---")

# VISUALIZACIÓN SIMPLIFICADA
st.subheader("📊 Visualización en Tiempo Real")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🌡️ Mapa de Calor"):
        try:
            response = requests.get("http://localhost:8000/heatmap", timeout=10)
            if response.status_code == 200:
                st.image(response.content, caption="Mapa de Calor General", use_column_width=True)
            else:
                webbrowser.open_new_tab("http://localhost:8000/heatmap")
        except Exception as e:
            st.warning(f"⚠️ Abriendo en navegador...")
            webbrowser.open_new_tab("http://localhost:8000/heatmap")

with col2:
    if st.button("🛣️ Trayectorias"):
        try:
            response = requests.get("http://localhost:8000/trajectories", timeout=10)
            if response.status_code == 200:
                st.image(response.content, caption="Mapa de Trayectorias", use_column_width=True)
            else:
                webbrowser.open_new_tab("http://localhost:8000/trajectories")
        except Exception as e:
            st.warning(f"⚠️ Abriendo en navegador...")
            webbrowser.open_new_tab("http://localhost:8000/trajectories")

with col3:
    if st.button("📈 Estadísticas"):
        try:
            response = requests.get("http://localhost:8000/stats", timeout=5)
            if response.status_code == 200:
                stats_data = response.json()
                st.json(stats_data)
                
                # Mostrar métricas
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Personas", stats_data.get("personas_detectadas", 0))
                with col_b:
                    st.metric("Frames", stats_data.get("frames_procesados", 0))
            else:
                st.error("❌ Error al obtener estadísticas")
        except Exception as e:
            st.error(f"❌ Error: {e}")

# Análisis por ID específico
st.markdown("### 🎯 Análisis por ID")
col1, col2 = st.columns([1, 2])

with col1:
    id_persona = st.number_input("ID:", min_value=0, step=1, value=0)

with col2:
    if st.button("🔍 Analizar ID"):
        try:
            response = requests.get(f"http://localhost:8000/heatmap/{int(id_persona)}", timeout=15)
            if response.status_code == 200:
                st.image(response.content, caption=f"Análisis ID {id_persona}", use_column_width=True)
            elif response.status_code == 404:
                st.warning(f"⚠️ No hay datos para ID {id_persona}")
            else:
                webbrowser.open_new_tab(f"http://localhost:8000/heatmap/{int(id_persona)}")
        except Exception as e:
            st.warning(f"⚠️ Abriendo en navegador...")
            webbrowser.open_new_tab(f"http://localhost:8000/heatmap/{int(id_persona)}")

st.markdown("---")

# Estado del sistema
st.subheader("🔧 Estado del Sistema")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Puerto 8000 (Online):**")
    try:
        response = requests.get("http://localhost:8000/", timeout=2)
        if response.status_code == 200:
            st.success("✅ Activo")
        else:
            st.error("❌ Error")
    except:
        st.warning("⚠️ Inactivo")

with col2:
    st.markdown("**Puerto 8001 (CAM):**")
    try:
        response = requests.get("http://localhost:8001/", timeout=2)
        if response.status_code == 200:
            st.success("✅ Activo")
        else:
            st.error("❌ Error")
    except:
        st.warning("⚠️ Inactivo")

# Footer
st.markdown("---")
st.markdown("🐟 **KOI Tracker System** - Detección y seguimiento de personas en tiempo real")