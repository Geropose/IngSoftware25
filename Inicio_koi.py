import streamlit as st
import subprocess
import os
import webbrowser
from yt_dlp import YoutubeDL
import requests
import time
import pandas as pd
import streamlit as st
import requests
import time
import cv2
import streamlit as st
import platform
import subprocess
import re
import json
def get_available_cameras():
    """Detecta las cámaras disponibles en el sistema"""
    available_cameras = []
    system = platform.system()
    
    # Método 1: Intentar abrir cada índice de cámara
    for i in range(10):  # Probar los primeros 10 índices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Intentar obtener el nombre de la cámara
            name = f"Cámara {i}"
            
            # En Windows, intentar obtener más información
            if system == "Windows":
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                name = f"Cámara {i} ({width}x{height})"
            
            available_cameras.append({"id": str(i), "name": name})
            cap.release()
    
    # Método 2: En Windows, intentar usar PowerShell para obtener más información
    if system == "Windows" and not available_cameras:
        try:
            cmd = "powershell -Command \"Get-CimInstance Win32_PnPEntity | Where-Object {$_.Caption -like '*camera*' -or $_.Caption -like '*webcam*'} | Select-Object Caption | ConvertTo-Json\""
            result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
            
            if result.returncode == 0 and result.stdout.strip():
                try:
                    cameras_info = json.loads(result.stdout)
                    # Asegurarse de que sea una lista
                    if not isinstance(cameras_info, list):
                        cameras_info = [cameras_info]
                        
                    for i, camera in enumerate(cameras_info):
                        if "Caption" in camera:
                            available_cameras.append({
                                "id": str(i),
                                "name": camera["Caption"]
                            })
                except json.JSONDecodeError:
                    pass
        except Exception:
            pass
    
    # Si no se encontraron cámaras, agregar opciones por defecto
    if not available_cameras:
        available_cameras = [
            {"id": "0", "name": "Cámara por defecto (0)"},
            {"id": "1", "name": "Cámara alternativa (1)"},
            {"id": "2", "name": "Cámara adicional (2)"}
        ]
    
    # Siempre añadir la opción de RTSP
    available_cameras.append({"id": "RTSP URL", "name": "Cámara IP (RTSP URL)"})
    
    return available_cameras

# Función para mostrar el selector de cámaras
def camera_selector(key_prefix="cam"):
    """Muestra un selector de cámaras disponibles"""
    # Obtener cámaras disponibles (con caché para no escanear cada vez)
    if "available_cameras" not in st.session_state:
        with st.spinner("Detectando cámaras disponibles..."):
            st.session_state.available_cameras = get_available_cameras()
    
    cameras = st.session_state.available_cameras
    
    # Crear opciones para el selectbox
    camera_options = [cam["name"] for cam in cameras]
    camera_ids = [cam["id"] for cam in cameras]
    
    # Mostrar el selectbox
    selected_index = st.selectbox(
        "📷 Seleccionar cámara:",
        options=range(len(camera_options)),
        format_func=lambda i: camera_options[i],
        key=f"{key_prefix}_camera_select"
    )
    
    selected_camera_id = camera_ids[selected_index]
    
    # Si es RTSP URL, mostrar campo para ingresar la URL
    if selected_camera_id == "RTSP URL":
        rtsp_url = st.text_input(
            "URL de la cámara RTSP:",
            value="rtsp://usuario:contraseña@192.168.1.100:554/stream",
            key=f"{key_prefix}_rtsp_url"
        )
        return rtsp_url
    else:
        # Para cámaras locales, mostrar información adicional
        st.info(f"✅ Seleccionada: {camera_options[selected_index]} (ID: {selected_camera_id})")
        return selected_camera_id

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

# Añadir esta función para formatear los datos de posición
def format_position_data(positions_data):
    """Formatea los datos de posición en el formato solicitado"""
    if not positions_data or "positions" not in positions_data:
        return "No hay datos disponibles"
        
    id_num = positions_data.get("id", "?")
    positions = positions_data.get("positions", [])
    
    # Crear el encabezado
    formatted_text = f"N° ID {id_num} "
    
    # Añadir cada posición en el formato solicitado
    for pos in positions:
        formatted_text += f"{{ frame: {pos['frame']}, x: {pos['x']}, y: {pos['y']} }}, "
    
    # Eliminar la última coma y espacio
    if formatted_text.endswith(", "):
        formatted_text = formatted_text[:-2]
        
    return formatted_text

st.set_page_config(page_title="KOI", layout="centered")

st.title("🐟 Bienvenido a KOI 🐟")
st.subheader("¡ATENCIÓN! ") 
st.error("Al seleccionar el modo KOI-EYE CAM o KOI-EYE ONLINE debe esperar a que se levante el servidor.")
st.text(" ⏳ Espera aproximada 10 segundos. ⏳ ")

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

tab1, tab2, tab3 = st.tabs(["OFFLINE", "CAMARA", "LIVE YOUTUBE"])

with tab1:
    # Sección OFFLINE
    st.subheader(" Modo OFFLINE ")
    if st.button("🚀 KOI-EYE OFFLINE"):
        st.success("Abriendo módulo KOI EYE OFFLINE")
        python_path = "python"
        koi_eye_path = os.path.abspath("./Codes/koi_eye.py")
        subprocess.Popen([python_path, "-m", "streamlit", "run", koi_eye_path])


with tab2:

    st.header("👁️ KOI Eye Cam")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        camera_option = camera_selector(key_prefix="eye_cam")
        
        if camera_option == "RTSP URL":
            camera_id = st.text_input("URL de la cámara:", "rtsp://ejemplo.com/stream")
        else:
            camera_id = camera_option

    with col2:
        tracker_option = st.selectbox("🧭 Algoritmo de tracking:", 
                                    ["Liviano (bytetrack) ", "Pesado (botsort)" ])

    with col3:
        fps_option_cam = st.slider("⏱️ FPS:", 
                            min_value=1, max_value=30, value=20, step=1, key="cam_fps_slider")

    if st.button("▶️ Iniciar KOI Eye Cam"):
        with st.spinner("Iniciando cámara con tracking..."):
            koi_eye_cam_path = os.path.abspath("./Codes/koi_eye_cam.py")
            subprocess.Popen([
                "python", 
                koi_eye_cam_path, 
                camera_id,
                tracker_option,
                str(fps_option_cam)
            ])
            webbrowser.open_new_tab("http://localhost:8001/video")
            st.success(f"KOI Eye Cam iniciado con {tracker_option} a {fps_option_cam} FPS")
            

    with col4:

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

        if st.button("🔄 Reiniciar CAM"):
            try:
                response = requests.get("http://localhost:8001/restart", timeout=5)
                if response.status_code == 200:
                    st.success("✅ Captura CAM reiniciada.")
                    # Esperar un momento para que el servicio se inicie
                    time.sleep(2)
                    # Abrir la pestaña de video
                    webbrowser.open_new_tab("http://localhost:8001/video")
                else:
                    st.warning("⚠️ No se pudo reiniciar la captura CAM.")
            except Exception as e:
                st.error(f"❌ Error al reiniciar captura CAM: {e}")

        if st.button("⬇️ Descargar Video CAM"):
            webbrowser.open_new_tab("http://localhost:8001/download")

    # Añadir esta sección después de la sección de KOI Eye Cam

    st.markdown("---")
    st.subheader("📊 Visualización CAM en Tiempo Real")

    if st.button("🌡️ Mapa de Calor CAM"):
        try:
            response = requests.get("http://localhost:8001/heatmap", timeout=10)
            if response.status_code == 200:
                st.image(response.content, caption="Mapa de Calor CAM", use_column_width=True)
            else:
                webbrowser.open_new_tab("http://localhost:8001/heatmap")
        except Exception as e:
            st.warning(f"⚠️ Abriendo en navegador...")
            webbrowser.open_new_tab("http://localhost:8001/heatmap")

    if st.button("🛣️ Trayectorias CAM"):
        try:
            response = requests.get("http://localhost:8001/trajectories", timeout=10)
            if response.status_code == 200:
                st.image(response.content, caption="Mapa de Trayectorias CAM", use_column_width=True)
            else:
                webbrowser.open_new_tab("http://localhost:8001/trajectories")
        except Exception as e:
            st.warning(f"⚠️ Abriendo en navegador...")
            webbrowser.open_new_tab("http://localhost:8001/trajectories")

    # Análisis por ID específico para CAM
    st.markdown("### 🎯 Análisis por ID (CAM)")

    id_persona_cam = st.number_input("ID CAM:", min_value=0, step=1, value=0, key="id_cam")

    if st.button("🔍 Analizar ID CAM"):
        try:
            response = requests.get(f"http://localhost:8001/heatmap/{int(id_persona_cam)}", timeout=15)
            if response.status_code == 200:
                st.image(response.content, caption=f"Análisis ID {id_persona_cam} (CAM)", use_column_width=True)
            elif response.status_code == 404:
                st.warning(f"⚠️ No hay datos para ID {id_persona_cam}")
            else:
                st.error(f"❌ Error al obtener datos: {response.status_code}")
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

    # Para la sección CAM
    st.markdown("### 📊 Datos de Posición CAM")
    cam_id_for_stats = st.number_input("ID para estadísticas (CAM):", min_value=0, step=1, value=0)

    if st.button("📊 Ver Datos CAM", key="simple_stats_cam"):
        try:
            response = requests.get(f"http://localhost:8001/id_positions/{cam_id_for_stats}", timeout=15)
            
            if response.status_code == 200:
                position_data = response.json()
                st.success(f"✅ ID {cam_id_for_stats}: {position_data.get('count', 0)} posiciones registradas")
                
                # Mostrar datos en formato solicitado
                positions = position_data.get("positions", [])
                formatted_text = f"N° ID {cam_id_for_stats} "
                
                for pos in positions:
                    formatted_text += f"{{ frame: {pos['frame']}, x: {pos['x']}, y: {pos['y']} }}, "
                    
                if formatted_text.endswith(", "):
                    formatted_text = formatted_text[:-2]
                    
                st.code(formatted_text, language="text")
                
            elif response.status_code == 404:
                st.warning(f"⚠️ No hay datos para ID {cam_id_for_stats}")
            else:
                st.error(f"❌ Error al obtener datos: {response.status_code}")
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
with tab3:
    # Sección ONLINE con YouTube - SIMPLIFICADA
    st.subheader("🌐 Modo Online con YouTube")
    col1, col2 = st.columns(2)

    with col1:
        tracker_option = st.selectbox("🧭 Elegí el algoritmo de seguimiento:", ["Liviano (bytetrack)", "Pesado (botsort)"])
        fps_live = st.slider("⏱️ FPS:", min_value=1, max_value=30, value=20, step=1, key="online_fps_slider")
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
                #subprocess.Popen(["python", koi_tracker_path, st.session_state.stream_link])
                # Agregar antes de llamar al subprocess

                # Luego llamar al subprocess con el valor seleccionado
                subprocess.Popen(["python", koi_tracker_path, st.session_state.stream_link, tracker_option, str(fps_live)])
                time.sleep(3)
                
                # Abrir múltiples pestañas
                webbrowser.open_new_tab("http://localhost:8000/video")
                #webbrowser.open_new_tab("http://localhost:8000/heatmap")
                #webbrowser.open_new_tab("http://localhost:8000/trajectories")


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

        if st.button("🔄 Reiniciar ONLINE"):
            try:
                response = requests.get("http://localhost:8000/restart", timeout=5)
                if response.status_code == 200:
                    st.success("✅ Stream Online reiniciado.")
                    # Esperar un momento para que el servicio se inicie
                    time.sleep(2)
                    # Abrir la pestaña de video
                    webbrowser.open_new_tab("http://localhost:8000/video")
                else:
                    st.warning("⚠️ No se pudo reiniciar el stream Online.")
            except Exception as e:
                st.error(f"❌ Error al reiniciar stream: {e}")
        

        if st.button("⬇️ Descargar Video Online"):
            webbrowser.open_new_tab("http://localhost:8000/download")



    # VISUALIZACIÓN SIMPLIFICADA
    st.subheader("📊 Visualización en Tiempo Real")

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
    # Para la sección Online
    st.markdown("### 📊 Datos de Posición Online")
    online_id_for_stats = st.number_input("ID para estadísticas (Online):", min_value=0, step=1, value=0)

    if st.button("📊 Ver Datos Online", key="simple_stats_online"):
        try:
            response = requests.get(f"http://localhost:8000/id_positions/{online_id_for_stats}", timeout=15)
            
            if response.status_code == 200:
                position_data = response.json()
                st.success(f"✅ ID {online_id_for_stats}: {position_data.get('count', 0)} posiciones registradas")
                
                # Mostrar datos en formato solicitado
                positions = position_data.get("positions", [])
                formatted_text = f"N° ID {online_id_for_stats} "
                
                for pos in positions:
                    formatted_text += f"{{ frame: {pos['frame']}, x: {pos['x']}, y: {pos['y']} }}, "
                    
                if formatted_text.endswith(", "):
                    formatted_text = formatted_text[:-2]
                    
                st.code(formatted_text, language="text")
                
            elif response.status_code == 404:
                st.warning(f"⚠️ No hay datos para ID {online_id_for_stats}")
            else:
                st.error(f"❌ Error al obtener datos: {response.status_code}")
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

    # Análisis por ID específico
    st.markdown("### 🎯 Análisis por ID")

    id_persona = st.number_input("ID:", min_value=0, step=1, value=0)

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

# Footer
st.markdown("---")
st.markdown("🐟 **KOI Tracker System** - Detección y seguimiento de personas en tiempo real")