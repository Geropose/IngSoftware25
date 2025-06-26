import streamlit as st
import subprocess
import os
import webbrowser
from yt_dlp import YoutubeDL
import requests
import time
import pandas as pd
import cv2
import platform
import re
import json

def get_available_cameras():
    """Detecta las cámaras disponibles en el sistema"""
    available_cameras = []
    system = platform.system()
    
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            name = f"Cámara {i}"
            
            if system == "Windows":
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                name = f"Cámara {i} ({width}x{height})"
            
            available_cameras.append({"id": str(i), "name": name})
            cap.release()
    
    if system == "Windows" and not available_cameras:
        try:
            cmd = "powershell -Command \"Get-CimInstance Win32_PnPEntity | Where-Object {$_.Caption -like '*camera*' -or $_.Caption -like '*webcam*'} | Select-Object Caption | ConvertTo-Json\""
            result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
            
            if result.returncode == 0 and result.stdout.strip():
                try:
                    cameras_info = json.loads(result.stdout)
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
    
    if not available_cameras:
        available_cameras = [
            {"id": "0", "name": "Cámara por defecto (0)"},
            {"id": "1", "name": "Cámara alternativa (1)"},
            {"id": "2", "name": "Cámara adicional (2)"}
        ]
    
    available_cameras.append({"id": "RTSP URL", "name": "Cámara IP (RTSP URL)"})
    
    return available_cameras

def camera_selector(key_prefix="cam"):
    """Muestra un selector de cámaras disponibles"""
    if "available_cameras" not in st.session_state:
        with st.spinner("Detectando cámaras disponibles..."):
            st.session_state.available_cameras = get_available_cameras()
    
    cameras = st.session_state.available_cameras
    
    camera_options = [cam["name"] for cam in cameras]
    camera_ids = [cam["id"] for cam in cameras]
    
    selected_index = st.selectbox(
        "📷 Seleccionar cámara:",
        options=range(len(camera_options)),
        format_func=lambda i: camera_options[i],
        key=f"{key_prefix}_camera_select"
    )
    
    selected_camera_id = camera_ids[selected_index]
    
    if selected_camera_id == "RTSP URL":
        rtsp_url = st.text_input(
            "URL de la cámara RTSP:",
            value="rtsp://usuario:contraseña@192.168.1.100:554/stream",
            key=f"{key_prefix}_rtsp_url"
        )
        return rtsp_url
    else:
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

def format_trajectory_events(events_data):
    """Formatea los eventos de trayectoria en el formato solicitado"""
    if not events_data or "eventos" not in events_data:
        return "No hay eventos de cambio de dirección disponibles"
        
    id_num = events_data.get("id", "?")
    eventos = events_data.get("eventos", [])
    
    if not eventos:
        return f"ID {id_num}: No hay cambios de dirección registrados"
    
    formatted_text = f"🔄 **Eventos de Cambio de Dirección - ID {id_num}**\n\n"
    
    for i, evento in enumerate(eventos, 1):
        formatted_text += f"**Evento {i}:**\n"
        formatted_text += f"• Frame {evento['frame']}: cambió su dirección de **{evento['direccion_anterior']}** hacia **{evento['direccion_nueva']}**\n"
        formatted_text += f"• Distancia recorrida: {evento['distancia']} píxeles\n"
        formatted_text += f"• Posición: ({evento['posicion_x']}, {evento['posicion_y']})\n\n"
        
    return formatted_text

def format_direction_events_new(events_data):
    """Formatea los eventos de dirección con la nueva estructura de datos"""
    if not events_data or "events" not in events_data:
        return "No hay eventos de cambio de dirección disponibles"
        
    id_num = events_data.get("id", "?")
    eventos = events_data.get("events", [])
    
    if not eventos:
        return f"ID {id_num}: No hay cambios de dirección registrados"
    
    formatted_text = f"🔄 **Eventos de Cambio de Dirección - ID {id_num}**\n\n"
    formatted_text += f"**Total de eventos:** {len(eventos)}\n\n"
    
    for i, evento in enumerate(eventos, 1):
        formatted_text += f"**Evento {i}:**\n"
        formatted_text += f"• {evento.get('description', 'N/A')}\n"
        
        # Información de posición
        position = evento.get('position', {})
        if position:
            formatted_text += f"• Posición: ({position.get('x', 0)}, {position.get('y', 0)})\n"
        
        # Información de ángulos
        angles = evento.get('angles', {})
        if angles:
            formatted_text += f"• Cambio de ángulo: {angles.get('change', 0)}°\n"
            formatted_text += f"• Ángulos: {angles.get('previous', 0)}° → {angles.get('current', 0)}°\n"
        
        # Timestamp si está disponible
        if 'timestamp' in evento:
            from datetime import datetime
            timestamp_str = datetime.fromtimestamp(evento['timestamp']).strftime('%H:%M:%S')
            formatted_text += f"• Tiempo: {timestamp_str}\n"
        
        formatted_text += "\n"
        
    return formatted_text

def get_ids_with_events(port):
    """Obtiene los IDs que tienen eventos de cambio de dirección"""
    try:
        response = requests.get(f"http://localhost:{port}/ids_with_events", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get("ids_with_events", [])
        else:
            return []
    except:
        return []

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
        tracker_display = st.selectbox("🧭 Algoritmo de tracking:", 
                                    ["Liviano (bytetrack)", "Pesado (botsort)"])
        
        tracker_mapping = {
            "Liviano (bytetrack)": "bytetrack",
            "Pesado (botsort)": "botsort"
        }
        tracker_option = tracker_mapping[tracker_display]

    with col3:
        fps_option_cam = st.slider("⏱️ FPS:", 
                            min_value=1, max_value=30, value=20, step=1, key="cam_fps_slider")

    # NUEVA SECCIÓN: Parámetros de detección de cambios de dirección para CAM
    st.markdown("### 🔄 Configuración de Detección de Cambios de Dirección (CAM)")
    col_cam_1, col_cam_2 = st.columns(2)
    
    with col_cam_1:
        umbral_angulo_cam = st.slider(
            "📐 Umbral de ángulo (CAM):", 
            min_value=15, max_value=90, value=30, step=5, 
            key="cam_umbral_angulo",
            help="Cambio mínimo de ángulo para considerar un cambio de dirección"
        )
    
    with col_cam_2:
        min_distancia_cam = st.slider(
            "📏 Distancia mínima (CAM):", 
            min_value=5, max_value=50, value=10, step=5, 
            key="cam_min_distancia",
            help="Distancia mínima entre puntos para calcular dirección"
        )

    if st.button("▶️ Iniciar KOI Eye Cam"):
        with st.spinner("Iniciando cámara con tracking..."):
            koi_eye_cam_path = os.path.abspath("./Codes/koi_eye_cam.py")
            subprocess.Popen([
                "python", 
                koi_eye_cam_path, 
                camera_id,
                tracker_option,
                str(fps_option_cam),
                str(umbral_angulo_cam),
                str(min_distancia_cam)
            ])
            time.sleep(10)
            webbrowser.open_new_tab("http://localhost:8001/video")
            st.success(f"KOI Eye Cam iniciado con {tracker_option} a {fps_option_cam} FPS")
            st.info(f"Parámetros: Umbral {umbral_angulo_cam}°, Distancia mín {min_distancia_cam}px")

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
                    time.sleep(2)
                    webbrowser.open_new_tab("http://localhost:8001/video")
                else:
                    st.warning("⚠️ No se pudo reiniciar la captura CAM.")
            except Exception as e:
                st.error(f"❌ Error al reiniciar captura CAM: {e}")

        if st.button("⬇️ Descargar Video CAM"):
            webbrowser.open_new_tab("http://localhost:8001/download")

    st.markdown("---")
    st.subheader("📊 Visualización CAM en Tiempo Real")

    if st.button("🌡️ Mapa de Calor CAM"):
        try:
            response = requests.get("http://localhost:8001/heatmap", timeout=10)
            if response.status_code == 200:
                st.image(response.content, caption="Mapa de Calor CAM", use_container_width=True)
            else:
                webbrowser.open_new_tab("http://localhost:8001/heatmap")
        except Exception as e:
            st.warning(f"⚠️ Abriendo en navegador...")
            webbrowser.open_new_tab("http://localhost:8001/heatmap")

    if st.button("🛣️ Trayectorias CAM"):
        try:
            response = requests.get("http://localhost:8001/trajectories", timeout=10)
            if response.status_code == 200:
                st.image(response.content, caption="Mapa de Trayectorias CAM", use_container_width=True)
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
                st.image(response.content, caption=f"Análisis ID {id_persona_cam} (CAM)", use_container_width=True)
            elif response.status_code == 404:
                st.warning(f"⚠️ No hay datos para ID {id_persona_cam}")
            else:
                st.error(f"❌ Error al obtener datos: {response.status_code}")
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

    # SECCIÓN ACTUALIZADA: Eventos de cambio de dirección para CAM
    st.markdown("### 🔄 Eventos de Cambio de Dirección (CAM)")
    
    # Obtener IDs con eventos para CAM (si el endpoint existe)
    ids_with_events_cam = get_ids_with_events(8001)
    
    if ids_with_events_cam:
        # Mostrar selector solo con IDs que tienen eventos
        cam_id_options = [f"ID {item['id']} ({item['event_count']} eventos)" for item in ids_with_events_cam]
        cam_id_values = [item['id'] for item in ids_with_events_cam]
        
        selected_cam_index = st.selectbox(
            "Seleccionar ID con eventos de dirección (CAM):",
            options=range(len(cam_id_options)),
            format_func=lambda i: cam_id_options[i],
            key="cam_events_selector"
        )
        
        cam_id_for_events = cam_id_values[selected_cam_index]
        
        if st.button("🔄 Ver Eventos de Dirección CAM", key="direction_events_cam"):
            try:
                response = requests.get(f"http://localhost:8001/direction_events/{cam_id_for_events}", timeout=15)
                
                if response.status_code == 200:
                    events_data = response.json()
                    st.success(f"✅ ID {cam_id_for_events}: {events_data.get('total_events', 0)} eventos de cambio de dirección")
                    
                    formatted_events = format_direction_events_new(events_data)
                    st.markdown(formatted_events)
                    
                elif response.status_code == 404:
                    st.warning(f"⚠️ No hay eventos de cambio de dirección para ID {cam_id_for_events}")
                else:
                    st.error(f"❌ Error al obtener eventos: {response.status_code}")
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    else:
        # Fallback al método anterior si no hay endpoint nuevo
        cam_id_for_events = st.number_input("ID para eventos de dirección (CAM):", min_value=0, step=1, value=0, key="cam_events_id")

        if st.button("🔄 Ver Eventos de Dirección CAM", key="trajectory_events_cam"):
            try:
                response = requests.get(f"http://localhost:8001/trajectory_events/{cam_id_for_events}", timeout=15)
                
                if response.status_code == 200:
                    events_data = response.json()
                    st.success(f"✅ ID {cam_id_for_events}: {events_data.get('total_eventos', 0)} eventos de cambio de dirección")
                    
                    formatted_events = format_trajectory_events(events_data)
                    st.markdown(formatted_events)
                    
                elif response.status_code == 404:
                    st.warning(f"⚠️ No hay eventos de cambio de dirección para ID {cam_id_for_events}")
                else:
                    st.error(f"❌ Error al obtener eventos: {response.status_code}")
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    # Datos de posición para CAM
    st.markdown("### 📊 Datos de Posición CAM")
    cam_id_for_stats = st.number_input("ID para estadísticas (CAM):", min_value=0, step=1, value=0)

    if st.button("📊 Ver Datos CAM", key="simple_stats_cam"):
        try:
            response = requests.get(f"http://localhost:8001/id_positions/{cam_id_for_stats}", timeout=15)
            
            if response.status_code == 200:
                position_data = response.json()
                st.success(f"✅ ID {cam_id_for_stats}: {position_data.get('count', 0)} posiciones registradas")
                
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
    st.subheader("🌐 Modo Online con YouTube")
    col1, col2 = st.columns(2)

    with col1:
        tracker_display = st.selectbox("🧭 Elegí el algoritmo de seguimiento:", 
                                 ["Liviano (bytetrack)", "Pesado (botsort)"])
    
        tracker_mapping = {
            "Liviano (bytetrack)": "bytetrack", 
            "Pesado (botsort)": "botsort"
        }
        tracker_option = tracker_mapping[tracker_display]
        fps_live = st.slider("⏱️ FPS:", min_value=1, max_value=30, value=20, step=1, key="online_fps_slider")
        video_url = st.text_input("📥 Pegá aquí el link del live de YOUTUBE:")

    # NUEVA SECCIÓN: Parámetros de detección de cambios de dirección para Online
    st.markdown("### 🔄 Configuración de Detección de Cambios de Dirección (Online)")
    col_online_1, col_online_2 = st.columns(2)
    
    with col_online_1:
        umbral_angulo_online = st.slider(
            "📐 Umbral de ángulo (Online):", 
            min_value=15, max_value=90, value=30, step=5, 
            key="online_umbral_angulo",
            help="Cambio mínimo de ángulo para considerar un cambio de dirección"
        )
    
    with col_online_2:
        min_distancia_online = st.slider(
            "📏 Distancia mínima (Online):", 
            min_value=5, max_value=50, value=10, step=5, 
            key="online_min_distancia",
            help="Distancia mínima entre puntos para calcular dirección"
        )

    if video_url:
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
        
        if st.button("🔥 KOI-TRACKER LIVE ONLINE"):
            st.success("Iniciando KOI-TRACKER LIVE ONLINE")
            koi_tracker_path = os.path.abspath("./Codes/koi_tracker_live.py")
            subprocess.Popen([
                "python", 
                koi_tracker_path, 
                st.session_state.stream_link, 
                tracker_option, 
                str(fps_live),
                str(umbral_angulo_online),
                str(min_distancia_online)
            ])
            time.sleep(15)
            webbrowser.open_new_tab("http://localhost:8000/video")
            st.info(f"Parámetros: Umbral {umbral_angulo_online}°, Distancia mín {min_distancia_online}px")

    with col2:
        st.markdown("**🌐 Control ONLINE (Puerto 8000):**")

        if st.button("🛑 Detener ONLINE"):
            try:
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
                    time.sleep(2)
                    webbrowser.open_new_tab("http://localhost:8000/video")
                else:
                    st.warning("⚠️ No se pudo reiniciar el stream Online.")
            except Exception as e:
                st.error(f"❌ Error al reiniciar stream: {e}")

        if st.button("⬇️ Descargar Video Online"):
            webbrowser.open_new_tab("http://localhost:8000/download")

    st.subheader("📊 Visualización en Tiempo Real")

    if st.button("🌡️ Mapa de Calor"):
        try:
            response = requests.get("http://localhost:8000/heatmap", timeout=10)
            if response.status_code == 200:
                st.image(response.content, caption="Mapa de Calor General", use_container_width=True)
            else:
                webbrowser.open_new_tab("http://localhost:8000/heatmap")
        except Exception as e:
            st.warning(f"⚠️ Abriendo en navegador...")
            webbrowser.open_new_tab("http://localhost:8000/heatmap")

    if st.button("🛣️ Trayectorias"):
        try:
            response = requests.get("http://localhost:8000/trajectories", timeout=10)
            if response.status_code == 200:
                st.image(response.content, caption="Mapa de Trayectorias", use_container_width=True)
            else:
                webbrowser.open_new_tab("http://localhost:8000/trajectories")
        except Exception as e:
            st.warning(f"⚠️ Abriendo en navegador...")
            webbrowser.open_new_tab("http://localhost:8000/trajectories")

    # SECCIÓN COMPLETAMENTE ACTUALIZADA: Eventos de cambio de dirección para Online
    st.markdown("### 🔄 Eventos de Cambio de Dirección Online")
    
    # NUEVO BOTÓN: Verificar IDs con cambios de dirección
    if st.button("🔍 Verificar IDs con Cambios de Dirección", key="verify_direction_changes_online"):
        with st.spinner("🔄 Verificando IDs con cambios de dirección..."):
            
            # Obtener IDs con eventos usando el endpoint
            ids_with_events_online = get_ids_with_events(8000)
            
            if ids_with_events_online:
                st.success(f"✅ Se encontraron {len(ids_with_events_online)} IDs con cambios de dirección")
                
                # Mostrar resumen de IDs encontrados
                for item in ids_with_events_online:
                    st.info(f"📍 ID {item['id']}: {item['event_count']} eventos detectados")
                
                # Guardar en session_state para usar en el selector
                st.session_state.verified_ids_online = ids_with_events_online
                st.session_state.verification_done_online = True
                
            else:
                st.warning("⚠️ No se encontraron IDs con cambios de dirección")
                st.markdown("""
                **Posibles razones:**
                - El tracker aún no ha procesado suficientes frames
                - Los objetos se mueven en línea recta
                - Los cambios de dirección son menores al umbral configurado
                - Las trayectorias son muy cortas
                """)
                st.session_state.verified_ids_online = []
                st.session_state.verification_done_online = True
    
    # Mostrar selector solo si se ha verificado y hay IDs con eventos
    if st.session_state.get('verification_done_online', False):
        verified_ids = st.session_state.get('verified_ids_online', [])
        
        if verified_ids:
            # Mostrar selector solo con IDs que tienen eventos
            online_id_options = [f"ID {item['id']} ({item['event_count']} eventos)" for item in verified_ids]
            online_id_values = [item['id'] for item in verified_ids]
            
            selected_online_index = st.selectbox(
                "Seleccionar ID con eventos de dirección (Online):",
                options=range(len(online_id_options)),
                format_func=lambda i: online_id_options[i],
                key="online_events_selector"
            )
            
            online_id_for_events = online_id_values[selected_online_index]
            
            # Verificar estado del tracker
            try:
                stats_response = requests.get("http://localhost:8000/stats", timeout=3)
                tracker_running = False
                if stats_response.status_code == 200:
                    stats_data = stats_response.json()
                    tracker_running = stats_data.get('status') == 'running'
            except:
                tracker_running = False
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔄 Ver Eventos de Dirección Online", key="direction_events_online"):
                    try:
                        response = requests.get(f"http://localhost:8000/direction_events/{online_id_for_events}", timeout=15)
                        
                        if response.status_code == 200:
                            events_data = response.json()
                            st.success(f"✅ ID {online_id_for_events}: {events_data.get('total_events', 0)} eventos de cambio de dirección")
                            
                            formatted_events = format_direction_events_new(events_data)
                            st.markdown(formatted_events)
                            
                        elif response.status_code == 404:
                            st.warning(f"⚠️ No hay eventos de cambio de dirección para ID {online_id_for_events}")
                        else:
                            st.error(f"❌ Error al obtener eventos: {response.status_code}")
                            
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            
            with col2:
                # Mostrar estado del tracker y botón de reporte
                if tracker_running:
                    st.warning("🟡 Tracker en ejecución")
                    st.info("📝 El reporte estará disponible cuando el tracker termine")
                else:
                    st.success("🟢 Tracker detenido")
                    
                    if st.button("📋 Generar Reporte TXT", key="generate_report_online"):
                        try:
                            response = requests.get(f"http://localhost:8000/direction_report/{online_id_for_events}", timeout=15)
                            
                            if response.status_code == 200:
                                # Crear botón de descarga
                                from datetime import datetime
                                filename = f"reporte_eventos_id_{online_id_for_events}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                                
                                st.download_button(
                                    label="⬇️ Descargar Reporte",
                                    data=response.content,
                                    file_name=filename,
                                    mime="text/plain"
                                )
                                st.success("✅ Reporte generado correctamente")
                                
                            elif response.status_code == 423:
                                st.error("❌ El reporte solo está disponible cuando el tracker está detenido")
                            elif response.status_code == 404:
                                st.warning(f"⚠️ No hay eventos para ID {online_id_for_events}")
                            else:
                                st.error(f"❌ Error al generar reporte: {response.status_code}")
                                
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
        else:
            st.info("🔍 No hay IDs con eventos de cambio de dirección detectados")
    else:
        st.info("👆 Presiona el botón 'Verificar IDs con Cambios de Dirección' para buscar IDs con eventos")

    # Datos de posición para Online
    st.markdown("### 📊 Datos de Posición Online")
    online_id_for_stats = st.number_input("ID para estadísticas (Online):", min_value=0, step=1, value=0)

    if st.button("📊 Ver Datos Online", key="simple_stats_online"):
        try:
            response = requests.get(f"http://localhost:8000/id_positions/{online_id_for_stats}", timeout=15)
            
            if response.status_code == 200:
                position_data = response.json()
                st.success(f"✅ ID {online_id_for_stats}: {position_data.get('count', 0)} posiciones registradas")
                
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
                st.image(response.content, caption=f"Análisis ID {id_persona}", use_container_width=True)
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
