import streamlit as st
from tracker import (
    procesar_video as procesar_video,
    generar_mapa_calor_general as generar_mapa_calor_general,
    generar_mapa_calor_por_id as generar_mapa_calor_por_id,
    generar_mapa_trayectorias as generar_mapa_trayectorias,
    detectar_grupos as detectar_grupos,
    generar_mapa_calor_grupos as generar_mapa_calor_grupos,
    calcular_proximidad as calcular_proximidad,
    generar_mapa_grupos as generar_mapa_grupos
)
from trackerDeepSort import (
    procesar_video_deepsort,
    generar_mapa_calor_general_deepsort,
    generar_mapa_calor_por_id_deepsort,
    generar_mapa_trayectorias_deepsort,
    detectar_grupos as detectar_grupos_deepsort,
    generar_mapa_calor_grupos as generar_mapa_calor_grupos_deepsort
)
import tempfile
import os
import time
import matplotlib.pyplot as plt
import cv2
import subprocess
from pathlib import Path
import numpy as np
import math

# Función para calcular el ángulo de dirección
def calcular_angulo_direccion(punto1, punto2):
    """
    Calcula el ángulo de dirección entre dos puntos
    
    Args:
        punto1: (x1, y1, frame1)
        punto2: (x2, y2, frame2)
    
    Returns:
        float: Ángulo en grados (0-360)
    """
    dx = punto2[0] - punto1[0]
    dy = punto2[1] - punto1[1]
    
    if dx == 0 and dy == 0:
        return None
    
    # Calcular ángulo en radianes y convertir a grados
    angulo_rad = math.atan2(dy, dx)
    angulo_deg = math.degrees(angulo_rad)
    
    # Normalizar a 0-360 grados
    if angulo_deg < 0:
        angulo_deg += 360
    
    return angulo_deg

def direccion_a_texto(angulo):
    """
    Convierte un ángulo a descripción textual de dirección
    
    Args:
        angulo: Ángulo en grados (0-360)
    
    Returns:
        str: Descripción de la dirección
    """
    if angulo is None:
        return "Sin movimiento"
    
    # Definir rangos de direcciones
    if 337.5 <= angulo or angulo < 22.5:
        return "derecha"
    elif 22.5 <= angulo < 67.5:
        return "abajo-derecha"
    elif 67.5 <= angulo < 112.5:
        return "abajo"
    elif 112.5 <= angulo < 157.5:
        return "abajo-izquierda"
    elif 157.5 <= angulo < 202.5:
        return "izquierda"
    elif 202.5 <= angulo < 247.5:
        return "arriba-izquierda"
    elif 247.5 <= angulo < 292.5:
        return "arriba"
    elif 292.5 <= angulo < 337.5:
        return "arriba-derecha"
    else:
        return "dirección desconocida"

def detectar_cambios_direccion(posiciones, umbral_angulo=30, min_distancia=10):
    """
    Detecta cambios significativos de dirección en las trayectorias
    
    Args:
        posiciones: Diccionario con posiciones por ID
        umbral_angulo: Cambio mínimo de ángulo para considerar cambio de dirección
        min_distancia: Distancia mínima entre puntos para calcular dirección
    
    Returns:
        dict: Eventos de cambio de dirección por ID
    """
    eventos_cambio = {}
    
    for id_obj, trayectoria in posiciones.items():
        if len(trayectoria) < 3:  # Necesitamos al menos 3 puntos
            continue
        
        # Ordenar por frame
        trayectoria_ordenada = sorted(trayectoria, key=lambda x: x[3])
        eventos_id = []
        
        direccion_anterior = None
        
        for i in range(1, len(trayectoria_ordenada)):
            punto_anterior = trayectoria_ordenada[i-1]
            punto_actual = trayectoria_ordenada[i]
            
            # Calcular distancia para filtrar movimientos muy pequeños
            distancia = math.sqrt(
                (punto_actual[0] - punto_anterior[0])**2 + 
                (punto_actual[1] - punto_anterior[1])**2
            )
            
            if distancia < min_distancia:
                continue
            
            # Calcular dirección actual
            angulo_actual = calcular_angulo_direccion(punto_anterior, punto_actual)
            
            if angulo_actual is None:
                continue
            
            # Comparar con dirección anterior
            if direccion_anterior is not None:
                diferencia_angulo = abs(angulo_actual - direccion_anterior)
                
                # Manejar el caso de cruce de 0/360 grados
                if diferencia_angulo > 180:
                    diferencia_angulo = 360 - diferencia_angulo
                
                # Si hay un cambio significativo de dirección
                if diferencia_angulo >= umbral_angulo:
                    direccion_texto = direccion_a_texto(angulo_actual)
                    
                    evento = {
                        'frame': punto_actual[3],
                        'posicion': (punto_actual[0], punto_actual[1]),
                        'direccion_anterior': direccion_a_texto(direccion_anterior),
                        'direccion_nueva': direccion_texto,
                        'angulo_anterior': round(direccion_anterior, 1),
                        'angulo_nuevo': round(angulo_actual, 1),
                        'cambio_angulo': round(diferencia_angulo, 1)
                    }
                    
                    eventos_id.append(evento)
            
            direccion_anterior = angulo_actual
        
        if eventos_id:
            eventos_cambio[id_obj] = eventos_id
    
    return eventos_cambio

# Función auxiliar para seleccionar las funciones correctas según el algoritmo
def get_tracker_functions(algoritmo='ByteTrack'):
    """
    Devuelve las funciones correspondientes al algoritmo seleccionado
    
    Args:
        algoritmo (str): Nombre del algoritmo ('ByteTrack' o 'DeepSORT')
        
    Returns:
        tuple: (procesar_video, generar_mapa_calor_general,
        generar_mapa_calor_por_id, generar_mapa_trayectorias,
        detectar_grupos_func, generar_mapa_calor_grupos_func)
    """
    if algoritmo == 'ByteTrack' or algoritmo == 'BotSort':
        return (
            procesar_video,
            generar_mapa_calor_general,
            generar_mapa_calor_por_id,
            generar_mapa_trayectorias,
            detectar_grupos,
            generar_mapa_calor_grupos
        )
    else:  # DeepSORT
        return (
            procesar_video_deepsort,
            generar_mapa_calor_general_deepsort,
            generar_mapa_calor_por_id_deepsort,
            generar_mapa_trayectorias_deepsort,
            detectar_grupos_deepsort,
            generar_mapa_calor_grupos_deepsort
        )

# Función para reiniciar el estado de procesamiento
def reset_processing_state():
    """
    Reinicia el estado relacionado con el procesamiento del video
    cuando se cambia de algoritmo de tracking
    """
    # Lista de claves que deben reiniciarse
    keys_to_reset = [
        'output_path', 
        'posiciones', 
        'video_dims', 
        'selected_id',  # Importante: reiniciar el ID seleccionado
        'algoritmo_usado',
        'tiempo_procesamiento',
        'num_ids_detectados',
        'total_puntos',
        'eventos_cambio_direccion'  # Agregar eventos de cambio de dirección
    ]
    
    # Eliminar estas claves del estado de la sesión
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]
    
    # Mantener la bandera 'processed' para indicar que hay un video procesado
    # pero forzar la regeneración de los mapas
    if 'processed' in st.session_state:
        st.session_state.processed = True
        st.session_state.maps_need_update = True

# Funciones para manejo de video
def convert_video_for_streamlit(input_path):
    """
    Convierte un video a un formato compatible con Streamlit
    """
    output_path = os.path.splitext(input_path)[0] + "_streamlit.mp4"
    
    # Intentar usar FFmpeg primero (más confiable)
    if is_ffmpeg_installed():
        try:
            cmd = [
                "ffmpeg", "-y", "-i", input_path,
                "-c:v", "libx264", "-preset", "fast",
                "-pix_fmt", "yuv420p", "-c:a", "aac",
                output_path
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return output_path
        except Exception as e:
            st.warning(f"Error al convertir con FFmpeg: {str(e)}. Intentando con OpenCV...")
    
    # Fallback a OpenCV si FFmpeg no está disponible o falla
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            st.error(f"No se pudo abrir el video: {input_path}")
            return input_path
        
        # Obtener propiedades del video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Configurar el codec y el writer
        fourcc = cv2.VideoWriter_fourcc(*'H264') if is_codec_available('H264') else cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Procesar el video frame por frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        
        # Liberar recursos
        cap.release()
        out.release()
        
        return output_path
    except Exception as e:
        st.error(f"Error al convertir video con OpenCV: {str(e)}")
        return input_path  # Devolver el original como último recurso

def is_ffmpeg_installed():
    """Verifica si FFmpeg está instalado en el sistema"""
    try:
        subprocess.run(
            ["ffmpeg", "-version"], 
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def is_codec_available(codec):
    """Verifica si un codec específico está disponible en OpenCV"""
    try:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        test_file = tempfile.NamedTemporaryFile(suffix='.mp4').name
        writer = cv2.VideoWriter(test_file, fourcc, 30, (640, 480))
        is_available = writer.isOpened()
        writer.release()
        if os.path.exists(test_file):
            os.remove(test_file)
        return is_available
    except:
        return False

def save_uploaded_video(uploaded_file):
    """
    Guarda un archivo de video subido y devuelve la ruta
    """
    # Crear un directorio temporal si no existe
    temp_dir = Path(tempfile.gettempdir()) / "streamlit_videos"
    temp_dir.mkdir(exist_ok=True)
    
    # Generar un nombre de archivo único
    file_extension = Path(uploaded_file.name).suffix
    temp_path = temp_dir / f"uploaded_video_{hash(uploaded_file.name)}{file_extension}"
    
    # Guardar el archivo
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return str(temp_path)

# Función para reiniciar el estado cuando se sube un nuevo video
def reset_session_state_for_new_video():
    # Mantener solo las configuraciones y preferencias del usuario
    keys_to_keep = ['current_tab', 'convert_video', 'show_video_always']
    
    # Guardar valores que queremos mantener
    saved_values = {}
    for key in keys_to_keep:
        if key in st.session_state:
            saved_values[key] = st.session_state[key]
    
    # Limpiar variables relacionadas con el video procesado
    for key in list(st.session_state.keys()):
        if key not in keys_to_keep:
            del st.session_state[key]
    
    # Restaurar valores guardados
    for key, value in saved_values.items():
        st.session_state[key] = value

# Inicializar el estado de la sesión si no existe
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "Procesamiento"

# Configuración de la página
st.set_page_config(
    page_title="Tracker + Heatmap", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos personalizados
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
    }
    /* Estilo para mejorar la visualización de videos */
    .stVideo > div {
        min-height: 300px;
    }
    /* Estilos para los logs de eventos */
    .evento-cambio {
        background-color: #f0f2f6;
        border-left: 4px solid #1f77b4;
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
    }
    .evento-frame {
        font-weight: bold;
        color: #1f77b4;
    }
    .evento-direccion {
        color: #d62728;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Título principal
st.title("KOI EYE")

# Barra lateral con configuraciones
with st.sidebar:
    st.header("Configuración")
    
    # Parámetros para los mapas de calor
    st.subheader("Mapas de Calor")
    sigma_general = st.slider("Suavizado mapa general", 5, 30, 15)
    sigma_id = st.slider("Suavizado mapa por ID", 5, 30, 10)
    
    # Parámetros para el tracker
    min_frames = 5#st.slider("Frames mínimos para trayectoria", 5, 50, 10)
    
    # Parámetros para detección de cambios de dirección
    st.subheader("Detección de Cambios de Dirección")
    umbral_angulo = st.slider("Umbral de cambio de ángulo (grados)", 15, 90, 30, 
                             help="Cambio mínimo de ángulo para considerar un cambio de dirección")
    min_distancia = st.slider("Distancia mínima de movimiento", 5, 50, 10,
                             help="Distancia mínima entre puntos para calcular dirección")

    # Parámetros para la detección de grupos
    st.subheader("Agrupamiento")
    max_distancia = st.slider("Distancia máxima para agrupar (píxeles)", 50, 200, 100)
    min_frames_grupo = st.slider("Mínimo de frames juntos", 5, 50, 10)
    
    # Selector de algoritmo de tracking
    st.subheader("Algoritmo de Tracking")
    algoritmo_tracking = st.radio(
        "Selecciona el algoritmo de tracking:",
        ["ByteTrack", "BotSort"],
        help=(
        "¿Qué opción elegir según tu computadora?\n"
        " No fuerza CPU ni GPU, detecta automáticamente el mejor dispositivo. \n"
        "- ByteTrack (rápido y liviano): Elegí esta opción si tu computadora no tiene placa de video (GPU) "
        "o si querés que el video se procese lo más rápido posible. Es ideal para la mayoría de los casos.\n"
        "- BoT-SORT (más preciso): Usá esta opción si tu computadora tiene una buena GPU y necesitás mayor precisión "
        "para seguir personas en lugares con mucha gente o movimiento.\n"
    )    
    )

    if 'last_algoritmo_tracking' not in st.session_state:
        st.session_state.last_algoritmo_tracking = algoritmo_tracking
    elif algoritmo_tracking != st.session_state.last_algoritmo_tracking:
        st.session_state.last_algoritmo_tracking = algoritmo_tracking
        reset_processing_state()
        st.rerun()

    # Guardar la selección en el estado de la sesión para mantenerla entre recargas
    st.session_state.algoritmo_tracking = algoritmo_tracking

    # Opciones de video
    st.subheader("Opciones de Video")
    st.session_state.convert_video = st.checkbox("Convertir video automáticamente", value=True, 
                               help="Convierte el video a un formato compatible con Streamlit")
    
    # Mostrar video procesado en todas las pestañas
    st.session_state.show_video_always = st.checkbox("Mostrar video en todas las pestañas", value=True,
                                  help="Mantiene visible el video procesado en todas las pestañas")
    

    st.subheader("Resolución del video")
    resoluciones_disponibles = {
        "Original": None,
        "1920x1080": (1920, 1080),
        "1280x720": (1280, 720),
        "854x480": (854, 480),
        "640x360": (640, 360)
    }

    resolucion_seleccionada = st.selectbox(
        "Elegí la resolución de salida:",
        options=list(resoluciones_disponibles.keys()),
    )

    # Detectar cambio de resolución
    if 'last_resolucion' not in st.session_state:
        st.session_state.last_resolucion = resolucion_seleccionada
    elif resolucion_seleccionada != st.session_state.last_resolucion:
        st.session_state.last_resolucion = resolucion_seleccionada
        reset_processing_state()  # Opcional, si querés limpiar procesamiento anterior
        st.rerun()

    # Guardar resolución seleccionada actual
    st.session_state.resolucion_seleccionada = resoluciones_disponibles[resolucion_seleccionada]


# Subir video
video_file = st.file_uploader("Subí un video", type=["mp4", "avi", "mov", "mkv", "webm"], 
                             on_change=reset_session_state_for_new_video)

if video_file:
    # Guardar el video subido de manera más robusta
    if 'temp_video_path' not in st.session_state:
        st.session_state.temp_video_path = save_uploaded_video(video_file)
        st.session_state.video_file_name = video_file.name  # Guardar el nombre del archivo para referencia
    
    # Convertir el video si es necesario
    if 'display_video_path' not in st.session_state:
        if st.session_state.get('convert_video', True):
            with st.spinner("Preparando video para visualización..."):
                st.session_state.display_video_path = convert_video_for_streamlit(st.session_state.temp_video_path)
                st.success("Video preparado correctamente")
        else:
            st.session_state.display_video_path = st.session_state.temp_video_path
    
    # Crear pestañas para organizar la interfaz
    tab_names = ["Procesamiento", "Mapas de Calor", "Trayectorias", "Eventos de Dirección", "Grupos"]
    tabs = st.tabs(tab_names)
    
    # Mostrar video procesado en la parte superior si está habilitado y procesado
    if (st.session_state.get('show_video_always', True) and 
        st.session_state.get('processed', False) and 
        st.session_state.current_tab != "Procesamiento"):
        
        st.subheader("Video Procesado con IDs")
        col1, col2 = st.columns([3, 1])
        with col1:
            try:
                st.video(st.session_state.output_path)
            except Exception as e:
                st.error(f"No se pudo mostrar el video procesado: {str(e)}")
        
        with col2:
            if 'posiciones' in st.session_state:
                st.write(f"IDs detectados: {len(st.session_state.posiciones)}")
                total_points = sum(len(points) for points in st.session_state.posiciones.values())
                st.write(f"Total de puntos: {total_points}")
                if 'grupos' in st.session_state:
                    total_grupos = sum(len(g) for g in st.session_state.grupos.values())
                    st.write(f"Grupos detectados: {total_grupos}")
                st.write(f"Video: {st.session_state.video_file_name}")
        
        st.markdown("---")
    
    with tabs[0]:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Video Original")
            # Mostrar el video convertido
            try:
                st.video(st.session_state.display_video_path)
            except Exception as e:
                st.error(f"No se pudo mostrar el video: {str(e)}")
                st.info("Intenta activar la opción 'Convertir video automáticamente' en la barra lateral.")
        
        with col2:
            st.subheader("Video Procesado")
            
            # Botón para iniciar procesamiento
            if st.button("Procesar Video", type="primary"):
                with st.spinner("Procesando video..."):
                    # Mostrar barra de progreso
                    progress_bar = st.progress(0)
                    
                    # Función para actualizar progreso (simulada)
                    def update_progress():
                        for i in range(100):
                            time.sleep(0.05)  # Simular procesamiento
                            progress_bar.progress(i + 1)
                    
                    # En una implementación real, el progreso vendría del procesamiento
                    # Aquí lo simulamos para la demo
                    update_progress()
                    
                    try:
                        # Procesar video (en una implementación real)
                        # Obtener las funciones correspondientes al algoritmo seleccionado
                        algoritmo_seleccionado = st.session_state.get('algoritmo_tracking', 'ByteTrack')
                        procesar_video, _, _, _, detectar_grupos_func, _ = get_tracker_functions(algoritmo_seleccionado)
    
                         # Procesar video con el algoritmo seleccionado -- Se agrega el siguiente parametro en la funcion procesar_video nueva_resolucion=st.session_state.resolucion_seleccionada -- Dia del cambio 2025-05-10
                        output_path, posiciones, video_dims = procesar_video(st.session_state.temp_video_path, nueva_resolucion=st.session_state.resolucion_seleccionada,algoritmo=algoritmo_seleccionado)
                        st.session_state.algoritmo_usado = algoritmo_seleccionado  # Guardar el algoritmo usado

                        # Convertir el video de salida para asegurar compatibilidad
                        if st.session_state.get('convert_video', True):
                            output_display_path = convert_video_for_streamlit(output_path)
                        else:
                            output_display_path = output_path
                        
                        # Detectar cambios de dirección
                        eventos_cambio = detectar_cambios_direccion(
                            posiciones, 
                            umbral_angulo=umbral_angulo, 
                            min_distancia=min_distancia
                        )
                        
                        # Guardar resultados en la sesión
                        st.session_state.output_path = output_display_path
                        st.session_state.posiciones = posiciones
                        st.session_state.video_dims = video_dims
                        st.session_state.grupos = detectar_grupos_func(posiciones)
                        st.session_state.eventos_cambio_direccion = eventos_cambio
                        st.session_state.processed = True
                        
                        st.success("Procesamiento terminado.")
                        
                        # Mostrar resumen de eventos detectados
                        total_eventos = sum(len(eventos) for eventos in eventos_cambio.values())
                        if total_eventos > 0:
                            st.info(f"Se detectaron {total_eventos} cambios de dirección en {len(eventos_cambio)} IDs")
                        else:
                            st.info("No se detectaron cambios significativos de dirección")
                            
                    except Exception as e:
                        st.error(f"Error al procesar el video: {str(e)}")
                        st.session_state.processed = False
                
                # Mostrar video procesado
                if st.session_state.get('processed', False):
                    try:
                        st.video(st.session_state.output_path)
                    except Exception as e:
                        st.error(f"No se pudo mostrar el video procesado: {str(e)}")
    
    # Pestaña de mapas de calor
    with tabs[1]:
        st.session_state.current_tab = "Mapas de Calor"
        
        if st.session_state.get('processed', False):
            st.subheader("Mapas de Calor")
            
            # Generar mapas de calor con algoritmo seleccionado
            _, generar_mapa_calor_general, generar_mapa_calor_por_id, _, _, generar_mapa_calor_grupos = get_tracker_functions(
            st.session_state.get('algoritmo_usado', 'ByteTrack')
            )
            # Crear columnas para los mapas
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Mapa de Calor General")
                fig_general = generar_mapa_calor_general(
                    st.session_state.posiciones, 
                    st.session_state.video_dims,
                    sigma=sigma_general
                )
                st.pyplot(fig_general)
                
                # Opción para descargar
                buf = tempfile.NamedTemporaryFile(suffix='.png')
                fig_general.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)  # Volver al inicio del archivo
                data = buf.read()  # Leer los datos binarios
                buf.close()  # Cerrar el archivo temporal
                
                st.download_button(
                    label="Descargar Mapa General",
                    data=data,
                    file_name="mapa_calor_general.png",
                    mime="image/png"
                )
            
            with col2:
                st.markdown("### Mapa de Calor por ID")
                
                # Obtener todos los IDs disponibles
                all_ids = sorted(st.session_state.posiciones.keys())
                
                if all_ids:
                    # Inicializar selected_id en session_state si no existe
                    if 'selected_id' not in st.session_state or st.session_state.selected_id not in all_ids:
                        st.session_state.selected_id = all_ids[0]
                    
                    # Selector de ID con información adicional
                    selected_id = st.selectbox(
                        "Seleccioná un ID", 
                        all_ids,
                        format_func=lambda x: f"ID: {x} ({len(st.session_state.posiciones[x])} puntos)",
                        key="id_selector"  # Clave única para el widget
                    )
                    
                    # Actualizar el ID seleccionado en el estado de la sesión
                    st.session_state.selected_id = selected_id
                    
                    # Generar y mostrar mapa para el ID seleccionado
                    fig_id = generar_mapa_calor_por_id(
                        st.session_state.posiciones, 
                        selected_id,
                        st.session_state.video_dims,
                        sigma=sigma_id
                    )
                    
                    if fig_id:
                        st.pyplot(fig_id)
                        
                        # Opción para descargar
                        buf = tempfile.NamedTemporaryFile(suffix='.png')
                        fig_id.savefig(buf, format='png', bbox_inches='tight')
                        buf.seek(0)  # Volver al inicio del archivo
                        data = buf.read()  # Leer los datos binarios
                        buf.close()  # Cerrar el archivo temporal
                        
                        st.download_button(
                            label=f"Descargar Mapa ID: {selected_id}",
                            data=data,
                            file_name=f"mapa_calor_id_{selected_id}.png",
                            mime="image/png"
                        )
                    else:
                        st.warning("No hay datos suficientes para este ID.")
                else:
                    st.warning("No se detectaron IDs en el video.")

            if 'grupos' in st.session_state and st.session_state.grupos:
                st.markdown("### Mapa de Grupos")
                fig_grupos = generar_mapa_calor_grupos(
                    st.session_state.posiciones,
                    st.session_state.grupos,
                    st.session_state.video_dims
                )
                st.pyplot(fig_grupos)
    
    # Pestaña de trayectorias
    with tabs[2]:
        st.session_state.current_tab = "Trayectorias"
        
        # Obtener la función de generación de trayectorias según el algoritmo usado
        _, _, _, generar_mapa_trayectorias, _, _ = get_tracker_functions(
        st.session_state.get('algoritmo_usado', 'ByteTrack')
        )

        if st.session_state.get('processed', False):
            st.subheader("Visualización de Trayectorias")
            
            # Generar mapa de trayectorias
            fig_trayectorias = generar_mapa_trayectorias(
                st.session_state.posiciones,
                st.session_state.video_dims,
                min_frames=min_frames
            )
            
            # Mostrar mapa de trayectorias
            st.pyplot(fig_trayectorias)
            
            # Opción para descargar
            buf = tempfile.NamedTemporaryFile(suffix='.png')
            fig_trayectorias.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)  # Volver al inicio del archivo
            data = buf.read()  # Leer los datos binarios
            buf.close()  # Cerrar el archivo temporal
            
            st.download_button(
                label="Descargar Mapa de Trayectorias",
                data=data,
                file_name="trayectorias.png",
                mime="image/png"
            )
            
            # Estadísticas de trayectorias
            st.subheader("Estadísticas de Trayectorias")
            
            # Crear tabla de estadísticas
            stats_data = []
            for id in st.session_state.posiciones:
                if len(st.session_state.posiciones[id]) >= min_frames:
                    trayectoria = sorted(st.session_state.posiciones[id], key=lambda x: x[3])
                    frames = [t[3] for t in trayectoria]
                    
                    stats_data.append({
                        "ID": id,
                        "Puntos": len(trayectoria),
                        "Primer Frame": frames[0],
                        "Último Frame": frames[-1],
                        "Duración (frames)": frames[-1] - frames[0]
                    })
            
            if stats_data:
                st.dataframe(stats_data)
            else:
                st.info("No hay trayectorias que cumplan con el mínimo de frames.")

    # Nueva pestaña de eventos de dirección
    with tabs[3]:
        st.session_state.current_tab = "Eventos de Dirección"
        
        if st.session_state.get('processed', False):
            st.subheader("Logs de Eventos de Cambio de Dirección")
            
            if 'eventos_cambio_direccion' in st.session_state and st.session_state.eventos_cambio_direccion:
                # Selector de ID para ver eventos específicos
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    ids_con_eventos = sorted(st.session_state.eventos_cambio_direccion.keys())
                    
                    if ids_con_eventos:
                        # Opción para ver todos los IDs o uno específico
                        mostrar_todos = st.checkbox("Mostrar eventos de todos los IDs", value=True)
                        
                        if not mostrar_todos:
                            id_seleccionado = st.selectbox(
                                "Seleccionar ID específico:",
                                ids_con_eventos,
                                format_func=lambda x: f"ID: {x} ({len(st.session_state.eventos_cambio_direccion[x])} eventos)"
                            )
                            ids_a_mostrar = [id_seleccionado]
                        else:
                            ids_a_mostrar = ids_con_eventos
                
                with col2:
                    # Estadísticas generales
                    total_eventos = sum(len(eventos) for eventos in st.session_state.eventos_cambio_direccion.values())
                    st.metric("Total de Eventos", total_eventos)
                    st.metric("IDs con Eventos", len(ids_con_eventos))
                
                # Mostrar eventos
                st.markdown("### Eventos Detectados")
                
                # Botón para descargar logs
                if st.button("📋 Generar Reporte de Eventos", type="secondary"):
                    reporte_texto = "REPORTE DE EVENTOS DE CAMBIO DE DIRECCIÓN\n"
                    reporte_texto += "=" * 50 + "\n\n"
                    
                    for id_obj in sorted(st.session_state.eventos_cambio_direccion.keys()):
                        eventos = st.session_state.eventos_cambio_direccion[id_obj]
                        reporte_texto += f"ID: {id_obj} ({len(eventos)} eventos)\n"
                        reporte_texto += "-" * 30 + "\n"
                        
                        for i, evento in enumerate(eventos, 1):
                            reporte_texto += f"{i}. Frame {evento['frame']}: "
                            reporte_texto += f"Cambió de {evento['direccion_anterior']} hacia {evento['direccion_nueva']} "
                            reporte_texto += f"(Δ{evento['cambio_angulo']}°)\n"
                            reporte_texto += f"   Posición: ({evento['posicion'][0]:.1f}, {evento['posicion'][1]:.1f})\n"
                        
                        reporte_texto += "\n"
                    
                    st.download_button(
                        label="Descargar Reporte Completo",
                        data=reporte_texto,
                        file_name="reporte_eventos_direccion.txt",
                        mime="text/plain"
                    )
                
                # Mostrar eventos por ID
                for id_obj in ids_a_mostrar:
                    eventos = st.session_state.eventos_cambio_direccion[id_obj]
                    
                    with st.expander(f"🎯 ID: {id_obj} ({len(eventos)} eventos)", expanded=len(ids_a_mostrar) == 1):
                        if eventos:
                            for i, evento in enumerate(eventos, 1):
                                # Crear un contenedor estilizado para cada evento
                                st.markdown(f"""
                                <div style="background-color: black; color: white; padding: 10px; border-radius: 8px; font-family: sans-serif;">
                                    <span style="font-weight: bold; color: #4da6ff;">Frame {evento['frame']}</span>: 
                                    Cambió su dirección de 
                                    <span style="font-weight: bold; color: red;">{evento['direccion_anterior']}</span> 
                                    hacia 
                                    <span style="font-weight: bold; color: lightgreen;">{evento['direccion_nueva']}</span>
                                    <br>
                                    <small style="color: #ccc;">
                                        📍 Posición: ({evento['posicion'][0]:.1f}, {evento['posicion'][1]:.1f}) | 
                                        📐 Cambio de ángulo: {evento['cambio_angulo']}° | 
                                        🧭 Ángulos: {evento['angulo_anterior']}° → {evento['angulo_nuevo']}°
                                    </small>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info("No se detectaron eventos para este ID.")
                
            else:
                st.info("No se detectaron cambios significativos de dirección en el video procesado.")
                st.markdown("""
                **Posibles razones:**
                - Los objetos se mueven en línea recta
                - Los cambios de dirección son menores al umbral configurado
                - Las trayectorias son muy cortas
                - Ajusta los parámetros en la barra lateral para mayor sensibilidad
                """)
        else:
            st.info("Primero debes procesar un video para ver los eventos de cambio de dirección.")
    # pestaña de grupos
    with tabs[4]:
            st.session_state.current_tab = "Grupos"
            if st.session_state.get('processed', False):
                st.subheader("Agrupamiento de Personas")
        
                # Calcular grupos
                grupos = calcular_proximidad(
                    st.session_state.posiciones,
                    max_distancia=max_distancia,
                    min_frames_conjuntos=min_frames_grupo
                )
        
                if grupos:
                    # lista compacta de grupos
                    st.markdown("**Grupos detectados (IDs):**")
                    cols = st.columns(3) 
                    for i, grupo in enumerate(grupos, 1):
                        with cols[i % 3]:  # Distribuye los grupos en columnas
                            st.markdown(f"🔹 **Grupo {i}**: `{', '.join(map(str, grupo))}`")
            
                    # Mapa de calor 
                    fig_grupos = generar_mapa_grupos(
                        st.session_state.posiciones,
                        grupos,
                        st.session_state.video_dims,
                        sigma=sigma_general
                    )
                    st.pyplot(fig_grupos, bbox_inches='tight')  
            
                    # Opción para descargar
                    buf = tempfile.NamedTemporaryFile(suffix='.png')
                    fig_grupos.savefig(buf, format='png', bbox_inches='tight')
                    st.download_button(
                        label="Descargar Mapa de Grupos",
                        data=buf.read(),
                        file_name="grupos.png",
                        mime="image/png"
                    )
                else:
                    st.warning("No se encontraron grupos con los criterios actuales.")
    
else:
    # Mensaje cuando no hay video
    st.info("👆 Subí un video para comenzar el análisis")
    
    # Mostrar ejemplo
    st.subheader("Ejemplo de Resultados")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Mapa de Calor General")
        st.image("https://via.placeholder.com/600x400.png?text=Ejemplo+Mapa+de+Calor")
    
    with col2:
        st.markdown("#### Trayectorias")
        st.image("https://via.placeholder.com/600x400.png?text=Ejemplo+Trayectorias")

# Pie de página
st.markdown("---")
st.markdown("Desarrollado por KOI𓆝")
