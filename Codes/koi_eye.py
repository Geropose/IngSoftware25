import streamlit as st
from tracker import (
    procesar_video as procesar_video, 
    generar_mapa_calor_general as generar_mapa_calor_general, 
    generar_mapa_calor_por_id as generar_mapa_calor_por_id,
    generar_mapa_trayectorias as generar_mapa_trayectorias
)
from trackerDeepSort import (
    procesar_video_deepsort,
    generar_mapa_calor_general_deepsort,
    generar_mapa_calor_por_id_deepsort,
    generar_mapa_trayectorias_deepsort
)
import tempfile
import os
import time
import matplotlib.pyplot as plt
import cv2
import subprocess
from pathlib import Path

# Función auxiliar para seleccionar las funciones correctas según el algoritmo
def get_tracker_functions(algoritmo='ByteTrack'):
    """
    Devuelve las funciones correspondientes al algoritmo seleccionado
    
    Args:
        algoritmo (str): Nombre del algoritmo ('ByteTrack' o 'DeepSORT')
        
    Returns:
        tuple: (procesar_video, generar_mapa_calor_general, generar_mapa_calor_por_id, generar_mapa_trayectorias)
    """
    if algoritmo == 'ByteTrack' or algoritmo == 'BotSort':
        return (
            procesar_video,
            generar_mapa_calor_general,
            generar_mapa_calor_por_id,
            generar_mapa_trayectorias
        )
    else:  # DeepSORT
        return (
            procesar_video_deepsort,
            generar_mapa_calor_general_deepsort,
            generar_mapa_calor_por_id_deepsort,
            generar_mapa_trayectorias_deepsort
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
        'total_puntos'
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
    
    # Selector de algoritmo de tracking
    st.subheader("Algoritmo de Tracking")
    algoritmo_tracking = st.radio(
        "Selecciona el algoritmo de tracking:",
        ["ByteTrack", "DeepSORT", "BotSort"],
        help=(
        "¿Qué opción elegir según tu computadora?\n"
        " No fuerza CPU ni GPU, detecta automáticamente el mejor dispositivo. \n"
        "- ByteTrack (rápido y liviano): Elegí esta opción si tu computadora no tiene placa de video (GPU) "
        "o si querés que el video se procese lo más rápido posible. Es ideal para la mayoría de los casos.\n"
        "- BoT-SORT (más preciso): Usá esta opción si tu computadora tiene una buena GPU y necesitás mayor precisión "
        "para seguir personas en lugares con mucha gente o movimiento.\n"
        "- DeepSORT (intermedio): Es una opción más antigua. Tarda más y en algunos casos suele funcionar relativamente mejor, "
        "pero no se recomienda."
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
    tab_names = ["Procesamiento", "Mapas de Calor", "Trayectorias"]
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
                        procesar_video, _, _, _ = get_tracker_functions(algoritmo_seleccionado)
    
                         # Procesar video con el algoritmo seleccionado -- Se agrega el siguiente parametro en la funcion procesar_video nueva_resolucion=st.session_state.resolucion_seleccionada -- Dia del cambio 2025-05-10
                        output_path, posiciones, video_dims = procesar_video(st.session_state.temp_video_path, nueva_resolucion=st.session_state.resolucion_seleccionada,algoritmo=algoritmo_seleccionado)
                        st.session_state.algoritmo_usado = algoritmo_seleccionado  # Guardar el algoritmo usado

                        # Convertir el video de salida para asegurar compatibilidad
                        if st.session_state.get('convert_video', True):
                            output_display_path = convert_video_for_streamlit(output_path)
                        else:
                            output_display_path = output_path
                        
                        # Guardar resultados en la sesión
                        st.session_state.output_path = output_display_path
                        st.session_state.posiciones = posiciones
                        st.session_state.video_dims = video_dims
                        st.session_state.processed = True
                        
                        st.success("Procesamiento terminado.")
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
            _, generar_mapa_calor_general, generar_mapa_calor_por_id, _ = get_tracker_functions(
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
    
    # Pestaña de trayectorias
    with tabs[2]:
        st.session_state.current_tab = "Trayectorias"
        
        # Obtener la función de generación de trayectorias según el algoritmo usado
        _, _, _, generar_mapa_trayectorias = get_tracker_functions(
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