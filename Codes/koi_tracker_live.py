from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse, Response, HTMLResponse
import cv2
from ultralytics import YOLO
import uvicorn
import sys
import os
import numpy as np
import io
import matplotlib.pyplot as plt
from collections import defaultdict
import threading
import time
from scipy.ndimage import gaussian_filter
import matplotlib
import math
matplotlib.use('Agg')  # Usar backend no interactivo

app = FastAPI()
model = YOLO("yolov8n.pt")  # Modelo ligero para detección de personas

# Variables globales
is_streaming = True
should_stop = False
video_writer = None
output_path = "stream_output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
posiciones = defaultdict(list)
eventos_cambio_direccion = defaultdict(list)  # Nueva variable para eventos
video_dims = (640, 480)
frame_count = 0
current_frame = None
lock = threading.Lock()
cap = None
processing_thread = None
tracker_algorithm = "bytetrack"  # Default tracker
target_fps = 25  # Default FPS

# Parámetros para detección de cambios de dirección
umbral_angulo = 30  # Cambio mínimo de ángulo para considerar cambio de dirección
min_distancia = 10  # Distancia mínima entre puntos para calcular dirección

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

def detectar_cambio_direccion_tiempo_real(id_obj, nueva_posicion):
    """
    Detecta cambios de dirección en tiempo real para un ID específico
    
    Args:
        id_obj: ID del objeto
        nueva_posicion: Nueva posición (x, y, conf, frame)
    """
    global eventos_cambio_direccion, posiciones
    
    # Necesitamos al menos 2 posiciones anteriores para detectar cambio
    if len(posiciones[id_obj]) < 2:
        return
    
    # Obtener las últimas 3 posiciones (incluyendo la nueva)
    ultimas_posiciones = posiciones[id_obj][-2:] + [nueva_posicion]
    
    if len(ultimas_posiciones) < 3:
        return
    
    # Calcular direcciones entre los puntos
    punto_anterior = ultimas_posiciones[-3]
    punto_medio = ultimas_posiciones[-2]
    punto_actual = ultimas_posiciones[-1]
    
    # Calcular distancias para filtrar movimientos muy pequeños
    distancia1 = math.sqrt(
        (punto_medio[0] - punto_anterior[0])**2 + 
        (punto_medio[1] - punto_anterior[1])**2
    )
    
    distancia2 = math.sqrt(
        (punto_actual[0] - punto_medio[0])**2 + 
        (punto_actual[1] - punto_medio[1])**2
    )
    
    if distancia1 < min_distancia or distancia2 < min_distancia:
        return
    
    # Calcular ángulos de dirección
    angulo_anterior = calcular_angulo_direccion(punto_anterior, punto_medio)
    angulo_actual = calcular_angulo_direccion(punto_medio, punto_actual)
    
    if angulo_anterior is None or angulo_actual is None:
        return
    
    # Calcular diferencia de ángulo
    diferencia_angulo = abs(angulo_actual - angulo_anterior)
    
    # Manejar el caso de cruce de 0/360 grados
    if diferencia_angulo > 180:
        diferencia_angulo = 360 - diferencia_angulo
    
    # Si hay un cambio significativo de dirección
    if diferencia_angulo >= umbral_angulo:
        direccion_texto_anterior = direccion_a_texto(angulo_anterior)
        direccion_texto_actual = direccion_a_texto(angulo_actual)
        
        evento = {
            'frame': punto_actual[3],
            'timestamp': time.time(),
            'posicion': (punto_actual[0], punto_actual[1]),
            'direccion_anterior': direccion_texto_anterior,
            'direccion_nueva': direccion_texto_actual,
            'angulo_anterior': round(angulo_anterior, 1),
            'angulo_nuevo': round(angulo_actual, 1),
            'cambio_angulo': round(diferencia_angulo, 1)
        }
        
        # Almacenar evento con lock para thread safety
        with lock:
            eventos_cambio_direccion[id_obj].append(evento)
            # Limitar histórico de eventos para no usar demasiada memoria
            if len(eventos_cambio_direccion[id_obj]) > 100:
                eventos_cambio_direccion[id_obj] = eventos_cambio_direccion[id_obj][-50:]
        
        print(f"🔄 ID {id_obj} - Frame {punto_actual[3]}: Cambió de {direccion_texto_anterior} hacia {direccion_texto_actual}")

# Validación de argumentos
if len(sys.argv) < 2:
    print("❌ No se proporcionó una URL de stream.")
    print("Uso: python koi_tracker_live.py <stream_url> [algoritmo] [fps] [umbral_angulo] [min_distancia]")
    sys.exit(1)

stream_url = sys.argv[1]
print(f"🎬 Conectando a: {stream_url}")

# Verificar si se proporcionó un algoritmo de tracking
if len(sys.argv) >= 3:
    tracker_algorithm = sys.argv[2].lower()
    if tracker_algorithm not in ["bytetrack", "botsort", "deepsort"]:
        print(f"⚠️ Algoritmo de tracking no reconocido: {tracker_algorithm}. Usando bytetrack por defecto.")
        tracker_algorithm = "bytetrack"
    print(f"🧭 Algoritmo de tracking: {tracker_algorithm}")

# Verificar si se proporcionó un valor de FPS
if len(sys.argv) >= 4:
    try:
        target_fps = float(sys.argv[3])
        if target_fps <= 0:
            print("⚠️ FPS debe ser mayor que 0. Usando 25 FPS por defecto.")
            target_fps = 25
        elif target_fps > 60:
            print("⚠️ FPS limitado a 60. Valores muy altos pueden afectar el rendimiento.")
            target_fps = 60
    except ValueError:
        print("⚠️ Valor de FPS no válido. Usando 25 FPS por defecto.")
        target_fps = 25
    print(f"⏱️ FPS objetivo: {target_fps}")

# Verificar parámetros de detección de cambios de dirección
if len(sys.argv) >= 5:
    try:
        umbral_angulo = float(sys.argv[4])
        if umbral_angulo < 5 or umbral_angulo > 90:
            print("⚠️ Umbral de ángulo debe estar entre 5 y 90 grados. Usando 30 por defecto.")
            umbral_angulo = 30
    except ValueError:
        print("⚠️ Valor de umbral de ángulo no válido. Usando 30 por defecto.")
        umbral_angulo = 30
    print(f"📐 Umbral de cambio de ángulo: {umbral_angulo}°")

if len(sys.argv) >= 6:
    try:
        min_distancia = float(sys.argv[5])
        if min_distancia < 1 or min_distancia > 100:
            print("⚠️ Distancia mínima debe estar entre 1 y 100 píxeles. Usando 10 por defecto.")
            min_distancia = 10
    except ValueError:
        print("⚠️ Valor de distancia mínima no válido. Usando 10 por defecto.")
        min_distancia = 10
    print(f"📏 Distancia mínima: {min_distancia} píxeles")

# Calcular intervalos de tiempo basados en FPS
frame_interval = 1.0 / target_fps
stream_interval = 1.0 / min(target_fps, 30)  # Limitar streaming a máximo 30 FPS para navegadores

def initialize_capture():
    """Inicializa la captura de video"""
    global cap, video_dims, video_writer
    
    # Configurar captura de video
    if stream_url == "0":  # Para webcam
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(stream_url)

    if not cap.isOpened():
        print("❌ No se pudo abrir el stream")
        return False

    # Obtener dimensiones del video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    video_dims = (width, height)

    print(f"📹 Dimensiones: {width}x{height} @ {fps}fps (fuente)")
    print(f"📹 Procesando a: {target_fps}fps (objetivo)")

    # Inicializar el escritor de video
    video_writer = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))
    
    return True

def cleanup_resources():
    """Limpia todos los recursos de manera segura"""
    global video_writer, cap
    
    print("🧹 Iniciando limpieza de recursos...")
    
    # Cerrar video writer
    if video_writer is not None:
        try:
            video_writer.release()
            video_writer = None
            print("✅ Video writer cerrado")
        except Exception as e:
            print(f"⚠️ Error cerrando video writer: {e}")
    
    # Cerrar captura
    if cap is not None:
        try:
            cap.release()
            cap = None
            print("✅ Captura cerrada")
        except Exception as e:
            print(f"⚠️ Error cerrando captura: {e}")
    
    # Liberar OpenCV
    try:
        cv2.destroyAllWindows()
        print("✅ Ventanas OpenCV cerradas")
    except Exception as e:
        print(f"⚠️ Error cerrando ventanas: {e}")
    
    print("✅ Limpieza completada")

def process_frames():
    """Procesa los frames del video con detección de cambios de dirección"""
    global is_streaming, should_stop, video_writer, posiciones, eventos_cambio_direccion, frame_count, current_frame, video_dims, tracker_algorithm
    
    print(f"🎬 Iniciando procesamiento con detección de cambios de dirección...")
    print(f"🔄 Parámetros: Umbral {umbral_angulo}°, Distancia mín {min_distancia}px")
    
    consecutive_fails = 0
    max_fails = 50
    last_frame_time = time.time()
    
    while is_streaming and not should_stop:
        current_time = time.time()
        elapsed = current_time - last_frame_time
        
        if elapsed < frame_interval:
            sleep_time = frame_interval - elapsed
            time.sleep(min(sleep_time, 0.1))
            continue
        
        last_frame_time = time.time()
        
        if cap is None:
            print("❌ Captura no disponible")
            break
            
        success, frame = cap.read()
        if not success:
            consecutive_fails += 1
            print(f"⚠️ Fallo al leer frame ({consecutive_fails}/{max_fails})")
            
            if consecutive_fails >= max_fails:
                print("❌ Demasiados fallos consecutivos, deteniendo...")
                break
                
            time.sleep(0.1)
            continue
        
        consecutive_fails = 0

        try:
            # Procesamiento con YOLO y tracking
            tracker_config = f"{tracker_algorithm}.yaml"
            results = model.track(frame, persist=True, tracker=tracker_config, classes=0)
            
            annotated_frame = frame.copy()
            
            if results[0].boxes is not None and results[0].boxes.id is not None:
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                
                for id, box, conf in zip(ids, boxes, confs):
                    if should_stop:
                        break
                        
                    x1, y1, x2, y2 = map(int, box)
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    
                    nueva_posicion = (cx, cy, conf, frame_count)
                    
                    # Detectar cambio de dirección ANTES de almacenar
                    detectar_cambio_direccion_tiempo_real(id, nueva_posicion)
                    
                    # Almacenar posición
                    with lock:
                        posiciones[id].append(nueva_posicion)
                        if len(posiciones[id]) > 1000:
                            posiciones[id] = posiciones[id][-500:]
                    
                    # Dibujar detección
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Mostrar información con indicador de cambios recientes
                    info_text = f"ID: {id} ({conf:.2f})"
                    
                    with lock:
                        if id in eventos_cambio_direccion and eventos_cambio_direccion[id]:
                            ultimo_evento = eventos_cambio_direccion[id][-1]
                            if frame_count - ultimo_evento['frame'] <= 30:
                                info_text += f" 🔄{ultimo_evento['direccion_nueva']}"
                    
                    cv2.putText(annotated_frame, info_text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            if should_stop:
                break
            
            # Información del sistema
            status_text = f"{tracker_algorithm.upper()} @ {target_fps}FPS - {('STOPPING' if should_stop else 'RECORDING')}"
            cv2.putText(annotated_frame, f"Frame: {frame_count} - {status_text}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # FPS real
            real_fps = 1.0 / (time.time() - last_frame_time) if (time.time() - last_frame_time) > 0 else 0
            cv2.putText(annotated_frame, f"FPS real: {real_fps:.1f}", (20, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Estadísticas de eventos
            with lock:
                total_eventos = sum(len(eventos) for eventos in eventos_cambio_direccion.values())
                if total_eventos > 0:
                    cv2.putText(annotated_frame, f"Cambios direccion: {total_eventos}", (20, 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
            # Guardar frame
            if video_writer is not None and not should_stop:
                video_writer.write(annotated_frame)
            
            # Actualizar frame actual
            with lock:
                current_frame = annotated_frame
                frame_count += 1
                
        except Exception as e:
            print(f"❌ Error procesando frame {frame_count}: {e}")
            continue
    
    print("🛑 Finalizando procesamiento...")
    cleanup_resources()
    is_streaming = False
    print("✅ Procesamiento finalizado")

# Resto de funciones (generar_mapa_calor, etc.) permanecen igual...
def generar_mapa_calor():
    """Genera un mapa de calor general con las posiciones actuales"""
    with lock:
        if not posiciones:
            return None
        local_posiciones = {k: v[:] for k, v in posiciones.items()}
        local_dims = video_dims
    
    width, height = local_dims
    heatmap = np.zeros((height, width))
    
    for id in local_posiciones:
        for cx, cy, conf, _ in local_posiciones[id]:
            if 0 <= cx < width and 0 <= cy < height:
                heatmap[cy, cx] += conf
    
    if np.max(heatmap) == 0:
        return None
    
    heatmap = gaussian_filter(heatmap, sigma=15)
    heatmap = heatmap / np.max(heatmap)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(heatmap, cmap='inferno', interpolation='nearest')
    ax.set_title(f'Mapa de Calor General ({tracker_algorithm.upper()} @ {target_fps}FPS)', fontsize=14, color='white')
    ax.axis('off')
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, facecolor='black')
    buf.seek(0)
    plt.close(fig)
    
    return buf

def generar_mapa_trayectorias():
    """Genera un mapa con todas las trayectorias de las personas"""
    with lock:
        if not posiciones:
            return None
        local_posiciones = {k: v[:] for k, v in posiciones.items()}
        local_dims = video_dims
    
    width, height = local_dims
    
    # Crear figura
    fig = plt.figure(figsize=(12, 8), facecolor='black')
    ax = fig.add_subplot(111, facecolor='black')
    
    # Configurar límites
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # Invertir eje Y
    
    # Colores para diferentes IDs
    colores = plt.cm.tab10(np.linspace(0, 1, 10))
    
    trayectorias_dibujadas = 0
    
    # Dibujar trayectorias
    for i, id in enumerate(local_posiciones):
        # Filtrar IDs con pocos frames
        if len(local_posiciones[id]) < 10:
            continue
            
        # Ordenar por número de frame
        trayectoria = sorted(local_posiciones[id], key=lambda x: x[3])
        coords = np.array([(x[0], x[1]) for x in trayectoria])
        
        if len(coords) > 1:
            color = colores[i % len(colores)]
            
            # Dibujar línea de trayectoria
            ax.plot(coords[:, 0], coords[:, 1], '-', color=color, 
                   linewidth=2, alpha=0.8, label=f"ID: {id}")
            
            # Marcar inicio y fin
            ax.plot(coords[0, 0], coords[0, 1], 'o', color=color, markersize=6)
            ax.plot(coords[-1, 0], coords[-1, 1], 's', color=color, markersize=6)
            
            trayectorias_dibujadas += 1
    
    if trayectorias_dibujadas == 0:
        ax.text(width/2, height/2, 'No hay trayectorias suficientes', 
                ha='center', va='center', color='white', fontsize=14)
    
    # Agregar leyenda si no hay demasiados IDs
    if trayectorias_dibujadas <= 10 and trayectorias_dibujadas > 0:
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    
    ax.set_title(f'Mapa de Trayectorias ({tracker_algorithm.upper()} @ {target_fps}FPS)', fontsize=14, color='white')
    ax.axis('off')
    
    # Convertir a imagen
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, facecolor='black')
    buf.seek(0)
    plt.close(fig)
    
    return buf

def generar_mapa_calor_por_id(id):
    """Genera un mapa de calor para un ID específico con trayectoria"""
    with lock:
        if id not in posiciones or not posiciones[id]:
            return None
        local_posiciones = posiciones[id][:]
        local_dims = video_dims
        
    width, height = local_dims
    
    # Crear una matriz vacía para el mapa de calor
    heatmap = np.zeros((height, width))
    
    # Extraer coordenadas y ordenarlas por número de frame
    trayectoria = sorted(local_posiciones, key=lambda x: x[3])
    
    # Agregar puntos al mapa de calor con pesos basados en confianza
    for cx, cy, conf, _ in trayectoria:
        if 0 <= cx < width and 0 <= cy < height:
            heatmap[cy, cx] += conf
    
    # Verificar si hay datos
    if np.max(heatmap) == 0:
        return None
    
    # Aplicar suavizado gaussiano
    heatmap = gaussian_filter(heatmap, sigma=10)
    
    # Normalizar para visualización
    heatmap = heatmap / np.max(heatmap)
    
    # Crear figura
    fig = plt.figure(figsize=(12, 8), facecolor='black')
    ax = fig.add_subplot(111, facecolor='black')
    
    # Mostrar mapa de calor
    ax.imshow(heatmap, cmap='viridis', interpolation='nearest')
    
    # Dibujar trayectoria con líneas
    coords = np.array([(x[0], x[1]) for x in trayectoria])
    if len(coords) > 1:
        ax.plot(coords[:, 0], coords[:, 1], 'r-', linewidth=2, alpha=0.7)
        
        # Marcar inicio y fin
        ax.plot(coords[0, 0], coords[0, 1], 'go', markersize=8)  # Inicio en verde
        ax.plot(coords[-1, 0], coords[-1, 1], 'ro', markersize=8)  # Fin en rojo
    
    ax.set_title(f'Análisis ID {id} - {len(trayectoria)} puntos ({tracker_algorithm.upper()} @ {target_fps}FPS)', fontsize=14, color='white')
    ax.axis('off')
    
    # Convertir a imagen
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, facecolor='black')
    buf.seek(0)
    plt.close(fig)

    return buf


def detectar_grupos(local_posiciones, distancia_minima=50, min_personas=2):
    """Detecta grupos de personas por proximidad."""
    frames = defaultdict(dict)
    for pid, puntos in local_posiciones.items():
        for x, y, _, frame in puntos:
            frames[frame][pid] = (x, y)

    grupos = defaultdict(list)
    for frame, id_pos in frames.items():
        sin_visitar = set(id_pos.keys())
        while sin_visitar:
            actual = sin_visitar.pop()
            grupo = {actual}
            cola = [actual]
            while cola:
                cid = cola.pop()
                cx, cy = id_pos[cid]
                for otro in list(sin_visitar):
                    ox, oy = id_pos[otro]
                    if np.hypot(cx - ox, cy - oy) <= distancia_minima:
                        grupo.add(otro)
                        cola.append(otro)
                        sin_visitar.remove(otro)
            if len(grupo) >= min_personas:
                grupos[frame].append(grupo)
    return grupos


def generar_mapa_calor_grupos(local_posiciones, grupos_por_frame, sigma=15):
    """Genera un mapa de calor para los centros de los grupos detectados."""
    width, height = video_dims
    heatmap = np.zeros((height, width))

    for frame, grupos in grupos_por_frame.items():
        for grupo in grupos:
            xs, ys = [], []
            for pid in grupo:
                for x, y, _, f in local_posiciones.get(pid, []):
                    if f == frame:
                        xs.append(x)
                        ys.append(y)
                        break
            if xs and ys:
                cx = int(np.mean(xs))
                cy = int(np.mean(ys))
                if 0 <= cx < width and 0 <= cy < height:
                    heatmap[cy, cx] += 1

    heatmap = gaussian_filter(heatmap, sigma=sigma)
    if np.max(heatmap) == 0:
        return None
    heatmap = heatmap / np.max(heatmap)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(heatmap, cmap='magma', interpolation='nearest')
    ax.set_title(f'Mapa de Grupos ({tracker_algorithm.upper()} @ {target_fps}FPS)', fontsize=14, color='white')
    ax.axis('off')

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, facecolor='black')
    buf.seek(0)
    plt.close(fig)

def gen_frames():
    """Generador para streaming de video"""
    last_frame_time = time.time()
    
    while is_streaming and not should_stop:
        current_time = time.time()
        elapsed = current_time - last_frame_time
        
        if elapsed < stream_interval:
            time.sleep(min(stream_interval - elapsed, 0.01))
            continue
            
        last_frame_time = time.time()
        
        with lock:
            if current_frame is not None:
                frame = current_frame.copy()
            else:
                time.sleep(0.1)
                continue
        
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Endpoints de la API
@app.get("/")
def root():
    return {
        "message": "KOI Tracker Live API con Detección de Cambios de Dirección", 
        "status": "running" if is_streaming else "stopped",
        "tracker": tracker_algorithm,
        "fps": target_fps,
        "direction_detection": {
            "angle_threshold": umbral_angulo,
            "min_distance": min_distancia
        },
        "endpoints": [
            "/video", "/direction_events", "/direction_events/{id}", 
            "/direction_stats", "/stats", "/stop", "/download",
            "/ids_with_events", "/direction_report/{id}"
        ]
    }

@app.get("/video")
def video():
    if not is_streaming:
        raise HTTPException(status_code=503, detail="El servicio está detenido")
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/heatmap")
def heatmap():
    try:
        buf = generar_mapa_calor()
        if buf is None:
            raise HTTPException(status_code=404, detail="No hay datos suficientes para generar mapa de calor")
        return Response(content=buf.getvalue(), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al generar mapa de calor: {str(e)}")

@app.get("/trajectories")
def trajectories():
    try:
        buf = generar_mapa_trayectorias()
        if buf is None:
            raise HTTPException(status_code=404, detail="No hay datos suficientes para generar trayectorias")
        return Response(content=buf.getvalue(), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al generar mapa de trayectorias: {str(e)}")

@app.get("/heatmap/{id}")
def heatmap_by_id(id: int):
    try:
        buf = generar_mapa_calor_por_id(id)
        if buf is None:
            raise HTTPException(status_code=404, detail=f"No hay datos para el ID {id}")
        return Response(content=buf.getvalue(), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al generar mapa de calor: {str(e)}")


@app.get("/group_heatmap")
def group_heatmap():
    try:
        with lock:
            if not posiciones:
                raise HTTPException(status_code=404, detail="No hay datos suficientes")
            local_pos = {k: v[:] for k, v in posiciones.items()}
        grupos = detectar_grupos(local_pos)
        if not grupos:
            raise HTTPException(status_code=404, detail="No se detectaron grupos")
        buf = generar_mapa_calor_grupos(local_pos, grupos)
        if buf is None:
            raise HTTPException(status_code=404, detail="No se pudo generar el mapa de grupos")
        return Response(content=buf.getvalue(), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al generar mapa de grupos: {str(e)}")

# NUEVOS ENDPOINTS AGREGADOS PARA LA FUNCIONALIDAD SOLICITADA

@app.get("/ids_with_events")
def ids_with_events():
    """Devuelve solo los IDs que tienen eventos de cambio de dirección"""
    with lock:
        if not eventos_cambio_direccion:
            return {
                "ids_with_events": [],
                "total_ids": 0,
                "tracker_status": "running" if is_streaming and not should_stop else "stopped"
            }
        
        ids_con_eventos = []
        for id_obj, eventos in eventos_cambio_direccion.items():
            if eventos:  # Solo IDs que tienen eventos
                ids_con_eventos.append({
                    "id": id_obj,
                    "event_count": len(eventos),
                    "last_event_frame": eventos[-1]['frame'] if eventos else None
                })
        
        # Ordenar por número de eventos (descendente)
        ids_con_eventos.sort(key=lambda x: x['event_count'], reverse=True)
        
        return {
            "ids_with_events": ids_con_eventos,
            "total_ids": len(ids_con_eventos),
            "tracker_status": "running" if is_streaming and not should_stop else "stopped"
        }

@app.get("/direction_report/{id}")
def direction_report(id: int):
    """Genera un reporte de texto para un ID específico (solo cuando el tracker está detenido)"""
    # Verificar si el tracker está detenido
    if is_streaming and not should_stop:
        raise HTTPException(status_code=423, detail="El reporte solo está disponible cuando el tracker está detenido")
    
    with lock:
        if id not in eventos_cambio_direccion or not eventos_cambio_direccion[id]:
            raise HTTPException(status_code=404, detail=f"No hay eventos de cambio de dirección para el ID {id}")
        
        eventos_id = eventos_cambio_direccion[id]
        
        # Generar reporte en texto
        from datetime import datetime
        
        reporte_texto = f"REPORTE DE EVENTOS DE CAMBIO DE DIRECCIÓN - ID {id}\n"
        reporte_texto += "=" * 60 + "\n\n"
        reporte_texto += f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        reporte_texto += f"ID: {id}\n"
        reporte_texto += f"Total de eventos: {len(eventos_id)}\n"
        reporte_texto += f"Algoritmo de tracking: {tracker_algorithm}\n"
        reporte_texto += f"FPS objetivo: {target_fps}\n"
        reporte_texto += f"Parámetros de detección:\n"
        reporte_texto += f"  - Umbral de ángulo: {umbral_angulo}°\n"
        reporte_texto += f"  - Distancia mínima: {min_distancia} píxeles\n\n"
        
        if eventos_id:
            reporte_texto += f"EVENTOS DETECTADOS:\n"
            reporte_texto += "-" * 30 + "\n"
            
            for i, evento in enumerate(eventos_id, 1):
                timestamp_str = datetime.fromtimestamp(evento['timestamp']).strftime('%H:%M:%S')
                reporte_texto += f"{i}. Frame {evento['frame']} ({timestamp_str})\n"
                reporte_texto += f"   Cambió de {evento['direccion_anterior']} hacia {evento['direccion_nueva']}\n"
                reporte_texto += f"   Posición: ({evento['posicion'][0]}, {evento['posicion'][1]})\n"
                reporte_texto += f"   Cambio de ángulo: {evento['cambio_angulo']}°\n"
                reporte_texto += f"   Ángulos: {evento['angulo_anterior']}° → {evento['angulo_nuevo']}°\n\n"
        else:
            reporte_texto += "No se encontraron eventos para este ID.\n"
        
        reporte_texto += "\n" + "=" * 60 + "\n"
        reporte_texto += "Reporte generado por KOI Tracker Live\n"
        
        # Devolver como descarga de archivo
        from fastapi.responses import Response
        return Response(
            content=reporte_texto,
            media_type="text/plain",
            headers={
                "Content-Disposition": f"attachment; filename=reporte_eventos_id_{id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            }
        )

@app.get("/direction_events")
def direction_events():
    """Devuelve todos los eventos de cambio de dirección detectados"""
    with lock:
        if not eventos_cambio_direccion:
            return {
                "message": "No se han detectado eventos de cambio de dirección",
                "total_events": 0,
                "ids_with_events": [],
                "events": {}
            }
        
        eventos_copia = {}
        total_eventos = 0
        
        for id_obj, eventos in eventos_cambio_direccion.items():
            eventos_copia[id_obj] = []
            for evento in eventos:
                evento_formateado = {
                    "frame": evento['frame'],
                    "timestamp": evento['timestamp'],
                    "position": {
                        "x": evento['posicion'][0],
                        "y": evento['posicion'][1]
                    },
                    "direction_change": {
                        "from": evento['direccion_anterior'],
                        "to": evento['direccion_nueva']
                    },
                    "angles": {
                        "previous": evento['angulo_anterior'],
                        "current": evento['angulo_nuevo'],
                        "change": evento['cambio_angulo']
                    },
                    "description": f"Frame {evento['frame']}: Cambió de {evento['direccion_anterior']} hacia {evento['direccion_nueva']}"
                }
                eventos_copia[id_obj].append(evento_formateado)
                total_eventos += 1
        
        return {
            "total_events": total_eventos,
            "ids_with_events": list(eventos_copia.keys()),
            "detection_params": {
                "angle_threshold": umbral_angulo,
                "min_distance": min_distancia
            },
            "events": eventos_copia
        }

@app.get("/direction_events/{id}")
def direction_events_by_id(id: int):
    """Devuelve los eventos de cambio de dirección para un ID específico"""
    with lock:
        if id not in eventos_cambio_direccion or not eventos_cambio_direccion[id]:
            raise HTTPException(status_code=404, detail=f"No hay eventos de cambio de dirección para el ID {id}")
        
        eventos_id = []
        for evento in eventos_cambio_direccion[id]:
            evento_formateado = {
                "frame": evento['frame'],
                "timestamp": evento['timestamp'],
                "position": {
                    "x": evento['posicion'][0],
                    "y": evento['posicion'][1]
                },
                "direction_change": {
                    "from": evento['direccion_anterior'],
                    "to": evento['direccion_nueva']
                },
                "angles": {
                    "previous": evento['angulo_anterior'],
                    "current": evento['angulo_nuevo'],
                    "change": evento['cambio_angulo']
                },
                "description": f"Frame {evento['frame']}: Cambió de {evento['direccion_anterior']} hacia {evento['direccion_nueva']}"
            }
            eventos_id.append(evento_formateado)
        
        return {
            "id": id,
            "total_events": len(eventos_id),
            "detection_params": {
                "angle_threshold": umbral_angulo,
                "min_distance": min_distancia
            },
            "events": eventos_id
        }

@app.get("/direction_stats")
def direction_stats():
    """Devuelve estadísticas de los eventos de cambio de dirección"""
    with lock:
        if not eventos_cambio_direccion:
            return {
                "total_events": 0,
                "ids_with_events": 0,
                "most_active_id": None,
                "direction_frequency": {},
                "recent_events": []
            }
        
        total_eventos = sum(len(eventos) for eventos in eventos_cambio_direccion.values())
        ids_con_eventos = len(eventos_cambio_direccion)
        
        # ID más activo
        id_mas_activo = max(eventos_cambio_direccion.keys(), 
                           key=lambda x: len(eventos_cambio_direccion[x]))
        
        # Frecuencia de direcciones
        frecuencia_direcciones = defaultdict(int)
        eventos_recientes = []
        
        for id_obj, eventos in eventos_cambio_direccion.items():
            for evento in eventos:
                frecuencia_direcciones[evento['direccion_nueva']] += 1
                eventos_recientes.append({
                    "id": id_obj,
                    "frame": evento['frame'],
                    "timestamp": evento['timestamp'],
                    "description": f"ID {id_obj} - Frame {evento['frame']}: {evento['direccion_anterior']} → {evento['direccion_nueva']}"
                })
        
        # Ordenar eventos recientes por timestamp
        eventos_recientes.sort(key=lambda x: x['timestamp'], reverse=True)
        eventos_recientes = eventos_recientes[:10]
        
        return {
            "total_events": total_eventos,
            "ids_with_events": ids_con_eventos,
            "most_active_id": {
                "id": id_mas_activo,
                "events": len(eventos_cambio_direccion[id_mas_activo])
            },
            "direction_frequency": dict(frecuencia_direcciones),
            "recent_events": eventos_recientes,
            "detection_params": {
                "angle_threshold": umbral_angulo,
                "min_distance": min_distancia
            }
        }

@app.get("/stats")
def stats():
    with lock:
        total_eventos_direccion = sum(len(eventos) for eventos in eventos_cambio_direccion.values())
        
        return {
            "personas_detectadas": len(posiciones),
            "frames_procesados": frame_count,
            "ids_activos": list(posiciones.keys()),
            "eventos_cambio_direccion": total_eventos_direccion,
            "ids_con_eventos_direccion": len(eventos_cambio_direccion),
            "status": "running" if is_streaming and not should_stop else "stopped",
            "tracker": tracker_algorithm,
            "fps": target_fps,
            "detection_params": {
                "angle_threshold": umbral_angulo,
                "min_distance": min_distancia
            },
            "video_file_exists": os.path.exists(output_path)
        }

@app.get("/stop")
def stop():
    global is_streaming, should_stop
    print("🛑 Recibida señal de parada...")
    should_stop = True
    
    max_wait = 10
    wait_count = 0
    
    while is_streaming and wait_count < max_wait:
        time.sleep(0.5)
        wait_count += 0.5
        print(f"⏳ Esperando parada... ({wait_count}s)")
    
    if is_streaming:
        print("⚠️ Forzando parada...")
        is_streaming = False
        cleanup_resources()
    
    with lock:
        total_eventos = sum(len(eventos) for eventos in eventos_cambio_direccion.values())
    
    return {
        "message": "✅ Stream detenido. El video se ha guardado.",
        "video_available": os.path.exists(output_path),
        "frames_processed": frame_count,
        "direction_events_detected": total_eventos,
        "tracker": tracker_algorithm,
        "fps": target_fps
    }

@app.get("/download")
def download():
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        if file_size > 0:
            return FileResponse(
                output_path, 
                filename=f"video_procesado_{tracker_algorithm}_{target_fps}fps.mp4", 
                media_type="video/mp4",
                headers={"Content-Length": str(file_size)}
            )
        else:
            raise HTTPException(status_code=422, detail="El archivo de video está vacío")
    else:
        raise HTTPException(status_code=404, detail="El archivo de video no está disponible")

# Función de inicialización
def initialize_system():
    """Inicializa todo el sistema"""
    if not initialize_capture():
        return False
    
    global processing_thread
    processing_thread = threading.Thread(target=process_frames, daemon=False)
    processing_thread.start()
    return True

# Inicializar el sistema
if not initialize_system():
    print("❌ Error inicializando el sistema")
    sys.exit(1)

if __name__ == "__main__":
    try:
        print(f"🚀 Iniciando servidor KOI Tracker Live en http://localhost:8000")
        print(f"📹 Stream: {stream_url}")
        print(f"🧭 Algoritmo de tracking: {tracker_algorithm}")
        print(f"⏱️ FPS objetivo: {target_fps}")
        print(f"🔄 Detección de cambios de dirección activada:")
        print(f"   - Umbral de ángulo: {umbral_angulo}°")
        print(f"   - Distancia mínima: {min_distancia} píxeles")
        print(f"📊 Endpoints principales:")
        print(f"- /video - Ver video en tiempo real")
        print(f"- /direction_events - Ver todos los eventos de cambio de dirección")
        print(f"- /direction_events/ID - Ver eventos de un ID específico")
        print(f"- /direction_stats - Ver estadísticas de cambios de dirección")
        print(f"- /ids_with_events - Ver solo IDs que tienen eventos")
        print(f"- /direction_report/ID - Generar reporte TXT para un ID (solo cuando está detenido)")
        print(f"- /stats - Ver estadísticas generales")
        print(f"- /stop - Detener procesamiento")
        print(f"- /download - Descargar video procesado")
        
        uvicorn.run("__main__:app", host="0.0.0.0", port=8000, log_level="warning")
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupción por teclado detectada")
        should_stop = True
        is_streaming = False
        cleanup_resources()
    finally:
        print("🏁 Servidor finalizado")
