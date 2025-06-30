from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse, Response
import cv2
from ultralytics import YOLO
import uvicorn
import sys
import threading
import time
import os
import numpy as np
import io
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.ndimage import gaussian_filter
import math
import matplotlib
matplotlib.use('Agg')  # Usar backend no interactivo

app = FastAPI()
model = YOLO("yolov8n.pt")  # Modelo base

# Variables globales
is_streaming = True
video_writer = None
output_path = "stream_output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
cap = None
tracker_algorithm = "bytetrack"  # Default tracker
target_fps = 25  # Default FPS
server_port = 8001  # Puerto por defecto

# Variables para mapas de calor
posiciones = defaultdict(list)  # Almacena posiciones de personas detectadas
frame_count = 0
video_dims = (640, 480)  # Se actualizará con las dimensiones reales
lock = threading.Lock()  # Para acceso seguro a variables compartidas


# Variables para detección de cambios de dirección y estado de movimiento
eventos_cambio_direccion = defaultdict(list)
estado_persona = defaultdict(lambda: "Moviendo")
contador_cambios_tray = defaultdict(int)
ultimo_movimiento = defaultdict(int)

# Parámetros de detección
umbral_quieto = 5        # Distancia mínima para considerar que hay movimiento
frames_quieto = 15       # Frames para marcar estado quieto
umbral_angulo = 30       # Umbral de cambio de dirección en grados
min_distancia = 10       # Distancia mínima entre puntos para calcular dirección


# Procesar argumentos de línea de comandos
if len(sys.argv) >= 2:
    camera_id = sys.argv[1]
    try:
        camera_id = int(camera_id)  # Intentar convertir a entero para cámaras locales
    except ValueError:
        pass  # Mantener como string si es una URL
    print(f"📷 Usando cámara: {camera_id}")
else:
    camera_id = 0
    print("📷 Usando cámara por defecto (0)")

# Verificar si se proporcionó un algoritmo de tracking
if len(sys.argv) >= 3:
    tracker_algorithm = sys.argv[2].lower()
    if tracker_algorithm not in ["bytetrack", "botsort", "deepsort"]:
        print(f"⚠️ Algoritmo de tracking no reconocido: {tracker_algorithm}. Usando bytetrack por defecto.")
        tracker_algorithm = "bytetrack"
    print(f"🧭 Algoritmo de tracking: {tracker_algorithm}")

# Parámetros adicionales: umbral de ángulo, distancia mínima y puerto opcional
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

# Verificar si se proporcionó un puerto personalizado
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

if len(sys.argv) >= 7:
    try:
        server_port = int(sys.argv[6])
        if server_port < 1024 or server_port > 65535:
            print("⚠️ Puerto debe estar entre 1024 y 65535. Usando 8001 por defecto.")
            server_port = 8001
    except ValueError:
        print("⚠️ Valor de puerto no válido. Usando 8001 por defecto.")
        server_port = 8001
    print(f"🔌 Puerto del servidor: {server_port}")

# Calcular intervalos de tiempo basados en FPS
frame_interval = 1.0 / target_fps
stream_interval = 1.0 / min(target_fps, 30)  # Limitar streaming a máximo 30 FPS para navegadores

def calcular_angulo_direccion(p1, p2):
    """Calcula el ángulo de dirección entre dos puntos"""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if dx == 0 and dy == 0:
        return None
    ang_rad = math.atan2(dy, dx)
    ang_deg = math.degrees(ang_rad)
    if ang_deg < 0:
        ang_deg += 360
    return ang_deg


def direccion_a_texto(angulo):
    """Convierte un ángulo a texto descriptivo"""
    if angulo is None:
        return "Sin movimiento"
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


def detectar_cambio_direccion_tiempo_real(pid, nueva_pos):
    """Detecta cambios de dirección para un ID específico"""
    if len(posiciones[pid]) < 2:
        return

    ultimas = posiciones[pid][-2:] + [nueva_pos]
    if len(ultimas) < 3:
        return

    p_ant = ultimas[-3]
    p_mid = ultimas[-2]
    p_act = ultimas[-1]

    dist1 = math.hypot(p_mid[0] - p_ant[0], p_mid[1] - p_ant[1])
    dist2 = math.hypot(p_act[0] - p_mid[0], p_act[1] - p_mid[1])
    if dist1 < min_distancia or dist2 < min_distancia:
        return

    ang_prev = calcular_angulo_direccion(p_ant, p_mid)
    ang_curr = calcular_angulo_direccion(p_mid, p_act)
    if ang_prev is None or ang_curr is None:
        return

    diff = abs(ang_curr - ang_prev)
    if diff > 180:
        diff = 360 - diff

    if diff >= umbral_angulo:
        evento = {
            'frame': p_act[3],
            'timestamp': time.time(),
            'posicion': (p_act[0], p_act[1]),
            'direccion_anterior': direccion_a_texto(ang_prev),
            'direccion_nueva': direccion_a_texto(ang_curr),
            'angulo_anterior': round(ang_prev, 1),
            'angulo_nuevo': round(ang_curr, 1),
            'cambio_angulo': round(diff, 1)
        }

        with lock:
            eventos_cambio_direccion[pid].append(evento)
            contador_cambios_tray[pid] += 1
            if len(eventos_cambio_direccion[pid]) > 100:
                eventos_cambio_direccion[pid] = eventos_cambio_direccion[pid][-50:]

def initialize_capture():
    """Inicializa la captura de video"""
    global cap, video_dims
    
    # Configurar captura de video
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print("❌ No se pudo abrir la cámara")
        return False
        
    # Obtener dimensiones del video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    
    # Actualizar dimensiones globales para mapas de calor
    video_dims = (width, height)
    
    print(f"📹 Dimensiones: {width}x{height} @ {fps}fps (fuente)")
    print(f"📹 Procesando a: {target_fps}fps (objetivo)")
    
    return True

def gen_frames():
    """Generador para streaming de video con control de FPS"""
    global is_streaming, video_writer, tracker_algorithm, target_fps, frame_count, posiciones
    
    if not initialize_capture():
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + b'Error al inicializar la camara' + b'\r\n')
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Inicializar el escritor de video
    video_writer = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))
    
    last_frame_time = time.time()
    
    while is_streaming:
        # Control de FPS - esperar hasta que sea tiempo para el siguiente frame
        current_time = time.time()
        elapsed = current_time - last_frame_time
        
        if elapsed < frame_interval:
            # Esperar el tiempo necesario para mantener el FPS objetivo
            time.sleep(min(frame_interval - elapsed, 0.01))  # Pequeña espera para no saturar CPU
            continue
            
        # Actualizar tiempo del último frame procesado
        last_frame_time = time.time()
        
        # Capturar frame
        success, frame = cap.read()
        if not success:
            print("❌ Error al leer frame")
            break

        try:
            # Procesamiento con YOLO y tracking
            tracker_config = f"{tracker_algorithm}.yaml"
            results = model.track(frame, persist=True, tracker=tracker_config, classes=0, verbose=False)
            
            # Dibujar detecciones
            annotated_frame = results[0].plot()
            
            # Almacenar posiciones para mapas de calor
            if results[0].boxes is not None and results[0].boxes.id is not None:
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                
                for id, box, conf in zip(ids, boxes, confs):
                    x1, y1, x2, y2 = map(int, box)
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    
                    nueva_pos = (cx, cy, conf, frame_count)

                    # Detectar cambio de dirección antes de almacenar
                    detectar_cambio_direccion_tiempo_real(id, nueva_pos)

                    # Almacenar posición y actualizar estado de movimiento
                    with lock:
                        if posiciones[id]:
                            ultimo_punto = posiciones[id][-1]
                            dist = math.hypot(nueva_pos[0] - ultimo_punto[0],
                                             nueva_pos[1] - ultimo_punto[1])
                            if dist > umbral_quieto:
                                ultimo_movimiento[id] = frame_count
                                estado_persona[id] = "Moviendo"
                        else:
                            ultimo_movimiento[id] = frame_count

                        posiciones[id].append(nueva_pos)
                        if len(posiciones[id]) > 1000:
                            posiciones[id] = posiciones[id][-500:]

                    # Mostrar información en pantalla
                    info_text = f"ID: {id} ({conf:.2f})"
                    with lock:
                        if id in eventos_cambio_direccion and eventos_cambio_direccion[id]:
                            ultimo_evento = eventos_cambio_direccion[id][-1]
                            if frame_count - ultimo_evento['frame'] <= 30:
                                info_text += f" 🔄{ultimo_evento['direccion_nueva']}"

                        if frame_count - ultimo_movimiento[id] >= frames_quieto:
                            estado_persona[id] = "Quieto"

                        info_text += f" | Estado: {estado_persona[id]}"
                        info_text += f" | cambios_tray: {contador_cambios_tray[id]}"

                    cv2.putText(annotated_frame, info_text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Calcular FPS real
            real_fps = 1.0 / (time.time() - last_frame_time) if (time.time() - last_frame_time) > 0 else 0
            
            # Agregar información de frame, estado, algoritmo y FPS
            status_text = f"{tracker_algorithm.upper()} @ {target_fps}FPS"
            cv2.putText(annotated_frame, f"Frame: {frame_count} - {status_text}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(annotated_frame, f"FPS real: {real_fps:.1f}", (20, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Guardar frame
            if video_writer:
                video_writer.write(annotated_frame)
                
            # Incrementar contador de frames
            with lock:
                frame_count += 1

            # Convertir a JPEG para streaming
            _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                   
        except Exception as e:
            print(f"❌ Error procesando frame: {e}")
            continue

    # Liberar recursos
    cleanup_resources()

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

# NUEVAS FUNCIONES PARA MAPAS DE CALOR

def generar_mapa_calor():
    """Genera un mapa de calor general con las posiciones actuales"""
    with lock:
        if not posiciones:
            return None
        local_posiciones = {k: v[:] for k, v in posiciones.items()}
        local_dims = video_dims
    
    width, height = local_dims
    
    # Crear una matriz vacía para el mapa de calor
    heatmap = np.zeros((height, width))
    
    # Agregar todas las posiciones al mapa de calor con sus pesos (confianza)
    for id in local_posiciones:
        for cx, cy, conf, _ in local_posiciones[id]:
            if 0 <= cx < width and 0 <= cy < height:
                heatmap[cy, cx] += conf
    
    # Verificar si hay datos
    if np.max(heatmap) == 0:
        return None
    
    # Aplicar suavizado gaussiano
    heatmap = gaussian_filter(heatmap, sigma=15)
    
    # Normalizar para visualización
    heatmap = heatmap / np.max(heatmap)
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(heatmap, cmap='inferno', interpolation='nearest')
    ax.set_title(f'Mapa de Calor General ({tracker_algorithm.upper()} @ {target_fps}FPS)', fontsize=14, color='white')
    ax.axis('off')
    
    # Convertir a imagen
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
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Configurar límites
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # Invertir eje Y
    ax.set_facecolor('black')
    
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
    fig, ax = plt.subplots(figsize=(12, 8))
    
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

    return buf


# ENDPOINTS ORIGINALES

@app.get("/")
def root():
    return {
        "message": "KOI Eye Cam API", 
        "status": "running" if is_streaming else "stopped",
        "tracker": tracker_algorithm,
        "fps": target_fps,
        "port": server_port,
        "direction_detection": {
            "angle_threshold": umbral_angulo,
            "min_distance": min_distancia
        },
        "endpoints": [
            "/video", "/stop", "/download", "/status",
            "/heatmap", "/trajectories", "/heatmap/{id}",
            "/group_heatmap", "/stats", "/movement_report"
        ]    }

@app.get("/video")
def video():
    global is_streaming
    
    # Si el servicio está detenido, intentar reiniciarlo automáticamente
    if not is_streaming:
        try:
            # Reiniciar el servicio
            restart_result = restart()
            if "status" in restart_result and restart_result["status"] == "running":
                # Reinicio exitoso
                pass
            else:
                # No se pudo reiniciar
                raise HTTPException(status_code=503, detail="No se pudo reiniciar el servicio")
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"El servicio está detenido y no se pudo reiniciar: {str(e)}")
    
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/stop")
def stop():
    global is_streaming
    is_streaming = False
    return {
        "message": "✅ Captura detenida. El video se ha guardado.",
        "video_available": os.path.exists(output_path),
        "tracker": tracker_algorithm,
        "fps": target_fps
    }

@app.get("/status")
def status():
    grupos = detectar_grupos({k: v[:] for k, v in posiciones.items()})
    total_grupos = sum(len(g) for g in grupos.values())
    return {
        "status": "running" if is_streaming else "stopped",
        "tracker": tracker_algorithm,
        "fps": target_fps,
        "port": server_port,
        "personas_detectadas": len(posiciones),
        "frames_procesados": frame_count,
        "grupos_detectados": total_grupos,
        "video_file_exists": os.path.exists(output_path)
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
        raise HTTPException(status_code=404, detail="El archivo de video no está disponible. Asegúrate de haber ejecutado el procesamiento.")

@app.get("/movement_report")
def movement_report():
    """Genera un reporte de texto con el estado de movimiento y cambios de trayectoria"""
    from datetime import datetime

    with lock:
        if not posiciones:
            raise HTTPException(status_code=404, detail="No hay datos para generar el reporte")

        ids = sorted(posiciones.keys())
        lines = ["REPORTE DE MOVIMIENTO\n", "=" * 60 + "\n\n", f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"]

        for pid in ids:
            state = estado_persona[pid]
            last_move = ultimo_movimiento[pid]
            change_count = contador_cambios_tray[pid]
            lines.append(f"ID {pid}\n")
            lines.append(f"  Estado: {state}\n")
            lines.append(f"  Último movimiento: Frame {last_move}\n")
            lines.append(f"  Cambios de trayectoria: {change_count}\n\n")

        report_text = "".join(lines)

    return Response(
        content=report_text,
        media_type="text/plain",
        headers={
            "Content-Disposition": f"attachment; filename=resumen_movimientos_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        }
    )



# NUEVOS ENDPOINTS PARA MAPAS DE CALOR

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
    
@app.get("/ids_with_events")
def ids_with_events():
    """Devuelve solo los IDs que tienen eventos de cambio de dirección"""
    try:
        with lock:
            if not eventos_cambio_direccion:
                return {
                    "ids_with_events": [],
                    "total_ids": 0,
                    "tracker_status": "running" if is_streaming and not should_stop else "stopped"
                }

            ids_con_eventos = []
            for id_obj, eventos in eventos_cambio_direccion.items():
                if eventos:
                    last_event = eventos[-1]
                    frame_value = int(last_event.get('frame', -1)) if isinstance(last_event, dict) else -1

                    ids_con_eventos.append({
                        "id": int(id_obj),  # <-- 🔥 ESTA línea soluciona el error
                        "event_count": int(len(eventos)),
                        "last_event_frame": frame_value
                    })

            ids_con_eventos.sort(key=lambda x: x['event_count'], reverse=True)

            return {
                "ids_with_events": ids_con_eventos,
                "total_ids": int(len(ids_con_eventos)),
                "tracker_status": "running" if is_streaming and not should_stop else "stopped"
            }
    except Exception as e:
        print(f"❌ Error en /ids_with_events: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {e}")

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

@app.get("/stats")
def stats():
    with lock:
        grupos = detectar_grupos({k: v[:] for k, v in posiciones.items()})
        total_grupos = sum(len(g) for g in grupos.values())
        return {
            "personas_detectadas": len(posiciones),
            "frames_procesados": frame_count,
            "ids_activos": list(posiciones.keys()),
            "grupos_detectados": total_grupos,
            "status": "running" if is_streaming else "stopped",
            "tracker": tracker_algorithm,
            "fps": target_fps,
            "video_file_exists": os.path.exists(output_path)
        }
    
@app.get("/restart")
def restart():
    """Reinicia el servicio de streaming"""
    global is_streaming, frame_count, posiciones
    
    # Si ya está en ejecución, no hacer nada
    if is_streaming:
        return {
            "message": "El servicio ya está en ejecución",
            "status": "running"
        }
    
    # Reiniciar variables importantes
    is_streaming = True
    
    # Limpiar posiciones antiguas si se desea
    # posiciones.clear()
    
    # Reiniciar la captura si es necesario
    if cap is None or not cap.isOpened():
        if not initialize_capture():
            raise HTTPException(status_code=500, detail="No se pudo reiniciar la captura")
    
    return {
        "message": "✅ Servicio reiniciado correctamente",
        "status": "running"
    }

@app.get("/id_positions/{id}")
def id_positions(id: int):
    """Devuelve todas las posiciones registradas para un ID específico"""
    with lock:
        if id not in posiciones or not posiciones[id]:
            raise HTTPException(status_code=404, detail=f"No hay datos para el ID {id}")
            
        # Obtener las posiciones para este ID
        positions_data = posiciones[id][:]
        
        # Convertir a formato más amigable para JSON
        formatted_positions = []
        for cx, cy, conf, frame_num in positions_data:
            formatted_positions.append({
                "frame": int(frame_num),
                "x": int(cx),
                "y": int(cy),
                "confidence": float(conf)
            })
        
        return {
            "id": id,
            "count": len(formatted_positions),
            "positions": formatted_positions
        }

if __name__ == "__main__":
    try:
        # Verificar si el puerto está en uso y buscar uno alternativo si es necesario
        import socket
        def is_port_in_use(port):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                return s.connect_ex(('localhost', port)) == 0
                
        def find_available_port(start_port, max_attempts=10):
            port = start_port
            attempts = 0
            while attempts < max_attempts:
                if not is_port_in_use(port):
                    return port
                port += 1
                attempts += 1
            return None
            
        if is_port_in_use(server_port):
            print(f"⚠️ Puerto {server_port} ya está en uso.")
            alternative_port = find_available_port(server_port + 1)
            if alternative_port:
                print(f"🔄 Usando puerto alternativo: {alternative_port}")
                server_port = alternative_port
            else:
                print("❌ No se encontraron puertos disponibles. Intenta cerrar otras aplicaciones.")
                sys.exit(1)
        
        print(f"🚀 Iniciando servidor KOI Eye Cam en http://localhost:{server_port}")
        print(f"🧭 Algoritmo de tracking: {tracker_algorithm}")
        print(f"⏱️ FPS objetivo: {target_fps}")
        print(f"🔄 Detección de cambios de dirección:")
        print(f"   - Umbral de ángulo: {umbral_angulo}°")
        print(f"   - Distancia mínima: {min_distancia} píxeles")
        print(f"📊 Endpoints disponibles:")
        print(f"- /video - Ver video en tiempo real")
        print(f"- /heatmap - Ver mapa de calor general")
        print(f"- /trajectories - Ver mapa de trayectorias")
        print(f"- /heatmap/ID - Ver mapa de calor para un ID específico")
        print(f"- /group_heatmap - Ver mapa de grupos")
        print(f"- /stats - Ver estadísticas de tracking")
        print(f"- /stop - Detener procesamiento")
        print(f"- /download - Descargar video procesado")
        print(f"- /movement_report - Descargar reporte de movimiento")
        
        uvicorn.run("__main__:app", host="0.0.0.0", port=server_port, log_level="warning")
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupción por teclado detectada")
        is_streaming = False
        cleanup_resources()
    finally:
        print("🏁 Servidor finalizado")