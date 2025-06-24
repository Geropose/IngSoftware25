from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse, Response
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
video_dims = (640, 480)
frame_count = 0
current_frame = None
lock = threading.Lock()
cap = None
processing_thread = None
tracker_algorithm = "bytetrack"  # Default tracker
target_fps = 25  # Default FPS

# Validación de argumentos
if len(sys.argv) < 2:
    print("❌ No se proporcionó una URL de stream.")
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
    """Procesa los frames del video con control de parada mejorado"""
    global is_streaming, should_stop, video_writer, posiciones, frame_count, current_frame, video_dims, tracker_algorithm
    
    print(f"🎬 Iniciando procesamiento de frames con algoritmo {tracker_algorithm} a {target_fps} FPS...")
    
    frame_skip_count = 0
    consecutive_fails = 0
    max_fails = 50  # Máximo de fallas consecutivas antes de parar
    
    last_frame_time = time.time()
    
    while is_streaming and not should_stop:
        # Control de FPS - esperar hasta que sea tiempo para el siguiente frame
        current_time = time.time()
        elapsed = current_time - last_frame_time
        
        if elapsed < frame_interval:
            # Esperar el tiempo necesario para mantener el FPS objetivo
            sleep_time = frame_interval - elapsed
            time.sleep(min(sleep_time, 0.1))  # Limitar a 100ms máximo para poder verificar parada
            continue
        
        # Actualizar tiempo del último frame procesado
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
        
        consecutive_fails = 0  # Reset counter on successful read

        try:
            # Procesamiento con YOLO y tracking usando el algoritmo seleccionado
            tracker_config = f"{tracker_algorithm}.yaml"
            results = model.track(frame, persist=True, tracker=tracker_config, classes=0)
            
            # Dibujar detecciones y almacenar posiciones
            annotated_frame = frame.copy()
            
            if results[0].boxes is not None and results[0].boxes.id is not None:
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                
                for id, box, conf in zip(ids, boxes, confs):
                    # Verificar si debe parar antes de procesar cada detección
                    if should_stop:
                        break
                        
                    x1, y1, x2, y2 = map(int, box)
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    
                    # Almacenar posición con peso (confianza) y frame
                    with lock:
                        posiciones[id].append((cx, cy, conf, frame_count))
                        # Limitar histórico para no usar demasiada memoria
                        if len(posiciones[id]) > 1000:
                            posiciones[id] = posiciones[id][-500:]
                    
                    # Dibujar en el frame
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"ID: {id} ({conf:.2f})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Verificar parada antes de continuar
            if should_stop:
                break
            
            # Agregar información de frame, estado, algoritmo y FPS
            status_text = f"{tracker_algorithm.upper()} @ {target_fps}FPS - {('STOPPING' if should_stop else 'RECORDING')}"
            cv2.putText(annotated_frame, f"Frame: {frame_count} - {status_text}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Calcular FPS real
            real_fps = 1.0 / (time.time() - last_frame_time) if (time.time() - last_frame_time) > 0 else 0
            cv2.putText(annotated_frame, f"FPS real: {real_fps:.1f}", (20, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Guardar frame en el video solo si no se está deteniendo
            if video_writer is not None and not should_stop:
                video_writer.write(annotated_frame)
            
            # Actualizar frame actual para streaming
            with lock:
                current_frame = annotated_frame
                frame_count += 1
                
        except Exception as e:
            print(f"❌ Error procesando frame {frame_count}: {e}")
            continue
    
    # Limpieza final
    print("🛑 Finalizando procesamiento...")
    cleanup_resources()
    
    # Marcar como no streaming
    is_streaming = False
    print("✅ Procesamiento finalizado")

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



def gen_frames():
    """Generador para streaming de video"""
    last_frame_time = time.time()
    
    while is_streaming and not should_stop:
        # Control de FPS para streaming
        current_time = time.time()
        elapsed = current_time - last_frame_time
        
        if elapsed < stream_interval:
            # Esperar el tiempo necesario para mantener el FPS objetivo para streaming
            time.sleep(min(stream_interval - elapsed, 0.01))  # Pequeña espera para no saturar CPU
            continue
            
        last_frame_time = time.time()
        
        with lock:
            if current_frame is not None:
                frame = current_frame.copy()
            else:
                time.sleep(0.1)
                continue
        
        # Convertir frame a JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.get("/")
def root():
    return {
        "message": "KOI Tracker Live API", 
        "status": "running" if is_streaming else "stopped",
        "tracker": tracker_algorithm,
        "fps": target_fps,
        "endpoints": ["/video", "/heatmap", "/trajectories", "/heatmap/{id}", "/group_heatmap", "/stats", "/stop", "/download", "/force_stop"]
    }

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
            "status": "running" if is_streaming and not should_stop else "stopped",
            "tracker": tracker_algorithm,
            "fps": target_fps,
            "video_file_exists": os.path.exists(output_path)
        }

@app.get("/stop")
def stop():
    global is_streaming, should_stop
    print("🛑 Recibida señal de parada...")
    should_stop = True
    
    # Esperar un poco para que el procesamiento termine
    max_wait = 10  # segundos
    wait_count = 0
    
    while is_streaming and wait_count < max_wait:
        time.sleep(0.5)
        wait_count += 0.5
        print(f"⏳ Esperando parada... ({wait_count}s)")
    
    if is_streaming:
        print("⚠️ Forzando parada...")
        is_streaming = False
        cleanup_resources()
    
    return {
        "message": "✅ Stream detenido. El video se ha guardado.",
        "video_available": os.path.exists(output_path),
        "frames_processed": frame_count,
        "tracker": tracker_algorithm,
        "fps": target_fps
    }

@app.get("/force_stop")
def force_stop():
    """Parada forzada inmediata"""
    global is_streaming, should_stop
    print("💥 PARADA FORZADA ACTIVADA")
    
    should_stop = True
    is_streaming = False
    
    # Limpiar recursos inmediatamente
    cleanup_resources()
    
    return {
        "message": "💥 Parada forzada ejecutada",
        "video_available": os.path.exists(output_path),
        "frames_processed": frame_count,
        "tracker": tracker_algorithm,
        "fps": target_fps
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
    
@app.get("/restart")
def restart():
    """Reinicia el servicio de streaming"""
    global is_streaming, should_stop, frame_count, current_frame, posiciones
    
    # Si ya está en ejecución, no hacer nada
    if is_streaming and not should_stop:
        return {
            "message": "El servicio ya está en ejecución",
            "status": "running"
        }
    
    # Reiniciar variables importantes
    is_streaming = True
    should_stop = False
    
    # Reiniciar la captura si es necesario
    if cap is None or not cap.isOpened():
        if not initialize_capture():
            raise HTTPException(status_code=500, detail="No se pudo reiniciar la captura")
    
    # Reiniciar el hilo de procesamiento
    global processing_thread
    if processing_thread is None or not processing_thread.is_alive():
        processing_thread = threading.Thread(target=process_frames, daemon=False)
        processing_thread.start()
        print("🔄 Hilo de procesamiento reiniciado")
    
    return {
        "message": "✅ Servicio reiniciado correctamente",
        "status": "running"
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

# Función de inicialización
def initialize_system():
    """Inicializa todo el sistema"""
    if not initialize_capture():
        return False
    
    global processing_thread
    processing_thread = threading.Thread(target=process_frames, daemon=False)  # No daemon para control completo
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
        print(f"📊 Endpoints disponibles:")
        print(f"- /video - Ver video en tiempo real")
        print(f"- /heatmap - Ver mapa de calor general")
        print(f"- /trajectories - Ver mapa de trayectorias")
        print(f"- /heatmap/ID - Ver mapa de calor para un ID específico")
        print(f"- /group_heatmap - Ver mapa de grupos")
        print(f"- /stats - Ver estadísticas de tracking")
        print(f"- /stop - Detener procesamiento")
        print(f"- /force_stop - Parada forzada")
        print(f"- /download - Descargar video procesado")
        
        uvicorn.run("__main__:app", host="0.0.0.0", port=8000, log_level="warning")
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupción por teclado detectada")
        should_stop = True
        is_streaming = False
        cleanup_resources()
    finally:
        print("🏁 Servidor finalizado")