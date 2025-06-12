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
should_stop = False  # Variable adicional para controlar parada
video_writer = None
output_path = "stream_output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
posiciones = defaultdict(list)  # Para almacenar posiciones de tracking
video_dims = (640, 480)  # Dimensiones predeterminadas
frame_count = 0
current_frame = None
lock = threading.Lock()
cap = None

# Validación de argumentos
if len(sys.argv) < 2:
    print("❌ No se proporcionó una URL de stream.")
    sys.exit(1)

stream_url = sys.argv[1]
print(f"🎬 Conectando a: {stream_url}")

# Configurar captura de video
if stream_url == "0":  # Para webcam
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("❌ No se pudo abrir el stream")
    sys.exit(1)

# Obtener dimensiones del video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 25
video_dims = (width, height)

print(f"📹 Dimensiones: {width}x{height} @ {fps}fps")

# Inicializar el escritor de video
video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

def process_frames():
    global is_streaming, should_stop, video_writer, posiciones, frame_count, current_frame, video_dims, cap
    
    while is_streaming and not should_stop:
        if cap is None:
            break
            
        success, frame = cap.read()
        if not success:
            print("⚠️ No se pudo leer frame")
            time.sleep(0.1)
            continue

        try:
            # Procesamiento con YOLO y tracking
            results = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=0)
            
            # Dibujar detecciones y almacenar posiciones
            annotated_frame = frame.copy()
            
            if results[0].boxes is not None and results[0].boxes.id is not None:
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                
                for id, box, conf in zip(ids, boxes, confs):
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
            
            # Agregar información de frame
            cv2.putText(annotated_frame, f"Frame: {frame_count}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Guardar frame en el video
            if video_writer is not None:
                video_writer.write(annotated_frame)
            
            # Actualizar frame actual para streaming
            with lock:
                current_frame = annotated_frame
                frame_count += 1
                
        except Exception as e:
            print(f"❌ Error procesando frame: {e}")
            continue
        
        # Pequeña pausa para no saturar CPU
        time.sleep(0.01)
        
        # Verificar si debe parar
        if should_stop:
            break
    
    # Liberar recursos al terminar
    print("🛑 Liberando recursos...")
    if video_writer is not None:
        video_writer.release()
        video_writer = None
    if cap is not None:
        cap.release()
        cap = None
    print("✅ Recursos liberados")

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
    ax.set_title('Mapa de Calor General', fontsize=14, color='white')
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
    
    ax.set_title('Mapa de Trayectorias', fontsize=14, color='white')
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
    
    ax.set_title(f'Análisis ID {id} - {len(trayectoria)} puntos', fontsize=14, color='white')
    ax.axis('off')
    
    # Convertir a imagen
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, facecolor='black')
    buf.seek(0)
    plt.close(fig)
    
    return buf

def gen_frames():
    """Generador para streaming de video"""
    while is_streaming and not should_stop:
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
        
        time.sleep(0.04)  # ~25 FPS

@app.get("/")
def root():
    return {
        "message": "KOI Tracker Live API", 
        "status": "running" if is_streaming else "stopped",
        "endpoints": ["/video", "/heatmap", "/trajectories", "/heatmap/{id}", "/stats", "/stop", "/download"]
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

@app.get("/stats")
def stats():
    with lock:
        return {
            "personas_detectadas": len(posiciones),
            "frames_procesados": frame_count,
            "ids_activos": list(posiciones.keys()),
            "status": "running" if is_streaming else "stopped"
        }

@app.get("/stop")
def stop():
    global is_streaming, should_stop
    print("🛑 Recibida señal de parada...")
    should_stop = True
    is_streaming = False
    
    # Dar tiempo para que el hilo termine
    time.sleep(1)
    
    return {"message": "✅ Stream detenido. El video se ha guardado."}

@app.get("/download")
def download():
    if os.path.exists(output_path):
        return FileResponse(output_path, filename="video_procesado.mp4", media_type="video/mp4")
    else:
        raise HTTPException(status_code=404, detail="El archivo aún no está disponible o no se ha generado.")

# Iniciar el procesamiento en un hilo separado
processing_thread = threading.Thread(target=process_frames, daemon=True)
processing_thread.start()

if __name__ == "__main__":
    print(f"🚀 Iniciando servidor KOI Tracker Live en http://localhost:8000")
    print(f"📹 Stream: {stream_url}")
    print(f"📊 Endpoints disponibles:")
    print(f"- /video - Ver video en tiempo real")
    print(f"- /heatmap - Ver mapa de calor general")
    print(f"- /trajectories - Ver mapa de trayectorias")
    print(f"- /heatmap/ID - Ver mapa de calor para un ID específico")
    print(f"- /stats - Ver estadísticas de tracking")
    print(f"- /stop - Detener procesamiento")
    print(f"- /download - Descargar video procesado")
    uvicorn.run("koi_tracker_live:app", host="0.0.0.0", port=8000)