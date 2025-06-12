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
video_writer = None
output_path = "stream_output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
posiciones = defaultdict(list)  # Para almacenar posiciones de tracking
video_dims = (640, 480)  # Dimensiones predeterminadas
frame_count = 0
current_frame = None
lock = threading.Lock()

# Validación de argumentos
if len(sys.argv) < 2:
    print("❌ No se proporcionó una URL de stream.")
    sys.exit(1)
stream_url = sys.argv[1]
cap = cv2.VideoCapture(stream_url)

# Obtener dimensiones del video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 25  # Por si no se detecta bien el FPS
video_dims = (width, height)

# Inicializar el escritor de video
video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

def process_frames():
    global is_streaming, video_writer, posiciones, frame_count, current_frame, video_dims
    
    while is_streaming:
        success, frame = cap.read()
        if not success:
            time.sleep(0.1)
            continue

        # Procesamiento con YOLO y tracking
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=0)  # Solo personas
        
        # Dibujar detecciones y almacenar posiciones
        annotated_frame = frame.copy()
        
        if results[0].boxes.id is not None:
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
        
        # Pequeña pausa para no saturar CPU
        time.sleep(0.01)
    
    # Liberar recursos al terminar
    if video_writer is not None:
        video_writer.release()
    cap.release()

def generar_mapa_calor():
    """Genera un mapa de calor general con las posiciones actuales"""
    with lock:
        local_posiciones = {k: v[:] for k, v in posiciones.items()}
        local_dims = video_dims
    
    width, height = local_dims
    
    # Crear una matriz vacía para el mapa de calor
    heatmap = np.zeros((height, width))
    
    # Agregar todas las posiciones al mapa de calor con sus pesos (confianza)
    for id in local_posiciones:
        for cx, cy, conf, _ in local_posiciones[id]:
            if 0 <= cx < width and 0 <= cy < height:
                heatmap[cy, cx] += conf  # Usar confianza como peso
    
    # Aplicar suavizado gaussiano
    heatmap = gaussian_filter(heatmap, sigma=15)
    
    # Normalizar para visualización
    if np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap)
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Usar un mapa de colores más efectivo
    ax.imshow(heatmap, cmap='inferno', interpolation='nearest')
    ax.axis('off')
    
    # Convertir a imagen
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close(fig)
    
    return buf

def generar_mapa_trayectorias():
    """Genera un mapa con todas las trayectorias de las personas"""
    with lock:
        local_posiciones = {k: v[:] for k, v in posiciones.items()}
        local_dims = video_dims
    
    width, height = local_dims
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Configurar límites
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # Invertir eje Y para coincidir con coordenadas de imagen
    
    # Colores para diferentes IDs
    colores = plt.cm.jet(np.linspace(0, 1, max(len(local_posiciones), 1)))
    
    # Dibujar trayectorias
    for i, id in enumerate(local_posiciones):
        # Filtrar IDs con pocos frames
        if len(local_posiciones[id]) < 5:  # Mínimo 5 frames para mostrar trayectoria
            continue
            
        # Ordenar por número de frame
        trayectoria = sorted(local_posiciones[id], key=lambda x: x[3])
        coords = np.array([(x[0], x[1]) for x in trayectoria])
        
        if len(coords) > 1:
            # Dibujar línea de trayectoria
            ax.plot(coords[:, 0], coords[:, 1], '-', color=colores[i % len(colores)], 
                   linewidth=2, alpha=0.7, label=f"ID: {id}")
            
            # Marcar inicio y fin
            ax.plot(coords[0, 0], coords[0, 1], 'o', color=colores[i % len(colores)], markersize=8)
            ax.plot(coords[-1, 0], coords[-1, 1], 's', color=colores[i % len(colores)], markersize=8)
    
    # Agregar leyenda si no hay demasiados IDs
    if len(local_posiciones) <= 15:
        ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
    
    ax.axis('off')
    
    # Convertir a imagen
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close(fig)
    
    return buf

def gen_frames():
    """Generador para streaming de video"""
    while is_streaming:
        with lock:
            if current_frame is not None:
                frame = current_frame.copy()
            else:
                time.sleep(0.1)
                continue
        
        # Convertir frame a JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.04)  # ~25 FPS

@app.get("/")
def root():
    return {"message": "KOI Tracker Live API", "endpoints": ["/video", "/heatmap", "/trajectories", "/stop", "/download"]}

@app.get("/video")
def video():
    global is_streaming
    is_streaming = True
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/heatmap")
def heatmap():
    try:
        buf = generar_mapa_calor()
        return Response(content=buf.getvalue(), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al generar mapa de calor: {str(e)}")

@app.get("/trajectories")
def trajectories():
    try:
        buf = generar_mapa_trayectorias()
        return Response(content=buf.getvalue(), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al generar mapa de trayectorias: {str(e)}")

@app.get("/stop")
def stop():
    global is_streaming
    is_streaming = False
    return {"message": "✅ Stream detenido. El video se ha guardado."}

@app.get("/download")
def download():
    if os.path.exists(output_path):
        return FileResponse(output_path, filename="video_procesado.mp4", media_type="video/x-msvideo")
    else:
        return {"error": "El archivo aún no está disponible o no se ha generado."}

# Iniciar el procesamiento en un hilo separado
threading.Thread(target=process_frames, daemon=True).start()

if __name__ == "__main__":
    uvicorn.run("koi_eye_live:app", host="0.0.0.0", port=8000)