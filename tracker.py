from ultralytics import YOLO

from collections import defaultdict

def buscar_id_existente(center_x, center_y, current_time, prev_centers, distancia_umbral=30, tiempo_umbral=0.5):
    """
    Reasigna el ID si la nueva posición es muy cercana a una anterior reciente.
    """
    for pid, historial in prev_centers.items():
        for px, py, t in reversed(historial[-5:]):
            distancia = ((center_x - px)**2 + (center_y - py)**2)**0.5
            if distancia < distancia_umbral and abs(current_time - t) < tiempo_umbral:
                return pid
    return None

import cv2
import time
import datetime
from collections import defaultdict

# Se utiliza el modelo pre-entrenado YOLOv8n
model = YOLO('yolov8n.pt')

# Carga el video
video_path = './supermercado.mp4'
cap = cv2.VideoCapture(video_path)


# Ruta del video original y de salida
input_path = 'video_original.mp4'
output_path = 'video_reducido.mp4'

# Leer propiedades del video
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Nueva resolución (ajustá según lo que necesites)
new_width = 640
new_height = 360

# Definir codec y VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))

# Redimensionar y guardar frame por frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    resized_frame = cv2.resize(frame, (new_width, new_height))
    out.write(resized_frame)

# Liberar recursos
cap.release()
out.release()
cap = cv2.VideoCapture(output_path)
print("✅ Video reducido guardado como:", output_path)

if not cap.isOpened():
    print("Error: No se pudo abrir el video.")
    exit()

# Obtener FPS del video para cálculos de tiempo
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"FPS del video: {fps}")

# Estructuras para almacenar datos por ID
positions = defaultdict(list)  # Posiciones por ID
timestamps = defaultdict(list)  # Timestamps por ID
first_seen = {}  # Primer frame en que se ve cada ID
last_seen = {}   # Último frame en que se ve cada ID
frame_count = 0

# lee frames
while True:
    ret, frame = cap.read()

    if not ret:
        break  # Sale del loop si no se puede leer el frame
    
    frame_count += 1
    current_time = frame_count / fps  # Tiempo en segundos basado en FPS
    
    # Detección y seguimiento de personas (clase 0)
    results = model.track(frame, persist=True, classes=[0])
    
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.int().cpu().numpy()
        
        # Diccionario para historial de centros (se mantiene entre frames)
        if 'prev_centers' not in globals():
            prev_centers = defaultdict(list)

        for i, box in enumerate(boxes):
            yolo_id = int(ids[i])
            x1, y1, x2, y2 = box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # Verificar si este ID podría ser una persona anterior
            id_real = buscar_id_existente(center_x, center_y, current_time, prev_centers)
            if id_real is None:
                id_real = yolo_id  # Usar el ID detectado por YOLO

            # Guardar datos
            positions[id_real].append((center_x, center_y))
            timestamps[id_real].append(current_time)

            if id_real not in first_seen:
                first_seen[id_real] = frame_count
            last_seen[id_real] = frame_count

            # Agregar al historial reciente
            prev_centers[id_real].append((center_x, center_y, current_time))
    
    # Visualización de resultados
    frame_ = results[0].plot()
    cv2.imshow('Tracking personas', frame_)

    # Salir si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera recursos
cap.release()
cv2.destroyAllWindows()

# Generar informe
print("\n===== INFORME DE SEGUIMIENTO =====")
print(f"Total de personas detectadas: {len(positions)}")

# Crear archivo de resumen
with open('resumen_tracking.txt', 'w') as f:
    f.write("===== INFORME DE SEGUIMIENTO =====\n")
    f.write(f"Total de personas detectadas: {len(positions)}\n\n")
    
    for person_id in sorted(positions.keys()):
        # Calcular tiempo total en pantalla (último timestamp - primero)
        if len(timestamps[person_id]) > 1:
            tiempo_total = timestamps[person_id][-1] - timestamps[person_id][0]
        else:
            tiempo_total = 1 / fps  # Duración de un frame
            
        # Calcular frames en pantalla
        frames_en_pantalla = last_seen[person_id] - first_seen[person_id] + 1
        
        # Escribir resumen para esta persona
        f.write(f"ID: {person_id}\n")
        f.write(f"  Tiempo total en pantalla: {tiempo_total:.2f} segundos\n")
        f.write(f"  Frames en pantalla: {frames_en_pantalla}\n")
        f.write(f"  Primera aparición (frame): {first_seen[person_id]}\n")
        f.write(f"  Última aparición (frame): {last_seen[person_id]}\n")
        f.write(f"  Número de detecciones: {len(positions[person_id])}\n")
        f.write(f"  Posiciones: {positions[person_id]}... (primeras 5)\n")
        f.write(f"  Timestamps: {[f'{t:.2f}' for t in timestamps[person_id]]}... (primeros 5)\n")
        f.write("\n")
        
        # Imprimir resumen en consola
        print(f"ID: {person_id}")
        print(f"  Tiempo total en pantalla: {tiempo_total:.2f} segundos")
        print(f"  Frames en pantalla: {frames_en_pantalla}")
        print(f"  Primera aparición (frame): {first_seen[person_id]}")
        print(f"  Última aparición (frame): {last_seen[person_id]}")
        print(f"  Número de detecciones: {len(positions[person_id])}")
        print()

print(f"Informe completo guardado en 'resumen_tracking.txt'")

# Opcionalmente, guardar datos completos para análisis posterior
import json
with open('datos_tracking_completos.json', 'w') as f:
    json.dump({
        'positions': {str(k): v for k, v in positions.items()},
        'timestamps': {str(k): v for k, v in timestamps.items()},
        'first_seen': {str(k): v for k, v in first_seen.items()},
        'last_seen': {str(k): v for k, v in last_seen.items()},
    }, f)

    