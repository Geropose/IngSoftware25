import cv2
import os
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter

# Carga el modelo YOLO
model = YOLO("yolov8n.pt")  # Usa el modelo ligero por velocidad

def procesar_video(video_path, output_path="output.mp4", nueva_resolucion=None,algoritmo="bytetrack"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("No se pudo abrir el video")

    # Resolución original del video
    width_original = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_original = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Resolución personalizada si se especificó
    if nueva_resolucion:
        width, height = nueva_resolucion
    else:
        width, height = width_original, height_original

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Almacenar dimensiones del video para mapas de calor
    video_dims = (width, height)

    # Diccionario para almacenar posiciones con pesos
    posiciones = defaultdict(list)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame is None or frame.shape[0] == 0 or frame.shape[1] == 0:
            print(f"Frame inválido en {frame_count}, se omite.")
            continue

        if frame.shape[:2] != (height, width):
            print(f"Resolución inconsistente en frame {frame_count}: {frame.shape[:2]}. Redimensionando.")
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        # Redimensionar el frame si es necesario
        #if nueva_resolucion:
         #   frame = cv2.resize(frame, (width, height))

        cv2.putText(frame, f'Resolucion: {width}x{height}', (20, 40), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # Ejecutar detección y tracking
        if algoritmo == "bytetrack":
            # Usar ByteTrack para detección y seguimiento
            results = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=0)# Solo personas
        else: 
            # Usar el algoritmo de seguimiento por defecto
            print("Usando algoritmo de seguimiento : ", algoritmo)
            results = model.track(frame, persist=True, tracker="botsort.yaml", classes=0)# Solo personas

        if results[0].boxes.id is not None:
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()  # Obtener confianzas

            for id, box, conf in zip(ids, boxes, confs):
                x1, y1, x2, y2 = map(int, box)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # Almacenar posición con peso (confianza) y frame
                posiciones[id].append((cx, cy, conf, frame_count))

                # Dibujar en el frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {id} ({conf:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Mostrar progreso
        if frame_count % 30 == 0:
            print(f"Procesando frame {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)")

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    return output_path, posiciones, video_dims



def generar_mapa_calor_general(posiciones, video_dims, sigma=15):
    """
    Genera un mapa de calor general con suavizado gaussiano
    """
    width, height = video_dims
    
    # Crear una matriz vacía para el mapa de calor
    heatmap = np.zeros((height, width))
    
    # Agregar todas las posiciones al mapa de calor con sus pesos (confianza)
    for id in posiciones:
        for cx, cy, conf, _ in posiciones[id]:
            if 0 <= cx < width and 0 <= cy < height:
                heatmap[cy, cx] += conf  # Usar confianza como peso
    
    # Aplicar suavizado gaussiano
    heatmap = gaussian_filter(heatmap, sigma=sigma)
    
    # Normalizar para visualización
    if np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap)
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Usar un mapa de colores más efectivo
    ax.imshow(heatmap, cmap='inferno', interpolation='nearest')
    ax.axis('off')
    
    return fig

def generar_mapa_calor_por_id(posiciones, id, video_dims, sigma=10):
    """
    Genera un mapa de calor para un ID específico con trayectoria
    """
    if id not in posiciones or not posiciones[id]:
        return None
        
    width, height = video_dims
    
    # Crear una matriz vacía para el mapa de calor
    heatmap = np.zeros((height, width))
    
    # Extraer coordenadas y ordenarlas por número de frame
    trayectoria = sorted(posiciones[id], key=lambda x: x[3])
    
    # Agregar puntos al mapa de calor con pesos basados en confianza
    for cx, cy, conf, _ in trayectoria:
        if 0 <= cx < width and 0 <= cy < height:
            heatmap[cy, cx] += conf
    
    # Aplicar suavizado gaussiano
    heatmap = gaussian_filter(heatmap, sigma=sigma)
    
    # Normalizar para visualización
    if np.max(heatmap) > 0:
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
    
    ax.axis('off')
    
    return fig

def generar_mapa_trayectorias(posiciones, video_dims, min_frames=10):
    """
    Genera un mapa con todas las trayectorias de las personas
    """
    width, height = video_dims
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Configurar límites
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # Invertir eje Y para coincidir con coordenadas de imagen
    
    # Colores para diferentes IDs
    colores = plt.cm.jet(np.linspace(0, 1, len(posiciones)))
    
    # Dibujar trayectorias
    for i, id in enumerate(posiciones):
        # Filtrar IDs con pocos frames
        if len(posiciones[id]) < min_frames:
            continue
            
        # Ordenar por número de frame
        trayectoria = sorted(posiciones[id], key=lambda x: x[3])
        coords = np.array([(x[0], x[1]) for x in trayectoria])
        
        if len(coords) > 1:
            # Dibujar línea de trayectoria
            ax.plot(coords[:, 0], coords[:, 1], '-', color=colores[i % len(colores)], 
                   linewidth=2, alpha=0.7, label=f"ID: {id}")
            
            # Marcar inicio y fin
            ax.plot(coords[0, 0], coords[0, 1], 'o', color=colores[i % len(colores)], markersize=8)
            ax.plot(coords[-1, 0], coords[-1, 1], 's', color=colores[i % len(colores)], markersize=8)
    
    # Agregar leyenda si no hay demasiados IDs
    if len(posiciones) <= 15:
        ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
    
    ax.axis('off')
    
    return fig