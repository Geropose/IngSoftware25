import cv2
import os
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter
from deep_sort_realtime.deepsort_tracker import DeepSort

tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0, max_cosine_distance=0.4)



# Carga el modelo YOLO
model = YOLO("yolov8n.pt") 

def procesar_video_deepsort(video_path, output_path="output.mp4", nueva_resolucion=None, mostrar_video=False, algoritmo="deepsort"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("No se pudo abrir el video")

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

    # Almacenar dimensiones del video para los mapas de calor
    video_dims = (width, height)
    
    # Diccionario para almacenar posiciones con pesos
    posiciones = defaultdict(list)
    frame_count = 0

    # Crear ventana para visualización si se solicita
    if mostrar_video:
        cv2.namedWindow('Procesamiento de Video', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Procesamiento de Video', width // 2, height // 2)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Redimensionar el frame si es necesario
        if nueva_resolucion:
            frame = cv2.resize(frame, (width, height))

        cv2.putText(frame, f'Resolucion: {width}x{height}', (20, 40), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        results = model(frame, classes=0)
        detections = []
        
        if results[0].boxes.xyxy is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            for box, conf in zip(boxes, confs):
                x1, y1, x2, y2 = box.astype(int)
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person'))  # formato: (bbox, conf, class_name)
            
            tracks = tracker.update_tracks(detections, frame=frame)  # lista de objetos Track
            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                
                # Guardar coordenadas
                posiciones[track_id].append((cx, cy, conf, frame_count))

                # Dibujar
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Mostrar progreso
        if frame_count % 30 == 0:
            print(f"Procesando frame {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)")

        # Mostrar frame procesado en ventana
        #if mostrar_video:
        #    frame_mostrar = cv2.resize(frame, (width // 2, height // 2))
        #    cv2.imshow('Procesamiento de Video', frame_mostrar)
            
        # Salir si se presiona 'q'
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break
        out.write(frame)
        frame_count += 1


    cap.release()
    out.release()

    if mostrar_video:
        cv2.destroyAllWindows()
    return output_path, posiciones, video_dims

def generar_mapa_calor_general_deepsort(posiciones, video_dims, sigma=15):
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

def generar_mapa_calor_por_id_deepsort(posiciones, id, video_dims, sigma=10):
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

def generar_mapa_trayectorias_deepsort(posiciones, video_dims, min_frames=10):
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

"""
if __name__ == "__main__":
    video_path = "pruebaAeropuerto.mp4" 
    
    # Procesar el video mostrando la visualización en tiempo real
    output_path, posiciones, video_dims = procesar_video(video_path, mostrar_video=True)
    
    # Generar y mostrar mapas de calor
    fig_general = generar_mapa_calor_general(posiciones, video_dims)
    fig_general.savefig("mapa_calor_general.png")
    plt.show()
    
    
    if posiciones:
        id_ejemplo = next(iter(posiciones))
        fig_id = generar_mapa_calor_por_id(posiciones, id_ejemplo, video_dims)
        if fig_id:
            fig_id.savefig(f"mapa_calor_id_{id_ejemplo}.png")
            plt.show()
    
    # Generar mapa de trayectorias
    fig_trayectorias = generar_mapa_trayectorias(posiciones, video_dims)
    fig_trayectorias.savefig("mapa_trayectorias.png")
    plt.show()
"""