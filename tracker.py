from ultralytics import YOLO
import cv2

# Se utiliza el modelo pre-entrenado YOLOv8n, optimizado para velocidad y precisión
model = YOLO('yolov8n.pt')

# Carga el video
video_path = './test.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: No se pudo abrir el video.")
    exit()

# lee frames
while True:
    ret, frame = cap.read()

    if not ret:
        break # Sale del loop si no se puede leer el frame

    # Detección y seguimiento de personas (clase 0)
    results = model.track(frame, persist=True, classes=[0])
    # persist=True: Mantiene los IDs de las personas entre cuadros.
    # classes=[0]: Filtra solo personas (clase 0 en el dataset COCO).

    # Visualización de resultados con cuadros delimitadores y tracking IDs
    frame_ = results[0].plot()
    cv2.imshow('Tracking personas', frame_)

    # Salir si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera recursos
cap.release()
cv2.destroyAllWindows()