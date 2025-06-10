from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse
import subprocess
import cv2
from ultralytics import YOLO
import uvicorn
import sys
import threading
import time
import os

app = FastAPI()
model = YOLO("yolov8n.pt")  # Cambiar por otro modelo si querés

# Validación de argumentos
if len(sys.argv) < 2:
    print("❌ No se proporcionó una URL de stream.")
    sys.exit(1)
stream_url = sys.argv[1]
cap = cv2.VideoCapture(stream_url)

# Globales para control
is_streaming = True
video_writer = None
output_path = "stream_output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  

def gen_frames():
    global is_streaming, video_writer

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25  # Por si no se detecta bien el FPS

    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while is_streaming:
        success, frame = cap.read()
        if not success:
            break

        # Procesamiento con YOLO 
        results = model(frame, verbose=False)[0]
        results.boxes = results.boxes[results.boxes.cls == 0]
        annotated = results.plot()

        # Guardar frame
        video_writer.write(annotated)

        _, buffer = cv2.imencode('.jpg', annotated)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    # liberar recursos
    if video_writer:
        video_writer.release()
    cap.release()

@app.get("/")
def root():
    return {"message": "Ir a /video para ver el stream"}

@app.get("/video")
def video():
    global is_streaming
    is_streaming = True
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

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

if __name__ == "__main__":
    uvicorn.run("koi_eye_live:app", host="0.0.0.0", port=8000)
