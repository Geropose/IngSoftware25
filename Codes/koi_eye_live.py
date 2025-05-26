from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import subprocess
import cv2
from ultralytics import YOLO
import uvicorn
import sys

app = FastAPI()
model = YOLO("yolov8n.pt")  # Cambiar por otro modelo si querés
if len(sys.argv) < 2:
    print("❌ No se proporcionó una URL de stream.")
    sys.exit(1)
stream_url = sys.argv[1]
cap = cv2.VideoCapture(stream_url)



def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        results = model(frame, verbose=False)[0]
        # Filtrar solo objetos de clase 'person' (ID 0)
        results.boxes = results.boxes[results.boxes.cls == 0]
        annotated = results.plot()
        _, buffer = cv2.imencode('.jpg', annotated)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.get("/")
def root():
    return {"message": "Ir a /video para ver el stream"}

@app.get("/video")
def video():
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

# ✅ Ejecutar directamente sin `reload=True` para evitar errores en Windows
if __name__ == "__main__":
    uvicorn.run("koi_eye_live:app", host="0.0.0.0", port=8000)
