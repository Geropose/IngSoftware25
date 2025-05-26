@echo off
echo  Instalando modulos necesarios...

python -m pip install --upgrade pip
python -m pip install --upgrade ultralytics deep_sort_realtime opencv-python matplotlib streamlit fastapi uvicorn python-multipart
python -m pip install -U yt-dlp

echo Todos los modulos fueron instalados.
pause