@echo off
echo  Instalando modulos necesarios...

python -m pip install --upgrade pip
python -m pip install --upgrade ultralytics deep_sort_realtime opencv-python matplotlib streamlit

echo Todos los modulos fueron instalados.
pause
