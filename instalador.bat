@echo off
echo 🔧 Instalando módulos necesarios...

python -m pip install --upgrade pip
python -m pip install ultralytics deep_sort_realtime opencv-python matplotlib streamlit

echo ✅ Todos los módulos fueron instalados.
pause
