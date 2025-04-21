import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
import re

# ======================= UTILIDAD =======================

def extraer_posiciones_por_id(file):
    texto = file.read().decode('latin-1')
    bloques = re.findall(r"ID:\s+(\d+)\n(.*?)(?=\nID:|\Z)", texto, re.DOTALL)
    datos = {}
    for pid, bloque in bloques:
        matches = re.findall(r'\((\d+),\s*(\d+)\)', bloque)
        if matches:
            pos = [(int(x), int(y)) for x, y in matches]
            datos[int(pid)] = pos
    return datos

# ======================= INTERFAZ =======================

st.set_page_config(layout="wide", page_title="Heatmap por ID")
st.title("🔥 Mapa de calor por ID (interactivo)")
st.caption("Subí el informe de tracking y la imagen de fondo para visualizar el recorrido de cada persona")

uploaded_tracking = st.file_uploader("📄 Subí el archivo resumen_tracking.txt", type=["txt"])
uploaded_image = st.file_uploader("🖼️ Subí la imagen de fondo (png/jpg)", type=["png", "jpg", "jpeg"])

if uploaded_tracking and uploaded_image:
    datos_por_id = extraer_posiciones_por_id(uploaded_tracking)
    imagen = Image.open(uploaded_image)
    ancho, alto = imagen.size

    st.sidebar.title("🎯 Selección de ID")
    ids_disponibles = sorted(datos_por_id.keys())
    id_seleccionado = st.sidebar.selectbox("Seleccioná un ID:", ids_disponibles)

    posiciones = datos_por_id.get(id_seleccionado, [])
    st.write(f"Cantidad de posiciones para ID {id_seleccionado}: {len(posiciones)}")
    if not posiciones:
        st.warning("No hay posiciones para este ID.")
    else:
        x, y = zip(*posiciones)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(imagen, extent=[0, ancho, alto, 0])
        sns.kdeplot(x=x, y=y, cmap='magma', fill=True, alpha=0.6, bw_adjust=1, ax=ax)
        ax.set_xlim(0, ancho)
        ax.set_ylim(alto, 0)
        ax.axis('off')
        st.pyplot(fig)
else:
    st.info("Esperando archivos...")
