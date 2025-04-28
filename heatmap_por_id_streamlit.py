import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
import re
import os

# ======================= UTILIDAD =======================

# Función para extraer las posiciones de cada ID desde el archivo de tracking
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

# Configuración inicial de la página
st.set_page_config(layout="wide", page_title="Heatmap por ID")
st.title("🔥 Mapa de calor por ID (interactivo)")
st.caption("Subí el informe de tracking y la imagen de fondo para visualizar el recorrido de cada persona")

# Carga de archivos
uploaded_tracking = st.file_uploader("📄 Subí el archivo resumen_tracking.txt", type=["txt"])
uploaded_image = st.file_uploader("🖼️ Subí la imagen de fondo (png/jpg)", type=["png", "jpg", "jpeg"])

if uploaded_tracking and uploaded_image:
    # Procesar datos de tracking e imagen
    datos_por_id = extraer_posiciones_por_id(uploaded_tracking)
    imagen = Image.open(uploaded_image)
    ancho, alto = imagen.size

    # Cargar imágenes de personas detectadas
    carpeta_personas = "personas_detectadas"
    imagenes_disponibles = {}
    if os.path.exists(carpeta_personas):
        for nombre_archivo in os.listdir(carpeta_personas):
            if nombre_archivo.startswith("ID_") and nombre_archivo.endswith(".jpg"):
                partes = nombre_archivo.split('_')
                if len(partes) >= 2:
                    id_str = partes[1]
                    if id_str.isdigit():
                        id_img = int(id_str)
                        if id_img not in imagenes_disponibles:
                            imagenes_disponibles[id_img] = []
                        imagenes_disponibles[id_img].append(os.path.join(carpeta_personas, nombre_archivo))

    # Sidebar para seleccionar el modo de búsqueda
    st.sidebar.title("🎯 Selección de ID o Imagen")
    modo = st.sidebar.radio("Seleccionar por:", ["ID", "Imagen"])

    if modo == "ID":
        ids_disponibles = sorted(datos_por_id.keys())
        id_seleccionado = st.sidebar.selectbox("Seleccioná un ID:", ids_disponibles)
    else:
        # MOSTRAR IMÁGENES CON BOTONES (NUEVA VERSIÓN)
        st.sidebar.subheader("Seleccioná una imagen:")
        imagen_seleccionada = None
        
        # Crear columnas para organizar las imágenes y botones
        cols = st.sidebar.columns(1)  # 2 columnas para mejor organización
        
        for i, (id_img, paths) in enumerate(imagenes_disponibles.items()):
            if paths:
                img_path = paths[0]
                col = cols[i % 1]  # Alternar entre columnas
                
                # Mostrar la imagen en miniatura
                try:
                    img = Image.open(img_path)
                    col.image(img, use_container_width=True, caption=f"ID {id_img}")
                    
                    # Botón para seleccionar
                    if col.button(f"Seleccionar ID {id_img}", key=f"btn_{id_img}"):
                        imagen_seleccionada = id_img
                except Exception as e:
                    col.error(f"Error al cargar imagen para ID {id_img}: {str(e)}")
        
        id_seleccionado = imagen_seleccionada

    if id_seleccionado is not None:
        # Mostrar cantidad de posiciones encontradas para el ID seleccionado
        posiciones = datos_por_id.get(id_seleccionado, [])
        st.write(f"Cantidad de posiciones para ID {id_seleccionado}: {len(posiciones)}")

        if not posiciones:
            st.warning("No hay posiciones para este ID.")
        else:
            # Mostrar mapa de calor
            x, y = zip(*posiciones)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(imagen, extent=[0, ancho, alto, 0])
            sns.kdeplot(x=x, y=y, cmap='magma', fill=True, alpha=0.6, bw_adjust=1, ax=ax)
            ax.set_xlim(0, ancho)
            ax.set_ylim(alto, 0)
            ax.axis('off')
            st.pyplot(fig)

            # Mostrar imágenes capturadas para el ID seleccionado
            st.subheader(f"🧑‍🦰 Imágenes capturadas para ID {id_seleccionado}")

            imagenes_id = imagenes_disponibles.get(id_seleccionado, [])

            if imagenes_id:
                for img_path in imagenes_id:
                    st.image(img_path, width=300, caption=os.path.basename(img_path))
            else:
                st.info("No se encontraron imágenes para este ID en 'personas_detectadas'.")
else:
    st.info("Esperando archivos...")