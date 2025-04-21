import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
from PIL import Image

def extraer_todas_las_posiciones(ruta_informe):
    posiciones = []
    with open(ruta_informe, 'r', encoding='latin-1') as archivo:
        lineas = archivo.readlines()
        for i in range(len(lineas)):
            if lineas[i].strip().startswith("Posiciones:"):
                # Buscar todas las tuplas (x, y) incluso en varias líneas
                raw_pos = lineas[i]
                j = i + 1
                while j < len(lineas) and not lineas[j].strip().startswith("Timestamps:") and not lineas[j].strip().startswith("ID:"):
                    raw_pos += lineas[j]
                    j += 1
                matches = re.findall(r'\((\d+),\s*(\d+)\)', raw_pos)
                for match in matches:
                    x, y = int(match[0]), int(match[1])
                    posiciones.append((x, y))
    return posiciones

# Ruta de tus archivos
informe_path = 'resumen_tracking.txt'
imagen_path = 'PLANO640X360.png'

# Extraer posiciones
posiciones = extraer_todas_las_posiciones(informe_path)

if not posiciones:
    print("⚠️ No se encontraron posiciones. Verificá el archivo.")
    exit()

# Cargar imagen
img = Image.open(imagen_path)
ancho, alto = img.size

# Separar coordenadas
x, y = zip(*posiciones)

# Crear heatmap
plt.figure(figsize=(10, 6))
sns.kdeplot(x=x, y=y, cmap='magma', fill=True, bw_adjust=1, alpha=0.6)

plt.imshow(img, extent=[0, ancho, alto, 0])
plt.xlim(0, ancho)
plt.ylim(alto, 0)
plt.axis('off')
plt.colorbar()
plt.tight_layout()
plt.show()
