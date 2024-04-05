import cv2
import numpy as np

# Cargar la imagen
imagen = cv2.imread('/home/nicolas/Github/Algoritmos_vision/imagen.png', cv2.IMREAD_COLOR)

# Ajustar el brillo de la imagen para simular un ISO más alto
imagen_brillante = cv2.convertScaleAbs(imagen, alpha=-0.1, beta=0)

# Añadir ruido gaussiano a la imagen para simular el ruido que se obtiene con un ISO más alto
# ruido = np.random.normal(0, 10, imagen_brillante.shape).astype(np.uint8)
# imagen_con_ruido = cv2.add(imagen_brillante, ruido)

# Mostrar la imagen original y la imagen con el ISO simulado
cv2.imshow('Imagen Original', imagen)
cv2.imshow('Imagen con ISO Simulado', imagen_brillante)

cv2.waitKey(0)
cv2.destroyAllWindows()
