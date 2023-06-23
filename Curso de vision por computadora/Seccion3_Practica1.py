#Práctica 1: Detección de bordes y esquinas

#Cargar una imagen y aplicar el detector de bordes de Canny para detectar los bordes en la imagen.
#Utilizar el algoritmo de detección de esquinas, como el algoritmo de Harris o el algoritmo de Shi-Tomasi, para detectar las esquinas en la imagen.
#Visualizar los bordes y las esquinas detectadas en la imagen.

import cv2

# Cargar la imagen
path = r'C:\Users\LENOVO\github\Algoritmos_vision\Curso de vision por computadora\imagen.jpg'
image = cv2.imread(path, 1)

# Convertir la imagen a escala de grises
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Aplicar el detector de bordes de Canny
edges = cv2.Canny(gray_image, 100, 200)

# Aplicar el algoritmo de detección de esquinas (por ejemplo, Harris)
corners = cv2.cornerHarris(gray_image, blockSize=2, ksize=3, k=0.04)

# Marcar los bordes y las esquinas en la imagen
image_with_edges = image.copy()
image_with_edges[edges != 0] = [0, 0, 255]
image_with_corners = image.copy()
image_with_corners[corners > 0.01 * corners.max()] = [0, 255, 0]

# Mostrar la imagen original, los bordes y las esquinas detectadas
cv2.imshow('Imagen original', image)
cv2.imshow('Bordes detectados', image_with_edges)
cv2.imshow('Esquinas detectadas', image_with_corners)
cv2.waitKey(0)