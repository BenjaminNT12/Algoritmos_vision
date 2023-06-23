#Práctica 5: Transformaciones geométricas

#Cargar una imagen y aplicar transformaciones geométricas, como la traslación, la rotación, la escala y la perspectiva.
#Visualizar la imagen transformada y compararla con la imagen original.
#Experimentar con diferentes parámetros y observar los efectos de cada transformación en la imagen.

import cv2
import numpy as np

# Cargar la imagen
path = r'C:\Users\LENOVO\github\Algoritmos_vision\Curso de vision por computadora\imagen.jpg'
image = cv2.imread(path, 1)

# Definir los puntos de destino para la transformación
rows, cols = image.shape[:2]
pts1 = np.float32([[50, 50], [cols - 50, 50], [50, rows - 50], [cols - 50, rows - 50]])
pts2 = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])

# Calcular la matriz de transformación
M = cv2.getPerspectiveTransform(pts1, pts2)

# Aplicar la transformación a la imagen
transformed_image = cv2.warpPerspective(image, M, (cols, rows))

# Mostrar la imagen original y la imagen transformada
cv2.imshow('Imagen original', image)
cv2.imshow('Imagen transformada', transformed_image)
cv2.waitKey(0)