#Práctica 4: Detección y mejora de bordes

#Cargar una imagen y aplicar algoritmos de detección de bordes, como el operador de Sobel o el detector de Canny.
#Visualizar los bordes detectados en la imagen original y comparar los resultados con diferentes parámetros.
#Aplicar técnicas para mejorar la calidad de los bordes detectados, como la supresión de no máximos y la detección de bordes en múltiples escalas.

import cv2

# Cargar la imagen
path = r'C:\Users\LENOVO\github\Algoritmos_vision\Curso de vision por computadora\imagen.jpg'
image = cv2.imread(path, 1)

# Convertir la imagen a escala de grises
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Aplicar el operador de Sobel para detección de bordes
sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
edges = cv2.sqrt(sobel_x**2 + sobel_y**2)

# Aplicar el detector de Canny para detección de bordes
canny_edges = cv2.Canny(gray_image, 100, 200)

# Mostrar las imágenes con los bordes detectados
cv2.imshow('Imagen original', image)
cv2.imshow('Bordes detectados con operador de Sobel', edges.astype('uint8'))
cv2.imshow('Bordes detectados con detector de Canny', canny_edges)
cv2.waitKey(0)