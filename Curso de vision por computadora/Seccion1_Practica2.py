#Práctica 2: Conversión de espacios de color

#Cargar una imagen y convertirla a diferentes espacios de color, como escala de grises, RGB, HSV, etc.
#Visualizar la imagen en cada espacio de color y comparar los resultados.
#Real0izar operaciones de manipulación de color, como cambiar la saturación o el brillo de la imagen.

import cv2

# Cargar la imagen
image = cv2.imread('imagen.jpg')

# Convertir la imagen a escala de grises
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Convertir la imagen a HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Mostrar las imágenes en diferentes espacios de color
cv2.imshow('Imagen original', image)
cv2.imshow('Imagen en escala de grises', gray_image)
cv2.imshow('Imagen en HSV', hsv_image)
cv2.waitKey(0)