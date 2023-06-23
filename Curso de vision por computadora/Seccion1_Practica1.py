#Curso de visión por computadora utilizando OpenCV y Python:

#Introducción a la visión por computadora y OpenCV

#Conceptos básicos de visión por computadora
#Introducción a OpenCV y sus características
#Configuración del entorno de desarrollo
#Manipulación de imágenes con OpenCV

#Carga y visualización de imágenes
#Operaciones básicas de manipulación de imágenes (recorte, redimensionamiento, rotación)
#Conversión entre espacios de color
#Filtrado y suavizado de imágenes
#Detección y descripción de características

#Detección de bordes y esquinas
#Descriptores locales (SIFT, SURF, ORB)
#Emparejamiento de características
#Transformaciones geométricas y calibración de cámaras

#Transformaciones afines y proyectivas
#Estimación de homografía
#Calibración de cámaras
#Seguimiento y detección de objetos

#Seguimiento de objetos en secuencias de video
#Detección de objetos utilizando clasificadores en cascada (Haar Cascade)
#Detección de objetos basada en aprendizaje profundo (YOLO, SSD)
#Segmentación de imágenes

#Umbralización y detección de contornos
#Segmentación por color y textura
#Uso de algoritmos de agrupamiento (K-means, Mean Shift)
#Reconocimiento de objetos y reconocimiento facial

#Extracción de características locales
#Aplicación de clasificadores (SVM, KNN)
#Reconocimiento facial utilizando Eigenfaces y Fisherfaces
#Procesamiento de imágenes en tiempo real

#Captura de video utilizando cámaras
#Procesamiento en tiempo real de secuencias de video
#Aplicaciones prácticas de visión por computadora en tiempo real
#Proyectos y prácticas

#Desarrollo de proyectos prácticos utilizando OpenCV y Python
#Resolución de problemas y desafíos en visión por computadora

# Práctica 1: Manipulación básica de imágenes

#Cargar una imagen y mostrarla en una ventana.
#Aplicar operaciones de recorte, redimensionamiento y rotación a la imagen.
#Guardar la imagen resultante en el disco.

import cv2

# Cargar la imagen
path = r'C:\Users\LENOVO\github\Algoritmos_vision\Curso de vision por computadora\imagen.jpg'
image = cv2.imread(path, 1)

# Mostrar la imagen en una ventana
cv2.imshow('Imagen original', image)
cv2.waitKey(0)

# Recortar la imagen
cropped_image = image[100:300, 200:400]

# Redimensionar la imagen
resized_image = cv2.resize(image, (500, 500))

# Rotar la imagen
rows, cols = image.shape[:2]
M = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 1)
rotated_image = cv2.warpAffine(image, M, (cols, rows))

# Guardar la imagen resultante
cv2.imwrite('imagen_recortada.jpg', cropped_image)
cv2.imwrite('imagen_redimensionada.jpg', resized_image)
cv2.imwrite('imagen_rotada.jpg', rotated_image)

# Mostrar las imágenes resultantes
cv2.imshow('Imagen recortada', cropped_image)
cv2.imshow('Imagen redimensionada', resized_image)
cv2.imshow('Imagen rotada', rotated_image)
cv2.waitKey(0)

