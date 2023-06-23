#Práctica 4: Extracción de características en imágenes panorámicas

#Cargar un conjunto de imágenes y realizar la alineación y el empalme de las imágenes para crear una imagen panorámica.
#Utilizar algoritmos de detección y descripción de características para extraer características en la imagen panorámica.
#Realizar el emparejamiento de características entre la imagen panorámica y las imágenes originales.
#Visualizar los resultados del emparejamiento de características en la imagen panorámica.


import cv2
import numpy as np

# Cargar las imágenes para la creación de la imagen panorámica
path1 = r'C:\Users\LENOVO\github\Algoritmos_vision\Curso de vision por computadora\imagen1.jpg'
image1 = cv2.imread(path1, 1)
path2 = r'C:\Users\LENOVO\github\Algoritmos_vision\Curso de vision por computadora\imagen2.jpg'
image2 = cv2.imread(path2, 1)

# Convertir las imágenes a escala de grises
gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Crear el objeto SIFT o SURF para la detección y descripción de características
detector = cv2.SIFT_create()
# detector = cv2.xfeatures2d.SURF_create()

# Detectar y describir las características en ambas imágenes
keypoints1, descriptors1 = detector.detectAndCompute(gray_image1, None)
keypoints2, descriptors2 = detector.detectAndCompute(gray_image2, None)

# Crear el objeto BFMatcher para el emparejamiento de características
matcher = cv2.BFMatcher()

# Realizar el emparejamiento de características
matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

# Filtrar los mejores emparejamientos según la relación de distancia
good_matches = []
for m, n in matches:
    if m.distance < 0.8 * n.distance:
        good_matches.append(m)

# Obtener los puntos correspondientes en ambas imágenes
points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Calcular la homografía entre las dos imágenes
homography, _ = cv2.findHomography(points2, points1, cv2.RANSAC)

# Crear la imagen panorámica a partir de la homografía
panorama = cv2.warpPerspective(image2, homography, (image1.shape[1] + image2.shape[1], image1.shape[0]))
panorama[:image1.shape[0], :image1.shape[1]] = image1

# Mostrar la imagen panorámica resultante
cv2.imshow('Imagen panorámica', panorama)
cv2.waitKey(0)
