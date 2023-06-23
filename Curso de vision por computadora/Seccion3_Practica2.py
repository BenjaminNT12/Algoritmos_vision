#Práctica 2: Descriptores locales

#Cargar dos imágenes y extraer los descriptores locales utilizando SIFT (Scale-Invariant Feature Transform) o SURF (Speeded-Up Robust Features).
#Comparar los descriptores de las dos imágenes utilizando el algoritmo de emparejamiento de características, como el algoritmo de fuerza bruta o el algoritmo de búsqueda de vecinos más cercanos.
#Visualizar los puntos de correspondencia entre las dos imágenes.

import cv2

# Cargar las dos imágenes
path1 = r'C:\Users\LENOVO\github\Algoritmos_vision\Curso de vision por computadora\imagen1.jpg'
image1 = cv2.imread(path1, 1)
path2 = r'C:\Users\LENOVO\github\Algoritmos_vision\Curso de vision por computadora\imagen2.jpg'
image2 = cv2.imread(path2, 1)

# Crear los objetos SIFT o SURF para la detección y descripción de características
detector = cv2.SIFT_create()
# detector = cv2.xfeatures2d.SURF_create()

# Detectar y describir las características en ambas imágenes
keypoints1, descriptors1 = detector.detectAndCompute(image1, None)
keypoints2, descriptors2 = detector.detectAndCompute(image2, None)

# Crear el objeto BFMatcher para el emparejamiento de características
matcher = cv2.BFMatcher()

# Realizar el emparejamiento de características
matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

# Filtrar los mejores emparejamientos según la relación de distancia
good_matches = []
for m, n in matches:
    if m.distance < 0.9*n.distance:
        good_matches.append(m)

# Dibujar los puntos de correspondencia en las imágenes
result = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Mostrar los puntos de correspondencia entre las imágenes
cv2.imshow('Emparejamiento de características', result)
cv2.waitKey(0)
