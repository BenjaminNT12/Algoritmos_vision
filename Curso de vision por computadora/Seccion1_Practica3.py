#Práctica 3: Filtrado y suavizado de imágenes

#Cargar una imagen y aplicar diferentes filtros para suavizarla, como el filtro de media, el filtro gaussiano y el filtro de mediana.
#Comparar los resultados y evaluar los efectos de cada filtro en la imagen.
#Aplicar técnicas de filtrado para reducir el ruido en imágenes.
import cv2

# Cargar la imagen
path = r'C:\Users\LENOVO\github\Algoritmos_vision\Curso de vision por computadora\imagen.jpg'
image = cv2.imread(path, 1)

# Aplicar el filtro de media
blurred_image = cv2.blur(image, (5, 5))

# Aplicar el filtro gaussiano
gaussian_blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

# Aplicar el filtro de mediana
median_blurred_image = cv2.medianBlur(image, 5)

# Mostrar las imágenes suavizadas
cv2.imshow('Imagen original', image)
cv2.imshow('Imagen suavizada con filtro de media', blurred_image)
cv2.imshow('Imagen suavizada con filtro gaussiano', gaussian_blurred_image)
cv2.imshow('Imagen suavizada con filtro de mediana', median_blurred_image)
cv2.waitKey(0)