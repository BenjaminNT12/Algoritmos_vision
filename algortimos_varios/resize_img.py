import cv2

# Carga tu imagen
img = cv2.imread('/home/nicolas/Downloads/esbe.jpg')

# Redimensiona la imagen a 640x480
resized_img = cv2.resize(img, (480, 640))

# Guarda la imagen redimensionada
cv2.imwrite('/home/nicolas/Downloads/esbe_resized.jpg', resized_img)

# Muestra la imagen redimensionada
cv2.imshow('Resized Image', resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()