import cv2

# Cargar el video
video = cv2.VideoCapture('/home/nicolas/Videos/VideosPruebasApr17/pruebas4.AVI')

while(video.isOpened()):
    # Leer el video frame por frame
    ret, frame = video.read()

    if ret:
        # Mostrar el frame
        cv2.imshow('Video', frame)

        # Salir si se presiona la tecla 'q'
        if cv2.waitKey(16) & 0xFF == ord('q'):
            break
    else:
        break

# Liberar los recursos y cerrar las ventanas
video.release()
cv2.destroyAllWindows()