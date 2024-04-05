import cv2

# Inicializa la cámara
cap = cv2.VideoCapture(2)

# Comprueba si la cámara se abrió correctamente
if not cap.isOpened():
    print("No se pudo abrir la cámara")
else:
    # Intenta modificar el tiempo de exposición
    # Nota: El valor de exposición puede variar dependiendo de la cámara y el controlador
    cap.set(cv2.CAP_PROP_EXPOSURE, -8)

    while True:
        # Captura cuadro por cuadro
        ret, frame = cap.read()
        # reduce el tamaño de la imagen
        frame = cv2.resize(frame, (640, 480))

        # Muestra el cuadro resultante
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cuando todo está hecho, libera la captura
    cap.release()
    cv2.destroyAllWindows()
