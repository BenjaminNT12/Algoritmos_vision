import cv2

# Inicializar el objeto de seguimiento MeanShift
tracker = cv2.TrackerMeanShift_create()

# Abrir el video de entrada
video = cv2.VideoCapture('ruta_del_video.mp4')

# Leer el primer frame del video
ret, frame = video.read()

if not ret:
    raise ValueError("No se pudo abrir el video")

# Seleccionar una región de interés (ROI) en el primer frame
bbox = cv2.selectROI("Seleccione el objeto a seguir", frame, fromCenter=False, showCrosshair=True)

# Inicializar el tracker con la ROI
tracker.init(frame, bbox)

while video.isOpened():
    ret, frame = video.read()

    if not ret:
        break

    # Actualizar el tracker para obtener la nueva ubicación del objeto
    success, bbox = tracker.update(frame)

    # Dibujar el cuadro delimitador alrededor del objeto rastreado
    if success:
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Mostrar el frame con el objeto rastreado
    cv2.imshow("Seguimiento MeanShift", frame)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos
video.release()
cv2.destroyAllWindows()
