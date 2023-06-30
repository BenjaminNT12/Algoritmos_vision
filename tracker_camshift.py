import cv2

# Inicializar el objeto de seguimiento KCF
tracker = cv2.TrackerKCF_create()

# Abrir el video de entrada
video = cv2.VideoCapture('/home/nicolas/Github/videos/video9.MOV')
start_second = 10

def get_frame_number(video, second):
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_number = int(fps * second)
    return frame_number

start_frame = get_frame_number(video, start_second)
video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

# Leer el primer frame del video
ret, frame = video.read()

if not ret:
    raise ValueError("No se pudo abrir el video")

# Seleccionar una región de interés (ROI) en el primer frame
bbox = cv2.selectROI("Seleccione el objeto a seguir", frame, fromCenter=False, showCrosshair=True)

# Inicializar el tracker con la ROI seleccionada
tracker.init(frame, bbox)

while video.isOpened():
    ret, frame = video.read()

    if not ret:
        break

    # Actualizar el tracker para obtener la nueva ubicación del objeto
    success, bbox = tracker.update(frame)

    # Si el seguimiento fue exitoso, dibujar el cuadro delimitador alrededor del objeto rastreado
    if success:
        x, y, w, h = [int(coord) for coord in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Mostrar el frame con el objeto rastreado
    cv2.imshow("Seguimiento KCF", frame)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Actualizar la región de interés (ROI) si el usuario presiona la tecla 'r'
    if cv2.waitKey(1) & 0xFF == ord('r'):
        bbox = cv2.selectROI("Seleccione el objeto a seguir", frame, fromCenter=False, showCrosshair=True)
        tracker.init(frame, bbox)

# Liberar los recursos
video.release()
cv2.destroyAllWindows()
