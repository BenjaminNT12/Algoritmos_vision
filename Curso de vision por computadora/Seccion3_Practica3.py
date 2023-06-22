# Práctica 3: Detección y seguimiento de características en video

# Capturar secuencias de video en tiempo real utilizando la cámara.
# Aplicar técnicas de detección y descripción de características en cada fotograma del video.
# Realizar el seguimiento de las características a lo largo de la secuencia de video utilizando algoritmos como el flujo óptico o el seguimiento basado en características.
# Visualizar los resultados del seguimiento de características en tiempo real.

import cv2

# Crear el objeto VideoCapture para capturar el video
video_capture = cv2.VideoCapture(0)  # Usar 0 para la cámara predeterminada

# Crear el objeto de detección de características (por ejemplo, SIFT o SURF)
detector = cv2.SIFT_create()
# detector = cv2.xfeatures2d.SURF_create()

# Variables para el seguimiento de características
old_gray = None
old_keypoints = None
old_descriptors = None

while True:
    # Leer el fotograma actual del video
    ret, frame = video_capture.read()

    # Convertir el fotograma a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if old_gray is None:
        # Establecer el primer fotograma como referencia
        old_gray = gray
        keypoints = detector.detect(old_gray, None)
        _, old_descriptors = detector.compute(old_gray, keypoints)
    else:
        # Realizar el seguimiento de características en los fotogramas sucesivos
        keypoints, descriptors = detector.detectAndCompute(gray, None)

        # Crear el objeto BFMatcher para el emparejamiento de características
        matcher = cv2.BFMatcher()
        matches = matcher.match(old_descriptors, descriptors)

        # Filtrar los mejores emparejamientos según la relación de distancia
        good_matches = []
        for m in matches:
            if m.distance < 0.75 * m.distance:
                good_matches.append(m)

        # Dibujar los puntos de correspondencia en el fotograma actual
        frame_with_matches = cv2.drawMatches(old_gray, old_keypoints, gray, keypoints, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # Mostrar el fotograma con los puntos de correspondencia
        cv2.imshow('Seguimiento de características', frame_with_matches)

        # Actualizar los fotogramas y descriptores antiguos
        old_gray = gray
        old_keypoints = keypoints
        old_descriptors = descriptors

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos y cerrar las ventanas
video_capture.release()
cv2.destroyAllWindows()
