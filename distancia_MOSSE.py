import cv2 as cv
import numpy as np
import math
import  matplotlib.pyplot as plt

path = '/home/nicolas/Github/videos/video9.MOV'
# path = 0

tracker = cv.legacy.TrackerMOSSE_create()

start_second = 18
restart_second = 18

ix, iy = 0, 0
xf, yf = 0, 0
selection = False
down = False

h_min, s_min, v_min = 0, 0, 0
h_max, s_max, v_max = 0, 0, 0

lower_color = np.array([h_min, s_min, v_min])
upper_color = np.array([h_max, s_max, v_max])

init_tracking = False

area_threshold = 20

cap = cv.VideoCapture(path)

rectangle_mask = np.zeros((int(cap.get(4)), int(cap.get(3)), 3), dtype=np.uint8)
mask = np.zeros((int(cap.get(4)), int(cap.get(3)), 3), dtype=np.uint8)

# Función para el evento del mouse
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, xf, yf, selection, down
    
    if event == cv.EVENT_LBUTTONDOWN:
        ix, iy = x, y
        down = True
    elif event == cv.EVENT_MOUSEMOVE and down == True:
        if down == True:
            rectangle_mask[:] = 0
            cv.rectangle(rectangle_mask,(ix,iy),(x,y),(0,255,0),1)
            xf,yf = x,y
    elif event == cv.EVENT_LBUTTONUP:
        down = False
        selection = True
    # return color_mask

def get_frame_number(video, second):
    fps = video.get(cv.CAP_PROP_FPS)
    frame_number = int(fps * second)
    return frame_number


def mejorar_imagen(frame):
    lab = cv.cvtColor(frame, cv.COLOR_BGR2LAB)

    l, a, b = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv.merge((l_enhanced, a, b))
    enhanced = cv.cvtColor(lab_enhanced, cv.COLOR_LAB2BGR)

    return enhanced


start_frame = get_frame_number(cap, start_second)
cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)


while True:
    
    ret, frame = cap.read()
    if ret == False: break

    frame = cv.flip(frame, 1)
    # frame = mejorar_imagen(frame)

    if cv.waitKey(1) & 0xFF == ord(' '):
        while True:
            temp_frame = np.zeros((int(cap.get(4)), int(cap.get(3)), 3), dtype=np.uint8)
            
            if selection == False:
                cv.namedWindow('frame')
                cv.setMouseCallback('frame', draw_rectangle)
                cv.addWeighted(frame, 1, rectangle_mask, 1, 0, temp_frame)

            if selection == True and init_tracking == False:
                start_frame = get_frame_number(cap, restart_second)
                init_tracking = True
                cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)
                bbox = np.array([ix, iy, xf - ix, yf - iy])
                tracker.init(frame,bbox)

            cv.imshow('frame', temp_frame)

            if cv.waitKey(1) & 0xFF == ord(' ') or init_tracking == True:
                break

    if init_tracking == True:
        success, bbox = tracker.update(frame)
        x, y, w, h = [int(coord) for coord in bbox]
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)   

    cv.imshow('frame', frame)

    if cv.waitKey(1) & 0xFF == 0x1B:
        break

cap.release()
cv.destroyAllWindows()

























# import cv2
# import numpy as np

# # Inicializar el objeto de seguimiento MOSSE
# tracker = cv2.legacy.TrackerMOSSE_create()

# # Abrir el video de entrada
# video = cv2.VideoCapture('/home/nicolas/Github/Algoritmos_vision/video9.MOV')

# # Leer el primer frame del video
# ret, frame = video.read()

# if not ret:
#     raise ValueError("No se pudo abrir el video")

# # Seleccionar una región de interés (ROI) en el primer frame
# bbox = cv2.selectionROI("Seleccione el objeto a seguir", frame, fromCenter=False, showCrosshair=True)

# # Inicializar el tracker con la ROI seleccionada
# tracker.init(frame, bbox)

# while video.isOpened():
#     ret, frame = video.read()

#     if not ret:
#         break

#     # Actualizar el tracker para obtener la nueva ubicación del objeto
#     success, bbox = tracker.update(frame)

#     # Si el seguimiento fue exitoso, dibujar el cuadro delimitador alrededor del objeto rastreado
#     if success:
#         x, y, w, h = [int(coord) for coord in bbox]
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

#     # Mostrar el frame con el objeto rastreado
#     cv2.imshow("Seguimiento MOSSE", frame)

#     # Salir si se presiona la tecla 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Liberar los recursos
# video.release()
# cv2.destroyAllWindows()