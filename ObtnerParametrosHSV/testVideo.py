import cv2 as cv
import numpy as np
import math
# import  matplotlib.pyplot as plt
# import time

# # path = '/home/nicolas/Github/videos/video9.MOV'
# path = '/home/nicolas/Github/Algoritmos_vision/Videos/marcadoresAzules.mp4'
path = 2

start_second = 10
restart_second = 10

ix, iy = 0, 0
xf, yf = 0, 0
selection = False
down = False

h_min, s_min, v_min = 88, 33, 214
h_max, s_max, v_max = 99, 172, 255

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
        init_tracking = True

def cut_segment(frame, x_start, y_start, x_end, y_end):
    """Corta un segmento de un frame.

    Args:
        frame (np.array): El frame a cortar.
        x_start (int): La coordenada x del inicio del segmento.
        y_start (int): La coordenada y del inicio del segmento.
        x_end (int): La coordenada x del final del segmento.
        y_end (int): La coordenada y del final del segmento.

    Returns:
        np.array: El segmento cortado del frame.
    """
    return frame[y_start:y_end, x_start:x_end]

def calculate_min_max_hsv(img):
    """Calcula los valores HSV mínimos y máximos de una imagen.

    Args:
        img (np.array): La imagen.

    Returns:
        dict: Un diccionario con los valores mínimos y máximos para cada canal de color.
    """
    # Convertir la imagen a HSV
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # Separar los canales de color
    h, s, v = cv.split(hsv)

    # Calcular los valores mínimos y máximos para cada canal
    min_max_values = {
        "H": {"min": np.min(h), "max": np.max(h)},
        "S": {"min": np.min(s), "max": np.max(s)},
        "V": {"min": np.min(v), "max": np.max(v)}
    }

    return min_max_values


seleccion_roi = False

while True:
    
    ret, frame = cap.read()
    if ret == False: 
        raise ValueError("No se pudo abrir el video")
        break

    frame = cv.flip(frame, 1)

    if cv.waitKey(1) & 0x00FF == ord('p') and selection == False:
        while True:
            temp_frame = np.zeros((int(cap.get(4)), int(cap.get(3)), 3), dtype=np.uint8)
            
            if selection is False:
                cv.namedWindow('frame')
                cv.setMouseCallback('frame', draw_rectangle)
                cv.addWeighted(frame, 1, rectangle_mask, 1, 0, temp_frame)
                # haz un recorte de la imagen original, usando las coordenadas del rectángulo
                cv.imshow('frame', temp_frame)
            if selection is True:
                roi = cut_segment(frame, ix, iy, xf, yf)
                seleccion_roi = True


            if cv.waitKey(1) & 0x00FF == ord('p') or seleccion_roi == True:
                break
    cv.imshow('frame', frame)
    if seleccion_roi is True:
        cv.imshow('frame_seleccionado', roi)
        print('RGB Max and Min values: ', calculate_min_max_hsv(roi))
    

    if cv.waitKey(1) & 0x00FF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
  




















# import cv2 as cv
# import numpy as np
# import math


# path = '/home/nicolas/Github/Algoritmos_vision/video3.mp4'
# video = cv.VideoCapture(path)
# # video = cv.VideoCapture(0)
# p = False
# down= False
# xi,yi = 0,0
# xf,yf = 0,0
# board = np.zeros((int(video.get(4)),int(video.get(3)),3),dtype = np.uint8)

# def paint(event,x,y,flags,params):
#     global down, xi , yi, xf, yf, p
#     if event == cv.EVENT_LBUTTONDOWN:
#         xi, yi = x, y
#         down = True
#     elif event == cv.EVENT_MOUSEMOVE and down == True:
#         if down == True:
#             board[:] = 0
#             cv.rectangle(board,(xi,yi),(x,y),(255,0,0),3)
#             xf,yf = x,y
#     elif event == cv.EVENT_LBUTTONUP:
#         down = False
#         p = True


# while True:
#     _, first_frame = video.read()
#     first_frame = cv.flip(first_frame, 1)
#     cv.namedWindow("Frame")
#     cv.setMouseCallback("Frame", paint)
#     res = cv.addWeighted(first_frame, 1, board, 1, 0)
#     cv.imshow("Frame", res)
#     first_key = cv.waitKey(1) & 0xFF
#     if first_key == 27 or p == True:
#         break


# roi = first_frame[yi: yf, xi : xf]
# x, y = xi, yi
# width = np.int32(math.fabs(xf - xi))
# height = np.int32(math.fabs(yf - yi))

# hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
# roi_hist = cv.calcHist([hsv_roi], [0], None, [180], [0, 180])
# roi_hist = cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)

# term_criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)


# print(x, y)
# print(width, height)

# while True:
#     _, frame = video.read()
#     frame = cv.flip(frame, 1)
#     hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
#     mask = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

#     _, track_window = cv.meanShift(mask, (x, y, width, height), term_criteria)
#     x, y, w, h = track_window
#     cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

#     cv.imshow("Mask", mask)
#     cv.imshow("Frame", frame)

#     key = cv.waitKey(1)
#     if key == 27:
#         break

# video.release()
# cv.destroyAllWindows()







