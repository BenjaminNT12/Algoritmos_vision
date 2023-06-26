import cv2 as cv
import numpy as np

path = '/home/nicolas/Github/Algoritmos_vision/video3.mp4'

start_second = 80

ix, iy = 0, 0
xf, yf = 0, 0
rectangle_complete = False
down = False

h_min, s_min, v_min = 0, 0, 0
h_max, s_max, v_max = 0, 0, 0

lower_color = np.array([h_min, s_min, v_min])
upper_color = np.array([h_max, s_max, v_max])

hsv_range_complete = False

cap = cv.VideoCapture(path)

rectangle_mask = np.zeros((int(cap.get(4)), int(cap.get(3)), 3), dtype=np.uint8)

# Función para el evento del mouse
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, xf, yf, rectangle_complete, down
    
    
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
        rectangle_complete = True

def rgb_to_hsv(frame):
    hsv_frame = cv.cvtColor(frame, cv.COLOR_RGB2HSV)
    return hsv_frame

def split_hsv_channels(frame):
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv_frame)
    return h, s, v

def color_tracking(frame, lower_color, upper_color):
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    color_mask = cv.inRange(hsv_frame, lower_color, upper_color)

    return color_mask

def get_frame_number(video, second):
    fps = video.get(cv.CAP_PROP_FPS)
    frame_number = int(fps * second)
    return frame_number
    
def get_hsv_range(hsv_frame, x_min, y_min, x_max, y_max):
    roi = hsv_frame[y_min:y_max, x_min:x_max, :]
    h_min = np.min(roi[:, :, 0])
    h_max = np.max(roi[:, :, 0])
    s_min = np.min(roi[:, :, 1])
    s_max = np.max(roi[:, :, 1])
    v_min = np.min(roi[:, :, 2])
    v_max = np.max(roi[:, :, 2])
    return [h_min, s_min, v_min], [h_max, s_max, v_max]    



start_frame = get_frame_number(cap, start_second)
cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)


while True:
    
    ret, frame = cap.read()
    if ret == False: break

    frame = cv.flip(frame, 1)
    
    frame_hsv = rgb_to_hsv(frame)

    cv.imshow('frame', frame)

    if cv.waitKey(1) & 0xFF == ord('p'):
        contador = 0
        while True:
            contador += 1
            temp_frame = frame
            print(contador)
            if rectangle_complete == False:
                cv.namedWindow('frame')
                cv.setMouseCallback('frame', draw_rectangle)
                cv.addWeighted(frame, 1, rectangle_mask, 1, 0, temp_frame)

            if rectangle_complete == True and hsv_range_complete == False:
                print('ix: ', ix, 'iy: ', iy, 'xf: ', xf, 'yf: ', yf)
                hsv_min, hsv_max = get_hsv_range(frame_hsv, ix, iy, xf, yf)
                print('hsv_min: ', hsv_min, 'hsv_max: ', hsv_max)
                hsv_range_complete = True

            cv.imshow('temp_frame', temp_frame)
            cv.imshow('frame', frame)

            if cv.waitKey(1) & 0xFF == ord('p'):
                break

    if cv.waitKey(1) & 0xFF == 0x1B:
        break
cap.release()
cv.destroyAllWindows()















































# import cv2 as cv
# import numpy as np

# # Variables globales
# drawing = False
# ix, iy = -1, -1
# selection_complete = False

# hue_min, saturation_min, value_min = 0, 0, 0
# hue_max, saturation_max, value_max = 0, 0, 0

# area_threshold = 100

# # Rango de colores a seguir en formato HSV
# lower_color = np.array([hue_min, saturation_min, value_min])
# upper_color = np.array([hue_max, saturation_max, value_max])


# # Función para el evento del mouse
# def draw_rectangle(event, x, y, flags, param):
#     global ix, iy, drawing, selection_complete

#     if event == cv.EVENT_LBUTTONDOWN:
#         drawing = True
#         ix, iy = x, y

#     elif event == cv.EVENT_LBUTTONUP:
#         drawing = False
#         selection_complete = True

# # Función para obtener el máximo y mínimo de los canales HSV en la región seleccionada
# def get_hsv_range(frame, x_min, y_min, x_max, y_max):
#     hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
#     roi = hsv_frame[y_min:y_max, x_min:x_max, :]
#     h_min = np.min(roi[:, :, 0])
#     h_max = np.max(roi[:, :, 0])
#     s_min = np.min(roi[:, :, 1])
#     s_max = np.max(roi[:, :, 1])
#     v_min = np.min(roi[:, :, 2])
#     v_max = np.max(roi[:, :, 2])
#     return (h_min, s_min, v_min), (h_max, s_max, v_max)

# # Función para realizar el seguimiento de color
# def color_tracking(frame, lower_color, upper_color):
#     # Convertir el frame de BGR a HSV
#     hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

#     # Aplicar una máscara para detectar el color en el rango especificado
#     color_mask = cv.inRange(hsv_frame, lower_color, upper_color)

#     # Aplicar operaciones morfológicas para eliminar ruido
#     color_mask = cv.erode(color_mask, None, iterations=2)
#     color_mask = cv.dilate(color_mask, None, iterations=2)

#     # Encontrar los contornos de los objetos de color
#     contours, _ = cv.findContours(color_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

#     # Realizar el seguimiento de los objetos de color
#     for contour in contours:
#         # Calcular el área del contorno
#         area = cv.contourArea(contour)

#         # Descartar contornos pequeños
#         if area > area_threshold:
#             # Calcular el centroide del contorno
#             M = cv.moments(contour)
#             centroid_x = int(M["m10"] / M["m00"])
#             centroid_y = int(M["m01"] / M["m00"])

#             # Dibujar un círculo en el centroide
#             cv.circle(frame, (centroid_x, centroid_y), 5, (0, 255, 0), -1)

#     return frame

# # Abrir el video de entrada
# video = cv.VideoCapture(0)

# cv.namedWindow('Video')
# cv.setMouseCallback('Video', draw_rectangle)

# while video.isOpened():
#     ret, frame = video.read()

#     if not ret:
#         break

#     frame = cv.flip(frame, 1)

#     # Copiar el frame para dibujar la región seleccionada
#     frame_copy = frame.copy()

#     if drawing and not selection_complete:
#         # Dibujar un rectángulo mientras se arrastra el mouse
#         cv.rectangle(frame, (ix, iy), (cv.CAP_PROP_POS_FRAMES, cv.CAP_PROP_POS_FRAMES), (0, 255, 0), 2)

#     if selection_complete:
#         # Obtener el rango HSV de la región seleccionada
#         hsv_min, hsv_max = get_hsv_range(frame, ix, iy, cv.CAP_PROP_POS_FRAMES, cv.CAP_PROP_POS_FRAMES)
#         print("HSV Min:", hsv_min)
#         print("HSV Max:", hsv_max)
#         selection_complete = False

#     # Realizar el seguimiento de color en el frame
#     tracked_frame = color_tracking(frame, lower_color, upper_color)

#     # Mostrar el frame con el seguimiento de color
#     cv.imshow('Color Tracking', tracked_frame)
#     cv.imshow('Original', frame)

#     # Salir si se presiona la tecla 'q'
#     if cv.waitKey(1) & 0xFF == ord('q'):
#         break

# # Liberar los recursos
# video.release()
# cv.destroyAllWindows()


