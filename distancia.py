import cv2 as cv
import numpy as np
import time

# path = '/home/nicolas/Github/Algoritmos_vision/video3.mp4'
# lower_color [135   2 254] upper_color [158   4 255]
path = 0

name = False
count = [0]

coordenadas = []

start_second = 80
restart_second = 80

ix, iy = 0, 0
xf, yf = 0, 0
rectangle_complete = False
down = False

h_min, s_min, v_min = 135, 2, 254
h_max, s_max, v_max = 158, 4, 255

lower_color = np.array([h_min, s_min, v_min])
upper_color = np.array([h_max, s_max, v_max])

hsv_range_complete = False

area_threshold = 20

cap = cv.VideoCapture(path)

rectangle_mask = np.zeros((int(cap.get(4)), int(cap.get(3)), 3), dtype=np.uint8)
mask = np.zeros((int(cap.get(4)), int(cap.get(3)), 3), dtype=np.uint8)

# Función para el evento del mouse
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, xf, yf, rectangle_complete, down
    
    if event == cv.EVENT_LBUTTONDOWN:
        ix, iy = x, y
        down = True
    elif event == cv.EVENT_MOUSEMOVE and down == True:
        if down == True:
            rectangle_mask[:] = 0
            cv.rectangle(rectangle_mask,(ix,iy),(x,y),(0,0,255),1)
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

def dentro_de_area(coordenadas, punto_referencia, radio):
    # Calcula la distancia euclidiana entre cada coordenada y el punto de referencia
    distancias = np.linalg.norm(coordenadas - punto_referencia, axis=1)

    # Verifica si la distancia es menor que el radio
    coordenadas_dentro_area = coordenadas[distancias < radio]

    return coordenadas_dentro_area

def color_tracking(frame, lower_color, upper_color):
    global name
    id = 0
    
    hsv_frame = cv.cvtColor(frame, cv.COLOR_RGB2HSV)
    # cv.imshow('hsv_frame', hsv_frame)
    color_mask = cv.inRange(hsv_frame, lower_color, upper_color)
    # cv.imshow('color_mask', color_mask)

    color_mask = cv.erode(color_mask, None, iterations = 2)
    color_mask = cv.dilate(color_mask, None, iterations = 2)

    # print("lower_color",lower_color, "upper_color", upper_color)
    # imprimir la dimension de la mascara
    # print(color_mask.shape)

    # Encontrar los contornos de los objetos de color
    contours, _ = cv.findContours(color_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Realizar el seguimiento de los objetos de color
    for contour in contours:
        # Calcular el área del contorno
        area = cv.contourArea(contour)

        # Descartar contornos pequeños
        if area > area_threshold:
            # Calcular el centroide del contorno
            M = cv.moments(contour)
            centroid_x = int(M["m10"] / M["m00"])
            centroid_y = int(M["m01"] / M["m00"])
            
            
            if name == False:
                coordenadas.append([centroid_x, centroid_y])
            else:
                coordenadas[id] = dentro_de_area(coordenadas, coordenadas[id], 25)  
                print("coordenadas[id]: ",coordenadas[id])  
            id += 1    
            
                # coordenadas[id] = [centroid_x, centroid_y]
            
            cv.putText(frame, str(id), (centroid_x - 25, centroid_y - 25), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv.circle(frame, (centroid_x, centroid_y), 12, (0, 255, 0), -1)
    print("coordenadas", coordenadas,"coordenadas 1", coordenadas[1])
    name = True

    return color_mask


def get_frame_number(video, second):
    fps = video.get(cv.CAP_PROP_FPS)
    frame_number = int(fps * second)
    return frame_number
    
def get_hsv_range(frame, x_min, y_min, x_max, y_max):
    global lower_color, upper_color

    hsv_frame = cv.cvtColor(frame, cv.COLOR_RGB2HSV)
    roi = hsv_frame[y_min:y_max, x_min:x_max, :]
    
    h_min = np.min(roi[:, :, 0])
    h_max = np.max(roi[:, :, 0])
    s_min = np.min(roi[:, :, 1])
    s_max = np.max(roi[:, :, 1])
    v_min = np.min(roi[:, :, 2])
    v_max = np.max(roi[:, :, 2])

    lower_color = np.array([h_min, s_min, v_min])
    upper_color = np.array([h_max, s_max, v_max]) 

    return lower_color, upper_color


def mejorar_imagen(frame):
    lab = cv.cvtColor(frame, cv.COLOR_BGR2LAB)

    l, a, b = cv.split(lab)

    # Aplicar el ajuste de contraste en el canal L utilizando CLAHE
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)

    # Fusionar los canales LAB mejorados
    lab_enhanced = cv.merge((l_enhanced, a, b))

    # Convertir la imagen de vuelta a BGR
    enhanced = cv.cvtColor(lab_enhanced, cv.COLOR_LAB2BGR)

    return enhanced

start_frame = get_frame_number(cap, start_second)
cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)


while True:
    
    ret, frame = cap.read()
    if ret == False: break

    frame = cv.flip(frame, 1)
    # frame = mejorar_imagen(frame)

    # if cv.waitKey(1) & 0xFF == ord(' '):
    #     while True:
    #         temp_frame = np.zeros((int(cap.get(4)), int(cap.get(3)), 3), dtype=np.uint8)
            
    #         if rectangle_complete == False:
    #             cv.namedWindow('frame')
    #             cv.setMouseCallback('frame', draw_rectangle)
    #             cv.addWeighted(frame, 1, rectangle_mask, 1, 0, temp_frame)

    #         if rectangle_complete == True and hsv_range_complete == False:
    #             get_hsv_range(frame, ix, iy, xf, yf)
    #             hsv_range_complete = True
    #             start_frame = get_frame_number(cap, restart_second)
    #             cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)

    #         cv.imshow('frame', temp_frame)

    #         if cv.waitKey(1) & 0xFF == ord(' ') or hsv_range_complete == True:
    #             break

    if hsv_range_complete == True or cv.waitKey(1) & 0xFF == ord('a'):
        hsv_range_complete = True
        mask = color_tracking(frame, lower_color, upper_color)
        # cv.imshow('mask', mask)
    
    cv.imshow('frame', frame)
    
    # time.sleep(0.02)

    if cv.waitKey(1) & 0xFF == 0x1B:
        break
cap.release()
cv.destroyAllWindows()


