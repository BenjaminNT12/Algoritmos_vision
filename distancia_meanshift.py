import cv2 as cv
import numpy as np
import math
import  matplotlib.pyplot as plt

path = '/home/nicolas/Github/Algoritmos_vision/video3.mp4'
# path = 0

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
    hsv_frame = cv.cvtColor(frame, cv.COLOR_RGB2HSV)
    color_mask = cv.inRange(hsv_frame, lower_color, upper_color)

    color_mask = cv.erode(color_mask, None, iterations = 2)
    color_mask = cv.dilate(color_mask, None, iterations = 2)

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

            # Dibujar un círculo en el centroide
            cv.circle(frame, (centroid_x, centroid_y), 5, (0, 255, 0), -1)

    return frame

    # return color_mask

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

def get_meanshift_hist(frame, x_min, y_min, x_max, y_max):
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    roi = hsv_frame[y_min:y_max, x_min:x_max, :]
    roi_hist = cv.calcHist([roi], [0], None, [180], [0, 180])
    roi_hist = cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)

    return roi_hist

def meanShiftTacker(frame, roi_h, xi, yi, xf, yf):
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    x, y = xi, yi
    width = np.int32(math.fabs(xf - xi))
    height = np.int32(math.fabs(yf - yi))

    term_criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)
    mask = cv.calcBackProject([hsv_frame], [0], roi_h, [0, 180], 1)
    _, track_window = cv.meanShift(mask, (x, y, width, height), term_criteria)

    return mask, track_window


start_frame = get_frame_number(cap, start_second)
cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)


while True:
    
    ret, frame = cap.read()
    if ret == False: break

    frame = cv.flip(frame, 1)
    frame_enhaced = mejorar_imagen(frame)

    if cv.waitKey(1) & 0xFF == ord(' '):
        while True:
            temp_frame = np.zeros((int(cap.get(4)), int(cap.get(3)), 3), dtype=np.uint8)
            
            if rectangle_complete == False:
                cv.namedWindow('frame')
                cv.setMouseCallback('frame', draw_rectangle)
                cv.addWeighted(frame_enhaced, 1, rectangle_mask, 1, 0, temp_frame)

            if rectangle_complete == True and hsv_range_complete == False:
                get_hsv_range(frame_enhaced, ix, iy, xf, yf)
                roi_hist_mean = get_meanshift_hist(frame_enhaced, ix, iy, xf, yf)
                hsv_range_complete = True
                start_second = 0
                start_frame = get_frame_number(cap, start_second)
                cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)

            cv.imshow('frame', temp_frame)

            if cv.waitKey(1) & 0xFF == ord(' ') or hsv_range_complete == True:
                break

    if hsv_range_complete == True:
        mask = color_tracking(frame_enhaced, lower_color, upper_color)
        _, track_window = meanShiftTacker(frame_enhaced, roi_hist_mean, ix, iy, xf, yf)
        x, y, w, h = track_window
        cv.rectangle(frame_enhaced, (x, y), (x + w, y + h), (0, 0, 255), 1)
        # cv.imshow('mask', mask)
        # cv.imshow('mask_mean', maskMean)
    
    cv.imshow('frame', frame_enhaced)
    
    # time.sleep(0.02)

    if cv.waitKey(1) & 0xFF == 0x1B:
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
