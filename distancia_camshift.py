import cv2 as cv
import numpy as np
import math
import  matplotlib.pyplot as plt

path = '/home/nicolas/Github/Algoritmos_vision/video3.mp4'
# path = 0

start_second = 80

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

def get_frame_number(video, second):
    fps = video.get(cv.CAP_PROP_FPS)
    frame_number = int(fps * second)
    return frame_number


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
    _, track_window = cv.CamShift(mask, (x, y, width, height), term_criteria)

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
        ret, track_window = meanShiftTacker(frame_enhaced, roi_hist_mean, ix, iy, xf, yf)
        print("ret: ",ret)
        print("track_window: ",track_window)
        x, y, w, h = track_window
        cv.rectangle(frame_enhaced, (x, y), (x + w, y + h), (0, 0, 255), 1)
        pts = cv.boxPoints(ret)
        pts = np.int0(pts)
        cv.polylines(frame_enhaced, [pts], True, (0, 255, 0), 1)
        # cv.imshow('mask', mask)
        # cv.imshow('mask_mean', maskMean)
    
    cv.imshow('frame', frame_enhaced)
    
    # time.sleep(0.02)

    if cv.waitKey(1) & 0xFF == 0x1B:
        break

cap.release()
cv.destroyAllWindows()
  






