import cv2 as cv
import numpy as np
import math
import time

# path = r'C:\Users\benja\github\Algoritmos_vision\video1.mp4'
# video = cv.VideoCapture(path)
video = cv.VideoCapture(0)
p = False
down= False
xi,yi = 0,0
xf,yf = 0,0
board = np.zeros((int(video.get(4)),int(video.get(3)),3),dtype = np.uint8)

def paint(event,x,y,flags,params):
    global down, xi , yi, xf, yf, p
    if event == cv.EVENT_LBUTTONDOWN:
        xi, yi = x, y
        down = True
    elif event == cv.EVENT_MOUSEMOVE and down == True:
        if down == True:
            board[:] = 0
            cv.rectangle(board,(xi,yi),(x,y),(255,0,0),3)
            xf,yf = x,y
    elif event == cv.EVENT_LBUTTONUP:
        down = False
        p = True


while True:
    _, first_frame = video.read()
    first_frame = cv.flip(first_frame, 1)
    cv.namedWindow("Frame")
    cv.setMouseCallback("Frame", paint)
    res = cv.addWeighted(first_frame, 1, board, 1, 0)
    cv.imshow("Frame", res)
    time.sleep(0.05)
    first_key = cv.waitKey(1) & 0xFF
    if first_key == 27 or p == True:
        break


roi = first_frame[yi: yf, xi : xf]
x, y = xi, yi
width = np.int32(math.fabs(xf - xi))
height = np.int32(math.fabs(yf - yi))

hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
roi_hist = cv.calcHist([hsv_roi], [0], None, [180], [0, 180])

cap = cv.VideoCapture(path)
# cap = cv.VideoCapture(0)

term_criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)

while True:
    _, frame = cap.read()
    frame = cv.flip(frame, 1)
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    mask = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
    time.sleep(0.05)
    ret, track_window = cv.CamShift(mask, (x, y, width, height), term_criteria)

    pts = cv.boxPoints(ret)
    pts = np.int0(pts)
    cv.polylines(frame, [pts], True, (255, 0, 0), 2)

    cv.imshow("mask", mask)
    cv.imshow("Frame", frame)

    key = cv.waitKey(1)
    if key == 27:
        break

cap.release()
cv.destroyAllWindows()
