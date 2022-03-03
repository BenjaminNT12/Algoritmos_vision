import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt


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
            #cv.rectangle(board,(xi,yi),(x,y),(255,0,0),3)
            #cv.circle(board,(xi,yi),10,(255,0,0),3)
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
    first_key = cv.waitKey(1) & 0xFF
    if first_key == 27 or p == True:
        break


roi = first_frame[yi: yf, xi : xf]
x, y = xi, yi
width = np.int32(math.fabs(xf - xi))
height = np.int32(math.fabs(yf - yi))

hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
cv.imshow('roi', roi)
mask = cv.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
cv.imshow('mask_2', mask)
roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv.normalize(roi_hist,roi_hist,0,255,cv.NORM_MINMAX)
color = ('b','g','r')

for i,col in enumerate(color):
    histr = cv.calcHist([first_frame[yi:yf,xi:xf]],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()

cap = cv.VideoCapture(0)

term_criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)
greenLower = (110, 100, 100)
greenUpper = (123, 100, 32)
while True:
    _, frame = cap.read()
    frame = cv.flip(frame, 1)
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    mask = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
    ret, track_window = cv.CamShift(mask, (x, y, width, height), term_criteria)

    cv.imshow('ret', track_window)
    pts = cv.boxPoints(ret)
    pts = np.int0(pts)
    cv.polylines(frame, [pts], True, (255, 0, 0), 2)

    #centerY = ((pts[0,1]-pts[2,1])/2)+pts[2,1]
    #centerX = ((pts[0,0]-pts[2,0])/2)+pts[2,0]

    #radio = abs(np.int0(pts[0,1]-pts[2,1]))

    #center = (np.int0(centerX),np.int0(centerY))
    #cv.circle(frame, center, radio,(255,0,0),3)

    cv.imshow("mask", mask)
    cv.imshow("Frame", frame)

    #for i,col in enumerate(color):
    #    histr = cv.calcHist([frame[yi:yf,xi:xf]],[i],None,[256],[0,256])
    #    plt.plot(histr,color = col)
    #    plt.xlim([0,256])
    #plt.show()

    key = cv.waitKey(1)
    if key == 27:
        break

cap.release()
cv.destroyAllWindows()
