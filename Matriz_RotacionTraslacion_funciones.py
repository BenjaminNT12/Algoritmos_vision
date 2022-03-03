import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
import time


path = r'C:\Users\benja\Desktop\cuadro.png'

frame = cv.imread(path,1)
dimensiones = frame.shape
w = dimensiones[0]
h = dimensiones[1]



dx = int(0.5*w)
dy = int(0.5*h)

cv.imshow('freno1',frame)


def traslacion(frame,dx,dy):
    dimensiones = frame.shape
    w = dimensiones[0]
    h = dimensiones[1]
    frame2 = np.zeros((w,h,3), dtype=np.uint8)
    for x in range(w):
        for y in range(h):
            if x+dx >= w or y+dy >= h or x+dx < 0 or y+dy < 0 :
                frame2[x,y] = 0
            else:
                frame2[x,y] = frame[x+dx,y+dy]
    cv.imshow('frame',frame2)



traslacion(frame, dx,dy)


def rotacion(frame, theta_g, x0, y0):
    dimensiones = frame.shape
    w = dimensiones[0]
    h = dimensiones[1]
    frame3 = np.zeros((w,h,3), dtype=np.uint8)

    theta = theta_g * math.pi/180
    for x in range(w):
        for y in range(h):
            if int(x*math.cos(theta)-y*math.sin(theta)+(x0*(1-math.cos(theta))+y0*math.sin(theta))) >= w or int(x*math.sin(theta)+y*math.cos(theta)+(y0*(1-math.cos(theta))-x0*math.sin(theta))) >= h:
                frame3[x,y] = 0
            else:
                frame3[x,y] = frame[int(x*math.cos(theta)-y*math.sin(theta)+(x0*(1-math.cos(theta))+y0*math.sin(theta))),int(x*math.sin(theta)+y*math.cos(theta)+(y0*(1-math.cos(theta))-x0*math.sin(theta)))]

    cv.imshow('freno3',frame3)
    # cv.waitKey(1)

rotacion(frame, 45, w/2, h/2)






# frame4 = np.zeros((w,h,3), dtype=np.uint8)
#
# dx = -int(0.5*w)
# dy = -int(0.5*h)
#
# for x in range(w):
#     for y in range(h):
#         if x+dx >= w or y+dy >= h or x+dx < 0 or y+dy < 0 :
#             frame4[x,y] = 0
#         else:
#             frame4[x,y] = frame3[x+dx,y+dy]
# cv.imshow('freno4',frame4)

print("presione una tecla para continuar \n")
cv.waitKey(0)
