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
dx = -int(0.5*w)
dy = -int(0.5*h)
x0 = 0#int(w/2)
y0 = 0#int(h/2)

cv.imshow('freno1',frame)
cv.circle(frame, (x0,y0), 1, (0,0,255), 2)
frame2 = np.zeros((w,h,3), dtype=np.uint8)

theta = 45 * math.pi/180
for x in range(w):
    for y in range(h):
        if (int(x*math.cos(theta)-y*math.sin(theta)+(x0*(1-math.cos(theta))+y0*math.sin(theta)))+int(dx)) >= w or (int(x*math.sin(theta)+y*math.cos(theta)+(y0*(1-math.cos(theta))-x0*math.sin(theta)))+int(dy)) >= h:
            frame2[x,y] = 0
        else:
            frame2[x,y] = frame[(int(x*math.cos(theta)-y*math.sin(theta)+(x0*(1-math.cos(theta))+y0*math.sin(theta)))+int(dx)),(int(x*math.sin(theta)+y*math.cos(theta)+(y0*(1-math.cos(theta))-x0*math.sin(theta)))+int(dy))]

cv.imshow('freno3',frame2)

print("presione una tecla para continuar \n")
cv.waitKey(0)
