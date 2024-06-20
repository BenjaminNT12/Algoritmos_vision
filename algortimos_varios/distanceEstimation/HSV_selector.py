

# coding: utf-8

# In[3]:


import cv2
import numpy as np
import matplotlib.pyplot as plt

PATH = '/home/nicolas/Videos/VideosPruebasMay21/pruebasMay21.AVI'

def resize_frame(frame, scale_percent):
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

def nothing(x):
    pass

def hsv_calc():
    cap = cv2.VideoCapture(PATH)
    cv2.namedWindow("Trackbars",)
    cv2.createTrackbar("lh","Trackbars",0,179,nothing)
    # cv2.createTrackbar("ls","Trackbars",102,255,nothing)
    cv2.createTrackbar("ls","Trackbars",132,255,nothing)
    # cv2.createTrackbar("lv","Trackbars",240,255,nothing)
    cv2.createTrackbar("lv","Trackbars",235,255,nothing)
    cv2.createTrackbar("uh","Trackbars",179,179,nothing)
    cv2.createTrackbar("us","Trackbars",255,255,nothing)
    cv2.createTrackbar("uv","Trackbars",255,255,nothing)
    while True:
        ret, frame = cap.read()
        frame = resize_frame(frame, 50)
        #frame = cv2.imread('candy.jpg')
        height, width = frame.shape[:2]
        #frame = cv2.resize(frame,(width/5, height/5), interpolation = cv2.INTER_CUBIC)
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

        lh = cv2.getTrackbarPos("lh","Trackbars")
        ls = cv2.getTrackbarPos("ls","Trackbars")
        lv = cv2.getTrackbarPos("lv","Trackbars")
        uh = cv2.getTrackbarPos("uh","Trackbars")
        us = cv2.getTrackbarPos("us","Trackbars")
        uv = cv2.getTrackbarPos("uv","Trackbars")

        l_blue = np.array([lh,ls,lv])
        u_blue = np.array([uh,us,uv])
        mask = cv2.inRange(hsv, l_blue, u_blue)
        result = cv2.bitwise_or(frame,frame,mask=mask)

        cv2.imshow("frame",frame)
        cv2.imshow("mask",mask)
        cv2.imshow("result",result)
        key = cv2.waitKey(30)
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

hsv_calc()
