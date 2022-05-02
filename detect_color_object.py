import cv2
import numpy as np

from time import sleep

cap = cv2.VideoCapture(0)

hsv_b_min = (79,114,158)
hsv_b_max = (99,250,226)

hsv_g_min = (54,93,183)
hsv_g_max = (81,255,255)

hsv_r_min = (156,98,157)
hsv_r_max = (176,190,255)

while (cap.isOpened() is True):
    ret, frame = cap.read()

    if ret is False:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BRG2HSV)
    mask_b = cv2.inRange(hsv,hsv_b_min,hsv_b_max)
    mask_g = cv2.inRange(hsv,hsv_g_min,hsv_g_max)
    mask_r = cv2.inRange(hsv,hsv_r_min,hsv_r_max)

    cv2.imshow("frame",mask_b)
    if cv2.waitKey(1) & 0xFF == 27: # tecla scape:
        break
