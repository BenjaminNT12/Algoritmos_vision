import cv2
import numpy as np

from time import sleep

# cap = cv2.VideoCapture(0)

hsv_b_min = (int(194/2),68,83)
hsv_b_max = (int(241/2),255,255)

hsv_g_min = (54,93,183)
hsv_g_max = (81,255,255)

hsv_r_min = (0,100,100)
hsv_r_max = (0,255,255)
print(hsv_r_min)
print(hsv_r_max)

path = "/home/cinvestav/Descargas/colore-rgb.jpg"

frame = cv2.imread(path)
# frame = cv2.flip(frame, 1)

# cv2.imshow("frame", frame)
# cv2.waitKey(0)

hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask_b = cv2.inRange(hsv,hsv_b_min,hsv_b_max)
mask_g = cv2.inRange(hsv,hsv_g_min,hsv_g_max)
mask_r = cv2.inRange(hsv,hsv_r_min,hsv_r_max)
#
cv2.imshow("frame b",mask_b)
cv2.imshow("frame r",mask_r)
cv2.imshow("frame g",mask_g)
cv2.imshow("frame",frame)
cv2.waitKey(0)
# if cv2.waitKey(1) & 0xFF == 27: # tecla scape:
    # break

# while (cap.isOpened() is True):
#     ret, frame = cap.read()
#     frame = cv2.flip(frame,1)
#
#     if ret is False:
#         break
#
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     mask_b = cv2.inRange(hsv,hsv_b_min,hsv_b_max)
#     mask_g = cv2.inRange(hsv,hsv_g_min,hsv_g_max)
#     mask_r = cv2.inRange(hsv,hsv_r_min,hsv_r_max)
#
#     cv2.imshow("frame b",mask_b)
#     cv2.imshow("frame r",mask_r)
#     cv2.imshow("frame g",mask_g)
#     cv2.imshow("frame2",frame)
#     if cv2.waitKey(1) & 0xFF == 27: # tecla scape:
#         break
