import cv2
import numpy as np

# path = r'C:\Users\benja\github\Algoritmos_vision\video1.mp4'
cap = cv2.VideoCapture(2)

while(1):

    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of white color in HSV
    # change it according to your need !
    lower_white = np.array([0,0,0], dtype=np.uint8)
    upper_white = np.array([0,0,255], dtype=np.uint8)

    # Threshold the HSV image to get only white colors
    mask = cv2.inRange(hsv, lower_white, upper_white)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)

    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()

#
# import numpy as np
# import cv2
#
#
#
#
#
#
# path = r'C:\Users\benja\github\Algoritmos_vision\circulos2.png'
# imageFrame = cv2.imread(path)
# # _, imageFrame = webcam.read()
#
#
#
#
#
# hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)
#
#
# #     lower_white = np.array([0,0,0], dtype=np.uint8)
# #     upper_white = np.array([0,0,255], dtype=np.uint8)
# red_lower = np.array([0,0,0], dtype=np.uint8)
# red_upper = np.array([0,0,255], dtype=np.uint8)
# red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)
#
#
#
# green_lower = np.array([0,0,0], dtype=np.uint8)
# green_upper = np.array([0,0,255], dtype=np.uint8)
# green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)
#
#
#
# blue_lower = np.array([0,0,0], dtype=np.uint8)
# blue_upper = np.array([0,0,255], dtype=np.uint8)
# blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)
#
#
#
#
#
# kernal = np.ones((5, 5), "uint8")
#
#
# red_mask = cv2.dilate(red_mask, kernal)
# res_red = cv2.bitwise_and(imageFrame, imageFrame,
#                           mask = red_mask)
#
#
# green_mask = cv2.dilate(green_mask, kernal)
# res_green = cv2.bitwise_and(imageFrame, imageFrame,
#                             mask = green_mask)
#
#
# blue_mask = cv2.dilate(blue_mask, kernal)
# res_blue = cv2.bitwise_and(imageFrame, imageFrame,
#                            mask = blue_mask)
#
#
# contours, hierarchy = cv2.findContours(red_mask,
#                                        cv2.RETR_TREE,
#                                        cv2.CHAIN_APPROX_SIMPLE)
#
# for pic, contour in enumerate(contours):
#     area = cv2.contourArea(contour)
#     if(area > 300):
#         x, y, w, h = cv2.boundingRect(contour)
#         imageFrame = cv2.rectangle(imageFrame,(x, y),
#                                    (x + w, y + h),
#                                    (0, 0, 255), 2)
#
#         cv2.putText(imageFrame, "Red Colour",(x, y),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1.0,
#                     (0, 0, 255))
#
#
# contours, hierarchy = cv2.findContours(green_mask,
#                                        cv2.RETR_TREE,
#                                        cv2.CHAIN_APPROX_SIMPLE)
#
# for pic, contour in enumerate(contours):
#     area = cv2.contourArea(contour)
#     if(area > 300):
#         x, y, w, h = cv2.boundingRect(contour)
#         imageFrame = cv2.rectangle(imageFrame,(x, y),
#                                    (x + w, y + h),
#                                    (0, 255, 0), 2)
#
#         cv2.putText(imageFrame, "Green Colour",(x, y),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     1.0,(0, 255, 0))
#
#
# contours, hierarchy = cv2.findContours(blue_mask,
#                                        cv2.RETR_TREE,
#                                        cv2.CHAIN_APPROX_SIMPLE)
# for pic, contour in enumerate(contours):
#     area = cv2.contourArea(contour)
#     if(area > 300):
#         x, y, w, h = cv2.boundingRect(contour)
#         imageFrame = cv2.rectangle(imageFrame,(x, y),
#                                    (x + w, y + h),
#                                    (255, 0, 0), 2)
#
#         cv2.putText(imageFrame, "Blue Colour",(x, y),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     1.0,(255, 0, 0))
#
#
# cv2.imshow("Multiple Color Detection in Real-TIme", imageFrame)
#
# cv2.waitKey(0)
# cap.release()
# cv2.destroyAllWindows()
