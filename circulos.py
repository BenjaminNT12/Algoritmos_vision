import cv2
import numpy as np
import math

# Load image, grayscale, Otsu's threshold
path = r'C:\Users\benja\github\Algoritmos_vision\circulos5.png'
image = cv2.imread(path)
# image2 = cv2.imread(path)
# image = cv2.bitwise_not(image)
# image = cv2.imread('1.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
# cv2.imshow('thresh 1', thresh)
# Find circles with HoughCircles
circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, 1, minDist=150, param1=200, param2=18, minRadius=20)

# Draw circles
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x,y,r) in circles:
        cv2.circle(image, (x,y), r, (0,255,0), 25)
        distancia = math.sqrt((x-x)**2+(y-y)**2)

cv2.line(image,(circles[0][0] , circles[0][1]), (circles[1][0] , circles[1][1]), (255,0,0),25)
cv2.line(image,(circles[4][0] , circles[4][1]), (circles[3][0] , circles[3][1]), (255,0,0),25)
cv2.line(image,(circles[2][0] , circles[2][1]), (circles[3][0] , circles[3][1]), (255,0,0),25)
cv2.line(image,(circles[2][0] , circles[2][1]), (circles[0][0] , circles[0][1]), (255,0,0),25)
cv2.line(image,(circles[3][0] , circles[3][1]), (circles[0][0] , circles[0][1]), (255,0,0),25)
cv2.line(image,(circles[2][0] , circles[2][1]), (circles[1][0] , circles[1][1]), (255,0,0),25)
cv2.line(image,(circles[4][0] , circles[4][1]), (circles[1][0] , circles[1][1]), (255,0,0),25)
cv2.line(image,(circles[4][0] , circles[4][1]), (circles[2][0] , circles[2][1]), (255,0,0),25)

print(image.shape)


image = cv2.resize(image, dsize=(int(image.shape[1]/4), int(image.shape[0]/4)), interpolation=cv2.INTER_CUBIC)


cv2.imshow('image', image)

cv2.waitKey()
