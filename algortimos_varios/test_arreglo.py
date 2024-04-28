import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt
import time
from PIL import Image
from array import array

list = [2, 23, 2, 5, 4, "benjamin"]
tup = ("benjamin", "nicolas", "Esbe", 3, 5)
diccionario = {"benjami":1, "Esbe": 2, "nombre": "Nicolas"}

print(type(list), list, list[5])
print(type(tup), tup, tup[3])
print(type(diccionario), diccionario, diccionario["nombre"])

a = [2, 4, 5]
b = [1, 3, 6]
c = [7, 2, 9, 5, 6, 7, 3, 2, 4, 8]

list = [a, b]

print(list)

list[1][0:3] = c
frame = np.zeros((2, 5, 10),dtype = np.uint8)
frame2 = np.zeros((5, 10, 2),dtype = np.uint8)
frame3 = np.zeros((5, 10),dtype = np.uint8)

frame[0][0:5][0] = c
# frame[1][0:3][0] = c

# frame3 = [[]]

frame2 = frame

print(list)
print(frame)
print("2")
print(frame2)
print("3")
print(frame3)
print(type(frame))
print(frame.shape)
