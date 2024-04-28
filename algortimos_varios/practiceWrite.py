import math 
import numpy as np
import cv2 as cv

PATH = '/home/alex/Downloads/opencv/samples/data/'

PTS_COMPLETE = False
COUNT = 0 
SET_ROTATION = 0.0

coordenadas = []
new_coord = np.array([])

START_SECOND = 80
REESTART_SECOND = 80

ix, iy = 0, 0
xf, yf = 0, 0 

RECTANGLE_COMPLETE = False
DOWN = False

h_min, s_min, v_min = 135, 2, 254
h_max, s_max, v_max = 158, 4, 255

lower_color = np.array([h_min, s_min, v_min])
uppper_color = np.array([h_max, s_max, v_max])

HSV_RANGE_COMPLETE = False

AREA_THRESHOLD = 50

cap = cv.VideoCapture(PATH)

rectangle_mask = np.zeros((int(cap.get(4)), int(cap.get(3)), 3), dtype=np.uint8)
mask = np.zeros((int(cap.get(4)), int(cap.get(3)), 3), dtype=np.uint8)

ancho_imagen = int(cap.get(4))
alto_imagen = int(cap.get(3))

K1 = 0.1148
K2 = 0.1148
P1 = 0.1148
K3 = 0.1148


FX, FY = ancho_imagen, alto_imagen
CX, CY = int(ancho_imagen/2),int(alto_imagen/2)

cameraMatriz = np.array([[190.0, 190.0, 0.0],
                         [0.0, 190.0, 0.0],
                         [95.0, 45.0, 0.0],
                         [190.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0]], dtype=np.float64)

def plot_points(frames_to_draw, points):
    for i in range(len(points)):
        cv.circle(frames_to_draw,(int(points[i][0]), int(points[i][1])), 5, (0, 0, 0), 3)
        cv.putText(frames_to_draw, "hola", (int(points[i][0]), int(points[i][1])), cv.FONT_HERSHEY_DUPLEX, 0.5, (0,0,255),2 )


def estimar_pose_3D(objPoints, imgPoints, camMatrix, distorCoeffs):
    success, rvec, tvec = cv.solvePnP(objPoints,imgPoints,camMatrix,distorCoeffs,flags=cv.SOLVEPNP_ITERATIVE)
    
    return success, rvec, tvec


def dentro_del_area():