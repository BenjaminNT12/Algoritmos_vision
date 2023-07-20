import numpy as np
import cv2 as cv
import glob
import os

path = "/home/nicolas/Github/Algoritmos_vision/fotosCalibracion4"
# termination criterio
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# preparar puntos de objeto, como (0,0,0,0), (1,0,0,0), (2,0,0,0)...., (6,5,0)
objp = np.zeros((4*4, 3), np.float32)
objp[:, :2] = np.mgrid[0:4, 0:4].T.reshape(-1, 2)

# Arrays para almacenar puntos de objeto y puntos de imagen de todas las imágenes.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

images = glob.glob(os.path.join(path, '*.jpg'))

def enhanced_contrast(frame):
    # Convertir a escala de grises
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Aumentar el contraste
    claheFilter = cv.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    frame = claheFilter.apply(frame)
    # frame = cv.equalizeHist(frame)
    return frame

# cap = cv.VideoCapture(2)

# while True:
#     _, img = cap.read()


count = 0
count2 = 0

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # gray = enhanced_contrast(img)
    # cv.imshow('gray', gray)
    # cv.imshow('gray_enhanced', gray)
    # cv.waitKey(10)
    ret, corners = cv.findChessboardCorners(gray, (4, 4), None)
    # Si se encuentran, añada puntos de objeto, puntos de imagen (después de refinarlos)fname
    print(ret)
    if count == 10:
        break
    if ret == True:
        count += 1
        print(count)
        objpoints.append(objp)
        
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Dibuja y muestra las esquinas
        img = cv.drawChessboardCorners(img, (4, 4), corners2, ret)
        # cv.imshow('img', img)
        # cv.waitKey(10)

cv.destroyAllWindows() 


ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
# print("ret: ", ret, "mtx :", mtx, "dist", dist, "rvecs: ", rvecs, "tvecs: ", tvecs)


img = cv.imread(os.path.join(path,'calibracion27.jpg'))
h, w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w,h))

# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibresult.png', dst)
cv.imshow('calibresult 1', dst)

# undistort
mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imshow('calibresult 2', dst)
# cv.imwrite('calibresult.png', dst)
cv.waitKey(0)