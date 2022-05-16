import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt
import time



path = r'C:\Users\benja\github\Algoritmos_vision\video3.mp4'
video = cv.VideoCapture(path)

p = False
down= False
xi,yi = 0,0
xf,yf = 0,0
board = np.zeros((int(video.get(4)),int(video.get(3)),3),dtype = np.uint8)
frame2 = np.zeros((int(video.get(4)),int(video.get(3)),3),dtype = np.uint8)
frame3 = np.zeros((int(video.get(4)),int(video.get(3)),3),dtype = np.uint8)
color = ('b','g','r')

tiempo_anterior = 0.0
periodo = 0.0
frecuencia = 0.0
################################################################################

def DarkChannel(im,sz):
    b,g,r = cv.split(im)
    dc = cv.min(cv.min(r,g),b);
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(sz,sz))
    dark = cv.erode(dc,kernel)
    return dark

def AtmLight(im,dark):
    [h,w] = im.shape[:2]
    imsz = h*w
    numpx = int(max(math.floor(imsz/1000),1))
    darkvec = dark.reshape(imsz);
    imvec = im.reshape(imsz,3);

    indices = darkvec.argsort();
    indices = indices[imsz-numpx::]

    atmsum = np.zeros([1,3])
    for ind in range(1,numpx):
       atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx;
    return A

def TransmissionEstimate(im,A,sz):
    omega = 0.95;
    im3 = np.empty(im.shape,im.dtype);

    for ind in range(0,3):
        im3[:,:,ind] = im[:,:,ind]/A[0,ind]

    transmission = 1 - omega*DarkChannel(im3,sz);
    return transmission

def Guidedfilter(im,p,r,eps):
    mean_I = cv.boxFilter(im, cv.CV_64F,(r,r));
    mean_p = cv.boxFilter(p, cv.CV_64F,(r,r));
    mean_Ip = cv.boxFilter(im*p, cv.CV_64F,(r,r));
    cov_Ip = mean_Ip - mean_I*mean_p;

    mean_II = cv.boxFilter(im*im, cv.CV_64F,(r,r));
    var_I   = mean_II - mean_I*mean_I;

    a = cov_Ip/(var_I + eps);
    b = mean_p - a*mean_I;

    mean_a = cv.boxFilter(a, cv.CV_64F,(r,r));
    mean_b = cv.boxFilter(b, cv.CV_64F,(r,r));

    q = mean_a*im + mean_b;
    return q;

def TransmissionRefine(im,et):
    gray = cv.cvtColor(im,cv.COLOR_BGR2GRAY);
    gray = np.float64(gray)/255;
    r = 60;
    eps = 0.0001;
    t = Guidedfilter(gray,et,r,eps);

    return t;

def Recover(im,t,A,tx = 0.1):
    res = np.empty(im.shape,im.dtype);
    t = cv.max(t,tx);

    for ind in range(0,3):
        res[:,:,ind] = (im[:,:,ind]-A[0,ind])/t + A[0,ind]

    return res
################################################################################
#
#
# def paint(event,x,y,flags,params):
#     global down, xi , yi, xf, yf, p
#     if event == cv.EVENT_LBUTTONDOWN:
#         xi, yi = x, y
#         down = True
#     elif event == cv.EVENT_MOUSEMOVE and down == True:
#         if down == True:
#             board[:] = 0
#             cv.rectangle(board,(xi,yi),(x,y),(0,0,255),3)
#             xf,yf = x,y
#     elif event == cv.EVENT_LBUTTONUP:
#         down = False
#         p = True
#
#
# while True:
#     _, frame = video.read()
#     # time.sleep(1/12)
# ################################################################################
#     I = frame.astype('float64')/255;
#
#     dark = DarkChannel(I,30);
#     A = AtmLight(I,dark);
#     te = TransmissionEstimate(I,A,30);
#     t = TransmissionRefine(frame,te);
#     frame = Recover(I,t,A,0.1);
#     frame = frame.astype('uint8')*255
#     # print(type(J), type(frame), J.shape, frame.shape, type(J[2][1][0]), type(frame[2][1][0]))
#
#     # cv.imshow('J',J);
# ################################################################################
#     # first_frame = cv.flip(first_frame, 1)
#     cv.namedWindow("Frame")
#     cv.setMouseCallback("Frame", paint)
#     res = cv.addWeighted(frame, 1, board, 1, 0)
#     cv.imshow("Frame", res)
#     first_key = cv.waitKey(1) & 0xFF
#     if first_key == 27 or p == True:
#         break
#
#
# roi = frame[yi: yf, xi : xf]
# x, y = xi, yi
# width = np.int32(math.fabs(xf - xi))
# height = np.int32(math.fabs(yf - yi))
#
# hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
# roi_hist = cv.calcHist([hsv_roi], [0], None, [180], [0, 180])

################################################################################

while True:
    _, frame = video.read()
    # frame = cv.flip(frame, 1)
    # hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # mask = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    # cv.imshow("mascara",mask)

    periodo = time.time() - tiempo_anterior
    frecuencia = 1/periodo
    print('Periodo: %5.2f, Frecuencia %5.2f'%(periodo, frecuencia))
    tiempo_anterior = time.time()
    # cv.normalize(frame,frame2,0,255,cv.NORM_MINMAX)
# ###############################################################################
#     # convert from RGB color-space to YCrCb
#     ycrcb_img = cv.cvtColor(frame, cv.COLOR_BGR2YCrCb)
#     # equalize the histogram of the Y channel
#     ycrcb_img[:, :, 0] = cv.equalizeHist(ycrcb_img[:, :, 0])
#     # convert back to RGB color-space from YCrCb
#     frame3 = cv.cvtColor(ycrcb_img, cv.COLOR_YCrCb2BGR)
###############################################################################
    # for i, c in enumerate(color):
    #     hist = cv.calcHist([frame], [i], None, [256], [0, 256])
    #     plt.plot(hist, color = c)
    #     plt.xlim([0,256])
    #     # cv.imshow("Frame2", frame2)
    # plt.show()
################################################################################
    # for i, c in enumerate(color):
    #     hist2 = cv.calcHist([frame3], [i], None, [256], [0, 256])
    #     plt.plot(hist2, color = c)
    #     plt.xlim([0,256])
    #     # cv.imshow("Frame2", frame2)
    # plt.show()
################################################################################
    I = frame.astype('float64')/255;

    dark = DarkChannel(I,30);
    A = AtmLight(I,dark);
    te = TransmissionEstimate(I,A,30);
    t = TransmissionRefine(frame,te);
    J = Recover(I,t,A,0.1);
    J2 = J.astype('uint8')*255
    cv.imshow("dark",dark);
    cv.imshow("t",t);
    # cv.imshow('I',frame2);
    cv.imshow('J',J);
    cv.imshow('J2',J2);
    # cv2.imwrite("J.png",J*255);
    # cv2.waitKey();
################################################################################
    cv.imshow("Frame", frame)
    # cv.imshow("Frame2", frame2)
    # cv.imshow("Frame3", frame3)

    key = cv.waitKey(1)
    if key == 27:
        break

video.release()
cv.destroyAllWindows()
