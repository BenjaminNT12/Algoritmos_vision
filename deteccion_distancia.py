import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt
import time

path = r'C:\Users\benja\github\Algoritmos_vision\video3.mp4'
video = cv.VideoCapture(path)

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

frame2 = np.zeros((int(video.get(4)),int(video.get(3)),3),dtype = np.uint8)
frame3 = np.zeros((int(video.get(4)),int(video.get(3)),3),dtype = np.uint8)
color = ('b','g','r')
while True:
    _, frame = video.read()
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

    cv.imshow("dark",dark);
    cv.imshow("t",t);
    # cv.imshow('I',frame2);
    cv.imshow('J',J);
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
