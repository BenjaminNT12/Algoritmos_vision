import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt
import time
from PIL import Image



# path = r'video2.mp4'
video = cv.VideoCapture(0)

p = False
down= False
xi,yi = 0,0
xf,yf = 0,0
board = np.zeros((int(video.get(4)), int(video.get(3)), 3), dtype = np.uint8)
frame2 = np.zeros((int(video.get(4)), int(video.get(3)), 3), dtype = np.uint8)
frame3 = np.zeros((int(video.get(4)), int(video.get(3)), 3), dtype = np.uint8)
frame4 = np.zeros((int(video.get(4)), int(video.get(3)), 3), dtype = np.uint8)
color = ('b','g','r')

tiempo_anterior = 0.0
periodo = 0.0
frecuencia = 0.0

result_r, result_g, result_b = [], [], []
target_hue = 0


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

def find_object(im, mask, color):
    cnts, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    x, y, w, h = 0, 0, 0, 0
    i = 0

    for pic, cnts in enumerate(cnts):
        area = cv.contourArea(cnts)
        if (area > 250):
            x, y, w, h = cv.boundingRect(cnts)

            cv.circle(im, (int(x+w/2),int(y+h/2)), 10, color,-1)
            i = i+1
            im = cv.putText(im, str(i) , (int(x+w/2),int(y+h/2)), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

    return(round(x+w/2), round(y+h/2))

while True:
    _, frame = video.read()

    periodo = time.time() - tiempo_anterior
    frecuencia = 1/periodo
    print('Periodo: %5.2f, Frecuencia %5.2f'%(periodo, frecuencia))
    tiempo_anterior = time.time()

    I = frame.astype('float64')/255;
    dark = DarkChannel(I,30);
    A = AtmLight(I,dark);
    te = TransmissionEstimate(I,A,30);
    t = TransmissionRefine(frame,te);
    J = Recover(I,t,A,0.1);
    J2 = J.astype('uint8')*255

    J2_gris = cv.cvtColor(J2, cv.COLOR_BGR2GRAY)
    thresh = 128
    retb, img_binary = cv.threshold(J2_gris, thresh, 255, cv.THRESH_BINARY)

    cv.imshow("Frame", frame)
    cv.imshow("dark",dark);
    cv.imshow("t",t);
    cv.imshow("J",J);
    cv.imshow('J2',J2);
    cv.imshow("J2_gris", J2_gris)
    cv.imshow("Binaria", img_binary)

################################################################################
    r, g, b = cv.split(J2)
    r = r*1
    g = g*0
    b = b*0

    image = cv.merge((r, g, b))
    cv.imshow("blue",b)
    cv.imshow("green",g)
    cv.imshow("red",r)
    cv.imshow('image',image);

    hsv_b_min = (int(194/2),68,83)
    hsv_b_max = (int(241/2),255,255)

    hsv_g_min = (54,93,183)
    hsv_g_max = (81,255,255)

    hsv_r_min = (0, 100, 20)
    hsv_r_max = (8, 255, 255)

    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    mask_blue = cv.inRange(hsv, hsv_b_min, hsv_b_max)
    pg = find_object(image, mask_blue, (0,0,255))
    # pg = find_object(image, mask_blue, (0,0,255))


    cv.imshow('Tracking',image);

################################################################################

    key = cv.waitKey(1)
    if key == 27:
        break

video.release()
# salida.release()
cv.destroyAllWindows()
