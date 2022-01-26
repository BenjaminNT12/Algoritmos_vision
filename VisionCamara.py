import numpy as np
import cv2


from matplotlib import pyplot as plt
from scipy.cluster import hierarchy
from numpy.core.multiarray import dtype

down= False
xi,yi = 0,0
xf,yf = 0,0

def paint(event,x,y,flags,params):
    global down, xi , yi, xf, yf, p
    if event == cv2.EVENT_LBUTTONDOWN:
        xi,yi=x,y
        down = True
    elif event == cv2.EVENT_MOUSEMOVE and down == True:
        if down == True:
            board[:] = 0
            cv2.rectangle(board,(xi,yi),(x,y),(255,0,0),3)
            xf,yf = x,y
    elif event == cv2.EVENT_LBUTTONUP:
        down = False
        p = True

def webcam(cam, color_avr):

    kernel = np.ones((5,5),np.uint8)
    def add(m,num):
        output = np.array([0,0,0],np.uint8)
        print("print m = ",m)
        for i,e in enumerate(m):
            print("print i = ", i)
            print("print e = ", e)

            q = e+num

            if q >= 0 and q <= 255:
                output[i] = q
            elif q > 255:
                output[i] = 255
            else:
                output[i] = 0
        return output
    rangomax = add(color_avr,30)
    rangomin = add(color_avr,-30)
    print("color_avr:\t",color_avr)
    print("rangomax:\t",rangomax)
    print("rangomin:\t",rangomax)
    while(True):
        ret,frame = cam.read(0)
        mascara = cv2.inRange(frame,rangomin,rangomax)
        cv2.imshow("mascara", mascara)
        opening = cv2.morphologyEx(mascara,cv2.MORPH_OPEN,kernel)
        variable,contours,hierarchy = cv2.findContours(opening,1,2)
        for cnt in contours:
            if np.size(cnt)>500:
                x,y,w,h = cv2.boundingRect(cnt)
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
                # cv2.circle(frame,(x+w,y+h),5,(0,0,255),-1)
        cv2.imshow("cam",frame)
        k=cv2.waitKey(1)&0xFF
        if k == 27:
            # plt.show()
            break
    cam.release()
    cv2.destroyAllWindows()

def main():
    global board,p
    p = False
    cam = cv2.VideoCapture(0)
    board = np.zeros((int(cam.get(4)),int(cam.get(3)),3),dtype = np.uint8)
    while(True):
        ret,frame=cam.read()
        # hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cv2.namedWindow('board')
        cv2.setMouseCallback('board',paint)
        dst = cv2.addWeighted(frame,1,board,1,0)
        cv2.imshow('board',dst)
        k=cv2.waitKey(1) & 0xFF
        if k == 27 or p:
            break
    print("primer paso \n")
    cv2.destroyAllWindows()

    color = ('b','g','r')

    for i,col in enumerate(color):
        histr = cv2.calcHist([frame[yi:yf,xi:xf]],[i],None,[256],[0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])
    plt.show()



    if xi != xf and yi != yf:
        ext = frame[yi:yf,xi:xf]
        s = np.array([0,0,0])
        for i in range(np.shape(ext)[0]):
            for j in range(np.shape(ext)[1]):
                s += ext[i][j]
        webcam(cam,s/((i+1)*(j+1)))

if __name__== '__main__':
    print("Comando")
    print("Salir [ESC]")
    main()
