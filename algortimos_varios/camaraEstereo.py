import cv2
import numpy as np
import threading

class camThread(threading.Thread):
    def __init__(self, previewName, camID):
        threading.Thread.__init__(self)
        self.previewName = previewName
        self.camID = camID
        self.frame = np.zeros((int(480) , int(640), 3), dtype=np.uint8)
    def run(self):
        print("Starting " + self.previewName)
        camPreview(self.previewName, self.camID, self.frame)


def camPreview(previewName, camID, frame):
    cv2.namedWindow(previewName)
    cam = cv2.VideoCapture(camID)
    cam.set(3, 640)
    cam.set(4, 480)
    if cam.isOpened():
        rval, frame = cam.read()
        frame = cv2.flip(frame, 0)
    else:
        rval = False

    while rval:
        cv2.imshow(previewName, frame)
        rval, frame = cam.read()
        frame = cv2.flip(frame, 1)
        key = cv2.waitKey(5)
        if key == 27:  # exit on ESC
            break
    cv2.destroyWindow(previewName)

# Create threads as follows
thread1 = camThread("Camera 1", 2)
thread2 = camThread("Camera 2", 4)
# thread3 = camThread("Camera 3", 0)


thread1.start()
thread2.start()
print("tipo de dato: \n")
cv2.imshow("frame 1",thread1.frame)
# thread1.getVideo()

print("Active threads", threading.activeCount())
