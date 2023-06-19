from ultralytics import YOLO
import cv2 as cv
import cvzone
import math

cap = cv.VideoCapture(0)

cap.set(3, 640)
cap.set(4, 480)

model = YOLO('../Yolo-Weigths/yolov8n.pt')


while True:
    ret, frame = cap.read()
    frame = cv.flip(frame, 1)
    results = model(frame, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w ,h = x2-x1, y2-y1

            bbbox = [x1, y1, w, h]
            cvzone.cornerRect(frame, bbbox)

            conf = math.ceil(box.conf[0]*100)/100
            print(conf)
            cvzone.putTextRect(frame, f'{conf}', (x1, y1-10), 2, 2, offset=10, border=5)


    cv.imshow('frame', frame)
    cv.waitKey(1)
    # if cv.waitKey(1) & 0xFF == 0x1B:    
    #     break