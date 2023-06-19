from ultralytics import YOLO
import cv2

model = YOLO('../Yolo-Weigths/yolov8l.pt')
results = model('motorcycles.jpg', show=True)

cv2.waitKey(0)  