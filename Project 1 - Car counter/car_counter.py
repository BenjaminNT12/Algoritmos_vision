from ultralytics import YOLO
import cv2 as cv
import cvzone
import math
from sort import *

cap = cv.VideoCapture("../Algoritmos_vision/Videos/cars.mp4")
mask = cv.imread("../Algoritmos_vision/Videos/mask.png")

# cv.imshow("mask", mask)
# cv.waitKey(0)

model = YOLO('../Yolo-Weigths/yolov8n.pt')

className = ["Persona", "bicicleta", "automovil", "motocicleta", "casa", "plátano", "pelota", "teléfono", "ordenador", "semaforo", "libro", 
                 "cuchara", "árbol", "sol", "lámpara", "camisa", "zapato", "sombrero", "taza", "piano", "mesa", "pantalla", "ratón", 
                 "manzana", "naranja", "reloj", "papel", "lápiz", "puerta", "ventana", "gafas", "bicicleta", "jardín", "cartas", 
                 "televisión", "maleta", "cama", "cámara", "aire acondicionado", "balón", "vaso", "reloj despertador", "flores", 
                 "silla de ruedas", "candado", "auriculares", "radio", "pintura", "ropa", "bolígrafo", "peine", "escritorio", 
                 "teclado", "percha", "laptop", "lente", "mochila", "abrelatas", "bombilla", "caja", "botella", "tetera", "cuchillo", 
                 "plato", "sartén", "tenedor", "tijeras", "microondas", "basura", "aspiradora", "secadora", "lavadora", "espejo", 
                 "pizarra", "horno", "cafetera", "frigorífico", "impresora", "altavoz"]


tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limits = [400, 297, 673, 297]

while True:
    ret, frame = cap.read()
    imgRegion = cv.bitwise_and(frame, mask)
    # frame = cv.flip(frame, 1)
    results = model(imgRegion, stream=True)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w ,h = x2-x1, y2-y1

            bbbox = [x1, y1, w, h]
            

            conf = math.ceil(box.conf[0]*100)/100
            print(conf)
            
            cls = int(box.cls[0])
            currentClass = className[cls]

            if currentClass == "automovil" or currentClass == "bus" or currentClass == "motocicleta" and conf > 0.3:
                # cvzone.cornerRect(frame, bbbox, l=9, rt=5)
                # cvzone.putTextRect(frame, f'{currentClass}  {conf}', (max(0, x1), max(35,y1)), scale=0.6, thickness=1,offset=3)
                
                currentArray = np.array([[x1, y1, x2, y2, conf]])
                detections = np.vstack([detections, currentArray])




    resultsTracker = tracker.update(detections)

    cv.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 2)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w ,h = x2-x1, y2-y1
        bbbox = [x1, y1, w, h]
        cvzone.cornerRect(frame, bbbox, l=9, rt=5, colorR=(255,0,0))
        cvzone.putTextRect(frame, f'{int(id)}', (max(0, x1), max(35,y1)), scale=2, thickness=3,offset=10)

        cx, cy = x1+w//2, y1+h//2, # integracion enteras

        cv.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

    cv.imshow('frame', frame)
    # cv.imshow('ImgRegion', imgRegion)
    # cv.waitKey(1)
    if cv.waitKey(1) & 0xFF == 0x1B:    
        break