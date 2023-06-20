from ultralytics import YOLO
import cv2 as cv
import cvzone
import math

# cap = cv.VideoCapture(0)
# cap.set(3, 640)
# cap.set(4, 480)

cap = cv.VideoCapture("../Algoritmos_vision/Videos/motorbikes.mp4")



model = YOLO('../Yolo-Weigths/yolov8l.pt')

className = ["Persona", "bicicleta", "automovil", "motocicleta", "casa", "plátano", "pelota", "teléfono", "ordenador", "semaforo", "libro", 
                 "cuchara", "árbol", "sol", "lámpara", "camisa", "zapato", "sombrero", "taza", "piano", "mesa", "pantalla", "ratón", 
                 "manzana", "naranja", "reloj", "papel", "lápiz", "puerta", "ventana", "gafas", "bicicleta", "jardín", "cartas", 
                 "televisión", "maleta", "cama", "cámara", "aire acondicionado", "balón", "vaso", "reloj despertador", "flores", 
                 "silla de ruedas", "candado", "auriculares", "radio", "pintura", "ropa", "bolígrafo", "peine", "escritorio", 
                 "teclado", "percha", "laptop", "lente", "mochila", "abrelatas", "bombilla", "caja", "botella", "tetera", "cuchillo", 
                 "plato", "sartén", "tenedor", "tijeras", "microondas", "basura", "aspiradora", "secadora", "lavadora", "espejo", 
                 "pizarra", "horno", "cafetera", "frigorífico", "impresora", "altavoz"]


while True:
    ret, frame = cap.read()
    # frame = cv.flip(frame, 1)
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
            

            cls = int(box.cls[0])

            cvzone.putTextRect(frame, f'{className[cls]}  {conf}', (max(0, x1), max(35,y1)), scale=0.7, thickness=1, colorB=(0,255,0))

    cv.imshow('frame', frame)
    cv.waitKey(1)
    # if cv.waitKey(1) & 0xFF == 0x1B:    
    #     break