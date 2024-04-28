import cv2 as cv 
import os

path = "/home/nicolas/Github/Algoritmos_vision/fotosCalibracion4"

def take_photo():
    # Abre la cámara
    cap = cv.VideoCapture(2)
    count = 0
    while True:
        # Captura un cuadro de la cámara
        ret, frame = cap.read()

        # Muestra el cuadro en una ventana llamada "Captura"
        cv.imshow("Captura", frame)

        # Espera a que se presione una tecla y obtiene el código ASCII
        key = cv.waitKey(1) & 0xFF

        # Si se presiona la tecla "t" (ASCII 116), toma la foto
        if key == ord("t"):
            # Guarda la foto con el nombre "calibracion.jpg"
            cv.imwrite(os.path.join(path, f"calibracion{count}.jpg"), frame)
            print(f"calibracion{count}.jpg")
            count += 1
            # break

        # Si se presiona la tecla "q" (ASCII 113), sale del bucle
        elif key == 0x1B:
            break

    # Libera la cámara y cierra todas las ventanas abiertas
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    take_photo()
