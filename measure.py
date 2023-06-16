import cv2 as cv
import numpy as np
import tensorflow as tf 


# HIGH_VALUE = 10000
# WIDTH = HIGH_VALUE
# HEIGHT = HIGH_VALUE


def captura_frame(origen):
  captura = cv.VideoCapture(origen)
  # print(captura.get(cv.CAP_PROP_FRAME_HEIGHT),captura.get(cv.CAP_PROP_FRAME_WIDTH),3);
  new_frame = np.zeros((int(captura.get(cv.CAP_PROP_FRAME_HEIGHT)), int(captura.get(cv.CAP_PROP_FRAME_WIDTH)),int(3)),dtype=np.uint8)

  while (captura.isOpened()):
    ret, imagen = captura.read()
    new_frame = imagen
    new_frame = np.fliplr(imagen) # matriz en espejo
    if ret == True: # si se lee el frame de manera correcta entra en el if
      cv.imshow('video', new_frame)
      if cv.waitKey(1) & 0xFF == 0x1B: # detect if press esc key
        break
    else:
      break 

  captura.release()
  cv.destroyAllWindows()

  return (new_frame)


if __name__ == "__main__":
  captura_frame(0)

  


