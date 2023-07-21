import cv2


# HIGH_VALUE = 10000
# WIDTH = HIGH_VALUE
# HEIGHT = HIGH_VALUE

captura = cv2.VideoCapture(2)

# captura.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('D','I','V','X'))
# captura.set(3, 1920)
# captura.set(4, 1080)
print(captura.get(3), captura.get(4))

while (captura.isOpened()):
  ret, imagen = captura.read()
  
  if ret == True:
    imagen = cv2.flip(imagen, 1)
    cv2.imshow('video', imagen)
    if cv2.waitKey(1) & 0xFF == 0x1B: # detect if press esc key
      break
  else: break
captura.release()
cv2.destroyAllWindows()
