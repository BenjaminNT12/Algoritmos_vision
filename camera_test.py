import cv2


# HIGH_VALUE = 10000
# WIDTH = HIGH_VALUE
# HEIGHT = HIGH_VALUE

captura = cv2.VideoCapture(2)

# captura.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('D','I','V','X'))
captura.set(3, 640)
captura.set(4, 480)

while (captura.isOpened()):
  ret, imagen = captura.read()
  # up_width = 600
  # up_height = 400
  # up_points = (up_width, up_height)
  # imagen = cv2.resize(imagen, up_points, interpolation= cv2.INTER_LINEAR)
  if ret == True:
    cv2.imshow('video', imagen)
    if cv2.waitKey(1) & 0xFF == ord('s'):
      break
  else: break
captura.release()
cv2.destroyAllWindows()
