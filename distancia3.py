import math
import numpy as np
import cv2 as cv

# PATH = '/home/nicolas/Github/Algoritmos_vision/Videos/video3.mp4'
# lower_color [135   2 254] upper_color [158   4 255]-
PATH = 0

PTS_COMPLETE = False
COUNT = 0
SET_ROTATION = 0.0

coordenadas = []
new_cordinates = np.array([])
START_SECOND = 80
RESTART_SECOND = 80

ix, iy = 0, 0
xf, yf = 0, 0
RECTANGLE_COMPLETE = False
DOWN = False

h_min, s_min, v_min = 255, 255, 255
h_max, s_max, v_max = 255, 255, 255


lower_color = np.array([h_min, s_min, v_min])
upper_color = np.array([h_max, s_max, v_max])

HSV_RANGE_COMPLETE = False

AREA_THRESHOLD = 600

cap = cv.VideoCapture(PATH)


rectangle_mask = np.zeros(
    (int(cap.get(4)), int(cap.get(3)), 3), dtype=np.uint8)
mask = np.zeros((int(cap.get(4)), int(cap.get(3)), 3), dtype=np.uint8)

ancho_imagen = int(cap.get(3))
alto_imagen = int(cap.get(4))

K1 = 0.11480806073904032
K2 = -0.21946985653851792
P1 = 0.0012002116999769957
P2 = 0.008564577708855225
K3 = 0.11274677130853494

FX, FY = ancho_imagen, alto_imagen
CX, CY = int(ancho_imagen/2), int(alto_imagen/2)

NUMERO_DE_PUNTOS = 4

cameraMatrix = np.array(
    [[FX, 0, CX],
     [0, FY, CY],
     [0,  0, 1]], dtype=np.float32)

distCoeffs = np.array([K1, K2, K1, P2, K3], dtype=np.float32)
# distCoeffs = np.zeros((4,1))

objectPoints = np.array(
    [[190.0, 190.0, 0.0],
     [0.0, 190.0, 0.0],
     [190.0, 0.0, 0.0],
     [0.0, 0.0, 0.0]], dtype=np.float32)


def draw_rectangle(event, x_coord, y_coord, flags, param):
    """_summary_

    Args:
        event (_type_): _description_
        x (_type_): _description_
        y (_type_): _description_
        flags (_type_): _description_
        param (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    global ix, iy, xf, yf, RECTANGLE_COMPLETE, DOWN

    if event == cv.EVENT_LBUTTONDOWN:
        ix, iy = x_coord, y_coord
        DOWN = True
    elif event == cv.EVENT_MOUSEMOVE and DOWN is True:
        if DOWN is True:
            rectangle_mask[:] = 0
            cv.rectangle(rectangle_mask, (ix, iy), (x_coord, y_coord), (0, 0, 255), 1)
            xf, yf = x_coord, y_coord
    elif event == cv.EVENT_LBUTTONUP:
        DOWN = False
        RECTANGLE_COMPLETE = True

def plot_points(frame_to_draw, points):
    """Funcion para plotear los puntos en la imagen

    Args:
        points (np.array): coordenadas de los puntos en 3 dimensiones
    """

    for i in range(len(points)):
        cv.circle(frame_to_draw, (int(points[i][0]), int(
            points[i][1])), 5, (0, 0, 0), -1)
        cv.putText(frame_to_draw, str(i), (int(points[i][0]), int(
            points[i][1])), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 2)


def estimar_pose_3D(objPoints, imgPoints, camMatrix, distorCoeffs):
    # Resolver PnP para estimar la pose 3D
    success, rvec, tvec = cv.solvePnP(objPoints,
                                      imgPoints,
                                      camMatrix,
                                      distorCoeffs,
                                      flags=cv.SOLVEPNP_ITERATIVE)

    return success, rvec, tvec


def dentro_de_area(coord, punto_referencia, radio):
    """_summary_

    Args:
        coord (_type_): _description_
        punto_referencia (_type_): _description_
        radio (_type_): _description_

    Returns:
        _type_: _description_
    """

    # Calcula la distancia euclidiana entre cada coordenada y el punto de referencia
    distancias = np.linalg.norm(coord - punto_referencia, axis=1)

    posiciones_dentro_area = np.where(distancias < radio)[0]

    return posiciones_dentro_area


def calcular_pose(puntos):
    """_summary_

    Args:
        puntos (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    global SET_ROTATION

    # Verificar que haya exactamente 5 puntos
    if puntos.shape != (NUMERO_DE_PUNTOS, 2):
        raise ValueError(
            "Se requiere una matriz de 4 filas y 2 columnas para calcular la pose.")

    # Extraer las coordenadas x e y de los puntos
    x_pts = puntos[:, 0]
    y_pts = puntos[:, 1]

    # Calcular el centro de masa de los puntos
    centro_x = np.mean(x_pts)
    centro_y = np.mean(y_pts)

    # Calcular la posición relativa de la pose
    posicion = np.array([centro_x, centro_y])

    x_pos = new_cordinates[NUMERO_DE_PUNTOS-1][1] - posicion[1]
    y_pos = new_cordinates[NUMERO_DE_PUNTOS-1][0] - posicion[0]

    orientacion_deg = math.atan2(y_pos, x_pos) * (180.0 / math.pi)
    if SET_ROTATION == 0.0:
        SET_ROTATION = orientacion_deg
    orientacion_deg = orientacion_deg - SET_ROTATION

    return posicion, orientacion_deg


def draw_line(img, start_point, end_point, color=(0, 0, 255), thickness=1):
    """_summary_

    Args:
        img (_type_): _description_
        start_point (_type_): _description_
        end_point (_type_): _description_
        color (tuple, optional): _description_. Defaults to (0, 0, 255).
        thickness (int, optional): _description_. Defaults to 1.
    """

    start_x = start_point[0]
    start_y = start_point[1]
    end_x = end_point[0]
    end_y = end_point[1]

    # Draw the line.
    cv.line(img, (int(start_x), int(start_y)),
            (int(end_x), int(end_y)), color, thickness)

# Funcion para calcular la distancia entre dos puntos dados?


def distance(point_1, point_2):
    """
    Calcula la distancia entre dos puntos dados.

    Parámetros:
        x1: La coordenada x del primer punto.
        y1: La coordenada y del primer punto.
        x2: La coordenada x del segundo punto.
        y2: La coordenada y del segundo punto.

    Devuelve:
        La distancia entre los dos puntos.
    """

    point1_x, point1_y = point_1
    point2_x, point2_y = point_2

    dist_x = point2_x - point1_x
    dist_y = point2_y - point1_y
    return math.sqrt(dist_x**2 + dist_y**2)

# Calcula la distancia entre dos puntos dados


def calculate_distance(cordinates):
    """_summary_
|
    Args:
        cordinates (_type_): _description_

    Returns:
        _type_: _description_
    """
    dist1 = distance(cordinates[0][:], cordinates[1][:])
    dist2 = distance(cordinates[1][:], cordinates[3][:])
    dist3 = distance(cordinates[3][:], cordinates[2][:])
    dist4 = distance(cordinates[2][:], cordinates[0][:])

    d1cm = 85.638 - 0.152*dist1  # polinomio de ajuste d1
    d2cm = 86.952 - 0.163*dist2  # polinomio de ajuste d2
    d3cm = 89.312 - 0.166*dist3  # polinomio de ajuste d3
    d4cm = 88.928 - 0.172*dist4  # polinomio de ajuste d4

    return d1cm, d2cm, d3cm, d4cm


def color_tracking(frame_to_track, lower_color_to_track, upper_color_to_track):
    """_summary_

    Args:
        frame (_type_): _description_
        lower_color (_type_): _description_
        upper_color (_type_): _description_

    Returns:
        _type_: _description_
    """
    global PTS_COMPLETE, coordenadas, new_cordinates

    position = 0
    translacion = 0

    kernel = np.ones((5,5),np.uint8)
    color_mask = cv.inRange(frame_to_track, lower_color_to_track, upper_color_to_track)

    opening = cv.morphologyEx(color_mask,cv.MORPH_OPEN,kernel)
    contours,_ = cv.findContours(opening,1,2)
    cv.drawContours(frame_to_track,contours,-1,(0,255,0),3)

#     # Encontrar los contornos de los objetos de color
#     contours, _ = cv.findContours(
#         color_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

#     # Realizar el seguimiento de los objetos de color
    for contour in contours:
        # Calcular el área del contorno
        area = cv.contourArea(contour)
        # print("area: ", area)
        # Descartar contornos pequeños
        print("area: ", area)
        if area > AREA_THRESHOLD:
            # Calcular el centroide del contorno
            moment = cv.moments(contour)
            centroid_x = int(moment["m10"] / moment["m00"])
            centroid_y = int(moment["m01"] / moment["m00"])

            if PTS_COMPLETE is False:
                coordenadas.append([centroid_x, centroid_y])
                new_cordinates = np.array(coordenadas)
            else:
                actual_coordinates = np.array([centroid_x, centroid_y])
                position = dentro_de_area(
                    new_cordinates, actual_coordinates, 50)
                np.put(new_cordinates, [len(actual_coordinates)*position,
                       len(actual_coordinates)*position + 1], actual_coordinates)

                translacion, angle = calcular_pose(new_cordinates)

                cv.circle(frame_to_track, (int(translacion[0]), int(
                    translacion[1])), 5, (0, 0, 0), -1)

                draw_line(frame_to_track, new_cordinates[0][:],
                          new_cordinates[1][:], thickness=3)
                draw_line(frame_to_track, new_cordinates[1][:],
                          new_cordinates[3][:], thickness=3)
                draw_line(frame_to_track, new_cordinates[3][:],
                          new_cordinates[2][:], thickness=3)
                draw_line(frame_to_track, new_cordinates[2][:],
                          new_cordinates[0][:], thickness=3)
############################################################################################################
                plot_points(frame_to_track, objectPoints)
                coordenadas_float = np.array(new_cordinates, dtype=np.float32)

                if (len(new_cordinates) > NUMERO_DE_PUNTOS-1):
                    _, rotacion3d, translacion3D = estimar_pose_3D(objectPoints,
                                                                   coordenadas_float,
                                                                   cameraMatrix,
                                                                   distCoeffs)
                    print("translacion: ", translacion3D)
                    cv.putText(frame_to_track,
                               "Rotacion X: " + str(int(math.degrees(rotacion3d[0]))),
                               (20, 60), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)
                    cv.putText(frame_to_track,
                               "Rotacion Y: " + str(int(math.degrees(rotacion3d[1]))),
                               (20, 80), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)
                    cv.putText(frame_to_track, 
                               "Rotacion Z: " + str(int(math.degrees(rotacion3d[2]))),
                               (20, 100), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)

                    
                    nose_end_point2D, jacobian = cv.projectPoints(
                        np.array([(0.0, 0.0, 1000.0)]),
                        rotacion3d,
                        translacion3D,
                        cameraMatrix,
                        distCoeffs)

                    point1 = (int(new_cordinates[0][0]),
                              int(new_cordinates[0][1]))

                    point2 = (int(nose_end_point2D[0][0][0]),
                              int(nose_end_point2D[0][0][1]))
                    print(nose_end_point2D)
                    cv.line(frame_to_track, point1, point2, (0, 0, 0), 2)
############################################################################################################
                d1cm, d2cm, d3cm, d4cm = calculate_distance(new_cordinates)

                cv.putText(frame_to_track, "Angulo: " + str(int(angle)) + " Grados",
                           (20, 20), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)

                cv.putText(frame_to_track, "Distancias: "
                           + str(int(d1cm)) + "cm ,"
                           + str(int(d2cm)) + "cm ,"
                           + str(int(d3cm)) + "cm ,"
                           + str(int(d4cm)) + "cm ",
                           (20, 40), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)

            cv.putText(frame_to_track, str(position), (centroid_x - 25, centroid_y - 25),
                       cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 2)
            cv.circle(frame_to_track, (centroid_x, centroid_y),
                      12, (0, 255, 0), -1)

    PTS_COMPLETE = True
    return color_mask


def get_frame_number(video, second):
    """_summary_

    Args:
        video (_type_): _description_
        second (_type_): _description_

    Returns:
        _type_: _description_
    """
    fps = video.get(cv.CAP_PROP_FPS)
    frame_number = int(fps * second)
    return frame_number


def webcam(color_avr):
    global lower_color, upper_color
    # kernel = np.ones((5,5),np.uint8)
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
    upper_color = add(color_avr,30)
    lower_color = add(color_avr,-30)
    print("color_avr:\t",color_avr)
    print("rangomax:\t",upper_color)
    print("rangomin:\t",lower_color)

def get_hsv_range(frame_rgb, x_min, y_min, x_max, y_max):
    """_summary_

    Args:
        frame (_type_): _description_
        x_min (_type_): _description_
        y_min (_type_): _description_
        x_max (_type_): _description_
        y_max (_type_): _description_

    Returns:
        _type_: _description_
    """

    ext = frame_rgb[y_min:y_max,x_min:x_max]
    s = np.array([0,0,0])
    for i in range(np.shape(ext)[0]):
        for j in range(np.shape(ext)[1]):
            s += ext[i][j]
    webcam(s/((i+1)*(j+1)))




    # global lower_color, upper_color

    # hsv_frame = cv.cvtColor(frame_rgb, cv.COLOR_RGB2HSV)
    # roi = hsv_frame[y_min:y_max, x_min:x_max, :]

    # h_channel_min = np.min(roi[:, :, 0])
    # h_channel_max = np.max(roi[:, :, 0])
    # s_channel_min = np.min(roi[:, :, 1])
    # s_channel_max = np.max(roi[:, :, 1])
    # v_channel_min = np.min(roi[:, :, 2])
    # v_channel_max = np.max(roi[:, :, 2])

    # low_color = np.array([h_channel_min, s_channel_min, v_channel_min])
    # up_color = np.array([h_channel_max, s_channel_max, v_channel_max])

    # return low_color, up_color


def enhance_image(frame_to_enhance):
    """_summary_

    Args:
        frame (_type_): _description_

    Returns:
        _type_: _description_
    """
    lab = cv.cvtColor(frame_to_enhance, cv.COLOR_BGR2LAB)

    l_channel, a_channel, b_channel = cv.split(lab)

    # Aplicar el ajuste de contraste en el canal L utilizando CLAHE
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)

    # Fusionar los canales LAB mejorados
    lab_enhanced = cv.merge((l_enhanced, a_channel, b_channel))

    # Convertir la imagen de vuelta a BGR
    enhanced = cv.cvtColor(lab_enhanced, cv.COLOR_LAB2BGR)

    return enhanced


start_frame = get_frame_number(cap, START_SECOND)
cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)


# fourcc = cv.VideoWriter_fourcc(*'XVID')
# out = cv.VideoWriter("orientacion.avi", fourcc, 30,
                    #  ((int(cap.get(4)), int(cap.get(3)))))

# while True:

#     ret, frame = cap.read()
#     # print("type(frame): ", type(frame))
#     if ret is False:
#         break

#     frame = cv.flip(frame, 1)
#     # frame = cv.resize(frame, (int(cap.get(3)/4), int(cap.get(4)/4)))
#     # frame = enhance_image(frame)

#     if HSV_RANGE_COMPLETE or cv.waitKey(1) & 0xFF == ord('a'):
#         HSV_RANGE_COMPLETE = True
#         print("Comienza la deteccion de color")
#         mask = color_tracking(frame, lower_color, upper_color)

#     cv.imshow('frame', frame)
#     # out.write(frame)
#     if cv.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# # out.release()
# cv.destroyAllWindows()



while True:

    ret, frame = cap.read()
    if ret == False: break

    frame = cv.flip(frame, 1)
    # frame = mejorar_imagen(frame)

    if cv.waitKey(1) & 0xFF == ord('p'):
        print("Selección de region de interés")
        while True:
            temp_frame = np.zeros((int(cap.get(4)), int(cap.get(3)), 3), dtype=np.uint8)

            if RECTANGLE_COMPLETE == False:
                cv.namedWindow('frame')
                cv.setMouseCallback('frame', draw_rectangle)
                cv.addWeighted(frame, 1, rectangle_mask, 1, 0, temp_frame)

            if RECTANGLE_COMPLETE == True and HSV_RANGE_COMPLETE == False:
                get_hsv_range(frame, ix, iy, xf, yf)
                HSV_RANGE_COMPLETE = True
                print("lower_color", lower_color, "upper_color", upper_color)
                # start_frame = get_frame_number(cap, RESTART_SECOND)
                # cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)

            cv.imshow('frame', temp_frame)

            if cv.waitKey(1) & 0xFF == ord('p') or HSV_RANGE_COMPLETE == True:
                break

    if HSV_RANGE_COMPLETE or cv.waitKey(1) & 0xFF == ord('a'):
        HSV_RANGE_COMPLETE = True
        mask = color_tracking(frame, lower_color, upper_color)
        cv.imshow('mask', mask)
        print("Comienza la deteccion de color")
        # cv.imshow('mask', mask)

    cv.imshow('frame', frame)

    # time.sleep(0.02)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv.destroyAllWindows()
