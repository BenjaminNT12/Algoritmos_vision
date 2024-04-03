import math
import numpy as np
import cv2 as cv

# PATH = '/home/nicolas/Github/Algoritmos_vision/video3.mp4'
# lower_color [135   2 254] upper_color [158   4 255]
PATH = 2

NAME = False
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

h_min, s_min, v_min = 135, 2, 254
h_max, s_max, v_max = 158, 4, 255

lower_color = np.array([h_min, s_min, v_min])
upper_color = np.array([h_max, s_max, v_max])

HSV_RANGE_COMPLETE = False

AREA_THRESHOLD = 50

cap = cv.VideoCapture(PATH)

rectangle_mask = np.zeros(
    (int(cap.get(4)), int(cap.get(3)), 3), dtype=np.uint8)
mask = np.zeros((int(cap.get(4)), int(cap.get(3)), 3), dtype=np.uint8)

# Función para el evento del mouse


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


def rgb_to_hsv(frame_to_convert):
    """_summary_

    Args:
        frame (_type_): _description_

    Returns:
        _type_: _description_
    """
    hsv_frame = cv.cvtColor(frame_to_convert, cv.COLOR_RGB2HSV)
    return hsv_frame


def split_hsv_channels(frame_to_split):
    """_summary_

    Args:
        frame (_type_): _description_

    Returns:
        _type_: _description_
    """
    hsv_frame = cv.cvtColor(frame_to_split, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv_frame)
    return h, s, v


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

    # Verifica si la distancia es menor que el radio
    coordenadas_dentro_area = coord[distancias < radio]

    # Obtiene la posición de las coordenadas dentro del arreglo
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
    if puntos.shape != (5, 2):
        raise ValueError(
            "Se requiere una matriz de 5 filas y 2 columnas para calcular la pose.")

    # Extraer las coordenadas x e y de los puntos
    x = puntos[:, 0]
    y = puntos[:, 1]

    # Calcular el centro de masa de los puntos
    centro_x = np.mean(x)
    centro_y = np.mean(y)

    # Calcular la posición relativa de la pose
    posicion = np.array([centro_x, centro_y])

    x_pos = new_cordinates[4][1] - posicion[1]
    y_pos = new_cordinates[4][0] - posicion[0]

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

    Args:
        cordinates (_type_): _description_

    Returns:
        _type_: _description_
    """
    dist1 = distance(cordinates[0][:], cordinates[1][:])
    dist2 = distance(cordinates[1][:], cordinates[4][:])
    dist3 = distance(cordinates[0][:], cordinates[3][:])
    dist4 = distance(cordinates[3][:], cordinates[4][:])

    d1cm = 85.638 - 0.152*dist1  # polinomio de ajuste d1
    d2cm = 86.952 - 0.163*dist2  # polinomio de ajuste d2
    d3cm = 89.312 - 0.166*dist3  # polinomio de ajuste d3
    d4cm = 88.928 - 0.172*dist4  # polinomio de ajuste d4

    return d1cm, d2cm, d3cm, d4cm


def color_tracking(frame, lower_color, upper_color):
    global NAME, coordenadas, new_cordinates, SET_ROTATION, COUNT

    position = 0
    translacion = 0

    hsv_frame = cv.cvtColor(frame, cv.COLOR_RGB2HSV)
    color_mask = cv.inRange(hsv_frame, lower_color, upper_color)

    color_mask = cv.erode(color_mask, None, iterations=2)
    color_mask = cv.dilate(color_mask, None, iterations=2)

    # Encontrar los contornos de los objetos de color
    contours, _ = cv.findContours(
        color_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Realizar el seguimiento de los objetos de color
    for contour in contours:
        # Calcular el área del contorno
        area = cv.contourArea(contour)

        # Descartar contornos pequeños
        if area >AREA_THRESHOLD:
            # Calcular el centroide del contorno
            M = cv.moments(contour)
            centroid_x = int(M["m10"] / M["m00"])
            centroid_y = int(M["m01"] / M["m00"])

            if NAME == False:
                coordenadas.append([centroid_x, centroid_y])
                new_cordinates = np.array(coordenadas)
            else:
                actual_coordinates = np.array([centroid_x, centroid_y])
                position = dentro_de_area(
                    new_cordinates, actual_coordinates, 50)
                np.put(new_cordinates, [len(actual_coordinates)*position,
                       len(actual_coordinates)*position+1], actual_coordinates)

                translacion, angle = calcular_pose(new_cordinates)

                cv.circle(frame, (int(translacion[0]), int(
                    translacion[1])), 5, (0, 0, 0), -1)

                draw_line(frame, new_cordinates[0][:],
                          new_cordinates[1][:], thickness=3)
                draw_line(frame, new_cordinates[1][:],
                          new_cordinates[4][:], thickness=3)
                draw_line(frame, new_cordinates[0][:],
                          new_cordinates[3][:], thickness=3)
                draw_line(frame, new_cordinates[3][:],
                          new_cordinates[4][:], thickness=3)

                d1cm, d2cm, d3cm, d4cm = calculate_distance(new_cordinates)

                cv.putText(frame, "Angulo: " + str(int(angle)) + " Grados",
                           (20, 20), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)

                cv.putText(frame, "Distancias: "
                           + str(int(d1cm)) + "cm ,"
                           + str(int(d2cm)) + "cm ,"
                           + str(int(d3cm)) + "cm ,"
                           + str(int(d4cm)) + "cm ",
                           (20, 40), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)

                # if cv.waitKey(1) & 0xFF == ord('r'):
                #     print("Distancias: ", round(d1,2), round(d2,2), round(d3,2), round(d4,2), "iteracion", COUNT)
                #     COUNT += 1
                #     break

            cv.putText(frame, str(position), (centroid_x - 25, centroid_y - 25),
                       cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 2)
            cv.circle(frame, (centroid_x, centroid_y), 12, (0, 255, 0), -1)

    NAME = True
    return color_mask


def get_frame_number(video, second):
    fps = video.get(cv.CAP_PROP_FPS)
    frame_number = int(fps * second)
    return frame_number


def get_hsv_range(frame, x_min, y_min, x_max, y_max):
    global lower_color, upper_color

    hsv_frame = cv.cvtColor(frame, cv.COLOR_RGB2HSV)
    roi = hsv_frame[y_min:y_max, x_min:x_max, :]

    h_min = np.min(roi[:, :, 0])
    h_max = np.max(roi[:, :, 0])
    s_min = np.min(roi[:, :, 1])
    s_max = np.max(roi[:, :, 1])
    v_min = np.min(roi[:, :, 2])
    v_max = np.max(roi[:, :, 2])

    lower_color = np.array([h_min, s_min, v_min])
    upper_color = np.array([h_max, s_max, v_max])

    return lower_color, upper_color


def mejorar_imagen(frame):
    lab = cv.cvtColor(frame, cv.COLOR_BGR2LAB)

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


fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter("orientacion.avi", fourcc, 30,
                     ((int(cap.get(4)), int(cap.get(3)))))

while True:

    ret, frame = cap.read()
    if ret is False:
        break

    frame = cv.flip(frame, 1)
    # frame = mejorar_imagen(frame)

    if HSV_RANGE_COMPLETE or cv.waitKey(1) & 0xFF == ord('a'):
        HSV_RANGE_COMPLETE = True
        mask = color_tracking(frame, lower_color, upper_color)

    cv.imshow('frame', frame)
    out.write(frame)
    if cv.waitKey(1) & 0xFF == 0x1B:
        break
cap.release()
out.release()
cv.destroyAllWindows()


# while True:

#     ret, frame = cap.read()
#     if ret == False: break

#     frame = cv.flip(frame, 1)
#     # frame = mejorar_imagen(frame)

#     # if cv.waitKey(1) & 0xFF == ord(' '):
#     #     while True:
#     #         temp_frame = np.zeros((int(cap.get(4)), int(cap.get(3)), 3), dtype=np.uint8)

#     #         if RECTANGLE_COMPLETE == False:
#     #             cv.NAMEdWindow('frame')
#     #             cv.setMouseCallback('frame', draw_rectangle)
#     #             cv.addWeighted(frame, 1, rectangle_mask, 1, 0, temp_frame)

#     #         if RECTANGLE_COMPLETE == True and hsv_range_complete == False:
#     #             get_hsv_range(frame, ix, iy, xf, yf)
#     #             hsv_range_complete = True
#     #             start_frame = get_frame_number(cap, RESTART_SECOND)
#     #             cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)

#     #         cv.imshow('frame', temp_frame)

#     #         if cv.waitKey(1) & 0xFF == ord(' ') or hsv_range_complete == True:
#     #             break

#     ifHSV_RANGE_COMPLETEcv.waitKey(1) & 0xFF == ord('a'):
#         hsv_range_complete = True
#         mask = color_tracking(frame, lower_color, upper_color)
#         # cv.imshow('mask', mask)

#     cv.imshow('frame', frame)

#     # time.sleep(0.02)

#     if cv.waitKey(1) & 0xFF == 0x1B:
#         break
# cap.release()
# cv.destroyAllWindows()
