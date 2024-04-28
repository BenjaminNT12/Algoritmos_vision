import math
import numpy as np
import cv2 as cv
import time

PATH = '/home/nicolas/Videos/VideosPruebasApr17/pruebas4.AVI'
# lower_color [135   2 254] upper_color [158   4 255]-
# PATH = 0

PTS_COMPLETE = False
COUNT = 0
SET_ROTATION = 0.0

coordenadas = []
new_cordinates = np.array([])
START_SECOND = 10
RESTART_SECOND = 33
MARGEN_COLOR = 10

ix, iy = 0, 0
xf, yf = 0, 0
RECTANGLE_COMPLETE = False
DOWN = False

h_min, s_min, v_min = 225, 221, 255
h_max, s_max, v_max = 255, 255, 255

lower_color = np.array([h_min, s_min, v_min])
upper_color = np.array([h_max, s_max, v_max])

color1 = np.array([200, 238, 251])
color2 = np.array([145, 146, 232])
color3 = np.array([143, 144, 222])
color4 = np.array([143, 144, 222])
color5 = np.array([89, 87, 254])
color6 = np.array([129, 134, 234])
color7 = np.array([107, 158, 253])
color8 = np.array([80, 89, 233])
color9 = np.array([250, 250, 250])

lower_color = color1 -15
upper_color = color1 +15

lower_color2 = color2-15
upper_color2 = color2+15

lower_color3 = color3-15
upper_color3 = color3+15

lower_color4 = color4-15
upper_color4 = color4+15

lower_color5 = color5-15
upper_color5 = color5+15

lower_color6 = color6-15
upper_color6 = color6+15

lower_color7 = color7-15
upper_color7 = color7+15

lower_color8 = color8-15
upper_color8 = color8+15

lower_color9 = color9-20
upper_color9 = color9+5



HSV_RANGE_COMPLETE = False

AREA_THRESHOLD = 500

cap = cv.VideoCapture(PATH)


rectangle_mask = np.zeros((int(cap.get(4)* 50 / 100), int(cap.get(3)* 50 / 100), 3), dtype=np.uint8)
mask = np.zeros((int(cap.get(4)* 50 / 100), int(cap.get(3)* 50 / 100), 3), dtype=np.uint8)

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
AREA_MINIMUM_THRESHOLD = 20

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
            cv.rectangle(rectangle_mask, (ix, iy), (x_coord, y_coord), (0, 255, 0), 3)
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

    # numeroPuntos = 0

    kernel = np.ones((5,5),np.uint8)
    color_mask = cv.inRange(frame_to_track, lower_color_to_track, upper_color_to_track)

    # Encontrar los contornos de los objetos de color
    opening = cv.morphologyEx(color_mask,cv.MORPH_OPEN,kernel)
    contours,_ = cv.findContours(opening,1,2)
    cv.drawContours(frame_to_track,contours,-1,(0,255,0),3)


    # Realizar el seguimiento de los objetos de color
    for contour in contours:
        # Calcular el área del contorno
        area = cv.contourArea(contour)
        # Descartar contornos pequeños
        if area > AREA_THRESHOLD:
            print("area: ", area)
            # numeroPuntos += 1
            # print(" aceptada")
    
            # Calcular el centroide del contorno
            moment = cv.moments(contour)
            centroid_x = int(moment["m10"] / moment["m00"])
            centroid_y = int(moment["m01"] / moment["m00"])

            if PTS_COMPLETE is False:
                coordenadas.append([centroid_x, centroid_y])
                new_cordinates = np.array(coordenadas)
            else:
                # if numeroPuntos == NUMERO_DE_PUNTOS:
                #     PTS_COMPLETE = True
                actual_coordinates = np.array([centroid_x, centroid_y])
                position = dentro_de_area(new_cordinates, actual_coordinates, 100)
                np.put(new_cordinates, [len(actual_coordinates)*position,len(actual_coordinates)*position + 1], actual_coordinates)

                translacion, angle = calcular_pose(new_cordinates) # Se calcula la translacion y el angulo

                cv.circle(frame_to_track, (int(translacion[0]), int(translacion[1])), 5, (0, 0, 0), -1)

                draw_line(frame_to_track, new_cordinates[0][:], new_cordinates[1][:], thickness=3)
                draw_line(frame_to_track, new_cordinates[1][:], new_cordinates[3][:], thickness=3)
                draw_line(frame_to_track, new_cordinates[3][:], new_cordinates[2][:], thickness=3)
                draw_line(frame_to_track, new_cordinates[2][:], new_cordinates[0][:], thickness=3)
############################################################################################################
                plot_points(frame_to_track, objectPoints)
                coordenadas_float = np.array(new_cordinates, dtype=np.float32)

                if (len(new_cordinates) > NUMERO_DE_PUNTOS-1):
                    _, rotacion3d, translacion3D = estimar_pose_3D(objectPoints, coordenadas_float, cameraMatrix, distCoeffs) # Se calcula la rotacion y la translacion en 3D
                    # print("translacion: ", translacion3D)
                    cv.putText(frame_to_track, "Rotacion X: " + str(int(math.degrees(rotacion3d[0]))), (20, 60), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)
                    cv.putText(frame_to_track, "Rotacion Y: " + str(int(math.degrees(rotacion3d[1]))), (20, 80), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)
                    cv.putText(frame_to_track, "Rotacion Z: " + str(int(math.degrees(rotacion3d[2]))), (20, 100), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)
                    
                    nose_end_point2D, jacobian = cv.projectPoints( np.array([(0.0, 0.0, 1000.0)]), rotacion3d, translacion3D, cameraMatrix, distCoeffs)

                    point1 = (int(new_cordinates[0][0]), int(new_cordinates[0][1]))
                    point2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

                    # print(nose_end_point2D)
                    cv.line(frame_to_track, point1, point2, (0, 0, 0), 2)
############################################################################################################
                d1cm, d2cm, d3cm, d4cm = calculate_distance(new_cordinates)

                cv.putText(frame_to_track, "Angulo: " + str(int(angle)) + " Grados", (20, 20), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)

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


def getHSVparameters(color_avr, bounds=50):
    global lower_color, upper_color

    def add(m,num):
        output = np.array([0,0,0],np.uint8)
        for i,e in enumerate(m):
            q = e+num

            if q >= 0 and q <= 255:
                output[i] = q
            elif q > 255:
                output[i] = 255
            else:
                output[i] = 0
        return output
    print(color_avr.shape)
    upper_color = add(color_avr, bounds)
    lower_color = add(color_avr, -bounds)

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
    getHSVparameters(s/((i+1)*(j+1)), MARGEN_COLOR)
    print("color promedio:\t",s/((i+1)*(j+1)))



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



def resize_frame(frame, scale_percent):
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv.resize(frame, dim, interpolation = cv.INTER_AREA)



def main():
    global lower_color, upper_color, HSV_RANGE_COMPLETE, RECTANGLE_COMPLETE, cap
    

    while True:
        start_time = time.time()
        ret, frame = cap.read()

        if ret == False: 
            break
        frame_real = resize_frame(frame, 50)
        frame_to_process = frame_real

        floor_mask = cv.inRange(frame_to_process, lower_color9, upper_color9)
        _, floor_mask_binary = cv.threshold(floor_mask, 245, 245, cv.THRESH_BINARY)

        _, _, red_Channel  = cv.split(frame_to_process)
        _, red_Channel_binary = cv.threshold(red_Channel, 245, 245, cv.THRESH_BINARY)
        frame_to_detect_binary = cv.subtract(red_Channel_binary, floor_mask_binary)
        
        frame_to_detect = cv.merge([frame_to_detect_binary, frame_to_detect_binary, frame_to_detect_binary])
        frame_to_detect = cv.medianBlur(frame_to_detect, 5)
        
        frame_to_detect_gray = cv.cvtColor(frame_to_detect, cv.COLOR_BGR2GRAY)
        contours, _ = cv.findContours(frame_to_detect_gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        areas = []
        lower_y = []

        for contour in contours:
            area = cv.contourArea(contour)
            if area > AREA_MINIMUM_THRESHOLD:
                print("area", area)
                (_, y_pos), _ = cv.minEnclosingCircle(contour)
                areas.append(area)
                lower_y.append(y_pos)

        max_area = sorted(areas, reverse=True)[:6]
        min_pos_area = sorted(lower_y, reverse=True)[:4]

        for contour in contours:
            area = cv.contourArea(contour)
            (x,y),radius = cv.minEnclosingCircle(contour)

            if area in max_area and y in min_pos_area:
                
                center, radius = (int(x),int(y)), int(radius)
                frame_to_detect = cv.circle(frame_to_detect,center,radius,(0,255,0),-1)
                frame_real = cv.circle(frame_real,center,radius,(0,255,0),2)
    

        if cv.waitKey(5) & 0xFF == ord('p'):
            print("Selección de region de interés")
            while True:
                temp_frame = np.zeros((int(cap.get(4)* 50 / 100), int(cap.get(3)* 50 / 100), 3), dtype=np.uint8)
                if RECTANGLE_COMPLETE == False:
                    cv.namedWindow('frame')
                    cv.setMouseCallback('frame', draw_rectangle)
                
                    if frame.shape == rectangle_mask.shape:
                        cv.addWeighted(frame_to_detect, 1, rectangle_mask, 1, 0, temp_frame)
                    else:
                        print("frame y rectangle_mask no tienen el mismo tamaño")

                if RECTANGLE_COMPLETE == True and HSV_RANGE_COMPLETE == False:
                    get_hsv_range(frame_to_detect, ix, iy, xf, yf)
                    HSV_RANGE_COMPLETE = True

                cv.imshow('frame', frame_to_detect)

                if cv.waitKey(1) & 0xFF == ord('p') or HSV_RANGE_COMPLETE == True:
                    break

        if HSV_RANGE_COMPLETE or cv.waitKey(5) & 0xFF == ord('a'):
            HSV_RANGE_COMPLETE = True
            mask = color_tracking(frame_to_detect, lower_color, upper_color)

        cv.imshow('frame', frame_to_detect)
        cv.imshow('frame Real', frame_real)

        if cv.waitKey(1) & 0xFF == ord("q"):
            break
        
        endTime = time.time()
        duration = endTime - start_time
        print("tiempo de ejecucion: {:.4f} FPS: {:.2f}".format(duration, 1/duration))        

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()