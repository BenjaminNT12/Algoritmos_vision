import math
import numpy as np
import cv2 as cv
import time

PATH = '/home/nicolas/Videos/VideosPruebasMay21/pruebasMay21.AVI'
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

SCALE = 50

HSV_RANGE_COMPLETE = False

AREA_THRESHOLD = 500

cap = cv.VideoCapture(PATH)


rectangle_mask = np.zeros((int(cap.get(4)* SCALE / 100), int(cap.get(3)* SCALE / 100), 3), dtype=np.uint8)
mask = np.zeros((int(cap.get(4)* SCALE / 100), int(cap.get(3)* SCALE / 100), 3), dtype=np.uint8)

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
    dist2 = distance(cordinates[1][:], cordinates[2][:])
    dist3 = distance(cordinates[2][:], cordinates[3][:])
    dist4 = distance(cordinates[3][:], cordinates[0][:])

    d1cm = 285.283 - 0.849372*dist1
    d2cm = 219.445 - 1.08559*dist2  # polinomio de ajuste d2
    d3cm = 267.427 - 0.761644*dist3  # polinomio de ajuste d3
    d4cm = 295.1866 - 1.827553*dist4  # polinomio de ajuste d4
    avg = (d1cm + d2cm + d3cm + d4cm) / 4

    return d1cm, d2cm, d3cm, d4cm, avg




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



def order_points(vec_x, vec_y):
    if len(vec_x) == 4 and len(vec_y) == 4:
        max_val = -float('inf')
        min_val = float('inf')
        max_index = min_index = -1

        for i in range(4):
            sum_val = vec_x[i] + vec_y[i]
            if sum_val > max_val:
                max_val = sum_val
                max_index = i
            if sum_val < min_val:
                min_val = sum_val
                min_index = i

        remaining_indices = [i for i in range(4) if i not in [max_index, min_index]]
        remaining_indices.sort(key=lambda i: vec_x[i])

        posiciones = [max_index, min_index] + remaining_indices

        points = [ [vec_x[posiciones[1]], vec_y[posiciones[1]]]
                ,[vec_x[posiciones[3]], vec_y[posiciones[3]]]
                ,[vec_x[posiciones[0]], vec_y[posiciones[0]]]
                ,[vec_x[posiciones[2]], vec_y[posiciones[2]]] ]

        return points
    else:
        return -1

def main():
    global lower_color, upper_color, HSV_RANGE_COMPLETE, RECTANGLE_COMPLETE, cap
    
    start_frame = get_frame_number(cap, START_SECOND)
    cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)

    start = 0
    end = 0
    sec = time.time()
    avg_radius = 0
    while True:
        ret, frame = cap.read()

        if ret == False: 
            break
        frame_real = resize_frame(frame, SCALE)
        frame_to_process = resize_frame(frame, SCALE)
        # cv.rectangle(frame_to_process, (0,0), (frame_to_process.shape[1], 50), (0, 255, 0), -1)
    
        end = time.time()
        cv.putText(frame_real, "Sec: {:.2f}".format(time.time()-sec)
                ,(10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)    
        cv.putText(frame_real, "FPS: {:.2f}".format(1 / (end - start))
                ,(10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)
        start = time.time()
        
        floor_mask = cv.inRange(frame_to_process, lower_color9, upper_color9)
        _, floor_mask_binary = cv.threshold(floor_mask, 245, 245, cv.THRESH_BINARY)
        cv.imshow('floor_mask_binary', floor_mask_binary)

        blue_channel_, green_channel_, red_Channel  = cv.split(frame_to_process)
        cv.imshow('red_Channel', red_Channel)
        cv.imshow('green_channel', green_channel_)
        cv.imshow('blue_channel', blue_channel_)
        _, red_Channel_binary = cv.threshold(red_Channel, 245, 245, cv.THRESH_BINARY)
        cv.imshow('red_Channel_binary', red_Channel_binary)
        frame_to_detect_binary = cv.subtract(red_Channel_binary, floor_mask_binary)
        cv.imshow('frame_to_process', frame_to_detect_binary)

        if int(avg_radius) <= 10:
            kernel = np.ones((5,5),np.uint8)
            mask_long_distance = cv.inRange(frame_to_process, lower_color2, upper_color2)
            mask_long_distance = cv.medianBlur(mask_long_distance, 5)
            dilation = cv.dilate(mask_long_distance, kernel, iterations = 1)
            cv.imshow('mask2', dilation)
            frame_to_detect_binary = cv.add(frame_to_detect_binary, mask_long_distance)

        
        frame_to_detect = cv.merge([frame_to_detect_binary, frame_to_detect_binary, frame_to_detect_binary])
        frame_to_detect = cv.medianBlur(frame_to_detect, 5)
        
        frame_to_detect_gray = cv.cvtColor(frame_to_detect, cv.COLOR_BGR2GRAY)
        contours, _ = cv.findContours(frame_to_detect_gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        areas = []
        lower_y = []

        for contour in contours:
            area = cv.contourArea(contour)
            if area > AREA_MINIMUM_THRESHOLD:
                (x_pos, y_pos), _ = cv.minEnclosingCircle(contour)
                areas.append(area)
                lower_y.append(y_pos)
        
        max_area = sorted(areas, reverse=True)[:6]
        min_pos_area = sorted(lower_y, reverse=True)[:4]

        vec_x = []
        vec_y = []
        avg_radius = 0
        mask_circulos = np.zeros((frame_to_detect.shape[0], frame_to_detect.shape[1]), np.uint8)
        for contour in contours:
            area = cv.contourArea(contour)
            (x,y),radius = cv.minEnclosingCircle(contour)

            if area in max_area and y in min_pos_area:
                vec_x.append(int(x))
                vec_y.append(int(y))
                center, radius = (int(x),int(y)), int(radius)
                avg_radius += radius

                if radius >= 25:
                    radius = 25
                    
                mask_circulos = cv.circle(mask_circulos,center,radius,(255,255,255),-1)
                frame_to_detect = cv.circle(mask_circulos,center,radius,(255,255,255),-1)
                frame_real = cv.circle(frame_real,center,radius,(0,255,0),2)
        
        points2 = order_points(vec_x, vec_y)
        avg_radius = avg_radius/4

        if len(vec_x) >= 4 and points2 != -1:
            
            for i in range(4):
                cv.putText(frame_real, str(i), (int(points2[i][0]), int(points2[i][1])),cv.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
                draw_line(frame_real, (int(points2[i][0]), int(points2[i][1])), (int(points2[(i+1)%4][0]), int(points2[(i+1)%4][1])))
                d1, d2, d3, d4, avg_distance = calculate_distance(points2)
                cv.putText(frame_real, "Promedio del radio: " + str(int(avg_radius)), (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)
                cv.putText(frame_real, "Distancias: "
                           + str(int(avg_distance)) + "cm ",
                           (10, 110), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)



        cv.imshow('frame', frame_to_detect)
        cv.imshow('frame Real', frame_real)

        if cv.waitKey(6) & 0xFF == ord("q"):
            break
    

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
