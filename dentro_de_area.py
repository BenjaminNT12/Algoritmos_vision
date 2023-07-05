import numpy as np
import math 

def dentro_de_area(coordenadas, punto_referencia, radio):
    # Calcula la distancia euclidiana entre cada coordenada y el punto de referencia
    distancias = np.linalg.norm(coordenadas - punto_referencia, axis=1)

    # Verifica si la distancia es menor que el radio
    coordenadas_dentro_area = coordenadas[distancias < radio]

    return coordenadas_dentro_area

def dentro_de_area2(coordenadas, punto_referencia, radio):

    # Calcula la distancia euclidiana entre cada coordenada y el punto de referencia
    distancias = np.linalg.norm(coordenadas - punto_referencia, axis=1)

    # Verifica si la distancia es menor que el radio
    coordenadas_dentro_area = coordenadas[distancias < radio]

    # Obtiene la posición de las coordenadas dentro del arreglo
    posiciones_dentro_area = np.where(distancias < radio)[0]

    return coordenadas_dentro_area, posiciones_dentro_area

def distance(point1, point2):
  """Calculates the distance between two points.

  Args:
    point1: A tuple of (x, y) coordinates.
    point2: A tuple of (x, y) coordinates.

  Returns:
    The distance between the two points.
  """

  x_diff = point1[0] - point2[0]
  y_diff = point1[1] - point2[1]
  return math.sqrt(x_diff**2 + y_diff**2)

def find_closest_point(array, coordinates):
  """Finds the index of the point in the array that is closest to the given coordinates.

  Args:
    array: A list of points, each represented as a tuple of (x, y) coordinates.
    coordinates: The coordinates of the point to find the closest point to.

  Returns:
    The index of the point in the array that is closest to the given coordinates.
  """

  # Calculate the distance between each point in the array and the given coordinates.
  distances = []
  for i, point in enumerate(array):
    distances.append(distance(point, coordinates))

  # Find the index of the point with the smallest distance.
  closest_index = distances.index(min(distances))

  return closest_index


coordenadas = np.array([[388, 429], [72, 424], [243, 192], [398, 130], [81, 105]])

punto = np.array([253, 195])

print("coordenadas", coordenadas)

coordenadas_nuevas, posicion = dentro_de_area2(coordenadas, punto, 50)
np.put(coordenadas, [len(punto)*posicion, len(punto)*posicion+1], punto)
print("posicion", posicion, "punto", punto[:])
print("coordenadas", coordenadas, "posicion", posicion)



coordenadas = np.array([[243, 192], [388, 429], [72, 424], [398, 130], [81, 105]])

punto = np.array([253, 195])

print("coordenadas", coordenadas)

coordenadas_nuevas, posicion = dentro_de_area2(coordenadas, punto, 50)
np.put(coordenadas, [len(punto)*posicion, len(punto)*posicion+1], punto)
print("posicion", posicion, "punto", punto[:])
print("coordenadas", coordenadas, "posicion", posicion)

pos = find_closest_point(punto, coordenadas)
print("pos", pos)