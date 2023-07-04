import numpy as np

def dentro_de_area(coordenadas, punto_referencia, radio):
    # Calcula la distancia euclidiana entre cada coordenada y el punto de referencia
    distancias = np.linalg.norm(coordenadas - punto_referencia, axis=1)

    # Verifica si la distancia es menor que el radio
    coordenadas_dentro_area = coordenadas[distancias < radio]

    return coordenadas_dentro_area

# Ejemplo de uso
coordenadas = np.array([[410, 381], [162, 375], [287, 191], [424, 128], [148, 122]])
punto_referencia = np.array([165, 355])
radio = 25

coordenadas[0] = dentro_de_area(coordenadas, coordenadas[1], 25) 

print("coordenadas", coordenadas,"test", coordenadas[0])

coordenadas_dentro = dentro_de_area(coordenadas, punto_referencia, radio)
print(coordenadas_dentro)