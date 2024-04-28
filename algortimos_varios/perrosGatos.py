import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt # importar libreria para poder plotear datos
import cv2
import numpy as np



TAMANO_IMG = 100
setattr(tfds.image_classification.cats_vs_dogs, '_URL',"https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip")
datos, metadatos = tfds.load('cats_vs_dogs', as_supervised=True, with_info=True) # carga el dataset
print(metadatos) # imprime las caracteristicas del dataset

tfds.as_dataframe(datos['train'].take(5), metadatos)
# tfds.show_examples(datos['train'], metadatos) # forma correcta de visualizar los datos de entrenamiento

plt.figure(figsize=(20,20))

for i, (imagen, etiqueta) in enumerate(datos['train'].take(25)):
    imagen = cv2.resize(imagen.numpy(), (TAMANO_IMG, TAMANO_IMG))
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(imagen, cmap='gray')

# plt.show()

datos_entramiento = []

for i, (imagen, etiqueta) in enumerate(datos['train']):
    imagen = cv2.resize(imagen.numpy(), (TAMANO_IMG, TAMANO_IMG)) # redimensiona la imagen
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY) # cambia la imagen a blanco y negro
    imagen = imagen.reshape(TAMANO_IMG, TAMANO_IMG, 1) # cambia el tamano a 100 x 100 x 1
    datos_entramiento.append([imagen, etiqueta])


print(datos_entramiento[0])
print(len(datos_entramiento))

#preparar los datos

x = []
y = []

for imagen, etiqueta in datos_entramiento:
    x.append(imagen) # agrego cada elemento al arreglo X
    y.append(etiqueta)  # agrego cada elemento al arreglo Y

#aqui comienzo a usar numpy

x = np.array(x).astype(float) /255 # se realiza un casting para cambiar el tipo de datos a flotante

# print(x) # aqui el vector x tiene un arreglo de 23262 , 100, 100, 1 debido al casting que se hizo en la parte superior
# print(y) # hasta este punto se tienen puros tensores en el vector Y, aunque no se sabe a ciencia cierta por que?

print(x.shape)
y = np.array(y)
print(y.shape)

#arquitectura redes neuronals convolucionales para clasificacion de datos
# MODELO DE LA RED NEURONAL

modeloDenso = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(100, 100, 1)), # capa de entrada recibiendo los datos de entrada en este caso los pixeles
    tf.keras.layers.Dense(150, activation ='relu'), # capas densas de la red neuronal
    tf.keras.layers.Dense(150, activation ='relu'), # capas densas de la red neuronal
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])

# segundo modelo es una red neuronal convolucional
modeloCNN = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,(3,3), activation='relu', input_shape = (100, 100, 1)), # se agregan 3 pares de capas convolucionales y de agrupacion maxima pasando por 32, 64, y 128 filtros
    tf.keras.layers.MaxPooling2D(2, 2), # capas de agrupacion maxima
    tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2, 2), # capas de agrupacion maxima
    tf.keras.layers.Conv2D(120, (3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation = 'relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])


# tercer modelo es una red neuronal convolucional
modeloCNN2 = tf.keras.models.Sequential([
tf.keras.layers.Conv2D(32,(3,3), activation='relu', input_shape = (100, 100, 1)), # se agregan 3 pares de capas convolucionales y de agrupacion maxima pasando por 32, 64, y 128 filtros
tf.keras.layers.MaxPooling2D(2, 2), # capas de agrupacion maxima
tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'),
tf.keras.layers.MaxPooling2D(2, 2), # capas de agrupacion maxima
tf.keras.layers.Conv2D(120, (3,3), activation = 'relu'),
tf.keras.layers.MaxPooling2D(2, 2),

tf.keras.layers.Dropout(0.5),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(250, activation = 'relu'),
tf.keras.layers.Dense(1, activation = 'sigmoid')
])

modeloDenso.compile(optimizer='adam',  # optimizador Adam revisar
                    loss='binary_crossentropy', # funcion de perdida revisar que es
                    metrics=['accuracy']) # y metricas de presicion

modeloCNN.compile(optimizer='adam',  # optimizador Adam revisar
                    loss='binary_crossentropy', # funcion de perdida revisar que es
                    metrics=['accuracy']) # y metricas de presicion

modeloCNN2.compile(optimizer='adam',  # optimizador Adam revisar
                    loss='binary_crossentropy', # funcion de perdida revisar que es
                    metrics=['accuracy']) # y metricas de presicion

from tensorflow.keras.callbacks import TensorBoard

# tensorboardDenso = TensorBoard(log_dir="logs/denso") # solo para verificar el funcionamiento de la red neuronal
# modeloDenso.fit(x, y, batch_size = 32, # se introducen las dos entradas x que contiene las imagnes y Y que contiene los nombres, con un tamano de lote de 32
#                 validation_split = 0.15, # se separa la cantidad de datos para validacion y pruebas training and testing, para pruebas 15 por ciento
#                 epochs = 100, #  indicamos que queremos 100 epocas
#                 callbacks = [tensorboardDenso])# agregamos un arreglo de callbacks, usamos el callback de tensorboard despues de cada una de las 100 epocas guardara en el archivo el resultado de cada epoca para que lo podamos visualizar despues
#                 # entrenar el modelo con la funciojn fit


tensorboardCNN = TensorBoard(log_dir="logs/CNN") # solo para verificar el funcionamiento de la red neuronal
modeloCNN.fit(x, y, batch_size = 32, # se introducen las dos entradas x que contiene las imagnes y Y que contiene los nombres, con un tamano de lote de 32
                validation_split = 0.15, # se separa la cantidad de datos para validacion y pruebas training and testing, para pruebas 15 por ciento
                epochs = 100, #  indicamos que queremos 100 epocas
                callbacks = [tensorboardCNN])# agregamos un arreglo de callbacks, usamos el callback de tensorboard despues de cada una de las 100 epocas guardara en el archivo el resultado de cada epoca para que lo podamos visualizar despues
                # entrenar el modelo con la funciojn fit


