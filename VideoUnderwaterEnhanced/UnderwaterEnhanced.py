from dataclasses import dataclass
import os
from matplotlib import image
import skimage.io
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as v1
import matplotlib.pyplot as plt
import modelo

# tf.reset_default_graph() # Clears the default graph stack and resets the global default graph.
# por el momento no se requiere, ya que para resetear un grafico se realiza de manera diferente  

input_path = 'C:/Users/benja/GitHubVsCode/Algoritmos_vision/VideoUnderwaterEnhanced/img/input/' # the path of testing images
results_path = 'C:/Users/benja/GitHubVsCode/Algoritmos_vision/VideoUnderwaterEnhanced/img/output/' # the path of enhanced results

def _parse_function(filename):
    image_string = tf.io.read_file(filename) # Reads the contents of file. 
    image_decode = tf.image.decode_png(image_string, channels=3) # Decode a PNG-encoded image to a uint8 or uint16 tensor.
    image_decode = tf.image.convert_image_dtype(image_decode, tf.float32) # Convert image to dtype, scaling its values if needed.
    return image_decode

# @tf.function
# def video_enhanced():
    

if __name__ == '__main__':

    filename = os.listdir(input_path) # cargamos el directorio de donde se encuentras las imagenes de entrada

    for i in range(len(filename)): # hacemos un ciclo de la longitud del numero de archivos que estan dentro del directorio
        print(filename[i]) # imprime el nombre de los archivos
        #hacemos la suma del nombre de la entrada con el nombre del archivo
        filename[i] = input_path + filename[i]
        print(filename[i])

    print(type(filename))
    # ahora convertimos la lista en tensores de tensorflow

    filename_tensor = tf.convert_to_tensor(filename, dtype=tf.string) # Converts the given value to a Tensor
    print("imprimir tensores")
    print(filename_tensor)
# The simplest way to create a dataset is to create it from a python list
# Crea el dataset para el entrenamientoxz
    dataset = tf.data.Dataset.from_tensor_slices((filename_tensor))
    dataset = dataset.map(_parse_function) # convierte una imagene a un mapeo flotante de 32bits
    # .prefetch This allows later elements to be prepared while the current element is being processed
    dataset = dataset.prefetch(buffer_size = 10)
    dataset = dataset.repeat(1)
    # In TF 2 datasets are Python iterables which means you can consume their 
    # elements using for elem in dataset: ... or by explicitly creating iterator 
    # via iterator = iter(dataset) and fetching its elements via values = next(iterator).
    iterator = iter(dataset)
    print("iterador")
    print(iterator.get_next())
    underwater = iterator.get_next()
    print("underwater")
##############################################
    output = modelo.Network(underwater)
    print("output")
    output = modelo.compressedHE(output)
##############################################    
    # output = model.Network(underwater)
    # output = model.compressedHE(output)

    # tf.clip_by_value(
    #     t, clip_value_min, clip_value_max, name=None
    # )

    # Given a tensor t, this operation returns a tensor of the same type and shape 
    # as t with its values clipped to clip_value_min and clip_value_max. Any values 
    # less than clip_value_min are set to clip_value_min. Any values greater than
    # clip_value_max are set to clip_value_max.

    output = tf.clip_by_value(output, 0., 1.)
    print(output)
    final = output[0, :, :, :]

###################################################
    config = v1.ConfigProto()
    config.gpu_options.allow_growth=True

    with v1.Session(config=config) as sess:

        print ("Loading model")
        all_vars = v1.trainable_variables()
        all_vars = v1.train.Saver(var_list = all_vars)
        all_vars.restore(sess,'C:/Users/benja/GitHubVsCode/Algoritmos_vision/VideoUnderwaterEnhanced/model/model')

        num_img = len(filename)
        for i in range(num_img):

            enhanced,ori = sess.run([final,underwater])
            enhanced = np.uint8(enhanced* 255.)

            index = imgName[i].rfind('.')
            name = imgName[i][:index]
            skimage.io.imsave(results_path + name +'.png', enhanced)
            print('%d / %d images processed' % (i+1,num_img))

        print('All finished')
    sess.close()
###################################################


    plt.subplot(1,2,1)
    plt.imshow(ori[0,:,:,:])
    plt.title('Underwater')
    plt.subplot(1,2,2)
    plt.imshow(enhanced)
    plt.title('Enhanced')
    plt.show()

    # _parse_function()