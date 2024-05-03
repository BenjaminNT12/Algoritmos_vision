import os
import skimage.io
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import modelo
import time

input_path = '/home/nicolas/Github/Algoritmos_vision/VideoUnderwaterEnhanced/img/input/' # the path of testing images
results_path = '/home/nicolas/Github/Algoritmos_vision/VideoUnderwaterEnhanced/img/output/' # the path of enhanced results

def _parse_function(filename):
    image_string = tf.io.read_file(filename) # Reads the contents of file.
    print("paso 1") 
    print(image_string)
    image_decode = tf.image.decode_png(image_string, channels=3) # Decode a PNG-encoded image to a uint8 or uint16 tensor.
    print("paso 2")
    print(image_string)
    image_decode = tf.image.convert_image_dtype(image_decode, tf.float32) # Convert image to dtype, scaling its values if needed.
    print("paso 3")
    print(image_string)
    return image_decode

if __name__ == '__main__':

    imgName = os.listdir(input_path)
    filename = [os.path.join(input_path, f) for f in os.listdir(input_path)]

    filename_tensor = tf.convert_to_tensor(filename, dtype=tf.string) # Converts the given value to a Tensor

    print(filename_tensor)

    dataset = tf.data.Dataset.from_tensor_slices((filename_tensor))
    print(dataset)
    dataset = dataset.map(_parse_function) # convierte una imagene a un mapeo flotante de 32bits
    print(dataset)
    dataset = dataset.prefetch(buffer_size = 10)
    
    dataset = dataset.batch(1).repeat()
    iterator = iter(dataset)
    underwater = next(iterator)

    output = modelo.Network(underwater)
    output = modelo.compressedHE(output)

    output = tf.clip_by_value(output, 0., 1.)
    final = output[0,:,:,:]

    num_img = len(filename)
    for i in range(num_img):
        tiempo_anterior = time.time()
        enhanced,ori = iterator.get_next()
        enhanced = np.uint8(enhanced* 255.)

        index = imgName[i].rfind('.')
        name = imgName[i][:index]
        print('%d / %d images processed' % (i+1,num_img))
        print("tiempo anterior", tiempo_anterior , "tiempo linea a linea: ",tiempo_anterior - time.time(), "Frecuencia: ",1/(tiempo_anterior - time.time()))
    print('All finished')

    plt.subplot(1,2,1)
    plt.imshow(ori[0,:,:,:])
    plt.title('Underwater')
    plt.subplot(1,2,2)
    plt.imshow(enhanced)
    plt.title('Enhanced')
    plt.show()