import cv2 as cv
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as v1
import modelo2
import time

#chinga tu madre

tiempo_anterior = 0.0
periodo = 0.0
frecuencia = 0.0

v1.disable_eager_execution()
v1.reset_default_graph()


def _parse_function(filename):
    image_decode = tf.image.convert_image_dtype(filename, tf.float32) # Convert image to dtype, scaling its values if needed.
    return image_decode

def image_to_tensor(image):
    filename_tensor = v1.convert_to_tensor(image)
    tensor = tf.expand_dims(filename_tensor , 0) # paso necesario para completar el tensor, agrega las dimensiones necesairas
    dataset = v1.data.Dataset.from_tensor_slices((tensor))
    dataset = dataset.map(_parse_function) # convierte una imagene a un mapeo flotante de 32bits
    dataset = dataset.prefetch(buffer_size = 1)        
    dataset = dataset.batch(1).repeat()
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()

if __name__ == '__main__':

    # path = 'C:/Users/benja/GitHubVsCode/Algoritmos_vision/video1.mp4'
    # path = '/home/nicolas/github/Algoritmos_vision/video1.mp4'
    path = '/home/nicolas/Github/Algoritmos_vision/video1.mp4'
    video = cv.VideoCapture(path)

    # tiempo_previo = time.time()
    # print("tiempo anterior", tiempo_previo , "tiempo linea a linea: ",tiempo_previo - time.time(), "Frecuencia: ",1/(tiempo_previo - time.time()))

    config = v1.ConfigProto()
    config.gpu_options.allow_growth=False

    firs_time = True
    with v1.Session(config = config) as sess:
        while True:
            tiempo_anterior = time.time()
            _, frame = video.read()
            
            underwater = image_to_tensor(frame)
        
            output = modelo2.Network(underwater)
            output = modelo2.compressedHE(output)
            
            if firs_time == True:
                all_vars = v1.trainable_variables()
                all_vars = v1.train.Saver(var_list = all_vars)
                all_vars.restore(sess, '/home/nicolas/Github/Algoritmos_vision/VideoUnderwaterEnhanced/model/model')
                # all_vars.restore(sess, '/home/nicolas/github/Algoritmos_vision/VideoUnderwaterEnhanced/model/model')
                # all_vars.restore(sess,'C:/Users/benja/GitHubVsCode/Algoritmos_vision/VideoUnderwaterEnhanced/model/model') # windows
                print("first time")
                firs_time = False
                

            output = tf.clip_by_value(output, 0., 1.) # escala los valores del tensor entre .0 y .1
            final = output[0,:,:,:]
            
            enhanced, ori = sess.run([final, underwater])
            enhanced = np.uint8(enhanced* 255.)
            print("tiempo anterior", tiempo_anterior , "tiempo linea a linea: ",tiempo_anterior - time.time(), "Frecuencia: ",1/(tiempo_anterior - time.time()))

            cv.imshow("enhanced", enhanced)
            key = cv.waitKey(1)
            if key == 27:
                break

    sess.close()

    video.release()
    cv.destroyAllWindows()


