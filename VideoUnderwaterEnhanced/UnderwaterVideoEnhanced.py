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

if __name__ == '__main__':

    path = 'C:/Users/benja/GitHubVsCode/Algoritmos_vision/video1.mp4'
    # path = '/home/nicolas/github/Algoritmos_vision/video1.mp4'
    video = cv.VideoCapture(path)

    config = v1.ConfigProto()
    config.gpu_options.allow_growth=False

    firs_time = True
    while True:

        _, frame = video.read()

        filename_tensor = v1.convert_to_tensor(frame)
        tensor = tf.expand_dims(filename_tensor , 0) # paso necesario para completar el tensor, agrega las dimensiones necesairas
        dataset = v1.data.Dataset.from_tensor_slices((tensor))
        dataset = dataset.map(_parse_function) # convierte una imagene a un mapeo flotante de 32bits
        dataset = dataset.prefetch(buffer_size = 1)        
        dataset = dataset.batch(1).repeat()
        iterator = dataset.make_one_shot_iterator()
        underwater = iterator.get_next()
        ##############################################
        output = modelo2.Network(underwater)
        output = modelo2.compressedHE(underwater)
        ##############################################    
        output = tf.clip_by_value(output, 0., 1.)
        final = output[0,:,:,:]
        ###################################################

        tiempo_previo = time.time()
        with v1.Session(config = config) as sess:
            all_vars = v1.trainable_variables()
            all_vars = v1.train.Saver(var_list = all_vars)
            # all_vars.restore(sess, '/home/nicolas/github/Algoritmos_vision/VideoUnderwaterEnhanced/model/model') # ubuntu
            all_vars.restore(sess,'C:/Users/benja/GitHubVsCode/Algoritmos_vision/VideoUnderwaterEnhanced/model/model') # windows
            enhanced, ori = sess.run([final,underwater])
            enhanced = np.uint8(enhanced* 255.)    
        sess.close()

        print("tiempo linea a linea: ",tiempo_previo - time.time(), "Frecuencia: ",1/(tiempo_previo - time.time()))

        cv.imshow("frame", enhanced)
        key = cv.waitKey(1)
        if key == 27:
            break

    video.release()
    cv.destroyAllWindows()