import os
import string
import cv2 as cv
import skimage.io
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as v1
import matplotlib.pyplot as plt
import modelo2
import scipy.misc


v1.disable_eager_execution()
v1.reset_default_graph()

# Windows
# input_path = 'C:/Users/benja/GitHubVsCode/Algoritmos_vision/VideoUnderwaterEnhanced/img/input/' # the path of testing images
# results_path = 'C:/Users/benja/GitHubVsCode/Algoritmos_vision/VideoUnderwaterEnhanced/img/output/' # the path of enhanced results

# Ubuntu
# input_path = '/home/nicolas/github/Algoritmos_vision/VideoUnderwaterEnhanced/img/input/' # the path of testing images
# results_path = '/home/nicolas/github/Algoritmos_vision/VideoUnderwaterEnhanced/img/output/' # the path of enhanced results


def _parse_function(filename):
    image_decode = tf.image.convert_image_dtype(filename, tf.float32) # Convert image to dtype, scaling its values if needed.
    return image_decode

if __name__ == '__main__':

    path = r'C:/Users/benja/GitHubVsCode/Algoritmos_vision/video3.mp4'
    video = cv.VideoCapture(path)

    while True:
        _, frame = video.read()
        # frame = cv.cvtColor(frame2, cv.COLOR_BAYER_BG2GRAY)

        scale_percent = 40 # percent of original size
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)
        
        # resize image
        frame = cv.resize(frame, dim, interpolation = cv.INTER_AREA)


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
        output = modelo2.compressedHE(output)
        ##############################################    
        output = tf.clip_by_value(output, 0., 1.)
        final = output[0,:,:,:]
        ###################################################
        config = v1.ConfigProto()
        config.gpu_options.allow_growth=True

        with v1.Session(config=config) as sess:
            all_vars = v1.trainable_variables()
            all_vars = v1.train.Saver(var_list = all_vars)
            all_vars.restore(sess,'C:/Users/benja/GitHubVsCode/Algoritmos_vision/VideoUnderwaterEnhanced/model/model')
            enhanced,ori = sess.run([final,underwater])
            enhanced = np.uint8(enhanced* 255.)
            # print('All finished')
        sess.close()

        cv.imshow("frame", enhanced)
        key = cv.waitKey(1)
        if key == 27:
            break

    video.release()
    cv.destroyAllWindows()