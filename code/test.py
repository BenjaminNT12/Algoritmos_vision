# This is the testing code of our paper and for non-commercial use only.
# X. Fu, and X. Cao " Underwater image enhancement with global-local networks and compressed-histogram equalization",
# Signal Processing: Image Communication, 2020. DOI: 10.1016/j.image.2020.115892

import os
import skimage.io
import numpy as np
import tensorflow.compat.v1 as tf # Para version 2 de tensorflow
import matplotlib.pyplot as plt
import model

input_path = '/home/nicolas/github/Algoritmos_vision/code/img/input/' # the path of testing images
results_path = '/home/nicolas/github/Algoritmos_vision/code/img/output/' # the path of enhanced results

def _parse_function(filename):
  image_string  = tf.io.read_file(filename)
  image_decoded = tf.image.decode_png(image_string, channels = 3)
  image_decoded = tf.image.convert_image_dtype(image_decoded, tf.float32)
  return image_decoded


if __name__ == '__main__':

   imgName = os.listdir(input_path)

   filename = os.listdir(input_path)
   print("filename")
   for i in range(len(filename)):
      filename[i] = input_path + filename[i]
      print(filename[i])

   print(type(filename))
   filename_tensor = tf.convert_to_tensor(filename, dtype=tf.string)
   print("filename_tensor")
   print(filename_tensor)
   dataset = tf.data.Dataset.from_tensor_slices((filename_tensor))

   print("element dataset")

   for element in dataset:
       print(element)

   dataset = dataset.map(_parse_function)
   print("element dataset after map")

   for element in dataset:
      print(element)
   dataset = dataset.prefetch(buffer_size = 10) #despues de crean el dataset se debe crear un prefetch
   # permite al siguiente elemento estar preparado mientras el elemento anterior esta siendo procesado
   print("prefetch dataset")
   for element in dataset:
       print(element)
   dataset = dataset.batch(1).repeat() # convina elementos consecutivos de este dataset en batches
   # iterator = dataset.make_one_shot_iterator()
   print("batch")
   # for element in dataset:
   #     print(element)
   iterator = iter(dataset) # enumera los elementos del dataset mismos que son los que meteremos dentro de un for loop

   print(iterator)
   # iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
   # iterator = eval_input_fn()

   underwater = iterator.get_next()
   # underwater = next(iterator)

   output = model.Network(underwater)
   print("llego hasta aqui 1")
   output = model.compressedHE(output)
   print("llego hasta aqui 2")
   output = tf.clip_by_value(output, 0., 1.)
   final = output[0,:,:,:]

   config = tf.ConfigProto()
   config.gpu_options.allow_growth=False
   print("llego hasta aqui 6")
   with tf.Session(config=config) as sess:

        print ("Loading model")
        all_vars = tf.trainable_variables()
        print("llego hasta aqui 7")
        print(all_vars)
        all_vars = tf.keras.Model.save_weights(var_list = all_vars)
        print("llego hasta aqui 8")
        all_vars.restore(sess,'/home/nicolas/github/Algoritmos_vision/code/model/model')

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

   plt.subplot(1,2,1)
   plt.imshow(ori[0,:,:,:])
   plt.title('Underwater')
   plt.subplot(1,2,2)
   plt.imshow(enhanced)
   plt.title('Enhanced')
   plt.show()
