import os
import skimage.io
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import model

tf.reset_default_graph()

input_path = '/home/nicolas/github/Algoritmos_vision/code/img/input/' # the path of testing images
results_path = '/home/nicolas/github/Algoritmos_vision/code/img/output/' # the path of enhanced results

def _parse_function(filename):
  image_string  = tf.read_file(filename)
  image_decoded = tf.image.decode_png(image_string, channels = 3)
  image_decoded = tf.image.convert_image_dtype(image_decoded, tf.float32)
  return image_decoded

if __name__ == '__main__':

   imgName = os.listdir(input_path)

   filename = os.listdir(input_path)

   print("filename")
   print(filename)
   print("filename2")

   for i in range(len(filename)):
        print(filename[i]) # imprime el nombre de los archivos
        #hacemos la suma del nombre de la entrada con el nombre del archivo
        filename[i] = input_path + filename[i]
        print(filename[i])


   print(type(filename))

   filename_tensor = tf.convert_to_tensor(filename, dtype=tf.string)

   print(type(filename_tensor))

   print("filename_tensor")
   print(filename_tensor)
   print("filename_tensor2")

   dataset = tf.data.Dataset.from_tensor_slices((filename_tensor))
   dataset = dataset.map(_parse_function)
   dataset = dataset.prefetch(buffer_size = 10)
   dataset = dataset.batch(1).repeat()
   iterator = dataset.make_one_shot_iterator()
   
   print("iterator")
   print(iterator)
   print("iterator2")

   underwater = iterator.get_next()


   print("underwater")
   print(underwater)
   print("underwater2")

   output = model.Network(underwater)
   output = model.compressedHE(output)

   output = tf.clip_by_value(output, 0., 1.)
   print("output")
   print(output)
   final = output[0,:,:,:]
   print("output2")
   config = tf.ConfigProto()
   config.gpu_options.allow_growth=True

   with tf.Session(config=config) as sess:

        print ("Loading model")
        all_vars = tf.trainable_variables()
        print("all_vars")
        print(all_vars)
        print("all_vars2")
        all_vars = tf.train.Saver(var_list = all_vars)
        print("all_var2")
        print(all_vars)
        print("all_vars3")
        all_vars.restore(sess,'/home/nicolas/github/Algoritmos_vision/code/model/model')
        print("all_vars4")
        print(all_vars)
        print("all_vars5")

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
