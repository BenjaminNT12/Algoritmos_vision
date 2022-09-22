# This is the testing code of our paper and for non-commercial use only. 
# X. Fu, and X. Cao " Underwater image enhancement with global-local networks and compressed-histogram equalization", 
# Signal Processing: Image Communication, 2020. DOI: 10.1016/j.image.2020.115892

import os
import skimage.io
import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf # Para version 2 de tensorflow 
import matplotlib.pyplot as plt
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import model
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# tf.reset_default_graph()


input_path = '/home/nicolas/Documentos/GitHub/Algoritmos_vision/code/img/input/' # the path of testing images

results_path = '/home/nicolas/Documentos/GitHub/Algoritmos_vision/code/img/output/' # the path of enhanced results



def _parse_function(filename):      
  image_string  = tf.io.read_file(filename)  
  image_decoded = tf.image.decode_png(image_string, channels = 3)  
  image_decoded = tf.image.convert_image_dtype(image_decoded, tf.float32) 
  return image_decoded 


if __name__ == '__main__':

   imgName = os.listdir(input_path)
   
   filename = os.listdir(input_path)
   for i in range(len(filename)):
      filename[i] = input_path + filename[i]
      
   filename_tensor = tf.convert_to_tensor(filename, dtype=tf.string)  
        
   dataset = tf.data.Dataset.from_tensor_slices((filename_tensor))
   dataset = dataset.map(_parse_function)    
   dataset = dataset.prefetch(buffer_size = 10)
   dataset = dataset.batch(1).repeat()  
   # iterator = dataset.make_one_shot_iterator()
   iterator = iter(dataset)
   # iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
   # iterator = eval_input_fn()
   
   underwater = iterator.get_next()    
   # underwater = next(iterator)
   output = model.Network(underwater)
   output = model.compressedHE(output)
   output = tf.clip_by_value(output, 0., 1.)
   final = output[0,:,:,:]

   config = tf.ConfigProto()
   config.gpu_options.allow_growth=True

   with tf.Session(config=config) as sess:
       
        print ("Loading model")
        all_vars = tf.trainable_variables() 
        all_vars = tf.train.Saver(var_list = all_vars)
        all_vars.restore(sess,'./model/model')     

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