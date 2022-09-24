# This is the testing code of our paper and for non-commercial use only.
# X. Fu, and X. Cao " Underwater image enhancement with global-local networks and compressed-histogram equalization",
# Signal Processing: Image Communication, 2020. DOI: 10.1016/j.image.2020.115892


# import tensorflow as tf
import tensorflow.compat.v1 as tf
import numpy as np


def compressedHE(_input):

  _input = tf.squeeze(_input)
  _input = tf.cast(_input*255., tf.int32)

  output = []
  values_range = tf.constant([0, 255], dtype = tf.int32)

  for i in range(3):
    image =  tf.expand_dims(_input[:,:,i], -1)
    histogram = tf.histogram_fixed_width(image, values_range, 256)

    histogram = tf.cast(histogram, tf.float32)
    histogram =  tf.log1p(histogram)
    histogram = histogram/tf.reduce_sum(histogram)

    cdf = tf.cumsum(histogram)

    px_map =  tf.round(cdf  *255.)
    px_map = tf.cast(px_map, tf.int32)

    out = tf.expand_dims(tf.gather_nd(px_map, tf.cast(image, tf.int32)), 2)
    output.append(out)

  final = tf.concat([output[0],output[1],output[2]],-1)
  final = tf.cast(final, tf.float32)/255.

  enhanced = tf.expand_dims(final, 0)

  return enhanced



def GuideBlock(H,miu,in_channels):

    WH = tf.layers.conv2d(H, in_channels, 3, padding="SAME", use_bias=False)
    b = tf.layers.dense(miu, in_channels, use_bias=False)
    out = WH + b

    return out



def Network(images,in_channels = 16):
    with tf.variable_scope('Network',  reuse=tf.compat.v1.AUTO_REUSE): # AUTO_REUSE crea las variables si no existen de tal manera que sean variables independientes

        mean, var = tf.nn.moments(images, [1, 2], keepdims=False)# calcula la media y la varianza de x
        sigma = tf.sqrt(var)
        CONCAT = tf.concat([mean,sigma],-1)

    with tf.variable_scope('avg'):
        h1 = tf.keras.layers.Dense(CONCAT, in_channels)
        h1 = tf.nn.relu(h1)

        h2 = tf.keras.layers.Dense(h1, in_channels)
        h2 = tf.nn.relu(h2)

        h3 = tf.keras.layers.Dense(h2, in_channels)
        h3 = tf.nn.relu(h3)

        h = tf.concat([h1,h2,h3],-1)

        res = tf.keras.layers.Dense(h,3)
        new_mean =  tf.nn.sigmoid(mean + res)


    with tf.variable_scope('local'):
        I_centered = images - mean

        conv1 = GuideBlock(I_centered, res, in_channels)
        conv1 = tf.nn.relu(conv1)

        conv2 = GuideBlock(conv1, res, in_channels)
        conv2 = tf.nn.relu(conv2)

        conv3 = GuideBlock(conv2, res, in_channels)
        conv3 = tf.nn.relu(conv3)

        conv = tf.concat([conv1,conv2,conv3],-1)
        J_centered = GuideBlock(conv,res,3)


    with tf.variable_scope('output'):
        J = tf.nn.relu(J_centered + new_mean)
        J = tf.minimum(J, tf.ones_like(J))

    return  J



if __name__ == '__main__':
    tf.compat.v1.reset_default_graph()

    input_x = tf.compat.v1.random_normal([1,201,201,3])

    output  = Network(input_x)
    var_list = tf.trainable_variables()
    print("Total trainable parameters' number: %d"
         %(np.sum([np.prod(v.get_shape().as_list()) for v in var_list])))
