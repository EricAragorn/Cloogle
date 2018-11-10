import tensorflow as tf

# Data
IMG_DIM = (500, 500, 3)
LABEL_SIZE = 10

# Model
KERNEL_SIZE = (3, 3)
CNN_INITIALIZER = tf.initializers.he_normal()
