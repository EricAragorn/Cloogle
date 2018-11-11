import tensorflow as tf

# Data
LABEL_PATH = "parsed_data"
IMG_PATH = "images"
IMG_DIM = (330, 495)
LABEL_SIZE = 46

# Model
KERNEL_SIZE = (3, 3)
CNN_INITIALIZER = tf.initializers.he_normal()

# Training
BATCH_SIZE = 32
MAX_EPOCH = 50
INITIAL_LR = 2e-5
LR_DECAY_RATE = 0.95
LR_DECAY_EPOCH = 30
