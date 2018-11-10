import tensorflow as tf
import numpy as np
import config


class res34_v0:
    def __init__(self):
        img_dim1, img_dim2, img_dim3 = config.IMG_DIM
        self.input = tf.placeholder(dtype=tf.float32, shape=(None, img_dim1, img_dim2, img_dim3))
        self.target = tf.placeholder(dtype=tf.float32, shape=(None, config.LABEL_SIZE))
        self.is_training = tf.placeholder(dtype=tf.bool, shape=[])
        self.lr = tf.placeholder(dtype=tf.float32, shape=[])

        with tf.variable_scope("init_conv"):
            init_conv = tf.layers.conv2d(inputs=self.input,
                                         filters=32,
                                         kernel_size=(7, 7),
                                         activation=tf.nn.leaky_relu)

        with tf.variable_scope("res1"):
            pool1 = tf.layers.max_pooling2d(inputs=init_conv, pool_size=(2, 2), strides=(2, 2))
            conv1 = tf.layers.conv2d(inputs=pool1,
                                     kernel_size=(3, 3),
                                     filters=32)
            res1 = stacked_res_blocks(inputs=conv1,
                                      kernel_size=config.KERNEL_SIZE,
                                      filters=32,
                                      count=2,
                                      is_training=self.is_training)

        with tf.variable_scope("res2"):
            pool2 = tf.layers.max_pooling2d(inputs=res1, pool_size=(2, 2), strides=(2, 2))
            conv2 = tf.layers.conv2d(inputs=pool2,
                                     kernel_size=(3, 3),
                                     filters=64)
            res2 = stacked_res_blocks(inputs=conv2,
                                      kernel_size=config.KERNEL_SIZE,
                                      filters=64,
                                      count=2,
                                      is_training=self.is_training)

        with tf.variable_scope("res3"):
            pool3 = tf.layers.max_pooling2d(inputs=res2, pool_size=(2, 2), strides=(2, 2))
            conv3 = tf.layers.conv2d(inputs=pool3,
                                     kernel_size=(3, 3),
                                     filters=64)
            res3 = stacked_res_blocks(inputs=conv3,
                                      kernel_size=config.KERNEL_SIZE,
                                      filters=64,
                                      count=3,
                                      is_training=self.is_training)

        with tf.variable_scope("res4"):
            pool4 = tf.layers.max_pooling2d(res3, pool_size=(2, 2), strides=(2, 2))
            conv4 = tf.layers.conv2d(inputs=pool4,
                                     kernel_size=(3, 3),
                                     filters=128)
            res4 = stacked_res_blocks(inputs=conv4,
                                      kernel_size=config.KERNEL_SIZE,
                                      filters=128,
                                      count=3,
                                      is_training=self.is_training)

        global_average_pooling = [tf.reduce_mean(res1, axis=(1, 2)), tf.reduce_mean(res3, axis=(1, 2)),
                                  tf.reduce_mean(res4, axis=(1, 2))]
        feature_vect = tf.concat(global_average_pooling, axis=1)

        self.output = tf.layers.dense(feature_vect, config.LABEL_SIZE)
        self.loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(self.target, self.output))
        self.prediction = tf.nn.sigmoid(self.output)
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def train(self, sess, image, target):
        _, loss = sess.run([self.train_op, self.loss], feed_dict={self.input: image, self.target: target})
        return loss

    def eval(self, sess, image, target):
        loss = sess.run([self.loss], feed_dict={self.input: image, self.target: target})
        return loss


def stacked_res_blocks(inputs, kernel_size, filters, count, is_training, bottleneck_filters=None, type="resblock"):
    if count < 1:
        raise ValueError("The number of stacked residual blocks should be positive")

    if type == "resblock":
        # Original ResBlock
        last_block = inputs
        for i in range(count - 1):
            last_block = resBlock(inputs=last_block,
                                  kernel_size=kernel_size,
                                  filters=filters,
                                  is_training=is_training,
                                  block_id=i)
            last_block = resBlock(inputs=last_block,
                                  kernel_size=kernel_size,
                                  filters=filters,
                                  is_training=is_training,
                                  block_id=count - 1,
                                  activation=True)
            return last_block


def resBlock(inputs, kernel_size, filters, block_id, is_training, strides=(1, 1), activation=False):
    with tf.variable_scope(f"ResBlock{block_id}"):
        bn_ac1 = bn_activation(inputs, is_training)
        conv1 = tf.layers.conv2d(inputs=bn_ac1,
                                 filters=filters,
                                 kernel_size=kernel_size,
                                 kernel_initializer=config.CNN_INITIALIZER,
                                 strides=strides,
                                 padding="same",
                                 name="conv1")
        bn_ac2 = bn_activation(conv1, is_training)
        conv2 = tf.layers.conv2d(inputs=bn_ac2,
                                 filters=filters,
                                 kernel_size=kernel_size,
                                 kernel_initializer=config.CNN_INITIALIZER,
                                 strides=strides,
                                 padding="same",
                                 name="conv2")
        output = conv2 + inputs
        if activation:
            output = tf.layers.batch_normalization(output, name="output_bn")
            output = tf.nn.leaky_relu(output, name="output_activation")
        return output


def bn_activation(inputs, is_training=True):
    bn = tf.layers.batch_normalization(inputs, training=is_training)
    return tf.nn.leaky_relu(bn)
