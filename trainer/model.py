import tensorflow as tf
import numpy as np
from trainer import config


class res34_v0:
    def __init__(self, dataset):
        img_dim1, img_dim2 = config.IMG_DIM
        train_x, train_y = dataset.train
        valid_x, valid_y = dataset.valid
        self.test_x, self.test_y = tf.placeholder(dtype=tf.float32, shape=(None, config.IMG_DIM[0], config.IMG_DIM[1], 3)), \
                         tf.placeholder(dtype=tf.float32, shape=(None, config.LABEL_SIZE))
        train_set = tf.data.Dataset.from_tensor_slices((train_x, train_y)) \
            .batch(config.BATCH_SIZE)
        valid_set = tf.data.Dataset.from_tensor_slices((valid_x, valid_y)) \
            .batch(config.BATCH_SIZE)
        test_set = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(1)
        iter = tf.data.Iterator.from_structure(train_set.output_types, train_set.output_shapes)
        self.train_init_op = iter.make_initializer(train_set)
        self.valid_init_op = iter.make_initializer(valid_set)
        self.test_init_op = iter.make_initializer(test_set)
        _input, _target = iter.get_next()
        self.input = _input
        self.target = _target
        self.is_training = tf.placeholder(dtype=tf.bool, shape=[])
        self.lr = tf.placeholder(dtype=tf.float32, shape=[])

        with tf.variable_scope("init_conv"):
            init_conv = tf.layers.conv2d(inputs=self.input,
                                         filters=32,
                                         kernel_size=(5, 5),
                                         kernel_initializer=config.CNN_INITIALIZER,
                                         activation=tf.nn.leaky_relu)

        with tf.variable_scope("res1"):
            pool1 = tf.layers.max_pooling2d(inputs=init_conv, pool_size=(2, 2), strides=(2, 2))
            dropout1 = tf.layers.dropout(pool1, training=self.is_training)
            conv1 = tf.layers.conv2d(inputs=dropout1,
                                     kernel_size=(3, 3),
                                     kernel_initializer=config.CNN_INITIALIZER,
                                     filters=32)
            res1 = stacked_res_blocks(inputs=conv1,
                                      kernel_size=config.KERNEL_SIZE,
                                      filters=32,
                                      count=2,
                                      is_training=self.is_training)

        with tf.variable_scope("res2"):
            pool2 = tf.layers.max_pooling2d(inputs=res1, pool_size=(2, 2), strides=(2, 2))
            dropout2 = tf.layers.dropout(pool2, training=self.is_training)
            conv2 = tf.layers.conv2d(inputs=dropout2,
                                     kernel_size=(3, 3),
                                     kernel_initializer=config.CNN_INITIALIZER,
                                     filters=64)
            res2 = stacked_res_blocks(inputs=conv2,
                                      kernel_size=config.KERNEL_SIZE,
                                      filters=64,
                                      count=2,
                                      is_training=self.is_training)

        with tf.variable_scope("res3"):
            pool3 = tf.layers.max_pooling2d(inputs=res2, pool_size=(2, 2), strides=(2, 2))
            dropout3 = tf.layers.dropout(pool3, training=self.is_training)
            conv3 = tf.layers.conv2d(inputs=dropout3,
                                     kernel_size=(3, 3),
                                     kernel_initializer=config.CNN_INITIALIZER,
                                     filters=64)
            res3 = stacked_res_blocks(inputs=conv3,
                                      kernel_size=config.KERNEL_SIZE,
                                      filters=64,
                                      count=3,
                                      is_training=self.is_training)

        with tf.variable_scope("res4"):
            pool4 = tf.layers.max_pooling2d(res3, pool_size=(2, 2), strides=(2, 2))
            dropout4 = tf.layers.dropout(pool4, training=self.is_training)
            conv4 = tf.layers.conv2d(inputs=dropout4,
                                     kernel_size=(3, 3),
                                     kernel_initializer=config.CNN_INITIALIZER,
                                     filters=128)
            res4 = stacked_res_blocks(inputs=conv4,
                                      kernel_size=config.KERNEL_SIZE,
                                      filters=128,
                                      count=3,
                                      is_training=self.is_training)

        global_average_pooling = [tf.reduce_mean(res1, axis=(1, 2)), tf.reduce_mean(res3, axis=(1, 2)),
                                  tf.reduce_mean(res4, axis=(1, 2))]
        feature_vect = tf.concat(global_average_pooling, axis=1)
        # flatten = tf.layers.flatten(res4)

        hid = tf.layers.dense(feature_vect, units=1000, activation=tf.nn.relu)

        self.output = tf.layers.dense(hid, config.LABEL_SIZE)
        pred_classes = tf.split(self.output, config.CATEGORY_COUNT, axis=1)
        target_classes = tf.split(self.target, config.CATEGORY_COUNT, axis=1)
        self.loss = tf.add_n([tf.reduce_mean(tf.losses.softmax_cross_entropy(target_classes[i], pred_classes[i])) for i in range(len(pred_classes))])
        tf.summary.scalar("Loss", self.loss)
        self.prediction = tf.concat([tf.one_hot(tf.argmax(pred, axis=1), pred.shape[1]) for pred in pred_classes], axis=1)
        print(self.prediction.shape)
        self.accuracy = [tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tf.nn.softmax(pred_classes[i]), axis=1), tf.argmax(target_classes[i], axis=1)), tf.float32)) for i in range(len(pred_classes))]
        # tf.summary.scalar("Accuracy", self.accuracy)
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
        self.summaries = tf.summary.merge_all()


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
    with tf.variable_scope("ResBlock{:d}".format(block_id)):
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
