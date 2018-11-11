import tensorflow as tf
from trainer.data import Dataset
from trainer.model import res34_v0

dataset = Dataset()
model = res34_v0(dataset)
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, "saved_models/model_v2.ckpt")

    sess.run(model.test_init_op, feed_dict={model.test_x: <img>, model.test_y: np.zeros<sameLength as IMG>})
    prediction = sess.run(model.prediction, feed_dict={model.is_training: False})