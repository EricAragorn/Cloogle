import tensorflow as tf
import numpy as np
from trainer import config
from trainer.data import Dataset
from trainer.model import res34_v0


def train(TrainedModel = None):
    dataset = Dataset()
    model = res34_v0(dataset)

    print("Start Training...")
    saver = tf.train.Saver()
    save_path = "../saved_models/model_v0.ckpt"
    with tf.Session() as sess:
        writer = tf.summary.FileWriter("../log", sess.graph)
        lr = config.INITIAL_LR

        best_valid_accuracy = 0
        global_steps = 0

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        for epoch in range(config.MAX_EPOCH):
            if (epoch + 1) >= config.LR_DECAY_EPOCH:
                lr *= config.LR_DECAY_RATE

            train_loss, valid_loss = [], []
            train_accuracy, valid_accuracy = [], []
            sess.run(model.train_init_op)
            for step in range(dataset.train_batch_count):
                _, loss, accuracy, summaries = sess.run([model.train_op, model.loss, model.accuracy, model.summaries], feed_dict={model.lr: lr, model.is_training: True})
                train_loss.append(loss)
                train_accuracy.append(accuracy)
                writer.add_summary(summaries, global_step=global_steps)
                global_steps += 1
            sess.run(model.valid_init_op)
            for step in range(dataset.valid_batch_count):
                loss, accuracy = sess.run([model.loss, model.accuracy], feed_dict={model.is_training: False})
                valid_loss.append(loss)
                valid_accuracy.append(accuracy)
            print("Epoch {:d}, Training_loss: {:.4f}, Training_accuracy: {:4f}, Validation_loss: {:.4f}, Validation_accuracy: {:4f}" \
                  .format(epoch + 1, np.mean(train_loss), np.mean(train_accuracy), np.mean(valid_loss), np.mean(valid_accuracy)))

            # save model if current valid score is better than the best
            if np.mean(valid_accuracy) > best_valid_accuracy:
                saver.save(sess, save_path)
                best_valid_accuracy = np.mean(valid_accuracy)


if __name__ == "__main__":
    train()
