import pandas as pd
import numpy as np
from trainer import config
import os
import cv2 as cv
import matplotlib.image as mpimg
from pandas.compat import StringIO
from sklearn.model_selection import train_test_split
from tensorflow.python.lib.io import file_io


class Dataset:
    def __init__(self):
        if config.USING_GCP:
            data = pd.read_csv(StringIO(file_io.FileIO("gs://cloogo/parsed_data/labels.csv", mode='r').read()))
        else:
            data = pd.read_csv(os.path.join(config.LABEL_PATH, "labels.csv")).sample(frac=1).head(2000)
        data = data.set_index("ID")
        data = data[config.COLS]

        # labels = np.array(data[data.columns.values].tolist()).reshape(-1, config.LABEL_SIZE)
        if config.USING_GCP:
            images = [mpimg.imread(file_io.FileIO("gs://cloogo/images/%s.png" % _id, mode='r')) for _id in data.index.values]
        else:
            images = [cv.imread(os.path.join(config.IMG_PATH, "%s.png" % _id), flags=cv.IMREAD_COLOR) for _id in
                      data.index.values]
        data["images"] = [np.array(cv.resize(img, config.IMG_DIM),
                                   dtype=np.float32) / 255 if img is not None else None for img in images]
        data = data.dropna()
        print(data)
        train_x, valid_x, train_y, valid_y = train_test_split(
            np.array(data["images"].tolist()).reshape(-1, config.IMG_DIM[0], config.IMG_DIM[1], 3),
            np.array(data.drop("images", axis=1).values, dtype=np.float32).reshape(-1, config.LABEL_SIZE),
            test_size=0.2, random_state=20181110
        )

        # train_x = np.append(train_x, [np.fliplr(x) for x in train_x], axis=0)
        # train_y = np.append(train_y, train_y, axis=0)
        # valid_x = np.append(valid_x, [np.fliplr(x) for x in valid_x], axis=0)
        # valid_y = np.append(valid_y, valid_y, axis=0)

        self.train_batch_count = int(train_x.shape[0] / config.BATCH_SIZE)
        self.valid_batch_count = int(valid_x.shape[0] / config.BATCH_SIZE)
        self.train = (train_x, train_y)
        self.valid = (valid_x, valid_y)


def split_minibatch(np_array, batchsize=config.BATCH_SIZE):
    batches = []
    base = 0
    while base < np_array.shape[0]:
        if base + batchsize >= np_array.shape[0]:
            batches.append(np_array[base: np_array.shape[0]])
        else:
            batches.append(np_array[base: base + batchsize])
        base += batchsize
    return batches