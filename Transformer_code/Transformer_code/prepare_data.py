import ast
import os
import sys
import warnings

import pandas as pd
from pandas.api.types import CategoricalDtype

if not sys.warnoptions:
    warnings.simplefilter("ignore")
import json

import numpy as np
from tensorflow.keras.utils import Sequence

from audio_processing import random_crop, random_mask

class DataGenerator(Sequence):
    def __init__(self, path_x_label_list, class_mapping, n_classes, batch_size=32):
        self.path_x_label_list = path_x_label_list

        self.batch_size = batch_size
        self.indexes = np.arange(len(self.path_x_label_list))
        self.class_mapping = class_mapping
        self.on_epoch_end()
        self.n_classes = n_classes

    def __len__(self):
        return int(np.floor(len(self.path_x_label_list) / self.batch_size / 10))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        batch_samples = [self.path_x_label_list[k] for k in indexes]

        x, y = self.__data_generation(batch_samples)

        return x, y

    def on_epoch_end(self):
        np.random.shuffle(self.indexes)

    def __data_generation(self, batch_samples):
        paths, labels = zip(*batch_samples)

        labels = [labels_to_vector(x, self.class_mapping, self.n_classes) for x in labels]
        crop_low = 128
        crop_high = 256
        # crop_size = np.random.randint(128, 256)
        crop_size = np.random.randint(crop_low, crop_high)

        a = np.array([np.load(x, allow_pickle=True) for x in paths])
        # print("mel",a.shape)
        X = np.array([random_crop(np.load(x, allow_pickle=True), crop_size=crop_size) for x in paths])
        # print("X:",X.shape)
        Y = np.array(labels)

        return X, Y[..., np.newaxis]


class PretrainGenerator(Sequence):
    def __init__(self, path_x_label_list, batch_size=32):
        self.path_x_label_list = path_x_label_list

        self.batch_size = batch_size
        self.indexes = np.arange(len(self.path_x_label_list))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.path_x_label_list) / self.batch_size / 10))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        batch_samples = [self.path_x_label_list[k] for k in indexes]

        x, y = self.__data_generation(batch_samples)

        return x, y

    def on_epoch_end(self):
        np.random.shuffle(self.indexes)

    def __data_generation(self, batch_samples):
        paths, _ = zip(*batch_samples)

        # crop_size = np.random.randint(128, 256)
        crop_size = np.random.randint(10240, 256)


        X = [random_crop(np.load(x, allow_pickle=True), crop_size=crop_size) for x in paths]
        Y = [random_mask(a) for a in X]

        X = np.array(X)
        Y = np.array(Y)

        return X, Y


def get_id_from_path(path):
    base_name = os.path.basename(path)

    return base_name.replace(".mp3", "").replace(".npy", "")

# one hot coding
def labels_to_vector(labels, mapping, n_classes):
    vec = [0] * n_classes
    vec[mapping[labels[0]]] = 1
    return vec


if __name__ == "__main__":
    track_info_path = "/home/xmo/download/music_genre.csv"

    # track name -> popularity
    pop_mapping_path = "/home/xmo/download/pop_mapping.json"
    title_id_path = "/home/xmo/download/title_id.json"

    df = pd.read_csv(track_info_path)
    df.reset_index(inplace=True)

    df['pop_quartile'] = 0
    conditions = [
    (df['popularity'] <= 34),
    (df['popularity'] > 34) & (df['popularity'] <= 45),
    (df['popularity'] > 45) & (df['popularity'] <= 56),
    (df['popularity'] > 56)
    ]
    values = [0, 1, 2, 3]
    df['pop_quartile'] = np.select( conditions , values)

    title_id = {
        k: v
        for k, v in zip(df["track_name"].to_list(), df["instance_id"].to_list())
    }

    pop_mapping = {
        k: v
        for k, v in zip(df["track_name"].tolist(), df["pop_quartile"].tolist())
    }
    # track_id -> genres
    json.dump(pop_mapping, open(pop_mapping_path, "w"), indent=1)
    json.dump(title_id, open(title_id_path, "w"), indent=1)
