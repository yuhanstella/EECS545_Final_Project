import json
from glob import glob

import numpy as np
import pandas as pd

from models import transformer_classifier
from prepare_data import random_crop
from audio_processing import load_audio_file


def chunker(seq, size):
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


if __name__ == "__main__":

    transformer_h5 = "transformer.h5"

    track_info_path = "/home/xmo/download/music_genre.csv"

    # track name -> popularity
    pop_mapping_path = "/home/xmo/download/pop_mapping.json"
    title_id_path = "/home/xmo/download/title_id.json"
    base_path = "/home/xmo/download/spotify"

    df = pd.read_csv(track_info_path)
    df.reset_index(inplace=True)

    CLASS_MAPPING = json.load(open(pop_mapping_path))

    # test audio
    base_path = "../audio"
    files = sorted(list(glob(base_path + "/*.mp3")))

    data = [load_audio_file(x, input_length=16000 * 120) for x in files]

    transformer_model = transformer_classifier(n_classes=len(CLASS_MAPPING))

    transformer_model.load_weights(transformer_h5)

    crop_size = np.random.randint(128, 512)
    repeats = 8

    transformer_Y = 0

    for _ in range(repeats):
        X = np.array([random_crop(x, crop_size=crop_size) for x in data])

        transformer_Y += transformer_model.predict(X) / repeats

    transformer_Y = transformer_Y.tolist()

    for path, pred in zip(files, transformer_Y):

        print(path)
        pred_tup = [(k, pred[v]) for k, v in CLASS_MAPPING.items()]
        pred_tup.sort(key=lambda x: x[1], reverse=True)

        for a in pred_tup[:5]:
            print(a)
