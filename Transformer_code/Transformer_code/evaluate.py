import json
from glob import glob
import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, average_precision_score
from sklearn.model_selection import train_test_split

# from models import rnn_classifier, transformer_classifier
from models import transformer_classifier

from prepare_data import get_id_from_path, labels_to_vector, random_crop
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve


def chunker(seq, size):
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


if __name__ == "__main__":
    from collections import Counter

    transformer_h5 = "transformer.h5"

    batch_size = 128
    epochs = 5
    n_classes = 4

    track_info_path = "/home/xmo/download/music_genre.csv"

    # track name -> popularity
    pop_mapping_path = "/home/xmo/download/pop_mapping.json"
    title_id_path = "/home/xmo/download/title_id.json"
    base_path = "/home/xmo/download/spotify"

    df = pd.read_csv(track_info_path)
    df.reset_index(inplace=True)

    CLASS_MAPPING = json.load(open(pop_mapping_path))

    base_path = "/home/xmo/download/spotify"
    files = sorted(list(glob(base_path + "/*/*.npy")))
 
    labels = [ get_id_from_path(x) for x in files]
    print(len(labels))

    expected_tracks_set = set(df["track_name"].to_list())
    downloaded_tracks_set = set(labels)

    valid_tracks_set = downloaded_tracks_set - (downloaded_tracks_set - expected_tracks_set)
    valid_tracks = list(valid_tracks_set)
    
    track_path_dic = {
        get_id_from_path(v): v for v in files
    }

    valid_tracks_path = [track_path_dic[x] for x in valid_tracks]

    print(len(valid_tracks_path))

    print(len(set(df["track_name"].to_list()) - set(labels)))
    valid_tracks_m = [[x] for x in valid_tracks]
    
    samples = list(zip(valid_tracks_path, valid_tracks_m))

    train, val = train_test_split(
        samples, test_size=0.2, random_state=1337
    )

    transformer_model = transformer_classifier(n_classes=4)

    transformer_model.load_weights(transformer_h5)

    all_labels = []
    transformer_all_preds = []

    for batch_samples in tqdm(
        chunker(val, size=batch_size), total=len(val) // batch_size
    ):
        paths, labels = zip(*batch_samples)

        # music name -> one hot coding
        all_labels += [labels_to_vector(x, CLASS_MAPPING, n_classes=4) for x in labels]

        crop_size = np.random.randint(128, 256)

        # print("crop size:", crop_size)

        repeats = 16

        transformer_Y = 0

        for _ in range(repeats):
            X = np.array([random_crop(np.load(x), crop_size=crop_size) for x in paths])

            transformer_Y += transformer_model.predict(X) / repeats

        # print("shape",transformer_Y.shape)

        transformer_all_preds.extend(transformer_Y.tolist())

        # print("len",len(transformer_all_preds))

    T_Y = np.array(transformer_all_preds)

    Y = np.array(all_labels)

    trsf_ave_auc_pr = 0

    total_sum = 0

    auc_record = {}
    auc_record_path = "/home/xmo/music_genre_classification/auc_record.json"
    precision_path =  "/home/xmo/music_genre_classification/precision"
    recall_path = "/home/xmo/music_genre_classification/recall"

    for i in range(n_classes):
        if np.sum(Y[:, i]) > 0:
            trsf_auc = average_precision_score(Y[:, i], T_Y[:, i])

            print(i, np.sum(Y[:, i]))
            print("transformer   :", trsf_auc)
            print("")

            trsf_ave_auc_pr += np.sum(Y[:, i]) * trsf_auc
            total_sum += np.sum(Y[:, i])

            print(i," PR curve")

            precision, recall, _ = precision_recall_curve(Y[:, i], T_Y[:, i])
            average_precision = average_precision_score(Y[:, i], T_Y[:, i])
            plt.figure()
            plt.step(recall, precision, where="post")

            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.title("Average precision score ".format(average_precision))
            fig_filename = str(i)+".png"
            plt.savefig( os.path.join("/home/xmo/music_popularity_classification/result", fig_filename))

                # np.save(precision_path, precision)
                # np.save(recall_path, recall_path)

    trsf_ave_auc_pr /= total_sum

    print("transformer micro-average     : ", trsf_ave_auc_pr)
