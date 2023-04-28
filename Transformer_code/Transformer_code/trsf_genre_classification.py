import json
from glob import glob

from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import pandas as pd

from models import transformer_classifier
from prepare_data import get_id_from_path, DataGenerator

if __name__ == "__main__":
    from collections import Counter

    h5_name = "transformer.h5"
    h5_pretrain_name = "transformer_pretrain.h5"
    batch_size = 32
    epochs = 50
    n_classes = 4    

    track_info_path = "/home/xmo/download/music_genre.csv"

    # track name -> popularity
    pop_mapping_path = "/home/xmo/download/pop_mapping.json"
    title_id_path = "/home/xmo/download/title_id.json"
    base_path = "/home/xmo/download/spotify"

    df = pd.read_csv(track_info_path)
    df.reset_index(inplace=True)

    CLASS_MAPPING = json.load(open(pop_mapping_path))
    # id_to_genres = json.load(open("/home/xmo/download/fma_metadata/tracks_genre.json"))
    # id_to_genres = {int(k): v for k, v in id_to_genres.items()}

    base_path = "/home/xmo/download/spotify"
    files = sorted(list(glob(base_path + "/*/*.npy")))
    # files = [x for x in files if [int(get_id_from_path(x))]]
    labels = [ get_id_from_path(x) for x in files]
    print(len(labels))

    expected_tracks_set = set(df["track_name"].to_list())
    downloaded_tracks_set = set(labels)

    valid_tracks_set = downloaded_tracks_set - (downloaded_tracks_set - expected_tracks_set)
    valid_tracks = list(valid_tracks_set)
    
    track_path_dic = {
        get_id_from_path(v): v for v in files
    }

    # print(len(valid_tracks))

    valid_tracks_path = [track_path_dic[x] for x in valid_tracks]

    print(len(valid_tracks_path))
    # print(valid_tracks_path[:10])

    # print(len(set(df["track_name"].to_list()) - set(labels)))
    valid_tracks_m = [[x] for x in valid_tracks]
    
    samples = list(zip(valid_tracks_path, valid_tracks_m))

    # print(samples[:3])
    # strat = [a[-1] for a in labels]
    # cnt = Counter(strat)
    # strat = [a if cnt[a] > 2 else "" for a in strat]

    train, val = train_test_split(
        samples, test_size=0.2, random_state=1337
    )


    model = transformer_classifier(n_classes=n_classes)

    try:
        model.load_weights(h5_pretrain_name, by_name=True)
        print("Weights Loaded")
    except:
        print("Could not load weights")

    checkpoint = ModelCheckpoint(
        h5_name,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        mode="min",
        save_weights_only=True,
    )
    reduce_o_p = ReduceLROnPlateau(
        monitor="val_loss", patience=10, min_lr=1e-7, mode="min"
    )

    model.fit_generator(
        DataGenerator(train, batch_size=batch_size, class_mapping=CLASS_MAPPING, n_classes=n_classes),
        validation_data=DataGenerator(
            val, batch_size=batch_size, class_mapping=CLASS_MAPPING, n_classes=n_classes
        ),
        epochs=epochs,
        callbacks=[checkpoint, reduce_o_p],
        use_multiprocessing=True,
        workers=12,
        verbose=2,
        max_queue_size=64,
    )
