import json
from glob import glob

from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import pandas as pd

from models import transformer_pretrain
from prepare_data import get_id_from_path, PretrainGenerator

if __name__ == "__main__":
    from collections import Counter

    h5_name = "transformer_pretrain.h5"
    batch_size = 32
    epochs = 100
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

    files = sorted(list(glob(base_path + "/*/*.npy")))
    # files = [x for x in files if [int(get_id_from_path(x))]]
    labels = [ get_id_from_path(x) for x in files]
    print("files num:", len(labels))

    expected_tracks_set = set(df["track_name"].to_list())
    downloaded_tracks_set = set(labels)

    valid_tracks_set = downloaded_tracks_set - (downloaded_tracks_set - expected_tracks_set)
    valid_tracks = list(valid_tracks_set)
    
    track_path_dic = {
        get_id_from_path(v): v for v in files
    }

    valid_tracks_path = [track_path_dic[x] for x in valid_tracks]

    print("valid tracks num", len(valid_tracks_path))
    # print(valid_tracks_path[:10])

    # print(len(set(df["track_name"].to_list()) - set(labels)))
    valid_tracks_m = [[x] for x in valid_tracks]
    
    samples = list(zip(valid_tracks_path, valid_tracks_m))

    train, val = train_test_split(
        samples, test_size=0.2, random_state=1337
    )

    model = transformer_pretrain()

    try:
        model.load_weights(h5_name, by_name=True)
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
        monitor="val_loss", patience=20, min_lr=1e-7, mode="min"
    )

    try:
        model.load_weights(h5_name)
    except:
        print("Could not load weights")

    model.fit_generator(
        PretrainGenerator(train, batch_size=batch_size),
        validation_data=PretrainGenerator(val, batch_size=batch_size),
        epochs=epochs,
        callbacks=[checkpoint, reduce_o_p],
        use_multiprocessing=True,
        workers=12,
        verbose=2,
        max_queue_size=64,
    )
