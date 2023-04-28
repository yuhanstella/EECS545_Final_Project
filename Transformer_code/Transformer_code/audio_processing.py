import sys
import warnings

import librosa
import numpy as np
import os

if not sys.warnoptions:
    warnings.simplefilter("ignore")

os.chdir("/home/xmo/music_popularity_classification")

input_length = 16000 * 30

n_mels = 128

bad_tracks = set()

def pre_process_audio_mel_t(audio, sample_rate=16000):
    # print("pre_process_audio_mel_t")
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=n_mels)
    mel_db = (librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40

    return mel_db.T


def load_audio_file(file_path, input_length=input_length):
    # print("load_audio_file")
    try:
        data = librosa.core.load(file_path, sr=16000)[0]  # , sr=16000
    except:
        data = []
        print("error",file_path)
        bad_tracks.add(file_path)
    # print("check point")
    if len(data) > input_length:

        max_offset = len(data) - input_length

        offset = np.random.randint(max_offset)

        data = data[offset : (input_length + offset)]

    else:
        if input_length > len(data):
            max_offset = input_length - len(data)

            offset = np.random.randint(max_offset)
        else:
            offset = 0

        data = np.pad(data, (offset, input_length - len(data) - offset), "constant")

    data = pre_process_audio_mel_t(data)
    return data


def random_crop(data, crop_size=128):
    # print(data.shape[0])
    start = np.random.randint(0, data.shape[0] - crop_size)
    return data[start : (start + crop_size), :]


def random_mask(data):
    new_data = data.copy()
    prev_zero = False
    for i in range(new_data.shape[0]):
        if np.random.uniform(0, 1) < 0.1 or (
            prev_zero and np.random.uniform(0, 1) < 0.5
        ):
            prev_zero = True
            new_data[i, :] = 0
        else:
            prev_zero = False

    return new_data


def save(path):
    # print("save")
    data = load_audio_file(path)
    np.save(path.replace(".mp3", ".npy"), data, allow_pickle=True)
    return True


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from glob import glob
    from multiprocessing import Pool

    p = Pool(8)

    base_path = "/home/xmo/download/spotify"

    for cnt in range(501):

        print("--------------------Now processing dir:", cnt,"/501--------------------")
        dir_name = str(cnt).zfill(3)

        files = sorted(list(glob( os.path.join(base_path, dir_name) + "/*.mp3")))

        files_transed = sorted(list(glob( os.path.join(base_path, dir_name) + "/*.npy")))

        for i in range(len(files_transed)):
            files_transed[i] = files_transed[i].replace(".npy",".mp3")
        try:
            print(files_transed[0])
        except:
            print(len(files_transed))

        remained_files = list(set(files) - set(files_transed))

        print("# tracks ",len(files))
        print("# files_transed",len(files_transed))
        print("# remaining tracks ",len(remained_files))

        print(len(files))
        print(files[:10])
        
        for i, _ in tqdm(enumerate(p.imap(save, remained_files))):
            if i % 1000 == 0:
                print(i)
    
    print(bad_tracks)