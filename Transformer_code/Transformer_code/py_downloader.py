# download splited list
import os
import pandas as pd
import tqdm
from multiprocessing import Pool
from itertools import repeat

# import spotdl
def dl_song(song, dir_name):

    if isinstance(song, str):
        # cmd = "cd " + os.path.join("/home/xmo/download/spotify", dir_name)
        # os.system(cmd)
        # cmd = "spotdl download " + "'" + song + "'" + " --output {title}"
        # print("bash command:", cmd)
        # TODO modify here
        a = os.path.join("/home/xmo/download/spotify", dir_name)
        cmd = "spotdl download " + "'" + song + "'" + " --output " + a +"/{title}"
        os.system(cmd)

    else:
        print("wrong song type")


# TODO modify .csv path
df = pd.read_csv("/home/xmo/download/music_genre.csv",index_col=0, header=[0])
df.reset_index(inplace=True)
# df.columns.values

id_to_title = {k: v for k, v in zip( df.instance_id.to_list(), df.track_name.to_list() )}

data = {
  k: v
  for k, v in zip(df["instance_id"].tolist(), df["popularity"].tolist())
}

download_list = df.track_name.to_list()

download_splited_list = [
    download_list[x:x+100] for x in range(0, len(download_list), 100)
]

print(len(download_list))

# TODO modify path
spotify_dir = "/home/xmo/download/spotify"

# TODO modify continue_dl to continue previous downloading, the initial value should be 0
continue_dl = 365
######################################

p = Pool(8)

for cnt in range(continue_dl ,len(download_splited_list)):
    
    os.chdir(spotify_dir)

    print(download_splited_list[cnt])
    splited_list = download_splited_list[cnt]
    
    dir_name = str(cnt).zfill(3)
    print("-----------",dir_name,"----------")
    cmd = "mkdir " + dir_name
    # # print(cmd)
    os.system(cmd)
    
    os.chdir( os.path.join(spotify_dir, dir_name) )
    
    # os.system("pwd")
    
    print("Now downloading list", dir_name, "/" , len(download_splited_list) , ", current list length:" , len(splited_list) )

    
    p.starmap(dl_song, zip(splited_list, repeat(dir_name)))
    # p.join()

    # for song in splited_list:
    #     dl_song(song)

    # #     print(type(song))
    #     # if isinstance(song, str):
    #     #     cmd = "spotdl download " + "'" + song + "'" + " --output {title}"
    #     #     print("bash command:", cmd)
    #     #     # cmd = "spotdl download " + "'" + song + "'" + " --output {title}"
    #     #     os.system(cmd)
    #     # else:
    #     #     print("wrong song type")
        


    #     # break
    # # print(cmd)

    # # cmd = "spotdl download" +" 'Shape of you'"+ " 'Closer'" + " --output {title}"
    
    cnt += 1
    # break

# bash background downloading

# python /home/xmo/download/spotify/py_downloader.py > /home/xmo/download/spotify/spotifspotify_dl_log.txt
# nohup python /home/xmo/download/spotify/py_downloader.py > /home/xmo/download/spotify/spotify_dl.log &

