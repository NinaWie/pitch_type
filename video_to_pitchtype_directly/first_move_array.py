from array_from_videos import VideoProcessor
import numpy as np
from os import listdir

def save_pitcher(new_files, path):
    #print(new_files)
    res_p = []
    #res_b = []
    fi = []
    for f in new_files[:50]:
        fi.append(f[:-4])
        pitcher, batter = vid.get_pitcherAndBatter_array(path+f)
        res_p.append(pitcher)
        #res_b.append(batter)
    print(np.array(res_p).shape)
    #print(np.array(res_b).shape)
    np.save("outs/pitcher_first_move",np.array(res_p))
    #np.save("outs/batter_first_move", np.array(res_b))
    np.save("outs/files_used_first_move", np.array(fi))
    print("Done!")

def save_bunch_of_date(date):
    vid = VideoProcessor(path_input="/scratch/nvw224/videos/atl",df_path="/scratch/nvw224/cf_data.csv", resize_width = 110, resize_height = 110)

    view = "side view/"
    if view == "side view/":
        end = ".m4v"
    else:
        end = ".mp4"
    p = vid.path_input+"/"+date+"/"+view
    new_files = []
    for f in listdir(p):
        if f[-4:]==end:
            new_files.append(f)
    save_pitcher(new_files, p)

# save_bunch_of_date('2017-06-07')

def save_batter_run(path):
    p = "/Volumes/Nina Backup/videos/atl/2017-05-03/center field/490509-3f640b3f-9ba5-4e89-8f72-97c9f333aa2c.mp4"
    vid = VideoProcessor(path_input="/Volumes/Nina Backup/videos/atl",df_path="/Users/ninawiedemann/Desktop/UNI/Praktikum/ALL/cf_data.csv", resize_width = 110, resize_height = 110)
    _, batter = vid.get_pitcherAndBatter_array(p)
    print(batter.shape)
    np.save("outs/batter_example.npy", batter)

def get_filesystem_dic(path, view):
    dic = {}
    for d in listdir(path):
        dic[d] = listdir(path+d+"/"+view)
    return dic

def get_paths_from_games(game_ids, view):
    dic = get_filesystem_dic("/scratch/nvw224/videos/atl/", view)
    #print(dic)
    dates_belonging = []
    for g in game_ids:
	if view == "side view":
            new = g+".m4v"
        else:
            new = g+".mp4"
        for key in dic.keys():
            if new in dic[key]:
                dates_belonging.append(key)
    assert(len(game_ids)==len(dates_belonging))
    np.save("outs/dates.npy", dates_belonging)

games = ["490972-1c5b045a-1e38-4c87-87ba-dee56a69d592", "490479-1ae89699-bb92-450f-a17a-0af44f3dc86d", "490479-3ace19fc-df17-4487-8958-03bbdb7a4b51","490732-1cdb7ed4-fae2-41d1-9666-bb2ddac27d00"]
get_paths_from_games(games, "side view")
