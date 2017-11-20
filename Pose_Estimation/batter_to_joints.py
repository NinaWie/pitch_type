
import time
from torch import np
import argparse
import pandas as pd
from os.path import isfile, join
from os import listdir
import codecs, json
import tensorflow as tf

from Functions import handle_one, df_coordinates, to_json
import ast
import cv2
from test import test

parser = argparse.ArgumentParser(description='Pose Estimation Baseball')
parser.add_argument('input_file', metavar='DIR', # Video file to be processed
                    help='folder where merge.csv are')
parser.add_argument('output_folder', metavar='DIR', # Video file to be processed
                    help='folder where merge.csv are')
restore_first_move = "/scratch/nvw224/pitch_type/saved_models/first_move_more"
restore_release = "/scratch/nvw224/pitch_type/saved_models/release_model"
restore_position = "/scratch/nvw224/pitch_type/saved_models/modelPosition"
sequ_len = 50

# example arg: /Volumes/Nina\ Backup/videos/atl/2017-04-15/center\ field/490266-0aeec26e-80f7-409f-8ae7-a40b834b3a81.mp4

#dates = ["2017-04-14", "2017-04-18", "2017-05-02", "2017-05-06"] # , "2017-05-19", "2017-05-23", "2017-06-06", "2017-06-10", "2017-06-18", "2017-06-22", "2017-07-04", "2017-07-16",
#"2017-04-15", "2017-04-19", "2017-05-03", "2017-05-07", "2017-05-20", "2017-05-24", "2017-06-07", "2017-06-11", "2017-06-19", "2017-06-23", "2017-07-05", "2017-07-17"]
# only first two rows von den im cluster angezeigten
# for date in dates:

args = parser.parse_args()
inp_dir = args.input_file

out_dir = args.output_folder

with open("center_dics.json", "r") as infile:
    centers = json.load(infile)

for fi in listdir(inp_dir): #__name__ == "__main__": #changed
    if fi[0]==".": # or fi[:-4]+"_joints.json" in listdir(out_dir) or fi[:-4]+".json" in listdir(out_dir+"handle_one/"):    #changed
    	print("already there", fi)
    	continue
    j=0
    center_dic={}
    tic=time.time()

    f = inp_dir+fi
    print(f)
    video_capture = cv2.VideoCapture(f)
    #x, y = centers[fi[:-4]]#np.array([abs(top_p+bottom_p)/2., abs(left_p+right_p)/2.])
    center_dic['Batter'] = centers[fi[:-4]] #np.array([y, x])

    print("center: ", center_dic["Batter"])
    df = pd.DataFrame(columns=['Frame', 'Batter'])
    tic1 = time.time()
    p=0
    handle_one_res = []
    events_dic = {}
    events_dic["video_directory"]= inp_dir
    events_dic["bbox_batter"] = [0,0,0,0]
    events_dic["bbox_pitcher"] = [0,0,0,0]
    events_dic["start_time"]=time.time()
    while p<5: #True:
# Capture frame-by-frame
        ret, frame = video_capture.read()
        if frame is None:
            print("end of video capture")
            break
        batter = frame
        # print(p, "in video")
        df.loc[p]=[int(p),handle_one(batter)]
	#handle_one_res.append(handle_one(batter).tolist())
	# print("handle one finished")
        p+=1
    print(events_dic)
    print("Time to read in video and handle one:", time.time()-tic1)

    print(len(handle_one_res))
    #game_id = f.split("/")[-1][:-4]
    #with open(out_dir+"handle_one/new_Functions.json", "w") as outfile:
#	json.dump(list(handle_one_res), outfile)

  #  df.to_csv(out_dir+"handle_one/"+game_id+".csv")
 ##   continue

    tic2 = time.time()
    df_res= df_coordinates(df,center_dic, ["Batter"], interpolate = True)
    print("time for df_coordinates", time.time()-tic2)
    batter_array = np.zeros((p, 18,2))
    for i in range(p):
        try:
            batter_array[i,:,:] = np.array(df_res['Batter_player'].values[i])
        except:
            batter_array[i,:,:] = batter_array[0,i-1,:,:]

    # NEW: JSON FORMAT
    f_per_sec = video_capture.get(cv2.CAP_PROP_FPS)
    print(f_per_sec)
    game_id = fi[:-4]
    file_path_batter = out_dir+game_id
    to_json(batter_array, events_dic, file_path_batter, frames_per_sec=f_per_sec)

    toctoc=time.time()
    print("Time for whole video to array: ", toctoc-tic)
