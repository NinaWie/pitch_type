import cv2
import time
from torch import np
import argparse
import pandas as pd
from os.path import isfile, join
from os import listdir
import codecs, json

from Functions import handle_one,df_coordinates, to_json
import ast


parser = argparse.ArgumentParser(description='Pose Estimation Baseball')
parser.add_argument('input_dir', metavar='DIR', help='folder where videos are')
parser.add_argument('output_dir', metavar='DIR', help='folder where to put joint outputs') # both directories must have / in the end
# example usage: python from_videos_to_joints.py ./atl/2017-05-06/center\ field/ out_joints/ # make directory out_joints first


args = parser.parse_args()
directory = args.input_dir
out_dir = args.output_dir

for day in listdir(directory):
    print("-------------new day----------", day)
    files = []
    subdirectory = directory+day
    game = listdir(subdirectory)[0]
    subdirectory = subdirectory+"/"+game+"/CENTERFIELD/"
    for ff in listdir(subdirectory):
        string_f = str(ff)
        if  not string_f.endswith("dat"):
            files.append(string_f)
    #already_done = listdir("/scratch/nvw224/pitch_type/Pose_Estimation/out_joints/pitcher/")
    #print("number files:", len(files))
    #print(already_done)
    for fi in files:
        #if (fi[:-4]+".json") in already_done:
    	       #continue
        f = subdirectory+fi
        print("file", fi, "directory", f)
        video_capture = cv2.VideoCapture(f)
        #x, y = centers[fi[:-4]]#np.array([abs(top_p+bottom_p)/2., abs(left_p+right_p)/2.])
        center_dic={}
	#center_dic['Batter'] = np.array([y, x])
        for i in open(f+".dat").readlines():
            datContent=ast.literal_eval(i)
        bottom_b=datContent['Batter']['bottom']
        left_b=datContent['Batter']['left']
        right_b=datContent['Batter']['right']
        top_b=datContent['Batter']['top']
        bottom_p=datContent['Pitcher']['bottom']
        left_p=datContent['Pitcher']['left']
        right_p=datContent['Pitcher']['right']
        top_p=datContent['Pitcher']['top']
        print(bottom_b, left_b, right_b, top_b)
        print(bottom_p, left_p, right_p, top_p)
        center_dic['Batter']=np.array([abs(left_b-right_b)/2., abs(bottom_b-top_b)/2.])
        center_dic['Pitcher']=np.array([abs(left_p-right_p)/2., abs(bottom_p-top_p)/2.])
        print("center b: ", center_dic["Batter"], "center b: ", center_dic["Pitcher"])
        df = pd.DataFrame(columns=['Frame', 'Pitcher', 'Batter'])
        tic1 = time.time()
        p=0
        events_dic = {}
        events_dic["video_directory"]= subdirectory
        events_dic["start_time"]=time.time()
        while True:
    # Capture frame-by-frame
            ret, frame = video_capture.read()
            if frame is None:
                print("end of video capture at frame", p)
                break
            pitcher = frame[top_p:bottom_p, left_p:right_p]
            batter = frame[top_b:bottom_b, left_b:right_b]
            df.loc[p]=[int(p),handle_one(pitcher), handle_one(batter)]
            p+=1
        print("events", events_dic)
        print("Time to read in video and handle one:", time.time()-tic1)

        df_res= df_coordinates(df,center_dic, ["Pitcher", "Batter"], interpolate = True)

        pitcher_array = np.zeros((p, 18,2))
        for i in range(p):
            try:
                frame = np.array(df_res['Pitcher_player'].values[i])
                frame[:,0]+= min(left_p, right_p)
                frame[:,1]+= min(bottom_p, top_p)
                pitcher_array[i,:,:] = frame
            except:
                pitcher_array[i,:,:] = pitcher_array[i-1,:,:]
                print("missing_frame", i)
        print("shape", pitcher_array.shape)

        batter_array = np.zeros((p, 18,2))
        for i in range(p):
            try:
                frame = np.array(df_res['Batter_player'].values[i])
                frame[:,0]+= min(left_b, right_b)
                frame[:,1]+= min(bottom_b, top_b)
                batter_array[i,:,:] = frame
            except:
                print("missing_frame", i)
                batter_array[i,:,:] = batter_array[0,i-1,:,:]

        # NEW: JSON FORMAT
        f_per_sec = video_capture.get(cv2.CAP_PROP_FPS)
        print("Frames_per_sec", f_per_sec)
        game_id = f.split("/")[-1][:-4]

        file_path_batter = out_dir+"batter_"+game+"_"+game_id
        to_json(batter_array, events_dic, file_path_batter, frames_per_sec=f_per_sec)

        file_path_pitcher = out_dir+"pitcher_"+game+"_"+game_id
        to_json(pitcher_array, events_dic, file_path_pitcher, frames_per_sec=f_per_sec)

        toctoc=time.time()
        print("Finished video ", toctoc-tic1)
