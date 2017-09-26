import cv2
import ast

import numpy as np
import matplotlib.pylab as plt
from os import listdir
import pandas as pd

import scipy

path_input = "videos/atl"
path_output = "arrays/"
#path_input_dat = "/Volumes/Nina Backup/videos/atl/2017-04-14/center field/490251-0f308987-60b4-480c-89b7-60421ab39106.mp4.dat"

df = pd.read_csv("cf_data.csv")
df = df[df["Player"]=="Pitcher"]

dates = ["2017-04-14", "2017-04-18", "2017-05-02", "2017-05-06"] # , "2017-05-19", "2017-05-23", "2017-06-06", "2017-06-10", "2017-06-18", "2017-06-22", "2017-07-04", "2017-07-16",
# "2017-04-15", "2017-04-19", "2017-05-03", "2017-05-07", "2017-05-20", "2017-05-24", "2017-06-07", "2017-06-11", "2017-06-19", "2017-06-23", "2017-07-05", "2017-07-17"]
# only first two rows von den im cluster angezeigten
# output_folder=args.output_dir

for date in dates:
    output = []
    labels = []
    # args = parser.parse_args()
    input_dir= path_input+"/"+date+"/center field/"
    list_files = listdir(input_dir)
    #print(list_files[0:3])
    # a  = np.array(range(len(list_files)))
    print(date)
    for f in list_files:
        if f[-4:]==".mp4":
            video_capture = cv2.VideoCapture(input_dir+f)
            game_id = f[:-4]
            print(file, game_id)
            line = df[df["Game"]==game_id]
            #print(labels)
            if len(line["Pitch Type"].values)!=1:
                print("PROBLEM: NO LABEL/ TOO MANY")
                print(line["Pitch Type"].values)
                continue
            else:
                labels.append(line["Pitch Type"].values[0])

            for i in open(input_dir+f+".dat").readlines():
                datContent=ast.literal_eval(i)
            bottom_p=datContent['Pitcher']['bottom']
            left_p=datContent['Pitcher']['left'] +30
            right_p=datContent['Pitcher']['right']-30
            top_p=datContent['Pitcher']['top']
            # bottom_b=datContent['Batter']['bottom']
            # left_b=datContent['Batter']['left']
            # right_b=datContent['Batter']['right']
            # top_b=datContent['Batter']['top']
            # center_dic['Pitcher']=np.array([abs(top_p-bottom_p)/2., abs(left_p-right_p)/2.])
            # center_dic['Batter']=np.array([abs(top_b-bottom_b)/2., abs(left_b-right_b)/2.])
            frames = np.zeros((167, 220, 220))
            i = 0
            while True:
                ret, frame = video_capture.read()
                if frame is None:
                    break
                pitcher = frame[top_p:bottom_p, left_p:right_p]
                pitcher = cv2.resize(np.mean(pitcher, axis = 2),(220, 220), interpolation = cv2.INTER_LINEAR)/255 #scipy.misc.imresize(pitcher, (109, 143, 3))/255
                # batter = frame[top_b:bottom_b, left_b:right_b]
                frames[i]= pitcher
                i+=1
#	    print(labels)
#	    print(frames)
            output.append(frames)

#print(output)
    np.save(path_output+"array_videos_"+date+".npy", np.array(output))
    np.save(path_output+"labels_videos_"+date+".npy", np.array(labels))
    print("arrays saved", date)

#print(labels)

print((np.array(outputs)).shape, "label shape: ", (np.array(labels)).shape)
