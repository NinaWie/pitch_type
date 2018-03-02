import time
from torch import np
import argparse
import pandas as pd
from os.path import isfile, join
from os import listdir
import os
import json
import tensorflow as tf

from Functions import *
import ast
import cv2
from test import test

parser = argparse.ArgumentParser(description='Pose Estimation Baseball')
parser.add_argument('input_file', metavar='DIR', help='Video file to be processed')
parser.add_argument('view', metavar='DIR', help='cf (center field) or sv (side view)')
parser.add_argument('output_folder', metavar='DIR', help='to drop outputs')
sequ_len = 50

args = parser.parse_args()
inp_dir = args.input_file
out_dir = args.output_folder
view = args.view
if not os.path.exists(out_dir):
    os.makedirs(out_dir)



for day in listdir(inp_dir):
    print("-------------new day----------", day)
    files = []
    first_subdirectory = inp_dir+day
    sub_list = listdir(first_subdirectory)
    if len(sub_list)==1:
        prefix = sub_list[0]+"-"
        # out_dir_first = out_directory+"new_videos/"
        if view =="cf":
            subdirectory = first_subdirectory+"/"+sub_list[0]+"/CENTERFIELD/"
            # out_dir_first = out_dir_first+"cf/"
            # out_dir = out_dir_first+sub_list[0]+"_"
        elif view == "sv":
            subdirectory = first_subdirectory+"/"+sub_list[0]+"/CH_HIGH_SIDEVIEW/"
            # out_dir_first = out_dir_first+"sv/"
            # out_dir = out_dir_first+sub_list[0]+"_"
        else:
            raise ValueError("bad view argument, needs to be cf or sv")
    else:
        prefix = ""
        # out_dir = out_directory+"old_videos/"
        if view =="cf":
            subdirectory = first_subdirectory+"/center field/"
            # out_dir = out_dir+"cf/"
        elif view == "sv":
            subdirectory = first_subdirectory+"/side view/"
            # out_dir = out_dir+"sv/"
        else:
            raise ValueError("bad view argument, needs to be cf or sv")
        # out_dir_first = out_dir

    #already_done = listdir(out_dir_first)
    for ff in listdir(subdirectory):
        string_f = str(ff)
        if  not string_f.endswith("dat"):
            files.append(string_f)

    print(len(files), out_dir)
    for fi in files:
        for pla in ["pitcher", "batter"]:
            if fi[0]=="." or prefix + fi[:-4]+"_"+pla+".json" in listdir(out_dir): # or fi[:-4]+".json" in listdir(out_dir+"handle_one/"): ["#8 RHP Cole Johnson (2).mp4"]: #
                print("already there or wrong ending", fi)
                continue

            ### for batter first movemnet
            # try:
            #     ind = files.index(fi[:-4])
            #     if "Hit into" not in outcomes[ind]:
            #         print("different play outcome", outcomes[ind])
            #         continue
            # except ValueError:
            #     print("not in csv")
            #     continue

            try:
                for i in open(subdirectory+fi+".dat").readlines():
                    datContent=ast.literal_eval(i)
            except IOError:
                print("dat file not found", fi)
                continue
            if pla=="pitcher":
                player= "Pitcher"
            else:
                player= "Batter"
            bottom_b=datContent[player]['bottom']
            left_b=datContent[player]['left']
            right_b=datContent[player]['right']
            top_b=datContent[player]['top']
            print(bottom_b, left_b, right_b, top_b)
            center = np.array([abs(left_b-right_b)/2., abs(bottom_b-top_b)/2.])

            ##############

            j=0
            #center_dic={}
            tic=time.time()
            f = subdirectory+fi
            print(f)
            video_capture = cv2.VideoCapture(f)
            #video_capture.set(cv2.CAP_PROP_POS_FRAMES, 100)
            #x, y = centers[fi[:-4]]#np.array([abs(top_p+bottom_p)/2., abs(left_p+right_p)/2.])
            # center = centers[fi[:-4]] # [1200,700] # center between hips - known position of target person
            #    print(fi, "center: ", center_dic["Pitcher"])

            tic1 = time.time()
            events_dic = {}
            events_dic["video_directory"]= inp_dir
            events_dic["bbox_batter"] = [0,0,0,0]
            events_dic["bbox_pitcher"] = [0,0,0,0]
            events_dic["start_time"]=time.time()
            rel = []
            #handle_one_res = []


            # LOCALIZATION
            old_norm=10000
            indices = [6,9] # hips to find right person in the first frame
            p=0
            found = False
            pitcher_array = []

            #### for batter first
            videos = []

            # Not found until there is a frame with a person detected
            while not found and p<30:
                #len(df[player][i])==0:
                ret, frame = video_capture.read()
                out = handle_one(frame[top_b:bottom_b, left_b:right_b]) # changed batter first move
                for person in range(len(out)):
                    hips=np.asarray(out[person])[indices]
                    hips=hips[np.sum(hips,axis=1)!=0]
                    if len(hips)==0:
                        continue
                    mean_hips=np.mean(hips,axis=0)
                    norm= abs(mean_hips[0]-center[0])+abs(mean_hips[1]-center[1]) #6 hip
                    if norm<old_norm:
                        found = True
                        loc=person
                        old_norm=norm
                p+=1
                if not found:
                    pitcher_array.append([[0,0] for j in range(18)])
                    print("no person detected in frame", p)
            if not found:
                print("no person detected in first 30 frames")
                continue

            #left_b = 0 # change for batters first move
            #top_b = 0
            first_frame = np.array(out[loc])
            first_frame[:,0]+=left_b
            first_frame[first_frame[:,0]==left_b] = 0 # if the output was 0 (missing value), reverse box addition
            first_frame[:,1]+=top_b
            first_frame[first_frame[:,1]==top_b] = 0

            # Save first frame (detection on whole frame)
            pitcher_array.append(first_frame)

            # boundaries to prevent bounding box from being larger than the frame
            boundaries = [0, len(frame[0]), 0, len(frame)]
            # print("boundaries = ", boundaries)

            # from first detection, form bbox for next fram
            bbox = define_bbox(first_frame, boundaries)

            # save detection to compare to next one
            globals()['old_array'] = first_frame #first_saved

            # save detection in a second array, in which the missing values are constantly filled with the last detection
            new_res = first_frame.copy()

            print("first output", pitcher_array[-1])

            # START LOOP OVER FRAMES
            while True:
            # Capture frame-by-frame

                ret, frame = video_capture.read()
                if frame is None:
                    print("end of video capture")
                    break
                pitcher = frame[bbox[2]:bbox[3], bbox[0]:bbox[1]]

                ### for batter first_
                # videos.append(np.mean(pitcher, axis = 2).tolist())

                # pose estimation network
                out = handle_one(pitcher)
                out = np.array(out)
                # print("out", [o.tolist() for o in out])

                # add bbox edges such that output coordinates are for whole frame
                out[:, :,0]+=bbox[0]
                out[out[:, :,0]==bbox[0]] = 0 # if the output was 0 (missing value), reverse box addition
                out[:, :,1]+=bbox[2]
                out[out[:, :,1]==bbox[2]] = 0
                # print("output network,", whole_box, "global array", globals()['old_array'])

                # old array is the detected person in the previous frame - localize which person of out corresponds
                out = player_localization(out, globals()['old_array'])
                #l = [np.sum(np.asarray(elem) ==0) for elem in whole_box] # in order to simply take detection with least missing values
                out = np.array(out)
                # print("localized", out.tolist())

                # if frame is missing: continue (do not update bounding box and previous frame detection)
                if np.all(out==0):
                    pitcher_array.append(np.array([[0,0] for j in range(18)]))
                else:
                    # update previous detection
                    globals()['old_array'] = out.copy()
                    pitcher_array.append(out)
                    # update bounding box: for missing values, take last detection of this joint (saved in new_res)
                    new_res[np.where(out!=0)]=out[np.where(out!=0)]
                    # print(new_res)
                    bbox = define_bbox(new_res, boundaries)
                    # print("bbox new", bbox)

                # print(p)
                p+=1

                #if p%50==0:
                #    save_inbetween(pitcher_array, fi, out_dir, events_dic)


            pitcher_array = np.array(pitcher_array)
            print("shape pitcher_array", pitcher_array.shape)

            ## for interpolation, mix right left and smoothing:
            pitcher_array = df_coordinates(pitcher_array, do_interpolate = True, smooth = True, fps = 20)

            # SAVE IN JSON FORMAT
            game_id = fi[:-4]
            file_path_pitcher = out_dir+prefix + game_id+ "_"+ pla
            print(events_dic, "saved as ", file_path_pitcher)
            to_json(pitcher_array, events_dic, file_path_pitcher)
            video_capture.release()

            ### batter first move
            # with open(out_dir+game_id+"_video.json", "w") as outfile:
            #      json.dump(videos, outfile)

            toctoc=time.time()
            print("Time for whole video to array: ", toctoc-tic)
