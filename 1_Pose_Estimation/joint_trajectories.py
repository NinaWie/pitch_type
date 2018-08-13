import time
import numpy as np
import argparse
import pandas as pd
from os.path import isfile, join
from os import listdir
import os
import codecs, json
import tensorflow as tf

from pose_estimation_script import handle_one
from data_processing import *
import ast
import cv2

parser = argparse.ArgumentParser(description='Pose Estimation Baseball')
parser.add_argument('input_file', metavar='DIR', # Video file to be processed
                    help='folder with video files to be processed')
parser.add_argument('output_folder', metavar='DIR', # output dir
                    help='folder where to store the json files with the output coordinates')
parser.add_argument('center',  #
                    help='specify what kind of file is used for specifying the center of the target person: either path_to_json_dictionary.json, or datPitcher, or datBatter')


args = parser.parse_args()
inp_dir = args.input_file
out_dir = args.output_folder
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# If the starting positions are saved in a json file (for high quality videos)
if args.center[-5:] == ".json":
    with open(args.center, "r") as infile:
        centers = json.load(infile)


for fi in listdir(inp_dir):
    # check if file was already processed
    if fi[0]=="." or (fi[-4:]!=".mp4" and fi[-4:]!=".m4v"):
        print("wrong input file (Must be .mp4 or .m4v!)", fi)
        continue

    f = os.path.join(inp_dir, fi)

    # if starting position is saved in dat files (for MLBAM videos)
    if args.center[:3] == "dat":
        target = args.center[3:]
        try:
            for i in open(f+".dat").readlines():
                datContent=ast.literal_eval(i)
        except IOError:
            print("dat file exists:", os.path.exists(f+".dat"))
            print("dat file not found", fi)
            continue
        # bounding box
        bottom_b=datContent[target]['bottom']
        left_b=datContent[target]['left']
        right_b=datContent[target]['right']
        top_b=datContent[target]['top']
        print("bounding box for starting position from dat file:", bottom_b, left_b, right_b, top_b)
        # center of bounding box = start position
        center = np.array([abs(left_b-right_b)/2., abs(bottom_b-top_b)/2.])
    # if starting position is saved in json file
    elif args.center[-5:] == ".json":
        try:
            center = centers[fi[:-4]]
        except KeyError:
            print("the coordinates for this video are not contained in the json file")
            continue
        left_b = 0
        top_b = 0
    # no dat and no json file put as argument
    else:
        print("wrong input for center argument: must be either the file path of a json file containing a dictionary, or datPitcher or datBatter")
        continue
    ##############

    tic=time.time()
    print("input file path of video:", f)
    video_capture = cv2.VideoCapture(f)

    tic1 = time.time()

    # LOCALIZATION
    old_norm=10000
    indices = [6,9] # hips to find right person in the first frame
    frame_counter = 0
    found = False
    pitcher_array = []


    # Not found until there is a frame with a person detected
    while not found:
        ret, frame = video_capture.read()
        if args.center[-5:] == ".json":
            bottom_b = len(frame)
            right_b = len(frame[0])
        out = handle_one(frame[top_b:bottom_b, left_b:right_b])
        for person in range(len(out)):
            hips=np.asarray(out[person])[indices]
            hips=hips[np.sum(hips,axis=1)!=0]
            if len(hips)==0:
                continue
            mean_hips=np.mean(hips,axis=0)
            norm= abs(mean_hips[0]-center[0])+abs(mean_hips[1]-center[1])
            if norm<old_norm:
                found = True
                loc=person
                old_norm=norm
        frame_counter+=1
        if not found:
            pitcher_array.append([[0,0] for j in range(18)])
            print("no person detected in frame", frame_counter)

    first_frame = np.array(out[loc])
    first_frame[:,0]+=left_b
    first_frame[first_frame[:,0]==left_b] = 0 # if the output was 0 (missing value), reverse box addition
    first_frame[:,1]+=top_b
    first_frame[first_frame[:,1]==top_b] = 0

    # Save first frame (detection on whole frame)
    pitcher_array.append(first_frame)

    # boundaries to prevent bounding box from being larger than the frame
    boundaries = [0, len(frame[0]), 0, len(frame)]

    # from first detection, form bbox for next fram
    bbox = define_bbox(first_frame, boundaries)

    # save detection to compare to next one
    globals()['old_array'] = first_frame #first_saved

    # save detection in a second array, in which the missing values are constantly filled with the last detection
    new_res = first_frame.copy()

    # START LOOP OVER FRAMES
    while True:
        # Read frame by frame
        ret, frame = video_capture.read()
        if frame is None:
            print("end of video capture")
            break
        pitcher = frame[bbox[2]:bbox[3], bbox[0]:bbox[1]]

        # pose estimation network
        out = handle_one(pitcher)
        out = np.array(out)

        # add bbox edges such that output coordinates are for whole frame
        out[:, :,0]+=bbox[0]
        out[out[:, :,0]==bbox[0]] = 0 # if the output was 0 (missing value), reverse box addition
        out[:, :,1]+=bbox[2]
        out[out[:, :,1]==bbox[2]] = 0

        # old array is the detected person in the previous frame - localize which person of out corresponds
        out = player_localization(out, globals()['old_array'])
        out = np.array(out)

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
            print("ROI used as input to pose estimation for next frame: rectangle spanned by", bbox[[0,2]], "and", bbox[[1,3]])

        print("Frame ", frame_counter)
        frame_counter+=1

        #if p%50==0:
        #    save_inbetween(pitcher_array, fi, out_dir, events_dic)

    # FURTHER PROCESSINg
    pitcher_array = np.array(pitcher_array)
    print("shape pitcher_array", pitcher_array.shape)

    # Swap back right and left if swapped
    pitcher_array = mix_right_left(pitcher_array, index_list)

    # interpolate missing values
    pitcher_array = interpolate(pitcher_array)

    # Lowpass filtering of joint trajectories (each individually)
    # specify frames per second for lowpass filtering
    fps = 20
    for k in range(len(pitcher_array[0])):
        for j in range(2):
            pitcher_array[:,k,j] = lowpass(pitcher_array[:,k,j]-pitcher_array[0,k,j], cutoff = 1, fs = fps)+pitcher_array[0,k,j]

    # SAVE IN JSON FORMAT
    game_id = fi[:-4]
    output_path = os.path.join(out_dir,game_id)
    to_json(pitcher_array, output_path)
    print("saved as", output_path)

    toctoc=time.time()
    print("Time for whole video to joint trajectories: ", toctoc-tic)
