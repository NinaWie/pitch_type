
import time
from torch import np
import argparse
import pandas as pd
from os.path import isfile, join
from os import listdir
import codecs, json
import tensorflow as tf

from Functions_new_version import handle_one,df_coordinates , to_json, player_localization
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

with open("center_dics.json", "r") as infile:
    centers = json.load(infile)

args = parser.parse_args()
inp_dir = args.input_file

out_dir = args.output_folder

def define_bbox(res, boundaries):
    """
    input is ouput of pose estimation and boundaries of frame
    takes minima and maxima of the pose as edges of bounding box b
    extends bounding box by adding the width of the box on all sides (up to boundaries)
    """
    joints_for_bbox = np.where(res[:,0]!=0)[0]
    bbox = np.array([np.min(res[joints_for_bbox, 0]), np.max(res[joints_for_bbox, 0]),
           np.min(res[joints_for_bbox, 1]), np.max(res[joints_for_bbox, 1])]).astype(int)
    width = bbox[1]-bbox[0]
    for i in range(len(bbox)):
        bbox[i]-=width
        width*=(-1)
        if (bbox[i]<boundaries[i] and width<0) or (bbox[i]>boundaries[i] and width>0):
            bbox[i]=boundaries[i]
    return bbox

def save_inbetween(arr, fi, out_dir, events_dic):
    pitcher_array = np.array(arr)
    print("shape pitcher_array", pitcher_array.shape)
    # NEW: JSON FORMAT
    game_id = fi[:-4]
    file_path_pitcher = out_dir+game_id
    print(events_dic, file_path_pitcher)
    to_json(pitcher_array, events_dic, file_path_pitcher)

for fi in listdir(inp_dir): # ["cole_loncar.mp4"]: #["40mph_1us_1.2f_170fps_40m_sun.avi"]:
    if fi[0]==".":# or fi[:-4]+"_joints.json" in listdir(out_dir) or fi[:-4]+".json" in listdir(out_dir+"handle_one/"):
    	print("already there", fi)
    	continue
    j=0
    #center_dic={}
    tic=time.time()
    f = inp_dir+fi
    print(f)
    video_capture = cv2.VideoCapture(f)
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, 70)
    #x, y = centers[fi[:-4]]#np.array([abs(top_p+bottom_p)/2., abs(left_p+right_p)/2.])
    center = centers[fi[:-4]] # [1200,700]
#    print(fi, "center: ", center_dic["Pitcher"])

    df = pd.DataFrame(columns=['Frame','Pitcher'])
    tic1 = time.time()
    events_dic = {}
    events_dic["video_directory"]= inp_dir
    events_dic["bbox_batter"] = [0,0,0,0]
    events_dic["bbox_pitcher"] = [0,0,0,0]
    events_dic["start_time"]=time.time()
    rel = []
    #handle_one_res = []


    # localization

    old_norm=10000
    indices=[6,9]
    #print df[player][0][0]
    p=0
    found = False
    pitcher_array = []
    while not found:
        #len(df[player][i])==0:
        ret, frame = video_capture.read()
        out = handle_one(frame)
        for person in range(len(out)):
            hips=np.asarray(out[person])[indices]
            hips=hips[np.sum(hips,axis=1)!=0]
            mean_hips=np.mean(hips,axis=0)
            norm= abs(mean_hips[0]-center[0])+abs(mean_hips[1]-center[1]) #6 hip
            if norm<old_norm:
                found = True
                loc=person
                old_norm=norm
        if found:
            break
        p+=1
        pitcher_array.append([[0,0] for j in range(18)])
        print("no person detected in frame", p)

    # print(out, out[loc])
    pitcher_array.append(out[loc])

    p+=1
    boundaries = [0, len(frame[0]), 0, len(frame)]
    print("boundaries = ", boundaries)

    bbox = define_bbox(np.array(out[loc]), boundaries)

    # first_saved = np.asarray(out[loc].copy())
    # first_saved[:,0]-=bbox[0]
    # first_saved[first_saved[:,0]<0] = 0
    # first_saved[:,1]-=bbox[2]
    # first_saved[first_saved[:,1]<0] = 0
    globals()['old_array'] = out[loc] #first_saved

    new_res = out[loc].copy()

    print("first output", pitcher_array[-1])
    while True:
# Capture frame-by-frame

        ret, frame = video_capture.read()
        if frame is None:
            print("end of video capture")
            break
        pitcher = frame[bbox[2]:bbox[3], bbox[0]:bbox[1]]

        # pose estimation network
        out = handle_one(pitcher)
        out = np.array(out)
        #print("out", out.tolist())

        # add bbox edges such that output coordinates are for whole frame
        out[:, :,0]+=bbox[0]
        out[out[:, :,0]==bbox[0]] = 0 # if the output was 0 (missing value), reverse box addition
        out[:, :,1]+=bbox[2]
        out[out[:, :,1]==bbox[2]] = 0
        # print("output network,", whole_box, "global array", globals()['old_array'])

        # old array is the detected person in the previous frame - localize which person of out corresponds
        out, _ = player_localization(out, globals()['old_array'])
        #l = [np.sum(np.asarray(elem) ==0) for elem in whole_box]
        out = np.array(out)
        # print("localized", out.tolist())

        # if frame is missing: continue (do not update bounding box and previous frame detection)
        if np.all(out==0):
            pitcher_array.append(np.array([[0,0] for j in range(18)]))
            continue
        # out = whole_box[np.argmin(l)]
        globals()['old_array'] = out.copy()

        pitcher_array.append(out)

        # update bounding box: for missing values, take last detection of this joint (saved in new_res)
        new_res[np.where(out!=0)]=out[np.where(out!=0)]
        # print(new_res)
        if not np.all(out==0):
            bbox = define_bbox(new_res, boundaries)
        print("bbox new", bbox)

        # print("res mit vorherigen eingefuellt", new_res)
        print(p)

        p+=1

        #if p%50==0:
        #    save_inbetween(pitcher_array, fi, out_dir, events_dic)

    ## for interpolation, mix right left and smoothing:
    # df_res= df_coordinates(df,center_dic, ["Pitcher"], interpolate = False)

    pitcher_array = np.array(pitcher_array)
    print("shape pitcher_array", pitcher_array.shape)


    # SAVE IN JSON FORMAT
    game_id = fi[:-4]
    file_path_pitcher = out_dir+game_id
    print(events_dic, file_path_pitcher)
    to_json(pitcher_array, events_dic, file_path_pitcher)

    toctoc=time.time()
    print("Time for whole video to array: ", toctoc-tic)


        # # FOR MULTIPLE BBOXES: Version 2: whole_box nehmen, lokalisieren, dann falls unter threshold ersetzen mit smaller boxen werten
        # whole_box = handle_one(pitcher)
        # # print("output network,", whole_box, "global array", globals()['old_array'])
        # # whole_box_person, _ = player_localization(whole_box, globals()['old_array'])
        # l = [np.sum(np.asarray(elem) ==0) for elem in whole_box]
        # whole_box_person = whole_box[np.argmin(l)]
        # if np.all(np.array(whole_box_person)==0):
        #     pitcher_array.append(np.array(whole_box_person))
        #     continue
        #
        # length = len(pitcher)//2
        # pit1 = pitcher[:length]
        # pit2 = pitcher[length:]
        # #print(pitcher.shape, pit1.shape, pit2.shape)
        # out1 = np.array(handle_one(pit1))
        # out2 = np.array(handle_one(pit2))
        # if len(out1)==0:
        #     out1 = np.zeros((1, 18,2))
        # if len(out2)==0:
        #     out2 = np.zeros((1, 18,2))
	    # #print("out2", out2)
        # #print("out1", out1)
        # out2[:,:,1]+=length
        # out2[out2[:,:,1]==length] = 0
        #
        # #print("before refine", whole_box_person)
        # new_res = whole_box_person.copy()
        # new_res[np.where(new_res==0)] = globals()['old_array'][np.where(new_res==0)]
        # # print("before and with zeros filled in", new_res)
        # for i in range(len(whole_box_person)):
        #     if i<6 or i>11:
        #         out_small = out1[:,i]
        #     else:
        #         out_small = out2[:, i]
        #     for j in range(len(out_small)):
        #         if np.linalg.norm(new_res[i]-out_small[j])<30:
        #             whole_box_person[i] = out_small[j]
        # # print("after refine", whole_box_person)
        # globals()['old_array'] = whole_box_person.copy()
        # out = whole_box_person
        # # for any number of bboxes
        # out[:,0]+=bbox[0]
        # out[out[:,0]==bbox[0]] = 0
        # out[:,1]+=bbox[2]
        # out[out[:,1]==bbox[2]] = 0
        # # print("out", out)
        # # # without box or one box
        # # out = handle_one(pitcher)
        # # res, globals()['old_array'] = player_localization(out, globals()['old_array'])
        # # res = np.array(res)
        # # pitcher_array.append(res)
        #
        # #print("res ohne vorherigen eingefuellt", res)
        #
        # pitcher_array.append(out)
        # old_res = pitcher_array[-2]
        # new_res = out.copy()
        # new_res[np.where(new_res==0)]=old_res[np.where(new_res==0)]
        # if not np.all(out==0):
        #     bbox = define_bbox(new_res, boundaries)
        # print(bbox)




    # # FOR MULTIPLE BBOXES: Version 1: meiste in out2, dann auffuellen mit irgendeinem aus out1, dann falls 0 auffuellen mit whole box
    # length = len(pitcher)//2
    # pit1 = pitcher[:length]
    # pit2 = pitcher[length:]
    # #print(pitcher.shape, pit1.shape, pit2.shape)
    # out1 = handle_one(pit1)
    # out2 = handle_one(pit2)
    # if len(out1)==0:
    #     out1 = np.zeros((1, 18,2))
    # if len(out2)==0:
    #     out2 = np.zeros((1, 18,2))
    # #print("out2", out2)
    # #print("out1", out1)
    # out2[:,:,1]+=length
    # out2[out2[:,:,1]==length] = 0
    # #print("out2 after length", out2)
    # l = [np.sum(np.asarray(elem) ==0) for elem in out2]
    # #print(l)
    # most = out2[np.argmin(l)]
    # print("most", most)
    # for k in range(len(most)):
    #     if most[k, 0]==0:
    #         for i in range(len(out1)):
    #             if out1[i, k,0]!=0:
    #                 most[k] = out1[i,k]
    # #print("out2 after fill with out1", most)
    #
    # whole_box = handle_one(pitcher)
    # l = [np.sum(np.asarray(elem) ==0) for elem in whole_box]
    # whole_box_person = whole_box[np.argmin(l)]
    # print("whole_box_person", whole_box_person)
    # for i in range(len(most)):
    #     if most[i,0] ==0:
    #         print("kleiner 40?", np.linalg.norm(whole_box_person[i]-pitcher_array[-1][i])<40)
    #     if most[i,0] ==0 and np.linalg.norm(whole_box_person[i]-pitcher_array[-1][i])<40:
    #         most[i] = whole_box_person[i]
    # #most[np.where(most==0)] = whole_box_person[np.where(most==0)]
    #
    # # print("out", out.shape)
    # # print("out", out)
    # out = np.array([most])
    # # for any number of bboxes
    # out[:,:,0]+=bbox[0]
    # out[out[:,:,0]==bbox[0]] = 0
    # out[:,:,1]+=bbox[2]
    # out[out[:,:,1]==bbox[2]] = 0
    # print("out", out)
