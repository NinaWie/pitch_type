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
parser.add_argument('view', metavar='DIR', help='cf (center field) or sv (side view)')
parser.add_argument('output_dir', metavar='DIR', help='folder where to put joint outputs') # both directories must have / in the end
# example usage: python from_videos_to_joints.py ./atl/2017-05-06/center\ field/ out_joints/ # make directory out_joints first

# either /scratch/nvw224/videos_new/05 OR /scratch/nvw224/videos/atl

args = parser.parse_args()
directory = args.input_dir
out_directory = args.output_dir
view = args.view

for day in listdir(directory):
    print("-------------new day----------", day)
    files = []
    first_subdirectory = directory+day
    sub_list = listdir(first_subdirectory)
    if len(sub_list)==1:
        out_dir_first = out_directory+"new_videos/"
        if view =="cf":
            subdirectory = first_subdirectory+"/"+sub_list[0]+"/CENTERFIELD/"
            out_dir_first = out_dir_first+"cf/"
            out_dir = out_dir_first+sub_list[0]+"_"
        elif view == "sv":
            subdirectory = first_subdirectory+"/"+sub_list[0]+"/CH_HIGH_SIDEVIEW/"
            out_dir_first = out_dir_first+"sv/"
            out_dir = out_dir_first+sub_list[0]+"_"
        else:
            raise ValueError("bad view argument, needs to be cf or sv")
    else:
        out_dir = out_directory+"old_videos/"
        if view =="cf":
            subdirectory = first_subdirectory+"/center field/"
            out_dir = out_dir+"cf/"
        elif view == "sv":
            subdirectory = first_subdirectory+"/side view/"
            out_dir = out_dir+"sv/"
        else:
            raise ValueError("bad view argument, needs to be cf or sv")
        out_dir_first = out_dir

    #already_done = listdir(out_dir_first)
    for ff in listdir(subdirectory):
        string_f = str(ff)
        if  not string_f.endswith("dat"):
            files.append(string_f)

    #already_done = listdir(out_dir_first)
    #print("number files:", len(files))
    #print(already_done)
    for fi in files:

        f = subdirectory+fi
        game_id = f.split("/")[-1][:-4]
        if isfile(out_dir+game_id+"_batter.json"): # in already_done:
            print("already_done", fi)
            continue
        print("---------------------------------------------")
        print("file", fi, "directory", f)
        video_capture = cv2.VideoCapture(f)
        #x, y = centers[fi[:-4]]#np.array([abs(top_p+bottom_p)/2., abs(left_p+right_p)/2.])
        center_dic={}
	#center_dic['Batter'] = np.array([y, x])
        try:
            for i in open(f+".dat").readlines():
                datContent=ast.literal_eval(i)
        except IOError:
            print("dat file not found", f)
            continue
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
        events_dic["video_directory"] = subdirectory
        events_dic["bbox_batter"] = [left_b, right_b, top_b, bottom_b]
        events_dic["bbox_pitcher"] = [left_p, right_p, top_p, bottom_p]
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

    	# if len(df["Batter"][0])==0 or len(df["Pitcher"][0])==0:
    	#     print("first frame not detected", f)
    	#     continue
        try:
            df_res= df_coordinates(df,center_dic, ["Pitcher", "Batter"], interpolate = True)
        except (KeyError, ValueError) as e:
            print("batter not detected in any frame")
            continue
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

        file_path_batter = out_dir+game_id+"_batter"
        to_json(batter_array, events_dic, file_path_batter, frames_per_sec=f_per_sec)

        file_path_pitcher = out_dir+game_id+"_pitcher"
        to_json(pitcher_array, events_dic, file_path_pitcher, frames_per_sec=f_per_sec)

        toctoc=time.time()
        print("Finished video ", toctoc-tic1)






#         j=0
#         center_dic={}
#         tic=time.time()
#
#
#         path_input_dat=f+'.dat'
#
#         video_capture = cv2.VideoCapture(f)
#         for i in open(path_input_dat).readlines():
#             datContent=ast.literal_eval(i)
#         bottom_p=datContent['Pitcher']['bottom']
#         left_p=datContent['Pitcher']['left']
#         right_p=datContent['Pitcher']['right']
#         top_p=datContent['Pitcher']['top']
#         bottom_b=datContent['Batter']['bottom']
#         left_b=datContent['Batter']['left']
#         right_b=datContent['Batter']['right']
#         top_b=datContent['Batter']['top']
#         center_dic['Pitcher']=np.array([abs(top_p-bottom_p)/2., abs(left_p-right_p)/2.])
#         center_dic['Batter']=np.array([abs(top_b-bottom_b)/2., abs(left_b-right_b)/2.])
#         df = pd.DataFrame(columns=['Frame','Pitcher','Batter'])
#         tic1 = time.time()
#         p=0
#         #frames_pitcher = np.zeros((167, 110, 110))
#         #frames_batter = np.zeros((167, 110, 110))
#         while p<167:
#             # Capture frame-by-frame
#             ret, frame = video_capture.read()
#             if frame is None:
#                 print("end of video capture")
#                 break
#             pitcher = frame[top_p:bottom_p, left_p:right_p]
#             batter = frame[top_b:bottom_b, left_b:right_b]
#          #   frames_pitcher[p] = cv2.resize(np.mean(pitcher, axis = 2),(110, 110), interpolation = cv2.INTER_LINEAR)/255
#           #  frames_batter[p] = cv2.resize(np.mean(batter, axis = 2),(110, 110), interpolation = cv2.INTER_LINEAR)/255
#             df.loc[p]=[int(p),handle_one(pitcher),handle_one(batter) ]
#             p+=1
#         #np.save("/scratch/nvw224/pitch_type/Pose_Estimation/out_joints/pitcher/"+fi[:-4]+"_video.npy", frames_pitcher)
#         #np.save("/scratch/nvw224/pitch_type/Pose_Estimation/out_joints/batter/"+fi[:-4]+"_video.npy", frames_batter)
#         print("Time to read in video and handle one:", time.time()-tic1)
#         #print("nach handle one shape ", df.loc[p-1]["Pitcher"].shape)
#         #try:
#         tic2 = time.time()
#         df_res= df_coordinates(df,center_dic)
#         print("time for df_coordinates", time.time()-tic2)
#         pitcher_array = np.zeros((167, 18,2))
#         for i in range(167):
#             try:
#                 pitcher_array[i,:,:] = np.array(df_res['Pitcher_player'].values[i])
#             except:
#                 pitcher_array[i,:,:] = pitcher_array[i-1,:,:]
#         print("shape", pitcher_array.shape)
#         batter_array = np.zeros((167, 18,2))
#         for i in range(167):
#             try:
#                 batter_array[i,:,:] = np.array(df_res['Batter_player'].values[i])
#             except:
#                 print(i, " IN EXCEPT")
#     	    batter_array[i,:,:] = batter_array[i-1,:,:]
#         print("shape b", batter_array.shape)
#
#         p = pitcher_array.tolist()
#         b = batter_array.tolist()
#         file_path_p = out_dir+"pitcher/"+fi[:-4]+".json"
#         file_path_b = out_dir+"batter/"+fi[:-4]+".json"
#         json.dump(p, codecs.open(file_path_p, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
#         json.dump(b, codecs.open(file_path_b, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
#         #ALL IN ONE FILE:
#         #cf = pd.read_csv("/scratch/nvw224/cf_data.csv")
#         #print("csv fuer labels eingelesen")
#         #all_classes = ['Changeup', 'Curveball', 'Fastball (2-seam)', 'Fastball (4-seam)', 'Fastball (Cut)', 'Knuckle curve', 'Knuckleball', 'Sinker', 'Slider']
#         #location_play = (cf[cf["Game"]==(f[f.rfind('/')+1:][:-4])].index.values)[0]
#         #label = np.array([(cf["Pitch Type"].values)[location_play]])
#         #runner = Runner()
#         #pitches, acc = runner.run(pitcher_array, label, all_classes, RESTORE="/scratch/nvw224/WHOLE/model1")
#         toctoc=time.time()
#         print("Time for whole video to array: ", toctoc-tic)
#
# print("FINISHED")