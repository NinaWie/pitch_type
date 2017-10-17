import cv2
import time
from torch import np
import argparse
import pandas as pd
from os.path import isfile, join
from os import listdir
import codecs, json

from Functions import handle_one,df_coordinates
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
    game_id = listdir(subdirectory)[0]
    subdirectory = subdirectory+"/"+game_id+"/CENTERFIELD/"
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
        print(fi)
        f = subdirectory+fi
        ext= f[-4:]
        j=0
        center_dic={}
        tic=time.time()


        path_input_dat=f+'.dat'

        video_capture = cv2.VideoCapture(f)
        for i in open(path_input_dat).readlines():
            datContent=ast.literal_eval(i)
        bottom_p=datContent['Pitcher']['bottom']
        left_p=datContent['Pitcher']['left']
        right_p=datContent['Pitcher']['right']
        top_p=datContent['Pitcher']['top']
        bottom_b=datContent['Batter']['bottom']
        left_b=datContent['Batter']['left']
        right_b=datContent['Batter']['right']
        top_b=datContent['Batter']['top']
        center_dic['Pitcher']=np.array([abs(top_p-bottom_p)/2., abs(left_p-right_p)/2.])
        center_dic['Batter']=np.array([abs(top_b-bottom_b)/2., abs(left_b-right_b)/2.])
        df = pd.DataFrame(columns=['Frame','Pitcher','Batter'])
        tic1 = time.time()
        p=0
        #frames_pitcher = np.zeros((167, 110, 110))
        #frames_batter = np.zeros((167, 110, 110))
        while p<167:
            # Capture frame-by-frame
            ret, frame = video_capture.read()
            if frame is None:
                print("end of video capture")
                break
            pitcher = frame[top_p:bottom_p, left_p:right_p]
            batter = frame[top_b:bottom_b, left_b:right_b]
         #   frames_pitcher[p] = cv2.resize(np.mean(pitcher, axis = 2),(110, 110), interpolation = cv2.INTER_LINEAR)/255
          #  frames_batter[p] = cv2.resize(np.mean(batter, axis = 2),(110, 110), interpolation = cv2.INTER_LINEAR)/255
            df.loc[p]=[int(p),handle_one(pitcher),handle_one(batter) ]
            p+=1
        #np.save("/scratch/nvw224/pitch_type/Pose_Estimation/out_joints/pitcher/"+fi[:-4]+"_video.npy", frames_pitcher)
        #np.save("/scratch/nvw224/pitch_type/Pose_Estimation/out_joints/batter/"+fi[:-4]+"_video.npy", frames_batter)
        print("Time to read in video and handle one:", time.time()-tic1)
        #print("nach handle one shape ", df.loc[p-1]["Pitcher"].shape)
        #try:
        tic2 = time.time()
        df_res= df_coordinates(df,center_dic)
        print("time for df_coordinates", time.time()-tic2)
        pitcher_array = np.zeros((167, 18,2))
        for i in range(167):
            try:
                pitcher_array[i,:,:] = np.array(df_res['Pitcher_player'].values[i])
            except:
                pitcher_array[i,:,:] = pitcher_array[i-1,:,:]
        print("shape", pitcher_array.shape)
        batter_array = np.zeros((167, 18,2))
        for i in range(167):
            try:
                batter_array[i,:,:] = np.array(df_res['Batter_player'].values[i])
            except:
                print(i, " IN EXCEPT")
    	    batter_array[i,:,:] = batter_array[i-1,:,:]
        print("shape b", batter_array.shape)

        p = pitcher_array.tolist()
        b = batter_array.tolist()
        file_path_p = out_dir+"pitcher/"+fi[:-4]+".json"
        file_path_b = out_dir+"batter/"+fi[:-4]+".json"
        json.dump(p, codecs.open(file_path_p, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
        json.dump(b, codecs.open(file_path_b, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
        #ALL IN ONE FILE:
        #cf = pd.read_csv("/scratch/nvw224/cf_data.csv")
        #print("csv fuer labels eingelesen")
        #all_classes = ['Changeup', 'Curveball', 'Fastball (2-seam)', 'Fastball (4-seam)', 'Fastball (Cut)', 'Knuckle curve', 'Knuckleball', 'Sinker', 'Slider']
        #location_play = (cf[cf["Game"]==(f[f.rfind('/')+1:][:-4])].index.values)[0]
        #label = np.array([(cf["Pitch Type"].values)[location_play]])
        #runner = Runner()
        #pitches, acc = runner.run(pitcher_array, label, all_classes, RESTORE="/scratch/nvw224/WHOLE/model1")
        toctoc=time.time()
        print("Time for whole video to array: ", toctoc-tic)

print("FINISHED")
