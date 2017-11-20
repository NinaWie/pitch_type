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
out_dir = args.output_dir
view = args.view


f = directory

#f = subdirectory+fi
#print("file", fi, "directory", f)
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
events_dic["video_directory"]= "test" #subdirectory
events_dic["bbox_batter"] = [left_b, right_b, top_b, bottom_b]
events_dic["bbox_pitcher"] = [left_p, right_p, top_p, bottom_p]
events_dic["start_time"]=time.time()
handle_one_res=[]
while True:
# Capture frame-by-frame
    ret, frame = video_capture.read()
    if frame is None:
        print("end of video capture at frame", p)
        break
    pitcher = frame[top_p:bottom_p, left_p:right_p] #falschrum??
    batter = frame[top_b:bottom_b, left_b:right_b]
    handle_one_res.append(handle_one(batter))
    df.loc[p]=[int(p),handle_one(pitcher), handle_one(batter)]
    p+=1
print("events", events_dic)
print("Time to read in video and handle one:", time.time()-tic1)

with open(f.split("/")[-1][:-4]+"_handle_one.json", "w") as outfile:
    json.dump(outfile, handle_one_res)

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

file_path_batter = out_dir+game_id+"_batter"
to_json(batter_array, events_dic, file_path_batter, frames_per_sec=f_per_sec)

file_path_pitcher = out_dir+game_id+"_pitcher"
to_json(pitcher_array, events_dic, file_path_pitcher, frames_per_sec=f_per_sec)

toctoc=time.time()
print("Finished video ", toctoc-tic1)
