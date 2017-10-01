import cv2
import time
from torch import np
import argparse
import pandas as pd
from os.path import isfile, join
from os import listdir
import codecs, json

from Functions import handle_one, df_coordinates, score_coordinates
import ast


parser = argparse.ArgumentParser(description='Pose Estimation Baseball')
parser.add_argument('input_file', metavar='DIR', # Video file to be processed
                    help='folder where merge.csv are')

# example arg: /Volumes/Nina\ Backup/videos/atl/2017-04-15/center\ field/490266-0aeec26e-80f7-409f-8ae7-a40b834b3a81.mp4

#dates = ["2017-04-14", "2017-04-18", "2017-05-02", "2017-05-06"] # , "2017-05-19", "2017-05-23", "2017-06-06", "2017-06-10", "2017-06-18", "2017-06-22", "2017-07-04", "2017-07-16",
#"2017-04-15", "2017-04-19", "2017-05-03", "2017-05-07", "2017-05-20", "2017-05-24", "2017-06-07", "2017-06-11", "2017-06-19", "2017-06-23", "2017-07-05", "2017-07-17"]
# only first two rows von den im cluster angezeigten
# for date in dates:

args = parser.parse_args()
f = args.input_file
ext= f[-4:] #args.extension
print("input file: ",f, ext)

if True: #__name__ == "__main__":
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
    while True:
# Capture frame-by-frame
        ret, frame = video_capture.read()
        if frame is None:
            print("end of video capture")
            break
        pitcher = frame[top_p:bottom_p, left_p:right_p]
        batter = frame[top_b:bottom_b, left_b:right_b]
        df.loc[p]=[int(p),handle_one(pitcher),handle_one(batter) ]
        p+=1
    print("Time to read in video and handle one:", time.time()-tic1)
    #print("nach handle one shape ", df.loc[p-1]["Pitcher"].shape)
    #try:
    tic2 = time.time()
    df_res= df_coordinates(df,center_dic)
    print("time for df_coordinates", time.time()-tic2)
    pitcher_array = np.zeros((1, 167, 18,2))
    for i in range(167):
        try:
            pitcher_array[0,i,:,:] = np.array(df_res['Pitcher_player'].values[i])
        except:
            pitcher_array[0,i,:,:] = pitcher_array[0,i-1,:,:]
    print("shape", pitcher_array.shape)
    b = pitcher_array.tolist()
    file_path = "pitcher_array.json"
    json.dump(b, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)

    score_coordinates()


# serialized = json.dumps(memfile.read().decode('latin-1'))
#np.save("pitcher_arr", pitcher_array)

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
