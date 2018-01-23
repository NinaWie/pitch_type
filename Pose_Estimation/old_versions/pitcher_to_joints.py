
import time
from torch import np
import argparse
import pandas as pd
from os.path import isfile, join
from os import listdir
import codecs, json
import tensorflow as tf

from Functions import handle_one,df_coordinates , to_json
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

### TENSORFLOW PART
# saver = tf.train.import_meta_graph(restore_release+'.meta')
# graph = tf.get_default_graph()
# try:
#     sess = tf.InteractiveSession()
# except:
#     sess = tf.Session()
# print("session started")
# saver.restore(sess, restore_release)
# print("session restored")
# out = tf.get_collection("out")[0]
# unique = tf.get_collection("unique")[0]#
###

for fi in listdir(inp_dir): #__name__ == "__main__":
    if fi[0]==".":# or fi[:-4]+"_joints.json" in listdir(out_dir) or fi[:-4]+".json" in listdir(out_dir+"handle_one/"):
    	print("already there", fi)
    	continue
    j=0
    center_dic={}
    tic=time.time()
    f = inp_dir+fi

    video_capture = cv2.VideoCapture(f)
    #x, y = centers[fi[:-4]]#np.array([abs(top_p+bottom_p)/2., abs(left_p+right_p)/2.])
    center_dic['Pitcher'] = centers[fi[:-4]] #np.array([y, x])
    #print(fi, "center: ", center_dic["Pitcher"])

    df = pd.DataFrame(columns=['Frame','Pitcher'])
    tic1 = time.time()
    events_dic = {}
    events_dic["video_directory"]= inp_dir
    events_dic["bbox_batter"] = [0,0,0,0]
    events_dic["bbox_pitcher"] = [0,0,0,0]
    events_dic["start_time"]=time.time()
    rel = []
    #handle_one_res = []
    found = False
    p=0
    while p<5:
# Capture frame-by-frame
        ret, frame = video_capture.read()
        if frame is None:
            print("end of video capture")
            break
        pitcher = frame #[top_p:bottom_p, left_p:right_p]
        # print(p, "in video")
        df.loc[p]=[int(p),handle_one(pitcher)]
	#handle_one_res.append(handle_one(pitcher).tolist())
	# print("handle one finished")
        # # FOR RELEASE FRAME
        # input_release_frame = cv2.resize(np.mean(pitcher, axis = 2),(55, 55), interpolation = cv2.INTER_LINEAR)/255
        # data = np.reshape(input_release_frame, (1, 55, 55, 1))
        # if not found:
        #     out_release_frame = sess.run(out, {"input:0":  data, "training:0": False})
        #     rel.append(out_release_frame[0,1])
        #     print(out_release_frame)
        #     if out_release_frame[0,1]>0.1:
        #         events_dic["release_frame"] = p
        	# found = True
        p+=1
    #sort = np.argsort(np.array(rel))
    # # FOR RELEASE FRAME
    # print(sort)
    # events_dic["release_frame"] = sort[-1]
    # print(events_dic)
    # print("Time to read in video and handle one:", time.time()-tic1)
    # sess.close()
    #print(len(handle_one_res[0]), len(handle_one_res[0][0]))
    #print(handle_one_res)
    #game_id = f.split("/")[-1][:-4]
    #with open(out_dir+"handle_one/test_functions.json", "w") as outfile: #+game_id+".json", "w") as outfile:
    #    json.dump(list(handle_one_res), outfile)

  #  df.to_csv(out_dir+"handle_one/"+game_id+".csv")
    #break #continue

    tic2 = time.time()
    df_res= df_coordinates(df,center_dic, ["Pitcher"], interpolate = False)

    print("time for df_coordinates", time.time()-tic2)
    pitcher_array = np.zeros((p, 18,2))
    for i in range(p):
        try:
            pitcher_array[i,:,:] = np.array(df_res['Pitcher_player'].values[i])
        except:
            pitcher_array[i,:,:] = pitcher_array[0,i-1,:,:]
    print("shape", pitcher_array.shape)

    # # NEW: Pitching Position:
    # pitcher_norm = normalize(pitcher_array)
    # data = pitcher_norm[:,:,:12,:]
    # print("shape position", data.shape)
    # pos, out = test(data, restore_position)
    # print("POSITION", pos)
    #
    # # NEW: FIRST MOVE
    # r = events_dic["release_frame"]
    # joints_array = pitcher_norm[:, r-sequ_len:r, [7,8,10,11], :]
    # if len(joints_array[0])==4:
	# joints_array = np.swapaxes(joints_array, 1, 2)
    # print("joints array", joints_array.shape)
    # lab, out = test(joints_array, restore_first_move)
    # label = r-sequ_len+lab[0]
    # print("FIRST MOVE", label)
    # events_dic["first_move"]=label

    # NEW: JSON FORMAT
    game_id = fi[:-4]
    file_path_pitcher = out_dir+game_id
    print(events_dic, file_path_pitcher)
    to_json(pitcher_array, events_dic, file_path_pitcher)

    toctoc=time.time()
    print("Time for whole video to array: ", toctoc-tic)
