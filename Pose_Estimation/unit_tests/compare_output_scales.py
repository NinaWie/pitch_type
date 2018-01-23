
import time
from torch import np
import argparse
import pandas as pd
from os.path import isfile, join
from os import listdir
import codecs, json
import tensorflow as tf
from scipy.misc import imresize

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

colors = [[0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 0], [255, 0, 255], [0, 255, 255], [0, 170, 255],
          [0, 0, 0],
      [255, 0, 85], [0, 255, 170],  [0, 170, 255], [0, 85, 255],  [85, 0, 255],
      [170, 0, 255], [255, 0, 170], [255, 0, 85], [255, 0, 85], [0, 255, 170],  [0, 170, 255],
          [0, 85, 255],  [85, 0, 255],
      [170, 0, 255], [255, 0, 170], [255, 0, 85], [255, 0, 85], [0, 255, 170],  [0, 170, 255], [0, 85, 255],  [85, 0, 255], \
      [170, 0, 255], [255, 0, 170], [255, 0, 85]]

with open("center_dics.json", "r") as infile:
    centers = json.load(infile)

args = parser.parse_args()
inp_dir = args.input_file

out_dir = args.output_folder

def define_bbox(res, boundaries):
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

for fi in listdir(inp_dir): #__name__ == "__main__":
    if fi[0]==".":# or fi[:-4]+"_joints.json" in listdir(out_dir) or fi[:-4]+".json" in listdir(out_dir+"handle_one/"):
    	print("already there", fi)
    	continue
    f = inp_dir+fi
    img = cv2.imread(f)
    ori_shape = img.shape[:2]
    ori_img = img.copy()
    out = handle_one(img)
    print(out)
    all_peaks = out
    tic2 = time.time()
    # df_res= df_coordinates(df,center_dic, ["Pitcher"], interpolate = False)
    for x in range(len(all_peaks)):
        for j in range(len(all_peaks[x])):
            cv2.circle(ori_img, (int(all_peaks[x,j,0]),int(all_peaks[x,j,1])) , 2, colors[x], thickness=-1)
    cv2.imwrite(out_dir+fi[:-4]+"normal.jpg", ori_img)

    boundaries = [0, img.shape[1], 0, img.shape[0]]

    bbox = define_bbox(all_peaks[0], boundaries)
    cropped = img[bbox[2]:bbox[3], bbox[0]:bbox[1]]
    out = handle_one(cropped)
    print(out)
    all_peaks = out
    tic2 = time.time()
    # df_res= df_coordinates(df,center_dic, ["Pitcher"], interpolate = False)
    for x in range(len(all_peaks)):
        for j in range(len(all_peaks[x])):
            cv2.circle(cropped, (int(all_peaks[x,j,0]),int(all_peaks[x,j,1])) , 8, colors[x], thickness=-1)
    cv2.imwrite(out_dir+fi[:-4]+"cropped.jpg", cropped)

# for fi in listdir(inp_dir): #__name__ == "__main__":
#     if fi[0]==".":# or fi[:-4]+"_joints.json" in listdir(out_dir) or fi[:-4]+".json" in listdir(out_dir+"handle_one/"):
#     	print("already there", fi)
#     	continue
#     f = inp_dir+fi
#     img = cv2.imread(f)
#     ori_shape = img.shape[:2]
#     for i in range(1, 20, 2):
#         scaled = imresize(img, (int(ori_shape[0]*(i/float(20))), int(ori_shape[1]*(i/float(20))), 3))
#         out = handle_one(scaled)
#         print(out)
#         all_peaks = out
#         tic2 = time.time()
#         # df_res= df_coordinates(df,center_dic, ["Pitcher"], interpolate = False)
#
#         for x in range(len(all_peaks)):
#             for j in range(len(all_peaks[x])):
#                 cv2.circle(scaled, (int(all_peaks[x,j,0]),int(all_peaks[x,j,1])) , 2, colors[x], thickness=-1)
#
#         cv2.imwrite(out_dir+fi[:-4]+"_%d.jpg"%i, scaled)
#         with open(out_dir+fi[:-4]+"_%d.json"%i, "w") as outfile:
#             json.dump(out.tolist(), outfile)
