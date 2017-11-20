import pandas as pd
from os import listdir
import codecs
import json
import numpy as np
from scipy import ndimage
import matplotlib.pylab as plt
from tools import Tools
import tensorflow as tf
from run_thread import Runner


csv = pd.read_csv("/scratch/nvw224/csv_gameplay.csv", delimiter = ";")

path="/scratch/nvw224/pitch_type/Pose_Estimation/out_joints/batter/"
save_path = "/scratch/nvw224/pitch_type/saved_models/batter_runs"

# joints_array_batter = []
# files = []
# hit_into = []
# #play_outcome = []
# #release = []
#
# #with open(path+"release_frames Kopie", "r") as infile:
#  #   release_frame = json.load(infile)
#
# for fi in listdir(path):
#     if fi[-5:]==".json":
#         files.append(fi[:-5])
#         line = csv[csv["play_id"]==fi[:-5]]
#         try:
#             hit_into.append(line["Hit into play?"].values[0])
# #          release.append(release_frame[fi[:-5]])
#         except IndexError:
#             continue
#         except KeyError:
#             continue
#
#         # play_outcome.append(line["Play Outcome"].values[0])
#         obj_text = codecs.open(path+fi, encoding='utf-8').read()
#         joints_array_batter.append(json.loads(obj_text))
#
# joints_array_batter = np.array(joints_array_batter)[:,:,:12,:]
# joints_array_batter = ndimage.filters.gaussian_filter1d(joints_array_batter, axis =1, sigma = 3)
# print(joints_array_batter.shape)
# # print(hit_into)
# assert(len(hit_into)==len(joints_array_batter))
# # print(release)
# data = Tools.normalize(joints_array_batter)


# csv = pd.read_csv("/Users/ninawiedemann/Desktop/UNI/Praktikum/csvs/csv_gameplay.csv", delimiter = ";")

# path="/Users/ninawiedemann/Desktop/UNI/Praktikum/ALL/video_to_pitchtype_directly/different_batters/batter/"
joints_array_batter = []
files = []
hit_into = []
play_outcome = []
#release = []

#with open(path+"release_frames Kopie", "r") as infile:
 #   release_frame = json.load(infile)

for fi in listdir(path):
    if fi[-5:]==".json":
        files.append(fi[:-5])
        line = csv[csv["play_id"]==fi[:-5]]
        try:
            hit  = line["Hit into play?"].values[0]
            out = line["Play Outcome"].values[0]
#          release.append(release_frame[fi[:-5]])
        except IndexError:
            continue

        obj_text = codecs.open(path+fi, encoding='utf-8').read()

        arr = json.loads(obj_text)
        if np.all(np.array(arr)==0):
            continue

        if "Foul" in out or "Swinging strike" in out:
            lab = "hit"
        elif "Ball/Pitcher" in out or "Called strike" in out:
            lab = "nothing"
        elif "Hit into play" in out:
            lab = "run"
        else:
            print("none of all: ", out)
            continue

        print(out, lab)

        play_outcome.append(lab)
        joints_array_batter.append(arr)
        hit_into.append(hit)

joints_array_batter = np.array(joints_array_batter)[:,:,:12,:]
joints_array_batter = ndimage.filters.gaussian_filter1d(joints_array_batter, axis =1, sigma = 3)
print(joints_array_batter.shape)
# print(hit_into)
assert(len(hit_into)==len(joints_array_batter))
assert(len(play_outcome)==len(joints_array_batter))
# print(release)
data = Tools.normalize(joints_array_batter)

runner = Runner(data, play_outcome, SAVE = save_path, BATCH_SZ=40, EPOCHS = 60, batch_nr_in_epoch = 10,
        act = tf.nn.relu, rate_dropout = 0,
        learning_rate = 0.0001, nr_layers = 4, n_hidden = 128, optimizer_type="adam", regularization=0,
        first_conv_filters=128, first_conv_kernel=5, second_conv_filter=128,
        second_conv_kernel=9, first_hidden_dense=128, second_hidden_dense=0,
        network = "adjustable conv1d")

runner.start()
