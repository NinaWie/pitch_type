import pandas as pd
import numpy as np
import tensorflow as tf
import scipy as sp
import scipy.stats
import json
from os import listdir
import cv2
import ast
import json

import matplotlib.pylab as plt
#from notebooks.code_to_json import from_json

from run_thread import Runner
from test import test
from data_preprocess import JsonProcessor
from tools import Tools


def training(save_path, data_path, csv, label_name= "Pitch Type", player = "pitcher", sequ_len = 160, max_shift=30):
    prepro = JsonProcessor()
    data, plays = prepro.get_data(data_path, sequ_len=sequ_len, player=player)
    assert len(data)==len(plays)
    # data, plays = prepro.get_data_concat(path_outputs+ "old_videos/cf/", path_outputs+ "old_videos/sv/", sequ_len=sequ_len, player="pitcher")
    label = prepro.get_label(plays, csv, label_name) # , csv_path+ "sv_data.csv"  "Pitch Type"
    #print(label)
    inds = np.where(np.array(label)=="Unknown")[0]
    print(inds)
    files = np.delete(plays, inds, axis = 0)
    data = np.delete(data, inds, axis = 0)
    label = np.delete(label, inds, axis = 0)
    print(data.shape, len(label), np.any(label=="Unknown"), np.any(pd.isnull(label)))
    both = data

    ## shift and flip
    data_old, label = Tools.shift_data(both, label, shift_labels = False, max_shift=max_shift)
    print(data_old.shape)
    data_new = Tools.flip_x_data(data_old.copy()) #[:len(data_old)//2]
    print(data_new.shape)

    data = np.append(data_new, data_old, axis=0)
    label = np.append(label, label)
    #np.save("saved_for_testing.npy", data)
    #np.save("saved_for_testing_label.npy", label)
    print(data.shape, len(label))

    runner = Runner(data, label, SAVE = save_path, files=files, BATCH_SZ=40, EPOCHS = 1000, batch_nr_in_epoch = 100,
            act = tf.nn.relu, rate_dropout = 0.4,
            learning_rate = 0.0005, nr_layers = 4, n_hidden = 128, optimizer_type="adam", regularization=0,
            first_conv_filters=128, first_conv_kernel=5, second_conv_filter=128,
            second_conv_kernel=9, first_hidden_dense=128, second_hidden_dense=0,
            network = "adjustable conv1d") #conv1d_big")
    runner.start()


# path_outputs = "/Volumes/Nina Backup/finished_outputs/"
# test_json_files = "/Volumes/Nina Backup/high_quality_outputs/"
# test_data_path = "/Users/ninawiedemann/Desktop/UNI/Praktikum/high_quality_testing/pitcher/"
# save =  "/Users/ninawiedemann/Desktop/UNI/Praktikum/ALL/saved_models/pitch_type_svcf"
# csv_path = "train_data/"

path_outputs = "/scratch/nvw224/pitch_type/Pose_Estimation/outputs/"
test_json_files = "/scratch/nvw224/pitch_type/Pose_Estimation/v0testing/"
test_data_path = "/scratch/nvw224/pitch_type/Pose_Estimation/high_quality_testing/pitcher/"
save =  "/scratch/nvw224/pitch_type/saved_models/position_extended"
csv_path = "/scratch/nvw224/"

input_data_list = [[path_outputs+ "boston/new_videos/sv/"],  [path_outputs+ "old_videos/cf/"], [path_outputs+ "old_videos/sv/"]]
csv_list = [csv_path + "cf_data.csv", csv_path + "csv_gameplay.csv", csv_path + "BOS_SV_metadata.csv"]

#testing(save, sequ_len=160)
training(save, label_name = "Pitching Position (P)", data_path = input_data_list, csv = csv_list, player="pitcher") #label_name = "Pitching Position (P)", data_path = "old_videos/cf/", csv = "cf_data_cut")
