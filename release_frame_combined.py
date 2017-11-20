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
from run_thread import Runner
from test import test
import matplotlib.pylab as plt
#from notebooks.code_to_json import from_json

from data_preprocess import JsonProcessor
path_outputs = "/Volumes/Nina Backup/finished_outputs/"
test_json_files = "/Volumes/Nina Backup/high_quality_outputs/"
test_data_path = "/Users/ninawiedemann/Desktop/UNI/Praktikum/high_quality_testing/pitcher/"
save =  "/Users/ninawiedemann/Desktop/UNI/Praktikum/ALL/saved_models/pitch_type_svcf"
csv_path = ""

# path_outputs = "/scratch/nvw224/pitch_type/Pose_Estimation/outputs/"
# test_json_files = "/scratch/nvw224/pitch_type/Pose_Estimation/v0testing/"
# test_data_path = "/scratch/nvw224/pitch_type/Pose_Estimation/high_quality_testing/pitcher/"
# save =  "/scratch/nvw224/pitch_type/saved_models/release_frame_general"
# csv_path = "/scratch/nvw224/"

def shift_data(data, labels, max_shift=30):
    new_data=[]
    for i in range(len(data)):
        shift = np.random.randint(-max_shift, max_shift)
        new = np.roll(data[i], shift, axis=0)
        labels[i] = labels[i]+shift-max_shift
        new_data.append(new[max_shift:len(new)-max_shift])
    return np.array(new_data), labels

def flip_x_data(data):
    for i in range(len(data)):
        mean = np.mean(data[i, :, :, 0])
        flipped = (data[i, :,:,0]-mean)*(-1)
        flipped+=mean
        data[i,:,:,0] = flipped
    return data

def get_test_data(inp_dir,  sequ_len, start, labels=None):
    #sequ_len=150
    data = []

    filenames = []
    for files in listdir(inp_dir):
        name = files.split(".")[0]
        if name+".mp4" in listdir(test_data_path):
            prepro = JsonProcessor()
            array = prepro.from_json(inp_dir+files)
            if labels is None:
                if len(array)>start+sequ_len:
                    data.append(array[start:start+sequ_len])
                    filenames.append(name)
            else:
                real = labels[name]
                data.append(array[real-70:real+30])
                filenames.append(name)
    return np.array(data), filenames

def training_pitchtype(save_path, sequ_len = 160, max_shift=30):
    prepro = JsonProcessor()
    data, plays = prepro.get_data([[path_outputs+ "old_videos/cf/"]], sequ_len=sequ_len, player="pitcher")
    # data, plays = prepro.get_data_concat(path_outputs+ "old_videos/cf/", path_outputs+ "old_videos/sv/", sequ_len=sequ_len, player="pitcher")
    label = prepro.get_label_pitchtype(plays, [csv_path+"cf_data_cut.csv"], "Pitch Type") # , csv_path+ "sv_data.csv"
    print(label)
    inds = np.where(np.array(label)=="Unknown Pitch Type")[0]
    print(inds)
    data = np.delete(data, inds, axis = 0)
    label = np.delete(label, inds, axis=0)
    print(data.shape, len(label), np.any(label=="Unknown Pitch Type"))
    runner = Runner(data, label, SAVE = save_path, BATCH_SZ=40, EPOCHS = 60, batch_nr_in_epoch = 100,
            act = tf.nn.relu, rate_dropout = 0,
            learning_rate = 0.0005, nr_layers = 4, n_hidden = 128, optimizer_type="adam", regularization=0,
            first_conv_filters=128, first_conv_kernel=5, second_conv_filter=128,
            second_conv_kernel=9, first_hidden_dense=128, second_hidden_dense=0,
            network = "adjustable conv1d")
    runner.start()


def testing(save_path, sequ_len=100, start = 80):
    dic_release_high_quality = {"#31 LHP Michael Chavez (2)": 165, "#00 RHP Devin Smith": 130, "#4 RHP Parker Swindell": 125,
                           "#5 RHP Matt Blais (2)": 335, "#5 RHP Matt Blais (3)":335, "#5 RHP Matt Blais (4)": 138,
                            "#5 RHP Matt Blais (5)": 305, "#5 RHP Matt Blais 5-3 GO (2)": 158, "#5 RHP Matt Blais 5-3 GO": 160,
                            "#8 RHP Cole Johnson (2)": 115, "#8 RHP Cole Johnson": 90, "#9 RHP Ryan King (2)": 170,
                            "#9 RHP Ryan King": 150, "#26 RHP Tim Willites (2)":190, "#26 RHP Tim Willites (3)":158,
                            "#26 RHP Tim Willites": 188, "#31 LHP Michael Chavez (3)": 160, "#31 LHP Michael Chavez":345,
                            "#31 LHP Michael Chavez 5-3 GO": 78, "#42 LHP Michael Chavez": 470, "#45 LHP Steffen Simmons (2)":230,
                            "#45 LHP Steffen Simmons": 230, "#48 RHP Tom Flippin 6-3 GO": 525, "#48 RHP Tom Flippin": 282,
                            "Menlo Park Legends #98 RHP Zac Grotz":290, "#5 RHP Matt Blais": 252}
    #with open("dic_release_high_quality.json", "r") as outfile:
    #    dic_release_high_quality = json.load(outfile)
    data, files = get_test_data(test_json_files, sequ_len, start, labels=dic_release_high_quality)
    data = (data-np.mean(data))/np.std(data)
    # plt.figure(figsize=(20,10))
    # plt.plot(data[:,:,0,0])
    # plt.show()
    # plt.figure(figsize=(20,10))
    # plt.plot(data[:,:,0,1])
    # plt.show()
    lab, out = test(data, save_path)
    for i in range(len(files)):
        try:
            real = dic_release_high_quality[files[i]]
            if real>start+sequ_len or real<start:
                print("output", lab[i], "but sequence was not containing release frame")
            else:
                print("output", lab[i], "real label: ", real-start)
        except KeyError:
            print("output", lab[i], "no real available")

def training(save_path, sequ_len = 100, max_shift=30):

    prepro = JsonProcessor()
    data, plays = prepro.get_data([[path_outputs+"old_videos/cf/", path_outputs+"new_videos/cf/"], [path_outputs+"old_videos/sv/"]], sequ_len+2*max_shift)
    label = prepro.get_label_release(plays, csv_path+"cf_data.csv", "pitch_frame_index", cut_off_min=70, cut_off_max=110)
    inds = np.where(label==0)[0]
    print(inds)
    data = np.delete(data, inds, axis = 0)
    label = np.delete(label, inds, axis=0)
    print(data.shape, len(label), np.any(label==0))

    data_old, label = shift_data(data, label, max_shift=max_shift)
    print(data_old.shape)
    data_new = flip_x_data(data_old.copy())
    print(data_new.shape)
    data = np.append(data_new, data_old, axis=0)
    label = np.append(label, label)
    print(data.shape)

    runner = Runner(data, np.reshape(label, (-1, 1)), SAVE = save_path, BATCH_SZ=40, EPOCHS = 300, batch_nr_in_epoch = 50,
            act = tf.nn.relu, rate_dropout = 0,
            learning_rate = 0.0005, nr_layers = 4, n_hidden = 128, optimizer_type="adam", regularization=0,
            first_conv_filters=12, first_conv_kernel=3, second_conv_filter=12,
            second_conv_kernel=3, first_hidden_dense=128, second_hidden_dense=56,
            network = "combined")
    runner.unique = [sequ_len-2*max_shift]
    runner.start()


training_pitchtype(save)
