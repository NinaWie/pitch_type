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
from run_events import Runner
from test import test
import matplotlib.pylab as plt
#from notebooks.code_to_json import from_json

from data_preprocess import JsonProcessor
from tools import Tools
# path_outputs = "/Volumes/Nina Backup/finished_outputs/"
# test_json_files = "/Volumes/Nina Backup/high_quality_outputs/"
# test_data_path = "/Users/ninawiedemann/Desktop/UNI/Praktikum/high_quality_testing/pitcher/"
# save =  "/Users/ninawiedemann/Desktop/UNI/Praktikum/ALL/saved_models/pitch_type_svcf"

path_outputs = "/scratch/nvw224/pitch_type/Pose_Estimation/outputs/"
test_json_files = "/scratch/nvw224/pitch_type/Pose_Estimation/v0testing/"
test_data_path = "/scratch/nvw224/pitch_type/Pose_Estimation/high_quality_testing/pitcher/"
save =  "/scratch/nvw224/pitch_type/saved_models/release_frame_general"
csv_path = "/scratch/nvw224/"


def norm0(data):
    return (data-np.mean(data))/np.std(data)
def norm1(data):
    # per example
    norm = data.copy()
    for i, exp in enumerate(data):
        norm[i] = (exp-np.mean(exp))/np.std(exp)
    return norm
def norm2(data):
    return (data-np.min(data))/(np.max(data)-np.min(data))
def norm3(data):
    # per trajectory
    means = np.mean(data, axis = 1)
    stds = np.mean(data, axis = 1)
    new = [(data[:,i]- means)/stds for i in range(len(data[0]))]
    return np.swapaxes(new, 0,1)
def norm4(data):
    #log transform
    log_transformed = np.log(data)
    #print(log_transformed)
    return norm2(log_transformed)

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
    shift= [50, 60, 70]
    prepro = JsonProcessor()
    arr = np.zeros((len(shift), 4))
    stds = np.zeros((len(shift), 4))
    for i, s in enumerate(shift):
        data, files = prepro.get_test_data(test_json_files, test_data_path, sequ_len, start, shift=s, labels=dic_release_high_quality)
        for n in range(5):
            data_new = eval("norm"+str(n))(data)
            lab, out = test(data_new, save_path)
            arr[i,n] = (np.sum(np.absolute(np.array(lab)-s)))/float(len(lab))
            stds[i,n] = np.std(np.array(lab))
    for x in range(5):
        print("mean error for norm ", x, np.mean(arr[:, x]))
        print("mean stds for norm ", x, np.mean(stds[:, x]))
    print(arr)
    # data = norm0(data)
    #data = scipy.ndimage.filters.gaussian_filter1d(data, axis = 1, sigma = 1)
    return 0

    print("normal verteilt", (np.sum(np.array(lab)-shift))/float(len(lab)))
    print("average error", (np.sum(np.absolute(np.array(lab)-shift)))/float(len(lab)))
    res = {}
    res["shift"]=shift
    for i in range(len(files)):
        print(lab[i], files[i])
        res[files[i]]= float(lab[i][0])
    return 0
	try:
        real = dic_release_high_quality[files[i]]
        if real>start+sequ_len or real<start:
            print("output", lab[i], "but sequence was not containing release frame")
        else:
            print("output", lab[i], "real label: ", real-start)
    except KeyError:
        print("output", lab[i], "no real available")
    with open("results_high_quality_smooth.json", "w") as outfile:
	json.dump(res, outfile)

def training(save_path, sequ_len = 100, max_shift=30):
    prepro = JsonProcessor()
    data, plays = prepro.get_data([[path_outputs+"old_videos/cf/", path_outputs+"new_videos/cf/"], [path_outputs+"old_videos/sv/"], [path_outputs+"boston/new_videos/sv/"]], sequ_len+2*max_shift)
    label = prepro.get_label_release(plays, csv_path+"cf_data.csv", "pitch_frame_index", cut_off_min=70, cut_off_max=110)
    inds = np.where(label==0)[0]
    print(inds)
    data = np.delete(data, inds, axis = 0)
    label = np.delete(label, inds, axis=0)
    print(data.shape, len(label), np.any(label==0))

    # EXTENDING
    ## squish and stretch
    # a = len(data)//4
    # faster = Tools.squish_data(data[:a], 3, required_length=sequ_len)
    # bit_faster = Tools.squish_data(data[a:2*a], 4, required_length=sequ_len)
    # fast = np.append(faster, bit_faster, axis=0)
    # print(fast.shape)
    # slower = Tools.stretch_data(data[2*a:3*a], 3)
    # bit_slower = Tools.stretch_data(data[3*a:], 4)
    # slow = np.append(slower,bit_slower, axis = 0)
    # print(slow.shape)
    # changed_time = np.append(fast, slow, axis = 0)
    # both = np.append(data, changed_time, axis = 0)
    # label = np.append(label, label)
    # print(both.shape)
    both = data

    ## shift and flip
    data_old, label = Tools.shift_data(both, label, max_shift=max_shift)
    print(data_old.shape)
    data_new = Tools.flip_x_data(data_old.copy()) #[:len(data_old)//2]
    print(data_new.shape)

    data = np.append(data_new, data_old, axis=0)
    label = np.append(label, label)
    print(data.shape, len(label))
    # data = scipy.ndimage.filters.gaussian_filter1d(data, axis = 1, sigma = 1)
    runner = Runner(data, np.reshape(label, (-1, 1)), SAVE = save_path, BATCH_SZ=40, EPOCHS = 1000, batch_nr_in_epoch = 50,
            act = tf.nn.relu, rate_dropout = 0,
            learning_rate = 0.001, nr_layers = 4, n_hidden = 128, optimizer_type="adam", regularization=0,
            first_conv_filters=56, first_conv_kernel=5, second_conv_filter=56,
            second_conv_kernel=3, first_hidden_dense=128, second_hidden_dense=56,
            network = "combined")
    runner.unique = [sequ_len]
    runner.start()


testing(save, sequ_len=100)
