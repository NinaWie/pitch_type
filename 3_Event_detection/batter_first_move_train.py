import pandas as pd
import numpy as np
import tensorflow as tf
import scipy as sp
import scipy.stats

import sys
sys.path.append("..")

from tools import Tools
from model import Model
import tflearn
from tflearn import DNN
import json
import argparse

from run_events import Runner
from test import test
from os import listdir
import codecs
from data_preprocess import JsonProcessor

from detect_event import first_move_batter_NN

from sklearn.metrics import mean_squared_error


"""
USAGE: Download training data from Google drive into train_data folder
drive link: https://drive.google.com/drive/folders/19K6yLft35w0QjsW2Y6XEVxhBpHBkueYx?usp=sharing
"""


def batter_testing(restore_path, data_path = "train_data/", output_path = "outputs/"):
    """
    runs test for testing data in directory batter_runs, selecting only the test data saved in the json file labels_first_batter_test
    saves the outputs as a dictionary containing the first movement frame index for each file
    """
    path = data_path+"batter_runs/" # change to data_path +
    joints_array_batter = []
    files = []
    labels = []
    prepro = JsonProcessor()
    with open(data_path+ "labels_first_batter_test", "r") as infile:
        labels_dic = json.load(infile)
    print(len(list(labels_dic.keys())))
    for fi in list(labels_dic.keys()):
        files.append(fi)
        joints_array_batter.append(prepro.from_json(path+fi+".json")[:160]) #json.loads(obj_text))
        labels.append(labels_dic[fi])
    joints_array_batter = np.array(joints_array_batter)

    lab = first_move_batter_NN(joints_array_batter.copy(), [90 for _ in range(len(joints_array_batter))], model = restore_path) # run test file for array, release frame is estimated 90 for old video data
    for l in range(len(lab)):
        print("predicted ", lab[l], "true", labels[l])
    print("mean of labels", np.mean(lab))

    print("mean squared error:", mean_squared_error(lab, labels), np.sqrt(np.sum((np.asarray(lab)-np.asarray(labels))**2)/float(len(labels))))
    dic = {}
    assert(len(lab)==len(files))
    for i in range(len(files)):
        dic[files[i]]= float(lab[i])
    #print(dic)
    with open(output_path + "batter_first_move_test_outputs", "w") as outfile: # save outputs to plot them in the "Batter first movement" jupyter notebook
        json.dump(dic, outfile)


def batter_training(save_path, shift = 20, data_path = "train_data/"):
    """
    train neural network to find the first movement (first step of batter when he starts to run) in a sequence of frames
    save_path: path to save the trained model
    """
    path = "/Volumes/Nina Backup/low_quality_testing/batter_runs/" #data_path+ "batter_runs/"
    joints_array_batter = []
    files = []
    labels = []
    prepro = JsonProcessor()

    # release = []

    #with open(path1+"release_frames Kopie", "r") as infile:
    #    release_frame = json.load(infile)

    with open(data_path+"labels_first_move_train", "r") as infile:
        labels_dic = json.load(infile)
    print("number of training data examples", len(list(labels_dic.keys())))

    for fi in list(labels_dic.keys()):
        # try:
        #     release.append(release_frame[fi])
        # except KeyError:
        #     print("no release frame")
        #     release.append(90)
        if labels_dic[fi] > 160-shift:
            print("label too high")
            continue
        files.append(fi)

        joints_array_batter.append(prepro.from_json(path+fi+".json")[:160]) #json.loads(obj_text))
        labels.append(labels_dic[fi])

    print("data sampled:")
    print(np.array(joints_array_batter).shape)
    print(labels, min(labels), max(labels))
    joints_array_batter = np.array(joints_array_batter)

    ## shift and flip
    shift1, label1 = Tools.shift_data(joints_array_batter, labels.copy(), shift_labels = True, max_shift=shift)
    # print(shift1.shape, len(label1))
    shift2, label2 = Tools.shift_data(joints_array_batter, labels.copy(), shift_labels = True, max_shift=shift)
    shift3, label3 = Tools.shift_data(joints_array_batter, labels.copy(), shift_labels = True, max_shift=shift)
    shift4, label4 = Tools.shift_data(joints_array_batter, labels.copy(), shift_labels = True, max_shift=shift)
    data_old = np.append(shift4, np.append(shift3 , np.append(shift1, shift2, axis = 0), axis = 0), axis = 0) #joints_array_batter[:, shift:len(joints_array_batter[0])-shift]
    label = np.append(label4, np.append(label3, np.append(label1, label2, axis = 0), axis = 0), axis = 0) # np.array(labels)-shift,
    print("data shifted:")
    print(data_old.shape)
    print(label, min(label), max(label))
    false = np.where(label>len(data_old[0]))[0]
    data_old = np.delete(data_old, false, axis = 0)
    label = np.delete(label, false, axis = 0)

    print("false indizes deleted:")
    print(false)
    print(data_old.shape)
    print(label, min(label), max(label))
    cutoff_min = int(min(label))
    cutoff_max = int(max(label))

    data_new = Tools.flip_x_data(data_old.copy(), x=1) #[:len(data_old)//2]
    data = np.append(data_old, data_new, axis = 0)[:, cutoff_min:cutoff_max, 6:12]
    # print("data shape", data.shape)
    # gradient = np.gradient(data, axis = 1)
    # print("gradient shape", gradient.shape)
    # data = np.append(data, gradient, axis = 3)
    # print("shape together", data.shape)
    label = np.append(label, label, axis = 0) - cutoff_min
    print("data flipped")
    print(data_new.shape)
    print(data.shape)
    print(len(label))

    ##  to save training data to inspect data and corresponding labels
    ## SEE CORRESPONDING OUTPUTS IN NOTEBOOK BATTER_FIRST_MOVEMENT last section - plotted examples and histogram of labels

    # np.save("outputs/batter_first_data", data)
    # np.save("outputs/batter_first_label", label)

    runner = Runner(np.array(data), np.reshape(label, (-1,1)), SAVE = save_path, BATCH_SZ=40, EPOCHS = 500, batch_nr_in_epoch = 50,
            act = tf.nn.relu, rate_dropout = 0,
            learning_rate = 0.0005, nr_layers = 4, n_hidden = 128, optimizer_type="adam", regularization=0,
            first_conv_filters=12, first_conv_kernel=3, second_conv_filter=12,
            second_conv_kernel=3, first_hidden_dense=128, second_hidden_dense=56,
            network = "rnn")
    runner.unique = [cutoff_max-cutoff_min]
    runner.start()

if __name__ == "__main__":
    def boolean_string(s):
        if s not in {'False', 'True'}:
            raise ValueError('Not a valid boolean string')
        return s == 'True'
    parser = argparse.ArgumentParser(description='Train neural network to find batter first step')
    parser.add_argument('-training', default= "True", type=boolean_string, help='if training, set True, if testing, set False')
    parser.add_argument('-model_save_path', default="saved_models/batter_first_rnn_10_40", type=str, help='if training, path to save model, it testing, path to restore model')
    args = parser.parse_args()

    train = args.training
    save_path = args.model_save_path

    if train:
        batter_training(save_path)
    else:
        batter_testing(save_path)
