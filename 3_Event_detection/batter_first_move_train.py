import pandas as pd
import numpy as np
import tensorflow as tf
import scipy as sp
import os
from os import listdir
import scipy.stats

import sys
sys.path.append("..")

from utils import Tools
from model import Model
import json
import argparse

from fmo_detection import from_json
from run_events import Runner
from test_script import test


from detect_event import first_move_batter_NN

from sklearn.metrics import mean_squared_error


"""
USAGE: Download training data from Google drive into train_data folder
drive link: https://drive.google.com/drive/folders/19K6yLft35w0QjsW2Y6XEVxhBpHBkueYx?usp=sharing
"""


def batter_testing(restore_path, data_path, output_path = "outputs/"):
    """
    runs test for testing data in directory batter_runs, selecting only the test data saved in the json file labels_first_batter_test
    saves the outputs as a dictionary containing the first movement frame index for each file
    """
    # path = os.path.join(data_path, "batter_runs/")
    joints_array_batter = []
    files = []
    labels = []
    # Load labels for test data
    with open(os.path.join(data_path, "labels_first_batter_test"), "r") as infile:
        labels_dic = json.load(infile)
    print("Loaded labels, length:", len(list(labels_dic.keys())))
    # For all files in the Label dictionary, load data
    for fi in list(labels_dic.keys()):
        files.append(fi)
        joints_array_batter.append(from_json(os.path.join(data_path, fi+".json"))[:160]) # Load joint trajectories
        labels.append(labels_dic[fi])

    # Data
    joints_array_batter = np.array(joints_array_batter)
    print("Loaded Data, shape:", joints_array_batter.shape)

    # first_move_batter_NN function restores the model and predicts the first step (with 90 as the release frame)
    res = first_move_batter_NN(joints_array_batter.copy(), [90 for _ in range(len(joints_array_batter))], model = restore_path) # run test file for array, release frame is estimated 90 for old video data

    # Display results
    print("OUTPUTS:")
    for l in range(len(res)):
        print("predicted frame:", res[l], " - true:", labels[l])
    print("mean of results", np.mean(res))

    print("mean squared error:", mean_squared_error(res, labels))

    # Make dictionary with results and save it
    dic = {}
    assert(len(res)==len(files))
    for i in range(len(files)):
        dic[files[i]]= float(res[i])
    with open(os.path.join(output_path, "batter_first_move_test_outputs"), "w") as outfile: # save outputs to plot them in the "Batter first movement" jupyter notebook
        json.dump(dic, outfile)


def batter_training(save_path, data_path, shift_left = 20, shift_right=20, cutoff_min = 100, cutoff_max = 140):
    """
    train neural network to find the first movement (first step of batter when he starts to run) in a sequence of frames
    save_path: path to save the trained model
    shift: how much the data is shifted maximally in each direction
    """
    joints_array_batter = []
    files = []
    labels = []

    # release = []

    #with open(path1+"release_frames Kopie", "r") as infile:
    #    release_frame = json.load(infile)

    with open(os.path.join(data_path,"labels_first_move_train"), "r") as infile:
        labels_dic = json.load(infile)
    print("number of training data examples", len(list(labels_dic.keys())))

    for fi in list(labels_dic.keys()):
        files.append(fi)

        joints_array_batter.append(from_json(os.path.join(data_path, fi+".json"))[:160]) #json.loads(obj_text))
        labels.append(labels_dic[fi])

    print("joint trajectories", np.array(joints_array_batter).shape)
    print("Length labels", len(labels))
    joints_array_batter = np.array(joints_array_batter)

    ## shift and flip
    shift1, label1 = Tools.shift_data(joints_array_batter.copy(), labels.copy(), shift_labels = True, shift_left=shift_left, shift_right=shift_right)
    shift2, label2 = Tools.shift_data(joints_array_batter.copy(), labels.copy(), shift_labels = True, shift_left=shift_left, shift_right=shift_right)
    shift3, label3 = Tools.shift_data(joints_array_batter.copy(), labels.copy(), shift_labels = True, shift_left=shift_left, shift_right=shift_right)
    shift4, label4 = Tools.shift_data(joints_array_batter.copy(), labels.copy(), shift_labels = True, shift_left=shift_left, shift_right=shift_right)
    data_old = np.append(shift4, np.append(shift3 , np.append(shift1, shift2, axis = 0), axis = 0), axis = 0) #joints_array_batter[:, shift:len(joints_array_batter[0])-shift]
    label = np.append(label4, np.append(label3, np.append(label1, label2, axis = 0), axis = 0), axis = 0) # np.array(labels)-shift,
    print("Shape of data after shifted randomly:", data_old.shape)
    # print(data_old.shape)
    print("min and max of label:", min(label), max(label))
    false = np.where(label>cutoff_max)[0]
    data_old = np.delete(data_old, false, axis = 0)
    label = np.delete(label, false, axis = 0)
    print("Delete data were the label is too high:")
    print("length of deleted:", len(false))

    false = np.where(label<cutoff_min)[0]
    data_old = np.delete(data_old, false, axis = 0)
    label = np.delete(label, false, axis = 0)


    print("Delete data where the label is too low:")
    print("length of deleted:", len(false))
    print("New data shape", data_old.shape)
    print("New min and max label:", min(label), max(label))

    data_new = Tools.flip_x_data(data_old.copy(), x=1) #[:len(data_old)//2]
    data = np.append(data_old, data_new, axis = 0)[:, cutoff_min:cutoff_max, :12] # Cut data to 40 frames
    # print("data shape", data.shape)
    # gradient = np.gradient(data, axis = 1)
    # print("gradient shape", gradient.shape)
    # data = np.append(data, gradient, axis = 3)
    # print("shape together", data.shape)
    label = np.append(label, label, axis = 0) - cutoff_min # Change labels to range 0,40
    print("data flipped")
    print("Flipped data shape", data_new.shape)
    print("flipped plus previous data shape:", data.shape)
    print("Length of labels", len(label))
    print("Min and max label:", min(label), max(label))

    ## save training data to inspect data and corresponding labels
    ## SEE CORRESPONDING OUTPUTS IN NOTEBOOK BATTER_MOVEMENT last section - plotted examples and histogram of labels
    np.save("../train_data/batter_first_data", data)
    np.save("../train_data/batter_first_label", label)

    runner = Runner(np.array(data), np.reshape(label, (-1,1)), SAVE = save_path, BATCH_SZ=40, EPOCHS = 200, batch_nr_in_epoch = 50,
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
    parser.add_argument('-model_save_path', default="../saved_models/batter_first_step", type=str, help='if training, path to save model, it testing, path to restore model')
    parser.add_argument('-data_path', default="../train_data/batter_runs", type=str, help='path to data for training and testing')
    args = parser.parse_args()

    train = args.training
    save_path = args.model_save_path

    if train:
        batter_training(save_path, args.data_path)
    else:
        batter_testing(save_path, args.data_path)
