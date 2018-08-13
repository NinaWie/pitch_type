import pandas as pd
import numpy as np
import tensorflow as tf
import scipy as sp
import scipy.stats
import json
import os
from os import listdir
import cv2
import time
import argparse
import json

from config import cfg
import sys
sys.path.append("..")

from run_thread import Runner
from test_script import test
from utils import Tools

def testing(data, labels, save_path):
    """
    Tests movement classification model on the first 5% of the data in the csv file (trained on last 95%)
    """

    print("Data shape", data.shape, "Mean of data", np.mean(data))
    tic = time.time()
    labs, out = test(data, save_path)
    toc = time.time()
    print("time for nr labels", len(labs), toc-tic)
    for i in range(20): #len(labs)):
        print(labs[i], np.around(out[i],2))

    #  To compare with labels
    print(labels.shape)
    for i in range(20): #len(labels)):
        print('{:20}'.format(labels[i]), '{:20}'.format(labs[i])) #, ['%.2f        ' % elem for elem in out_test[i]])
    print("Accuracy:",Tools.accuracy(np.asarray(labs), labels))
    print("Balanced accuracy:", Tools.balanced_accuracy(np.asarray(labs), labels))


def training(data, labels, save_path):
    runner = Runner(data, labels, SAVE = save_path, BATCH_SZ=cfg.batch_size, EPOCHS = cfg.epochs, batch_nr_in_epoch = cfg.batches_per_epoch,
            act = tf.nn.relu, rate_dropout =  cfg.dropout,
            learning_rate = cfg.learning_rate, nr_layers = cfg.layers_lstm, n_hidden = cfg.hidden_lstm, optimizer_type="adam",
            first_conv_filters=cfg.first_filters, first_conv_kernel=cfg.first_kernel, second_conv_filter=cfg.second_conv_filter,
            second_conv_kernel=cfg.second_conv_kernel, first_hidden_dense=cfg.first_hidden_dense, second_hidden_dense=cfg.second_hidden_dense,
            network = "adjustable conv1d") #conv1d_big")
    runner.start()

if __name__ == "__main__":
    def boolean_string(s):
        if s not in {'False', 'True'}:
            raise ValueError('Not a valid boolean string')
        return s == 'True'
    parser = argparse.ArgumentParser(description='Train/test neural network for recognizing pitch type from joint trajectories')
    parser.add_argument('-training', default= "True", type=boolean_string, help='if training, set True, if testing, set False')
    parser.add_argument('-label', default="Pitch Type", type=str, help='Pitch Type, Play Outcome or Pitching Position (P) possible so far')
    parser.add_argument('save_path', default="/scratch/nvw224/pitch_type/new_models/position", type=str, help='usually training to classify pitch type, but can also be used for pitching position (with the right model)')
    args = parser.parse_args()

    save = args.save_path
    train_data_path = os.path.join("..", "train_data")

    if args.label=="Pitch Type" or args.label=="Pitching Position (P)":
        csv_path = os.path.join(train_data_path, "cf_pitcher.csv")
    elif args.label=="Play Outcome":
        csv_path = os.path.join(train_data_path, "cf_batter.csv")
    else:
        print("USAGE: WRONG INPUT FOR -label ARGUMENT (Pitch Type, Play Outcome or Pitching Position (P))")
        import sys
        sys.exit()

    label_name = args.label
    csv = pd.read_csv(csv_path)
    print("Number of data:", len(csv.index))

    # TEST DATA: 5% of the csv file is used as test data. Random indices are saved and excluded during training,
    # and then the network can be tested on these 5%.
    test_data_cutoff = len(csv.index)//20
    # TO SAVE NEW TEST INDICES:
    # test_data_indices = np.random.choice(np.arange(len(csv.index)), size = test_data_cutoff, replace=False)
    # print(test_data_indices)
    # np.save("test_indices.npy", test_data_indices)
    test_data_indices = np.load("test_indices.npy")

    if args.training:
        csv = csv.drop(csv.index[test_data_indices])
        # csv = csv.head(len(csv.index)-test_data_cutoff) # csv.drop(csv.index[np.arange(test_data_cutoff)])
        print("Number of data used for training", len(csv.index))
    else:
        csv = csv.iloc[test_data_indices]
        # csv = csv.drop(csv.index[np.arange(len(csv.index)-test_data_cutoff)]) # csv.head(test_data_cutoff)
        print("Number of data used for testing", len(csv.index))

    # DATA PREPARATION:
    # 1. cut to certain Pitching position?
    if len(cfg.position) > 0:
        assert cfg.position=="Windup" or cfg.position=="Stretch", "Wrong pitching position filtering in config file"
        csv = csv[csv["Pitching Position (P)"]==cfg.position]
        print("Only pitching position ", cfg.position, "included in data")

    # 2. the pitch type "eephus" is excluded because it only occurs once in the data
    if label_name=="Pitch Type":
        csv = csv[csv["Pitch Type"]!="Eephus"]

    # 3. cut to the 5 players with most data
    if cfg.five_players:
        csv = Tools.cut_csv_to_pitchers(csv)
        print("Only the five players with most data are included")

    # GET DATA
    data, labels = Tools.get_data_from_csv(csv, label_name, min_length = cfg.nr_frames)
    print("Data shape:", data.shape)
    # data = np.load("data_test.npy")
    # labels = np.load("labels_test.npy")

    # 4. Change labels to super classes (only for the pitch type!)
    if cfg.super_classes:
        labels = Tools.labels_to_classes(labels)
        print("Labels are transformed to superclasses")

    ## POSSIBLE TO SHIFT AND FLIP DATA TO TRAIN ON MORE GENERAL DATA
    # data_old, _ = Tools.shift_data(data, labels, shift_labels = False, max_shift=30)
    # data_new = Tools.flip_x_data(data_old.copy()) #[:len(data_old)//2]
    # data = np.append(data_new, data_old, axis=0)
    # labels = np.append(labels, labels, axis=0)
    # print(data.shape, labels.shape)

    data = Tools.normalize(data)

    if args.training:
        training(data, labels, args.save_path)
    else:
        testing(data, labels, args.save_path)
