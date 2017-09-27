import pandas as pd
import numpy as np
import tensorflow as tf
import scipy as sp
import scipy.stats
import json
from os import listdir
import cv2
import ast

path_input = "/scratch/nvw224/videos/atl"
path_output = "/scratch/nvw224/arrays/"
cf_data_path = "/scratch/nvw224/cf_data.csv"
#path_input_dat = "/Volumes/Nina Backup/videos/atl/2017-04-14/center field/490251-0f308987-60b4-480c-89b7-60421ab39106.mp4.dat"

from run_thread import Runner
from test import test
from video_to_pitchtype_directly import VideoProcessor

cut_off_min = 80
cut_off_max= 110

dates = ["2017-04-15", "2017-04-19", "2017-05-03", "2017-05-07", "2017-05-20", "2017-05-24", "2017-06-07",
 "2017-06-11", "2017-06-19", "2017-06-23", "2017-07-05", "2017-07-17", "2017-04-14", "2017-04-18",
  "2017-05-02", "2017-05-06", "2017-05-19", "2017-05-23", "2017-06-06", "2017-06-10", "2017-06-18", "2017-06-22", "2017-07-04", "2017-07-16"]

test_dates = ['2017-06-08', '2017-06-17', '2017-05-21', '2017-06-21', '2017-07-19', '2017-06-09', '2017-07-15', '2017-05-01',
 '2017-06-16', '2017-04-16', '2017-05-05', '2017-04-20', '2017-05-18', '2017-06-24', '2017-06-20', '2017-05-25',
  '2017-05-17', '2017-05-04', '2017-06-05', '2017-06-06', '2017-04-17', '2017-05-22', '2017-07-18', '2017-07-14']


def get_test_data(input_dir, f):
    process = VideoProcessor(path_input=path_input, df_path = cf_data_path)
    label = process.get_labels(f, "pitch_frame_index")
    if label is not None:
        data = process.get_pitcher_array(input_dir, f)
        return data[cut_off_min:cut_off_max], label-cut_off_min

def testing(test_dates, restore_path):
    for date in test_dates:
        output = []
        labels = []
        input_dir= path_input+"/"+date+"/center field/"
        list_files = listdir(input_dir)
        print(date)
        for f in list_files:
            if f[-4:]==".mp4":
                data, label = get_test_data(input_dir, f)
                for elem in data:
                    output.append(elem)
                labels.append(label)
                break
        break
    examples, width, height = output.shape
    data = np.reshape(output, (examples, width, height, 1))
    print(data.shape, len(data)/30, len(labels), labels)
    lab, out = test(data, restore_path)

    for i in range(len(data)/30):
        print([round(elem,2) for elem in out[30*i:30*(i+1), 1]])
        highest = np.argmax(out[30*i:30*(i+1), 1])
        print("frame index predicted: ", highest)
        #np.save("predicted_frame", data[highest])
        #np.save("all_frames", data)


def get_train_data(dates):
    pos = []
    neg = []
    for date in dates:
        arr = np.load(path_output+"array_videos_"+date+".npy")[:, cut_off_min:cut_off_max, :, :]
        lab =  np.load(path_output+"labels_release_"+date+".npy")
        for i, pitch in enumerate(arr):
            rel = lab[i]
            if rel>cut_off_min and rel<cut_off_max and not np.isnan(rel):
                ind = int(rel)-cut_off_min
                pos.append(pitch[ind])
                rest = np.delete(pitch, int(rel)-cut_off_min, axis = 0)
                indizes = np.random.choice(len(rest), 3, replace = False)
                for elem in rest[indizes]:
                    neg.append(elem)
            else:
                print("false", rel)
                #print(pitch.shape, pos[-1].shape, neg[-1].shape)
    print(np.array(pos).shape, np.array(neg).shape)

    pos = pos[93:]

    labels = np.ones((len(pos)), dtype=np.int).tolist()+np.zeros((len(neg)), dtype=np.int).tolist()
    print(len(pos), len(neg), len(labels))
    data = np.append(np.array(pos), np.array(neg), axis = 0)
    # np.save("positive.npy", np.array(pos))
    # print("positive saved")
    #np.save("negative.npy", np.array(neg)[:10])
    examples, width, height = data.shape
    data = np.reshape(data, (examples, width, height, 1))
    print(data.shape)
    return data, np.array(labels)

def training(dates, save_path):
    data, labels = get_train_data(dates)
    runner = Runner(data, labels, SAVE = save_path, BATCH_SZ=40, EPOCHS = 40, batch_nr_in_epoch = 100,
            act = tf.nn.relu, rate_dropout = 0,
            learning_rate = 0.0005, nr_layers = 4, n_hidden = 128, optimizer_type="adam", regularization=0,
            first_conv_filters=128, first_conv_kernel=9, second_conv_filter=128,
            second_conv_kernel=9, first_hidden_dense=128, second_hidden_dense=0,
            network = "adjustable conv2d")
    runner.start()

# training(dates, "/scratch/nvw224/pitch_type/saved_models/release_model")
testing(test_dates, "/scratch/nvw224/pitch_type/saved_models/release_model")
