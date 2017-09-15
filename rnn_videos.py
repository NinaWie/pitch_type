#import pandas as pd
import numpy as np
import tensorflow as tf
from tools import Tools
from model import Model
import tflearn
from tflearn import DNN
from run import Runner

PATH = "/scratch/nvw224/arrays/"

dates = ["2017-06-18", "2017-06-22", "2017-07-04", "2017-07-16", "2017-04-14", "2017-04-18", "2017-05-02", "2017-05-06", "2017-05-19", "2017-05-23", "2017-06-06", "2017-06-10"]
data = np.load(PATH+"array_videos_"+dates[0]+".npy")
labels_string = np.load(PATH+"labels_videos_"+dates[0]+".npy")

for date in dates[1:]:
    arr = np.load(PATH+"array_videos_"+date+".npy")
    data = np.concatenate((data, arr), axis = 0)
    lab =  np.load(PATH+"labels_videos_"+date+".npy")
    labels_string = np.concatenate((labels_string, lab), axis = 0)

#data = np.load("array_videos_2017-04-14.npy")
#labels_string = np.load("labels_videos_2017-04-14.npy")

print(data.shape)
# data = np.load("interpolated.npy") #("array_videos.npy")
# labels_string = np.load("labels.npy")

runner = Runner()
runner.run(data, labels_string, np.unique(labels_string), RESTORE = None, BATCH_SZ=40, EPOCHS = 60, batch_nr_in_epoch = 100, align = False,
        act = tf.nn.relu, rate_dropout = 0,
        learning_rate = 0.0005, nr_layers = 4, n_hidden = 128, optimizer_type="adam", regularization=0,
        first_conv_filters=128, first_conv_kernel=9, second_conv_filter=128,
        second_conv_kernel=5, first_hidden_dense=128, second_hidden_dense=0,
        network = "rnn")

# network is tflearn or rnn
