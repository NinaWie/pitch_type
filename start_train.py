import pandas as pd
import numpy as np
import tensorflow as tf
import scipy as sp
import scipy.stats

import tflearn
from tflearn import DNN

#from sklearn.preprocessing import StandardScaler
from data_preprocess import Preprocessor
from tools import Tools
from model import Model
from run_thread import Runner

import shutil

PATH = "cf"
CUT_OFF_Classes = 10
leaky_relu = lambda x: tf.maximum(0.2*x, x)
align = False
normalize = True

# PREPROCESS DATA
if PATH is "cf" or PATH is "concat":
    prepro = Preprocessor("cf_data.csv")
else:
    prepro = Preprocessor("sv_data.csv")

prepro.remove_small_classes(CUT_OFF_Classes)
# ONE PLAYER
players, _ = prepro.get_list_with_most("Pitcher")
prepro.select_movement("Windup")

prepro.cut_file_to_listof_pitcher(players)  # change_restore

# prepro.set_labels_toWindup()

if PATH is not "concat":
    data_raw = prepro.get_coord_arr(None) #PATH+"_all_coord.npy")
    # data_raw = np.load("cf_all_coord.npy")
    print("data loaded")
else:
    data_raw = prepro.concat_with_second("sv_data.csv", PATH+"_all_coord.npy")

data = data_raw[:,:,:12,:]


labels_string = prepro.get_labels()

labels_string = Tools.labels_to_classes(labels_string)

if align:
    data = Tools.align_frames(data, prepro.get_release_frame(60, 120), 60, 40)

if  normalize:
     data = Tools.normalize( data)

print(data.shape, len(labels_string), np.unique(labels_string))

runner = Runner(data, labels_string, SAVE = "/Users/ninawiedemann/Desktop/UNI/Praktikum/saved_models/modelCarlos2", BATCH_SZ=40, EPOCHS = 10, batch_nr_in_epoch = 100,
        act = tf.nn.relu, rate_dropout = 0,
        learning_rate = 0.0005, nr_layers = 4, n_hidden = 128, optimizer_type="adam", regularization=0,
        first_conv_filters=128, first_conv_kernel=9, second_conv_filter=128,
        second_conv_kernel=9, first_hidden_dense=128, second_hidden_dense=0,
        network = "adjustable conv1d")

runner.start()
