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
from test import *

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

# prepro.select_movement("Windup")

# players, _ = prepro.get_list_with_most("Pitcher")
# prepro.cut_file_to_listof_pitcher(players)

prepro.set_labels_toWindup()

if PATH is not "concat":
    data_raw = prepro.get_coord_arr(None) #PATH+"_all_coord.npy")
    # data_raw = np.load("/Users/ninawiedemann/Desktop/UNI/Praktikum/numpy arrays/carlos.npy")
    print("data loaded")
else:
    data_raw = prepro.concat_with_second("sv_data.csv", None)

data = data_raw[:,:,:12,:]


labels_string = prepro.get_labels()

# labels_string = Tools.labels_to_classes(labels_string)

if align:
    data = Tools.align_frames(data, prepro.get_release_frame(60, 120), 60, 40)

if  normalize:
     data = Tools.normalize( data)

print(data.shape, len(labels_string), np.unique(labels_string))


runner = Runner(data, labels_string, SAVE = "saved_models/modelPositionSV", BATCH_SZ=40, EPOCHS = 20, batch_nr_in_epoch = 100,
        act = tf.nn.relu, rate_dropout = 0,
        learning_rate = 0.0005, nr_layers = 4, n_hidden = 128, optimizer_type="adam", regularization=0,
        first_conv_filters=128, first_conv_kernel=9, second_conv_filter=128,
        second_conv_kernel=9, first_hidden_dense=128, second_hidden_dense=0,
        network = "adjustable conv1d")

runner.start()

# pitches_test, out_test = test(data, "saved_models/modelPosition")
# print(Tools.accuracy(pitches_test, labels_string))
# print("True                   Test                 ", np.unique(labels_string))
# # print(np.swapaxes(np.append([labels_string_test], [pitches_test], axis=0), 0,1))
# for i in range(len(labels_string)):
#     print('{:20}'.format(labels_string[i]), '{:20}'.format(pitches_test[i]), ['%.2f        ' % elem for elem in out_test[i]])
