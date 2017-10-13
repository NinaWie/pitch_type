import pandas as pd
import numpy as np
import tensorflow as tf
#import scipy as sp
#import scipy.stats
import json

import tflearn
from tflearn import DNN

#from sklearn.preprocessing import StandardScaler
from data_preprocess import Preprocessor
from tools import Tools
from model import Model
from run_thread import Runner
from test import *

from os import listdir
import codecs


# TO GET PARAMETERS FOR PREPROCESSING FOR TESTING:
# with open(save_path+'_preprocessing.json', "r") as fin:
#     dic = json.load(fin)

data_folder = "/Users/ninawiedemann/Desktop/UNI/Praktikum/numpy arrays/pitcher/"
label_folder = "/Users/ninawiedemann/Desktop/UNI/Praktikum/csvs/csv_gameplay.csv"
save_path = "saved_models/modelPitchTypeCFwindup"
label = "Pitch Type"
head_out = True
align = False
unify_classes = False

csv = pd.read_csv(label_folder, delimiter = ";")

games = csv["play_id"].values.tolist()
label_column = csv[label].values
data = []
labels = []
for f in listdir(data_folder):
    game = f[:-5]
    ind = games.index(game)
    if label_column[ind]=="Unknown Pitch Type":
        continue
    labels.append(label_column[ind])
    obj_text = codecs.open(data_folder+f, encoding='utf-8').read()
    data.append(json.loads(obj_text))

data = np.array(data)
labels_string = np.array(labels)

if head_out:
    data = data[:,:,:12,:]

if align:
    data = Tools.align_frames(data, prepro.get_release_frame(60, 120), 60, 40)

data = Tools.normalize( data)

if unify_classes:
    labels_string = Tools.labels_to_classes(labels_string)



print(data.shape, len(labels_string), np.unique(labels_string))




# runner = Runner(data, labels_string, SAVE = save_path, BATCH_SZ=40, EPOCHS = 60, batch_nr_in_epoch = 100,
#         act = tf.nn.relu, rate_dropout = 0,
#         learning_rate = 0.0005, nr_layers = 4, n_hidden = 128, optimizer_type="adam", regularization=0,
#         first_conv_filters=128, first_conv_kernel=5, second_conv_filter=128,
#         second_conv_kernel=9, first_hidden_dense=128, second_hidden_dense=0,
#         network = "adjustable conv1d")
#
# runner.start()

pitches_test, out_test = test(data, save_path)

# save_in_csv(pitches_test)
print(Tools.accuracy(pitches_test, labels_string))
print("True                   Test                 ", np.unique(labels_string))
# print(np.swapaxes(np.append([labels_string_test], [pitches_test], axis=0), 0,1))
for i in range(20):
    print('{:20}'.format(labels_string[i]), '{:20}'.format(pitches_test[i]), ['%.2f        ' % elem for elem in out_test[i]])
