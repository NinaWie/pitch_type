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


# TO GET PARAMETERS FOR PREPROCESSING FOR TESTING:
# with open(save_path+'_preprocessing.json', "r") as fin:
#     dic = json.load(fin)

load_preprocessing_parameters = "saved_models/modelCarlos2_preprocessing.json"
save_path = "saved_models/modelCarlos2"
head_out = True
CUT_OFF_Classes = 10

if load_preprocessing_parameters is None:
    PATH = "cf"
    CUT_OFF_Classes = 10
    align = False
    normalize = True
    position = "Stretch"
    unify_classes = False
else:
    with open(load_preprocessing_parameters, 'r') as fin:
        params = json.load(fin)
    PATH = params["path"]
    align = params["align"]
    players = params["players"]
    if players!=[]:
        prepro.cut_file_to_listof_pitcher(players)
    normalize = params["normalize"]
    position = params["position"]
    unify_classes = params["unify classes"]

# PREPROCESS DATA
if PATH == "cf" or PATH == "concat":
    prepro = Preprocessor("cf_data.csv")
    print("cf eingelsen")
else:
    prepro = Preprocessor("sv_data.csv")

prepro.remove_small_classes(CUT_OFF_Classes)

if position is not None:
    prepro.select_movement(position)

if load_preprocessing_parameters is None:
    players = []
    #players, _ = prepro.get_list_with_most("Pitcher")
    #prepro.cut_file_to_listof_pitcher(players)

if save_path is not None and load_preprocessing_parameters is None:
    processing_requirements = {"path":PATH, "align":align, "normalize":normalize, "position": position, "players":players, "unify classes": unify_classes}
    print(processing_requirements)
    with open(save_path+'_preprocessing.json', 'w') as fout:
        json.dump(processing_requirements, fout)
    print("saved preprocessing requirements")

# FOR POSITION CLASSIFICATION
# prepro.set_labels_toWindup()

if PATH is not "concat":
    data = prepro.get_coord_arr(None) #PATH+"_all_coord.npy")
    # data_raw = np.load("/Users/ninawiedemann/Desktop/UNI/Praktikum/numpy arrays/carlos.npy")
    print("data loaded")
else:
    data = prepro.concat_with_second("sv_data.csv", None)


if head_out:
    data = data[:,:,:12,:]

if align:
    data = Tools.align_frames(data, prepro.get_release_frame(60, 120), 60, 40)

if  normalize:
     data = Tools.normalize( data)

labels_string = prepro.get_labels()

if unify_classes:
    labels_string = Tools.labels_to_classes(labels_string)



print(data.shape, len(labels_string), np.unique(labels_string))




# runner = Runner(data, labels_string, SAVE = save_path, BATCH_SZ=40, EPOCHS = 40, batch_nr_in_epoch = 100,
#         act = tf.nn.relu, rate_dropout = 0,
#         learning_rate = 0.0005, nr_layers = 4, n_hidden = 128, optimizer_type="adam", regularization=0,
#         first_conv_filters=128, first_conv_kernel=9, second_conv_filter=128,
#         second_conv_kernel=9, first_hidden_dense=128, second_hidden_dense=0,
#         network = "adjustable conv1d")
#
# runner.start()

def save_in_csv(labels):
    dic ={}
    try:
        old_df = pd.read_csv("pitchtypes_test.csv")
        for col in old_df.columns.tolist():
            if col[:3]!="Unn":
                dic[col] = old_df[col].values
    except:
        dic["play_id"] = prepro.cf["play_id"].values

    games_in_cf = prepro.cf["play_id"].values.tolist()
    old_col = dic["CF3classes5players"]
    new_col = []
    games_in_dic = dic["play_id"]
    #dic["new_col"]=lab
    for i in range(len(games_in_dic)):
        game = games_in_dic[i]
        try:
            ind = games_in_cf.index(game)
            new_col.append(labels[ind])
        except:
            new_col.append(old_col[i]) #None)
    #print(new_col)
    dic["CF3classes5players"] = new_col
    df = pd.DataFrame.from_dict(dic)
    df.to_csv("pitchtypes_test.csv")

pitches_test, out_test = test(data, save_path)
save_in_csv(pitches_test)
print(Tools.accuracy(pitches_test, labels_string))
print("True                   Test                 ", np.unique(labels_string))
# print(np.swapaxes(np.append([labels_string_test], [pitches_test], axis=0), 0,1))
for i in range(20):
    print('{:20}'.format(labels_string[i]), '{:20}'.format(pitches_test[i]), ['%.2f        ' % elem for elem in out_test[i]])
