import pandas as pd
import numpy as np
import tensorflow as tf
#import scipy as sp
#import scipy.stats
from scipy import ndimage
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

load_preprocessing_parameters = None #"saved_models/modelPitchtype_new.json"
save_path = "saved_models/modelPitchtype_smoothed"
head_out = True
CUT_OFF_Classes = 10

if load_preprocessing_parameters is None:
    PATH = "cf"
    CUT_OFF_Classes = 10
    align = False
    normalize = True
    position = None # "Stretch"
    unify_classes = False
else:
    with open(load_preprocessing_parameters, 'r') as fin:
        params = json.load(fin)
    print(params)
    PATH = params["path"]
    align = params["align"]
    players = params["players"]
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
# prepro.remove_wrong_games("490266")

if position is not None:
    prepro.select_movement(position)

if load_preprocessing_parameters is None:
    players = []
    #players, _ = prepro.get_list_with_most("Pitcher")
    #prepro.cut_file_to_listof_pitcher(players)
else:
    players = []
    #if players!=[]:
    #    prepro.cut_file_to_listof_pitcher(players)

if save_path is not None and load_preprocessing_parameters is None:
    processing_requirements = {"path":PATH, "align":align, "normalize":normalize, "position": position, "players":players, "unify classes": unify_classes}
    print(processing_requirements)
    with open(save_path+'_preprocessing.json', 'w') as fout:
        json.dump(processing_requirements, fout)
    print("saved preprocessing requirements")

# FOR POSITION CLASSIFICATION
# prepro.set_labels_toWindup()


if PATH == "cf" or PATH == "sv":
    data = prepro.get_coord_arr(None) #PATH+"_all_coord.npy")
elif PATH == "concat":
    data = prepro.concat_with_second("sv_data.csv", None)
elif PATH[-3:]=="npy":
    data = np.load(PATH)
    print("data loaded")
else:
    print("Wrong path")
    import sys
    sys.exit()

data = ndimage.filters.gaussian_filter1d(data, axis = 1, sigma = 3)


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




runner = Runner(data, labels_string, SAVE = save_path, BATCH_SZ=40, EPOCHS = 60, batch_nr_in_epoch = 100,
        act = tf.nn.relu, rate_dropout = 0,
        learning_rate = 0.0005, nr_layers = 4, n_hidden = 128, optimizer_type="adam", regularization=0,
        first_conv_filters=128, first_conv_kernel=5, second_conv_filter=128,
        second_conv_kernel=9, first_hidden_dense=128, second_hidden_dense=0,
        network = "adjustable conv1d")

runner.start()

def save_in_csv(labels):
    dic ={}
    try:
        old_df = pd.read_csv("pitch_types.csv")
        for col in old_df.columns.tolist():
            if col[:3]!="Unn":
                dic[col] = old_df[col].values
    except:
        dic["Game"] = prepro.cf["Game"].values

    games_in_cf = prepro.cf["Game"].values.tolist()
    # old_col = dic["Pitchtype_3classes_5player"]
    new_col = []
    games_in_dic = dic["Game"]
    #dic["new_col"]=lab
    for i in range(len(games_in_dic)):
        game = games_in_dic[i]
        try:
            ind = games_in_cf.index(game)
            new_col.append(labels[ind])
        except:
            new_col.append(None) #old_col[i]) #None)
    #print(new_col)
    dic["Combined"] = new_col
    df = pd.DataFrame.from_dict(dic)
    df.to_csv("pitch_types.csv")

# pitches_test, out_test = test(data, save_path)

#pitches_test = np.load("/Users/ninawiedemann/Desktop/UNI/Praktikum/numpy arrays/pitches_test.npy")
#out_test = np.load("/Users/ninawiedemann/Desktop/UNI/Praktikum/numpy arrays/out_test.npy")

def compare_to_superclasses(labels_new, out):
    uniq = np.unique(labels_new)
    # print(labels_new[:20])
    df = pd.read_csv("pitch_types.csv")
    superclasses = df["Pitchtype_3classes_Allplayer"]
    assert(len(superclasses)==len(labels_new))
    super_labels = Tools.labels_to_classes(labels_new)
    # print(super_labels[:20])
    not_same = np.where(super_labels!=superclasses)[0]
    # print(not_same)
    for i in not_same:
        if not pd.isnull(superclasses[i]):
            j = -2
            while(super_labels[i]!=superclasses[i]):
                second_best = np.argsort(out_test[i])[j]
                # print(out_test[i])
                # print("aus csv ", superclasses[i])
                second_best_lab = uniq[second_best]
                # print("label vorher", labels_new[i])
                labels_new[i] = second_best_lab
                # print("label nachher", labels_new[i])
                super_labels[i] = Tools.labels_to_classes(np.array([second_best_lab]))[0]
                j-=1
    return labels_new

#np.save("/Users/ninawiedemann/Desktop/UNI/Praktikum/numpy arrays/pitches_test.npy",pitches_test)
#np.save("/Users/ninawiedemann/Desktop/UNI/Praktikum/numpy arrays/out_test.npy", out_test)

# pitches_test = compare_to_superclasses(pitches_test, out_test)
# save_in_csv(pitches_test)
# print(Tools.accuracy(pitches_test, labels_string))
# print("True                   Test                 ", np.unique(labels_string))
# # print(np.swapaxes(np.append([labels_string_test], [pitches_test], axis=0), 0,1))
# for i in range(20):
#     print('{:20}'.format(labels_string[i]), '{:20}'.format(pitches_test[i]), ['%.2f        ' % elem for elem in out_test[i]])
