import pandas as pd
import numpy as np
import tensorflow as tf
#import scipy as sp
#import scipy.stats
from scipy import ndimage
import json
import argparse

import sys
sys.path.append("/Users/ninawiedemann/Desktop/UNI/Praktikum/ALL")

#from sklearn.preprocessing import StandardScaler
from data_preprocess import Preprocessor
from tools import Tools
from model import Model
from run_thread import Runner
from test import test
from config import cfg

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

if __name__ == "__main__":
    def boolean_string(s):
        if s not in {'False', 'True'}:
            raise ValueError('Not a valid boolean string')
        return s == 'True'
    parser = argparse.ArgumentParser(description='Train/test neural network for recognizing pitch type from joint trajectories')
    parser.add_argument('-training', default= "True", type=boolean_string, help='if training, set True, if testing, set False')
    parser.add_argument('-model_path', default="saved_models/batter_first_rnn_10_40", type=str, help='if training, path to save model, it testing, path to restore model')
    parser.add_argument('-classify_position', default=False, type=boolean_string, help='usually training to classify pitch type, but can also be used for pitching position (with the right model)')
    args = parser.parse_args()

    train = args.training
    save_path = args.model_path

    head_out = cfg.head_out
    CUT_OFF_Classes = cfg.min_class_members

    if train:
        PATH = cfg.data_path
        align = cfg.align
        position = cfg.filter_position
        unify_classes = cfg.superclasses
        players = []
    else:
        try:
            with open(save_path+"_preprocessing.json", 'r') as fin:
                params = json.load(fin)
            print(params)
            PATH = params["path"]
            align = params["align"]
            players = params["players"]
            position = params["position"]
            unify_classes = params["unify classes"]
        except:
            print("WARNING: testing but no preprocessing file found, so parameters are set according to config")
            PATH = cfg.data_path
            align = cfg.align
            position = cfg.filter_position
            unify_classes = cfg.superclasses
            players = []

    # PREPROCESS DATA
    if "concat" in PATH:
        prepro = Preprocessor("train_data/cf_data.csv")
    else:
        if "cf" in PATH:
            PATH = "train_data/cf_data.csv"
        elif "sv" in PATH:
            PATH = "train_data/sv_data.csv"
        else:
            print("ABORTED, Wrong path")
            import sys
            sys.exit()
        prepro = Preprocessor(PATH)

    prepro.remove_small_classes(CUT_OFF_Classes)
    # prepro.remove_wrong_games("490266")

    if position is "Windup" or position is "Stretch":
        prepro.select_movement(position)

    if len(players)>0:
        players = []
        prepro.cut_file_to_listof_pitcher(players)

    if train:
        processing_requirements = {"path":PATH, "align":align, "position": position, "players":players, "unify classes": unify_classes}
        print(processing_requirements)
        with open(save_path+'_preprocessing.json', 'w') as fout:
            json.dump(processing_requirements, fout)
        print("saved preprocessing requirements")

    # FOR PITCHING POSITION CLASSIFICATION
    if args.classify_position or "position" in save_path:
        prepro.set_labels_toWindup()

    if PATH[-3:]=="npy":
        data = np.load(PATH)
        print("data loaded")
    elif "concat" in PATH:
        data = prepro.concat_with_second("train_data/sv_data.csv", None)
    else:
        data = prepro.get_coord_arr("testing_purposes.npy") # put filename as argument to save data_array


    if head_out:
        data = data[:,:,:12,:]

    if align:
        data = Tools.align_frames(data, prepro.get_release_frame(60, 120), 60, 40)

    labels_string = prepro.get_labels()

    if unify_classes:
        labels_string = Tools.labels_to_classes(labels_string)

    # data = Tools.do_pca(data, 5) # pca did not really help
    print(data.shape, len(labels_string), np.unique(labels_string))

    if train:
        runner = Runner(data, labels_string, SAVE = save_path, BATCH_SZ=40, EPOCHS = 3, batch_nr_in_epoch = 100,
                act = tf.nn.relu, rate_dropout = 0,
                learning_rate = 0.0005, nr_layers = 4, n_hidden = 128, optimizer_type="adam", regularization=0,
                first_conv_filters=128, first_conv_kernel=5, second_conv_filter=128,
                second_conv_kernel=9, first_hidden_dense=128, second_hidden_dense=0,
                network = "adjustable conv1d")
        runner.start()
    else:
        data = Tools.normalize(data)
        pitches_test, out_test = test(data, save_path)
        print(Tools.accuracy(pitches_test, labels_string))
        print("True                   Test                 ", np.unique(labels_string))
        # print(np.swapaxes(np.append([labels_string_test], [pitches_test], axis=0), 0,1))
        for i in range(20):
            print('{:20}'.format(labels_string[i]), '{:20}'.format(pitches_test[i]), ['%.2f        ' % elem for elem in out_test[i]])


    ## optional:
    # pitches_test = compare_to_superclasses(pitches_test, out_test)
    # save_in_csv(pitches_test)
