import pandas as pd
import numpy as np
import tensorflow as tf
import scipy as sp
import scipy.stats
import json
import os
from os import listdir
import cv2
import argparse
import json


from config import cfg
import sys
sys.path.append("..")

from run_thread import Runner
from test import test
from utils import Tools

def training(save_path, csv_path, label_name= "Pitch Type", sequ_len = 160, max_shift=30):
    csv = pd.read_csv(csv_path)
    if len(cfg.position) > 0:
        assert cfg.position=="Windup" or cfg.position=="Stretch", "Wrong pitching position filtering in config file"
        csv = csv[csv["Pitching Position (P)"]==position]

    # the pitch type "eephus" is excluded because it only occurs once in the data
    if label_name=="Pitch Type":
        csv = csv[csv["Pitch Type"]!="Eephus"]

    if cfg.five_players:
        csv = Tools.cut_csv_to_pitchers(csv)

    try:
        print("Pitching positions in data:", np.unique(csv["Pitching Position (P)"].values),
        "Pitcher IDs included in data:", np.unique(csv["Pitcher"].values))
    except TypeError:
        print(len(csv))
    data, labels = Tools.get_data_from_csv(csv, label_name, min_length = sequ_len)

    print(data.shape)

    # data = np.load("data_test.npy")
    # labels = np.load("labels_test.npy")

    if cfg.super_classes:
        labels = Tools.labels_to_classes(labels)

    ## POSSIBLE TO SHIFT AND FLIP DATA TO TRAIN ON MORE GENERAL DATA
    # data_old, _ = Tools.shift_data(data, labels, shift_labels = False, max_shift=30)
    # data_new = Tools.flip_x_data(data_old.copy()) #[:len(data_old)//2]
    # data = np.append(data_new, data_old, axis=0)
    # labels = np.append(labels, labels, axis=0)
    # print(data.shape, labels.shape)

    data = Tools.normalize(data)

    # for nr_stacked in [1,2,4,8]:
        # for nr_hidden in [32, 56, 160, 256]:
    print()
    print("NEW CONFIGURATION:")
    print("rnn") #, nr_stacked, "hidden units", nr_hidden)
    runner = Runner(data, labels, SAVE = save_path, BATCH_SZ=cfg.batch_size, EPOCHS = 100, batch_nr_in_epoch = cfg.batches_per_epoch,
        act = tf.nn.relu, rate_dropout =  cfg.dropout,
        learning_rate = cfg.learning_rate, nr_layers = 8, n_hidden = 256, optimizer_type="adam",
        first_conv_filters=cfg.first_filters, first_conv_kernel=cfg.first_kernel, second_conv_filter=cfg.second_conv_filter,
        second_conv_kernel=cfg.second_conv_kernel, first_hidden_dense=cfg.first_hidden_dense, second_hidden_dense=cfg.second_hidden_dense,
        network = "rnn") # adjustable conv1d") #conv1d_big")
    runner.run()

if __name__ == "__main__":
    def boolean_string(s):
        if s not in {'False', 'True'}:
            raise ValueError('Not a valid boolean string')
        return s == 'True'
    parser = argparse.ArgumentParser(description='Train/test neural network for recognizing pitch type from joint trajectories')
    # parser.add_argument('-training', default= "True", type=boolean_string, help='if training, set True, if testing, set False')
    parser.add_argument('-label', default="Pitch Type", type=str, help='Pitch Type, Play Outcome or Pitching Position (P) possible so far')
    parser.add_argument('save_path', default="/scratch/nvw224/pitch_type/new_models/position", type=str, help='usually training to classify pitch type, but can also be used for pitching position (with the right model)')
    parser.add_argument('-view', default="cf", type=str, help='either cf (center field) or sv (side view)')
    args = parser.parse_args()

    save = args.save_path
    csv_path = os.path.join("..", "train_data")#  "/scratch/nvw224/"
    # print("training", args.training)
    # input_data_list = [[path_outputs+ "old_videos/cf/"]] # , [path_outputs+ "old_videos/sv/"]], [path_outputs+ "new_videos/cf/",
    # csv_list = [csv_path + "cf_data.csv"] #, csv_path + "csv_gameplay.csv", csv_path + "BOS_SV_metadata.csv"]
    if True: #args.training:
        if args.label=="Pitch Type" or args.label=="Pitching Position (P)":
            csv = os.path.join(csv_path, args.view +"_pitcher.csv")
        elif args.label=="Play Outcome":
            csv= os.path.join(csv_path, args.view +"_batter.csv")
            # csv = csv_path +args.view +"_batter.csv"
        else:
            print("USAGE: WRONG INPUT FOR -label ARGUMENT (Pitch Type, Play Outcome or Pitching Position (P))")
            import sys
            sys.exit()
        training(save, csv, label_name = args.label) #, position="Stretch")
    #else:
    #    testing(save)

# TEST ON HIGH QUALITY VIDEOS

# def testing(save_path, sequ_len = 100):
#     with open("dic_release_high_quality.json", "r") as outfile:
#         dic_release_high_quality = json.load(outfile)
#     test_json_files = "/scratch/nvw224/pitch_type/Pose_Estimation/v0testing/"
#     test_data_path = "/scratch/nvw224/pitch_type/Pose_Estimation/high_quality_testing/pitcher/"
#     prepro = JsonProcessor()
#     labels = []
#     for s in [50,60,70]:
#         # print("----shift:", s, "-----------")
#         data, files = prepro.get_test_data(test_json_files, test_data_path, sequ_len, 0, shift=s, labels=dic_release_high_quality)
#         data = Tools.normalize(data)
#         lab, out = test(data, save_path)
#         labels.append(lab)
#     labels = np.array(labels)
#     for i in range(len(lab)):
#         print(files[i], labels[:,i].astype(str).tolist())


## to save parts of the data for testing:

# M,N,nr_joints,nr_coordinates = data.shape
# SEP = int(M*0.9)
# print("Test set size: ", len_test, " train set size: ", len_train)
# print("Shapes of train_x", train_x.shape, "shape of test_x", test_x.shape)
# ind = np.random.permutation(len(data))
# train_ind = ind[:SEP]
# test_ind = ind[SEP:]
# data_part = data[test_ind]
# labels_part = labels[test_ind]
# train_data = data[train_ind]
# train_labels = labels[train_ind]
# print("for validation data", data_part.shape, train_data.shape, labels_part.shape, train_labels.shape)
#np.save("data_test.npy", data_part)
#np.save("labels_test.npy", labels_part)

### TO USE JSON FILES WITH JsonProcessor

#testing(save, sequ_len=160)
#training(save, label_name = "Pitch Type", data_path = input_data_list, csv = csv_list, player=p) #label_name = "Pitching Position (P)", data_path = "old_videos/cf/", csv = "cf_data_cut")


# path_outputs = "/Volumes/Nina Backup/finished_outputs/"
# test_json_files = "/Volumes/Nina Backup/high_quality_outputs/"
# test_data_path = "/Users/ninawiedemann/Desktop/UNI/Praktikum/high_quality_testing/pitcher/"
# save =  "/Users/ninawiedemann/Desktop/UNI/Praktikum/saved_models/pitch_type_svcf"
# csv_path = "train_data/"

# path_outputs = "/scratch/nvw224/pitch_type/Pose_Estimation/outputs/"
# test_json_files = "/scratch/nvw224/pitch_type/Pose_Estimation/v0testing/"
# test_data_path = "/scratch/nvw224/pitch_type/Pose_Estimation/high_quality_testing/pitcher/"
# save =  "/scratch/nvw224/pitch_type/saved_models/position_unextended"
# csv_path = "/scratch/nvw224/"


# def training(save_path, data_path, csv, label_name= "Pitch Type", player = "pitcher", sequ_len = 160, max_shift=30):
#     prepro = JsonProcessor()
#     data, plays = prepro.get_data(data_path, sequ_len=sequ_len, player=player)
#     assert len(data)==len(plays)
#     # data, plays = prepro.get_data_concat(path_outputs+ "old_videos/cf/", path_outputs+ "old_videos/sv/", sequ_len=sequ_len, player="pitcher")
#     label = prepro.get_label(plays, csv, label_name) # , csv_path+ "sv_data.csv"  "Pitch Type"
#     #print(label)
#     inds = np.where(np.array(label)=="Unknown")[0]
#     print(inds)
#     files = np.delete(plays, inds, axis = 0)
#     data = np.delete(data, inds, axis = 0)
#     label = np.delete(label, inds, axis = 0)
#     print(data.shape, len(label), np.any(label=="Unknown"), np.any(pd.isnull(label)))
#     both = data
#
#     ## shift and flip
#     # data_old, label = Tools.shift_data(both, label, shift_labels = False, max_shift=max_shift)
#     # print(data_old.shape)
#     # data_new = Tools.flip_x_data(data_old.copy()) #[:len(data_old)//2]
#     # print(data_new.shape)
#
#     # data = np.append(data_new, data_old, axis=0)
#     # label = np.append(label, label)
#
#     #np.save("saved_for_testing.npy", data)
#     #np.save("saved_for_testing_label.npy", label)
#     data = Tools.normalize(self.data)
#     print(data.shape, len(label))
#
#     runner = Runner(data, label, SAVE = save_path, files=files, BATCH_SZ=40, EPOCHS = 1000, batch_nr_in_epoch = 100,
#             act = tf.nn.relu, rate_dropout = 0.4,
#             learning_rate = 0.0005, nr_layers = 4, n_hidden = 128, optimizer_type="adam", regularization=0,
#             first_conv_filters=128, first_conv_kernel=5, second_conv_filter=128,
#             second_conv_kernel=9, first_hidden_dense=128, second_hidden_dense=0,
#             network = "adjustable conv1d") #conv1d_big")
#     runner.start()