import pandas as pd
import numpy as np
import tensorflow as tf
import scipy as sp
import scipy.stats
import json
from os import listdir
import cv2
import ast
import json
import argparse

from run_thread import Runner
from test import test
from data_preprocess import JsonProcessor
from tools import Tools

if __name__ == "__main__":
    def boolean_string(s):
        if s not in {'False', 'True'}:
            raise ValueError('Not a valid boolean string')
        return s == 'True'
    parser = argparse.ArgumentParser(description='Train/test neural network for recognizing pitch type from joint trajectories')
    parser.add_argument('-training', default= "True", type=boolean_string, help='if training, set True, if testing, set False')
    parser.add_argument('-data_path', default="/scratch/nvw224/CapsNet-Tensorflow/data/cf_unextended/", type=str, help='if training, path to save model, it testing, path to restore model')
    parser.add_argument('-save_path', default="/scratch/nvw224/pitch_type/saved_models/position_unextended", type=str, help='usually training to classify pitch type, but can also be used for pitching position (with the right model)')
    args = parser.parse_args()

    save_path = args.save_path

    if args.training:
        data = np.load(args.data_path+"train_x.npy")
        label = np.load(args.data_path+"train_t.npy")
        data = np.reshape(data, (-1, 160, 12, 2))
        print(data.shape, label.shape)

        runner = Runner(data, label, SAVE = save_path, files=[], BATCH_SZ=40, EPOCHS = 100, batch_nr_in_epoch = 100,
                act = tf.nn.relu, rate_dropout = 0.4,
                learning_rate = 0.0005, nr_layers = 4, n_hidden = 128, optimizer_type="adam", regularization=0,
                first_conv_filters=128, first_conv_kernel=5, second_conv_filter=128,
                second_conv_kernel=9, first_hidden_dense=128, second_hidden_dense=0,
                network = "adjustable conv1d") #conv1d_big")
        runner.start()
    else:
        data = np.load(args.data_path+"test_x.npy")
        label = np.load(args.data_path+"test_t.npy")



# path_outputs = "/Volumes/Nina Backup/finished_outputs/"
# test_json_files = "/Volumes/Nina Backup/high_quality_outputs/"
# test_data_path = "/Users/ninawiedemann/Desktop/UNI/Praktikum/high_quality_testing/pitcher/"
# save =  "/Users/ninawiedemann/Desktop/UNI/Praktikum/saved_models/pitch_type_svcf"
# csv_path = "train_data/"
#
# # path_outputs = "/scratch/nvw224/pitch_type/Pose_Estimation/outputs/"
# # test_json_files = "/scratch/nvw224/pitch_type/Pose_Estimation/v0testing/"
# # test_data_path = "/scratch/nvw224/pitch_type/Pose_Estimation/high_quality_testing/pitcher/"
# # save =  "/scratch/nvw224/pitch_type/saved_models/position_unextended"
# # csv_path = "/scratch/nvw224/"
#
# input_data_list = [[path_outputs+ "old_videos/cf/"]] # , [path_outputs+ "old_videos/sv/"]], [path_outputs+ "new_videos/cf/",
# csv_list = [csv_path + "cf_data.csv"] #, csv_path + "csv_gameplay.csv", csv_path + "BOS_SV_metadata.csv"]
#
# #testing(save, sequ_len=160)
# training(save, label_name = "Pitch Type", data_path = input_data_list, csv = csv_list, player="pitcher") #label_name = "Pitching Position (P)", data_path = "old_videos/cf/", csv = "cf_data_cut")
