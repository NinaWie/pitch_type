import pandas as pd
import numpy as np
import tensorflow as tf
import scipy as sp
import scipy.stats

import sys
sys.path.append("/Users/ninawiedemann/Desktop/UNI/Praktikum/ALL")

from tools import Tools
from model import Model
import tflearn
from tflearn import DNN
import json

from run_events import Runner
from test import test
from os import listdir
import os
import codecs
from data_preprocess import JsonProcessor


def testing2(files, dic, restore_path, sequ_len):
    joints, labels = get_joint_array(path_sideview_labels, files, "Pitcher", dic)
    joints_rel = joints[:, :, [7,8,10,11],:]

    joints_rel = Tools.normalize(joints_rel)

    print(joints_rel.shape, np.array(labels).shape)
    bags = 167//sequ_len -2
    print("bags", bags)
    data = []
    for bsp in joints_rel:
        for i in range(bags):
            data.append(bsp[i*sequ_len:(i+2)*sequ_len])
    data = np.array(data)
    #np.save("/Users/ninawiedemann/Desktop/UNI/Praktikum/ALL/notebooks/bsp_data.npy", data)
    #np.save("/Users/ninawiedemann/Desktop/UNI/Praktikum/ALL/notebooks/bsp_labels.npy", labels)
    print(data.shape)
    # data = Tools.normalize(data)
    lab, out = test(data, restore_path)
    print("out", out.shape)
    print(out)
    truth_val = out[:, 1]
    r = 0
    for l in range(len(labels)):
        res = []
        for i in range(bags):
            res.append(truth_val[(l*bags)+i])
        # try:
        #     result = (np.where(np.array(res)>0.9)[0][0]+1)*10
        # except IndexError:
        result = (np.argmax(res)+1)*sequ_len
        print("result ", result, labels[l], "label")
        if abs(result-labels[l])< sequ_len:
            r+=1
        else:
            print(res)
    print("Accuracy: ", r/float(len(labels)))



def get_sequences(first_move_labels, joint_array, sequ_len):
    """
    Train first move by using a simple binary classification of positiove sequences (where a first movement occurs)
    and negative sequences (where no first movement occurs)
    returns data(array of size nr_data, sequ_len, nr_joints, nr_coordinates) and labels (size nr_data - containing 1 for pos and 0 for neg sequences)
    """
    pos = []
    neg = []
    for i in range(len(first_move_labels)):
        # lab = int(first_move_labels[f])
        lab = first_move_labels[i]
        pos.append(joint_array[i, lab-sequ_len:lab+sequ_len])
        # print("pos sequ: ", np.arange(lab-sequ_len, lab+sequ_len, 1))
        surround = np.delete(np.arange(sequ_len, len(joint_array[i])-sequ_len, 1), np.arange(lab-sequ_len, lab+sequ_len, 1))
        # print("label", lab, "surround", surround)
        neg_inds = np.random.choice(surround, 3, replace = False)
        for j in neg_inds:
            # print("neg sequ: ", np.arange(j-sequ_len, j+sequ_len,1))
            neg.append(joint_array[i, j-sequ_len:j+sequ_len])
    #print(np.any(labels==0))
    labels = np.ones((len(pos)), dtype=np.int).tolist()+np.zeros((len(neg)), dtype=np.int).tolist()
    print(len(pos), len(neg), len(labels))
    data = np.append(np.array(pos), np.array(neg), axis = 0)
    # np.save("positive.npy", np.array(pos))
    # print("positive saved")
    # np.save("negative.npy", np.array(neg)[:10])
    print(data.shape)
    return data, np.array(labels)


def other_method_predict_sequ():
    SEP = int(len(data)*0.9)
    ind = np.random.permutation(len(data))
    train_ind = ind[:SEP]
    test_ind = ind[SEP:]

    train_x = data[train_ind]
    test_x = data[test_ind]
    train_t= labels[train_ind]
    test_t = labels[test_ind]

    # return train_x, test_x, train_t, test_t
    model = Model()
    out, network = model.RNN_network_tflearn(frames, input_size, input_size, nr_layers, n_hidden, dropout, loss = "mean_square", act = "tanh")

    m = DNN(network)
    m.fit(train_x, train_t, validation_set=(test_x, test_t), show_metric=True, batch_size=BATCHSIZE, snapshot_step=100, n_epoch=EPOCHS)

    output = m.predict(test_x[:20])
    print(output)
    np.save("out_coord", output)
    np.save("seque_coord", test_x[:20])


def get_data(index, cf):
    begin_cf = cf.columns.get_loc("0")
    data_array = cf.iloc[index, begin_cf:begin_cf+167].values
    N = len(data_array)

    nr_joints = len(eval(data_array[0]))

    data = np.zeros((N,nr_joints,2))

    for j in range(N):
        if not pd.isnull(data_array[j]):
            data[j]=np.array(eval(data_array[j]))
        else:
            data[j] = data[j-1]
    return data

def get_joint_array(csv_file_path, files, player, dic_test):
    cf = pd.read_csv(csv_file_path)
    second_csv = pd.read_csv("/Users/ninawiedemann/Desktop/UNI/Praktikum/ALL/cf_data.csv")
    second_csv_player = second_csv[second_csv["Player"]==player]
    # cf = pd.read_csv("/Users/ninawiedemann/Desktop/UNI/Praktikum/ALL/cf_data.csv")
    cf_player = cf[cf["Player"]==player]
    games = cf_player["Game"].values.tolist()
    second_games = second_csv_player["Game"].values.tolist()
    releases = second_csv_player["pitch_frame_index"].values
    nonzero = np.logical_not(pd.isnull(releases))
    not_null = releases[nonzero]
    over_min = not_null[np.where(not_null>60)]
    below_max = over_min[np.where(over_min<130)]
    mean = np.mean(below_max)
    print("mean", mean)
    releases[np.where(np.isnan(releases))] = mean
    releases[np.where(releases<=60)] = mean
    releases[np.where(releases>=130)] = mean
    print(releases)

    positions = cf_player["Pitching Position (P)"].values
    joints_array = []
    lab_test = []
    position = []
    rel = []
    # pitch_type = []
    for i, game in enumerate(files):
        # print(game)
        try:
            ind = games.index(game)
            second_ind = second_games.index(game)
            if dic_test[game] >= releases[second_ind]:
                print(dic_test[game])
                continue
            position.append(positions[ind])
            rel.append(releases[second_ind])
            #pitch_type.append(pitchtypes[ind])
            joints_array.append(get_data(ind, cf_player))
            lab_test.append(dic_test[game])
        except:
            continue
    return np.array(joints_array), lab_test, rel

def get_sequences(first_move_labels, joint_array, sequ_len):
    pos = []
    neg = []
    for i in range(len(first_move_labels)):
        # lab = int(first_move_labels[f])
        lab = first_move_labels[i]
        pos.append(joint_array[i, lab-sequ_len:lab+sequ_len])
        # print("pos sequ: ", np.arange(lab-sequ_len, lab+sequ_len, 1))
        surround = np.delete(np.arange(sequ_len, len(joint_array[i])-sequ_len, 1), np.arange(lab-sequ_len, lab+sequ_len, 1))
        # print("label", lab, "surround", surround)
        neg_inds = np.random.choice(surround, 3, replace = False)
        for j in neg_inds:
            # print("neg sequ: ", np.arange(j-sequ_len, j+sequ_len,1))
            neg.append(joint_array[i, j-sequ_len:j+sequ_len])
    #print(np.any(labels==0))
    labels = np.ones((len(pos)), dtype=np.int).tolist()+np.zeros((len(neg)), dtype=np.int).tolist()
    print(len(pos), len(neg), len(labels))
    data = np.append(np.array(pos), np.array(neg), axis = 0)
    # np.save("positive.npy", np.array(pos))
    # print("positive saved")
    #np.save("negative.npy", np.array(neg)[:10])
    print(data.shape)
    return data, np.array(labels)


def training(files, dic_lab, save_path, sequ_len):
    nr_layers = 3
    n_hidden = 128
    BATCHSIZE = 256
    EPOCHS = 500
    dropout = 0.2
    joints_list = ["right_shoulder", "left_shoulder", "right_elbow", "right_wrist","left_elbow", "left_wrist",
            "right_hip", "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle", "neck ",
            "right_eye", "right_ear","left_eye", "left_ear"]
    relevant_joints = ["right_knee", "right_ankle", "left_knee", "left_ankle"]

    joints, lab_test, release = get_joint_array(path_sideview_labels, files, "Pitcher", dic_lab)
    print(min(lab_test), max(lab_test))
    print("joints array", joints.shape)
    joints_rel = joints[:, :, [7,8,10,11],:]
    # print("labels before", lab_test)
    release_joints = []
    print("lengths same?", len(release), len(joints_rel), len(lab_test), len(files))
    labels = np.array(lab_test)
    for j in range(len(joints_rel)):
        # print(release[j])
        r = int(release[j])
        # print("end frame", r, sequ_len, labels[j], "start frame", r-sequ_len, "new lab", labels[j]-r+sequ_len, files[j])
        labels[j] = labels[j]-r+sequ_len
        release_joints.append(joints_rel[j, r-sequ_len:r])
    release_joints = np.array(release_joints)
    print("new shape joints array ", release_joints.shape)

    release_joints = Tools.normalize(release_joints)

    # data, labels = get_sequences(lab_test, joints_rel, sequ_len)
    data = release_joints

    print(labels)
    # data = Tools.normalize(data)
    # np.save("/Users/ninawiedemann/Desktop/UNI/Praktikum/ALL/notebooks/bsp_data.npy", data)
    # np.save("/Users/ninawiedemann/Desktop/UNI/Praktikum/ALL/notebooks/bsp_labels.npy", labels)
    # print("saved")
    runner = Runner(data, labels, SAVE = save_path, BATCH_SZ=40, EPOCHS = EPOCHS, batch_nr_in_epoch = 50,
            act = tf.nn.relu, rate_dropout = 0,
            learning_rate = 0.0005, nr_layers = 4, n_hidden = 128, optimizer_type="adam", regularization=0,
            first_conv_filters=12, first_conv_kernel=3, second_conv_filter=12,
            second_conv_kernel=3, first_hidden_dense=128, second_hidden_dense=56,
            network = "conv 1st move")
    runner.unique = np.arange(0, 50,1).tolist()
    runner.start()


def testing(files, dic, restore_path, sequ_len):
    joints, labels, release = get_joint_array(path_sideview_labels, files, "Pitcher", dic)
    joints_rel = joints[:, :, [7,8,10,11],:]

    joints_rel = Tools.normalize(joints_rel)

    print(joints_rel.shape, np.array(labels).shape)

    # for_position = Tools.normalize(joints[:,:,:12,:])
    # lab, out = test(for_position, "/Users/ninawiedemann/Desktop/UNI/Praktikum/saved_models/modelPosition")
    # print("FINISHED FIRST", lab)

    new = []
    for j in range(len(joints_rel)):
        r = int(release[j])
        new.append(joints_rel[j, r-sequ_len:r])
    joints_array = np.array(new)
    print("joints array", joints_array.shape)
    lab, out = test(joints_array, restore_path)
    print("lab", lab)


# define data paths
path_sideview_labels = os.path.join("train_data", "sv_data.csv")
path_centerfield_labels = os.path.join("train_data","cf_data.csv")

with open("train_data/pitcher_first_move_labels", "r") as infile:
    dic_lab = json.load(infile)
all_files = list(dic_lab.keys())
print(len(all_files))
# print(dic_lab.values())
# files = all_files[:-10]
# print(len(all_files))
# print(all_files[:10])
SEQU_LEN = 50
testing(all_files[:20], dic_lab, "saved_models/first_move_more", SEQU_LEN)
