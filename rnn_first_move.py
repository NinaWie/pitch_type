import pandas as pd
import numpy as np
import tensorflow as tf
import scipy as sp
import scipy.stats

from tools import Tools
from model import Model
import tflearn
from tflearn import DNN
import json

from run_thread import Runner
from test import test
from os import listdir
import codecs

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
    # cf = pd.read_csv("/Users/ninawiedemann/Desktop/UNI/Praktikum/ALL/cf_data.csv")
    cf_player = cf[cf["Player"]==player]
    games = cf_player["Game"].values.tolist()
    positions = cf_player["Pitching Position (P)"].values
    joints_array = []
    lab_test = []
    position = []
    # pitch_type = []
    for i, game in enumerate(files):
        # print(game)
        try:
            ind = games.index(game)
            position.append(positions[ind])
            #pitch_type.append(pitchtypes[ind])
            joints_array.append(get_data(ind, cf_player))
            lab_test.append(dic_test[game])
        except:
            continue
    return np.array(joints_array), lab_test

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

    joints, lab_test = get_joint_array("/Users/ninawiedemann/Desktop/UNI/Praktikum/ALL/sv_data.csv", files, "Pitcher", dic_lab)
    print("joints array", joints.shape)
    joints_rel = joints[:, :, [7,8,10,11],:]

    joints_rel = Tools.normalize(joints_rel)

    # data, labels = get_sequences(lab_test, joints_rel, sequ_len)
    data = joints_rel
    labels = np.array(lab_test)

    # data = Tools.normalize(data)
    # np.save("/Users/ninawiedemann/Desktop/UNI/Praktikum/ALL/notebooks/bsp_data.npy", data)
    # np.save("/Users/ninawiedemann/Desktop/UNI/Praktikum/ALL/notebooks/bsp_labels.npy", labels)
    print("saved")
    runner = Runner(data, labels, SAVE = save_path, BATCH_SZ=40, EPOCHS = EPOCHS, batch_nr_in_epoch = 50,
            act = tf.nn.relu, rate_dropout = 0,
            learning_rate = 0.0005, nr_layers = 4, n_hidden = 128, optimizer_type="adam", regularization=0,
            first_conv_filters=12, first_conv_kernel=3, second_conv_filter=12,
            second_conv_kernel=3, first_hidden_dense=128, second_hidden_dense=56,
            network = "conv 1st move")
    runner.unique = np.arange(0, 167,1).tolist()
    runner.start()

def extend_data(joints_array_batter, labels):
    def shift_trajectory(play, label, shift):
        new = np.roll(play, shift, axis = 0)
        for i in range(abs(shift)):
            if shift>0:
                new[i]=new[shift]
            else:
                new[-i-1]=new[shift-1]
        return new, label+shift

    shifts = [-15, -12, -10, -7, -5, 5]
    stretch = [0.5, 0.75, 1.25, 1.5]
    variations = []
    for s in shifts:
        for st in stretch:
            variations.append((s, st))
    # variations = [(-15, 0.5), (-15, 0.75), (-15, 1.25), (-15, 1.5), (-10, 0.5),
    #               (-10, 0.75), (-10, 1.25), (-10, 1.5), (-5, 0.5), (-5, 0.75),
    #               (-5, 1.25), (-5, 1.5), (5, 0.5), (5, 0.75), (5, 1.25), (5, 1.5),
    #               (10, 0.5), (10, 0.75), (10, 1.25), (10, 1.5), (15, 0.5), (15, 0.75), (15, 1.25), (15, 1.5)]
    norm = Tools.normalize01(joints_array_batter).tolist()
    #labels = labels.tolist()
    M, N, j, xy = joints_array_batter.shape
    more_data = np.zeros((M*6, N, j, xy))
    more_data[:M]=np.array(norm)
    ind = M
    for i in range(len(norm)):
        var = np.array(variations)[np.random.permutation(len(variations))[:5]]
        for j in var:
            #print(i,j)
            new_data, new_label = shift_trajectory(np.array(norm[i]**j[1]), labels[i], int(j[0]))
            more_data[ind] = new_data
            labels.append(new_label)
            ind+=1
    return more_data, labels

def batter_testing(restore_path, cutoff):
    path = "/Users/ninawiedemann/Desktop/UNI/Praktikum/ALL/video_to_pitchtype_directly/out_testing_batter/"
    joints_array_batter = []
    files = []
    release = []

    with open(path+"release_frames", "r") as infile:
        release_frame = json.load(infile)
    print(release_frame)

    for fi in listdir(path):
        if fi[-5:]==".json":
            try:
                release.append(release_frame[fi[:-5]])
            except KeyError:
                # print("relase frame not in json")
                continue
            files.append(fi[:-5])
            obj_text = codecs.open(path+fi, encoding='utf-8').read()
            joints_array_batter.append(json.loads(obj_text))
    joints_array_batter = np.array(joints_array_batter)[:,:,:12,:]
    print(joints_array_batter.shape)

    ## new: with release
    cut_to_release = []
    assert(len(release)==len(joints_array_batter))
    for i, rel in enumerate(release):
        if rel>75 and rel<110:
            r = int(rel)
        else:
            r = int(np.median(release))
            release[i] = r
            print("mean")
        print(r)
        cut_to_release.append(joints_array_batter[i,r:r+cutoff])
    #labels = np.array(labels)
    # joints_array_batter = ndimage.filters.gaussian_filter1d(joints_array_batter, axis =1, sigma = 3)
    joints_array_batter  = np.array(cut_to_release)
    ## new: with release


    data = Tools.normalize01(joints_array_batter)
    lab, out = test(data, restore_path)
    print(lab)
    dic = {}
    assert(len(lab)==len(files))
    for i in range(len(files)):
        dic[files[i]]= float(lab[i]+release[i])
    print(dic)
    with open("/Users/ninawiedemann/Desktop/UNI/Praktikum/ALL/notebooks/labels_testing", "w") as outfile:
        json.dump(dic, outfile)


def batter_training(save_path, cutoff):
    csv = pd.read_csv("/Users/ninawiedemann/Desktop/UNI/Praktikum/csvs/csv_new_videos.csv")
    path1="/Users/ninawiedemann/Desktop/UNI/Praktikum/ALL/video_to_pitchtype_directly/out_new_batter/"
    path2 = "/Users/ninawiedemann/Desktop/UNI/Praktikum/ALL/video_to_pitchtype_directly/out_new_new_batter/"
    joints_array_batter = []
    files = []
    labels = []
    #play_outcome = []
    release = []

    with open(path1+"release_frames Kopie", "r") as infile:
        release_frame = json.load(infile)

    with open("/Users/ninawiedemann/Desktop/UNI/Praktikum/ALL/notebooks/labels_first_move", "r") as infile:
        labels_dic = json.load(infile)

    for fi in list(labels_dic.keys()):
        try:
            release.append(release_frame[fi])
        except KeyError:
            continue
        files.append(fi)
        if fi+".json" in listdir(path1):
            path = path1
        elif fi+".json" in listdir(path2):
            path = path2
        else:
            print("file not found")
        obj_text = codecs.open(path+fi+".json", encoding='utf-8').read()
        joints_array_batter.append(json.loads(obj_text))
        labels.append(labels_dic[fi])

    print(np.array(joints_array_batter).shape)
    print(labels)
    print(release)
    joints_array_batter = np.array(joints_array_batter)[:,:,:12,:]
    cut_to_release = []
    assert(len(release)==len(joints_array_batter))
    for i, rel in enumerate(release):
        if rel>75 and rel<110:
            r = int(rel)
        else:
            r = int(np.median(release))
            print("mean")
        print(r)
        cut_to_release.append(joints_array_batter[i,r:r+cutoff])
        print("old label", labels[i])
        labels[i] = labels[i]-r
        print("new label", labels[i])
    #labels = np.array(labels)
    # joints_array_batter = ndimage.filters.gaussian_filter1d(joints_array_batter, axis =1, sigma = 3)
    joints_array_batter  = np.array(cut_to_release)
    print(joints_array_batter.shape)
    print(labels)
    more_data, new_labels = extend_data(joints_array_batter, labels)

    print(more_data.shape)

    # runner = Runner(more_data, new_labels, SAVE = save_path, BATCH_SZ=40, EPOCHS = 400, batch_nr_in_epoch = 50,
    #         act = tf.nn.relu, rate_dropout = 0,
    #         learning_rate = 0.0005, nr_layers = 4, n_hidden = 128, optimizer_type="adam", regularization=0,
    #         first_conv_filters=12, first_conv_kernel=3, second_conv_filter=12,
    #         second_conv_kernel=3, first_hidden_dense=128, second_hidden_dense=56,
    #         network = "conv 1st move")
    # runner.unique = np.arange(0, cutoff,1).tolist()
    # runner.start()

    runner = Runner(more_data, np.reshape(new_labels, (-1, 1)), SAVE = save_path, BATCH_SZ=40, EPOCHS = 500, batch_nr_in_epoch = 50,
            act = tf.nn.relu, rate_dropout = 0,
            learning_rate = 0.0005, nr_layers = 4, n_hidden = 128, optimizer_type="adam", regularization=0,
            first_conv_filters=12, first_conv_kernel=3, second_conv_filter=12,
            second_conv_kernel=3, first_hidden_dense=128, second_hidden_dense=56,
            network = "adjustable conv1d")
    runner.unique = [cutoff]
    runner.start()



def testing(files, dic, restore_path, sequ_len):
    joints, labels = get_joint_array("/Users/ninawiedemann/Desktop/UNI/Praktikum/ALL/sv_data.csv", files, "Pitcher", dic)
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

# with open("/Users/ninawiedemann/Desktop/UNI/Praktikum/ALL/notebooks/first_move_frames.json", "r") as infile:
#     dic_lab = json.load(infile)
# all_files = list(dic_lab.keys())
# print(len(all_files))
# # files = all_files[:-10]
# # print(len(all_files))
# # print(all_files[:10])
# SEQU_LEN = 5
# training(all_files, dic_lab, "/Users/ninawiedemann/Desktop/UNI/Praktikum/saved_models/first_move_more", SEQU_LEN)

batter_training("/Users/ninawiedemann/Desktop/UNI/Praktikum/saved_models/batter_first_move", 66)
