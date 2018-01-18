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


### FIRST METHOD: CONVOLUTIONAL NETWORK:
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

    data, labels = get_sequences(lab_test, joints_rel, sequ_len)
    #data = joints_rel
    #labels = np.array(lab_test)

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
    runner.start()


# SECOND APPROACH: TFLEARN RNN
def get_sequences(data_in, inter, length):
    M, N, J, C =data_in.shape
    print(inter.shape,data_in.shape)
    zehner = []

    for i in range(M):
        counter = 0
        sample = []
        for j in range(N):
            if (data_in[i,j]==0).all():
                counter=0
                sample = []
            else:
                sample.append(inter[i,j])
                counter+=1
                if counter==length:
                    zehner.append(sample)
                    counter = 0
                    sample = []

    arr = Tools.normalize(np.array(zehner))  # NORMALIZATION AENDERN?
    M, N, J, C = arr.shape
    print(arr.shape)
    arr = arr.reshape(M, N, J*C)
    data = arr[:, :length-1, :]
    labels = arr[:, length-1, :]
    print(data.shape, labels.shape)
    #print(np.any(labels==0))

    SEP = int(len(data)*0.9)
    ind = np.random.permutation(len(data))
    train_ind = ind[:SEP]
    test_ind = ind[SEP:]

    train_x = data[train_ind]
    test_x = data[test_ind]
    train_t= labels[train_ind]
    test_t = labels[test_ind]

    return train_x, test_x, train_t, test_t

def training_tflearn_old()::
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

nr_layers = 3
n_hidden = 128
BATCHSIZE = 256
EPOCHS = 20
SEQU_LEN = 20
dropout = 0.2

data_in= np.load("unpro_all_coord.npy")
data_in=data_in[:,:,:12,:]
inter = np.load("interpolated.npy")

train_x, test_x, train_t, test_t = get_sequences(data_in, inter, SEQU_LEN)
print(train_x.shape, test_x.shape, train_t.shape, test_t.shape)
_, frames, input_size = train_x.shape

# np.save("train_x.npy", train_x[99])
# np.save("train_t.npy", train_t[99])
# print("saved")
#
# net = tflearn.input_data(shape=[None, frames, input_size])
# net = tflearn.lstm(net, n_hidden, dropout=0.2, return_seq=True)
# net = tflearn.lstm(net, n_hidden)
# out = tflearn.fully_connected(net, input_size, activation='tanh')
# network = tflearn.regression(out, optimizer='adam', loss='mean_square', name='output1')
# return
model = Model()
out, network = model.RNN_network_tflearn(frames, input_size, input_size, nr_layers, n_hidden, dropout, loss = "mean_square", act = "tanh")

m = DNN(network)
m.fit(train_x, train_t, validation_set=(test_x, test_t), show_metric=True, batch_size=BATCHSIZE, snapshot_step=100, n_epoch=EPOCHS)

output = m.predict(test_x[:20])
print(output)
np.save("out_coord", output)
np.save("seque_coord", test_x[:20])
