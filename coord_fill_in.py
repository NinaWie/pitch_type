import pandas as pd
import numpy as np
import tensorflow as tf
import scipy as sp
import scipy.stats
from tools import Tools
from model import Model
import tflearn
from tflearn import DNN

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
