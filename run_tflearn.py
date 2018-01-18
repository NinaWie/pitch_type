
import pandas as pd
import numpy as np
import tensorflow as tf
import scipy as sp
import scipy.stats
import threading

from tools import Tools
# from model import Model

import tflearn
from tflearn import DNN

def RNN_network_tflearn(self, frames, input_size, num_classes, nr_layers, n_hidden, dropout_rate, loss = "categorical_crossentropy", act = "softmax"):
    try:
        import tflearn
        from tflearn import DNN
        """Create a one-layer LSTM"""
        net = tflearn.input_data(shape=[None, frames, input_size])
        for i in range(nr_layers-1):
            net = tflearn.lstm(net, n_hidden, dropout=dropout_rate, return_seq=True)
        net = tflearn.lstm(net, n_hidden)
        out = tflearn.fully_connected(net, num_classes, activation=act)
        trainer = tflearn.regression(out, optimizer='adam', loss=loss, name='output1')
        return out, trainer
    except ImportError:
        print("Tflearn not installed")

def bidirectional_lstm(self, frames, input_size, num_classes, nr_layers, n_hidden, dropout_rate, loss = "categorical_crossentropy", act = "softmax"):
    try:
        import tflearn
        from tflearn.data_utils import to_categorical, pad_sequences
        from tflearn.datasets import imdb
        from tflearn.layers.core import input_data, dropout, fully_connected
        from tflearn.layers.embedding_ops import embedding
        from tflearn.layers.recurrent import bidirectional_rnn, BasicLSTMCell
        from tflearn.layers.estimator import regression
        """Create a one-layer LSTM"""
        net = input_data(shape= np.array([None, frames, 6, 2]))
        print("1", net)
        #net = embedding(net, input_dim=frames*input_size, output_dim=128)
        for i in range(nr_layers-1):
            net = bidirectional_rnn(net, BasicLSTMCell(n_hidden), BasicLSTMCell(n_hidden), return_seq=True)
            print("layer", i, net)
            net = dropout(net, 0.5)
        net = bidirectional_rnn(net, BasicLSTMCell(n_hidden), BasicLSTMCell(n_hidden))
        print("nach schleife", net)
        net = dropout(net, 0.5)
        out = fully_connected(net, num_classes, activation=act)
        print("out", out)
        trainer = regression(out, optimizer='adam', loss=loss)
        return out, trainer
    except ImportError:
        print("Tflearn not installed")

class Runner(threading.Thread):

    def __init__(self, data, labels_string, SAVE = None, files = [], BATCH_SZ=40, EPOCHS = 60, batch_nr_in_epoch = 100,
            act = tf.nn.relu, rate_dropout = 0,
            learning_rate = 0.0005, nr_layers = 4, n_hidden = 128, optimizer_type="adam", regularization=0,
            first_conv_filters=128, first_conv_kernel=9, second_conv_filter=128,
            second_conv_kernel=5, first_hidden_dense=128, second_hidden_dense=0,
            network = "adjustable conv1d"):
        threading.Thread.__init__(self)
        self.data = data
        self.files = files
        self.labels_string = np.array(labels_string)
        self.unique  = np.unique(labels_string).tolist()
        self.SAVE = SAVE
        self.BATCH_SZ=BATCH_SZ
        self.EPOCHS = EPOCHS
        self.batch_nr_in_epoch = batch_nr_in_epoch
        self.act = act
        self.rate_dropout = rate_dropout
        self.learning_rate = learning_rate
        self.nr_layers = nr_layers
        self.n_hidden = n_hidden
        self.optimizer_type = optimizer_type
        self.regularization= regularization
        self.first_conv_filters = first_conv_filters
        self.first_conv_kernel = first_conv_kernel
        self.second_conv_filter = second_conv_filter
        self.second_conv_kernel = second_conv_kernel
        self.first_hidden_dense = first_hidden_dense
        self.second_hidden_dense = second_hidden_dense
        self.network = network


    def run(self):
        try:
            shutil.rmtree("/Users/ninawiedemann/Desktop/UNI/Praktikum/logs")
            print("logs removed")
        except:
            print("logs could not be removed")

        tf.reset_default_graph()
        sess = tf.InteractiveSession()

        nr_classes = len(self.unique)
        print("classes", self.unique)

        model = Model()

        M,N,nr_joints,nr_coordinates = self.data.shape
        SEP = int(M*0.9)
        # print("Test set size: ", len_test, " train set size: ", len_train)
        # print("Shapes of train_x", train_x.shape, "shape of test_x", test_x.shape)
        ind = np.random.permutation(len(self.data))
        train_ind = ind[:SEP]
        test_ind = ind[SEP:]

        train_x = self.data[train_ind]
        testing = self.data[test_ind]
        test_x = testing[:-40]
        val_x = testing[-40:]
        labels_string_train = self.labels_string[train_ind]
        labels_string_testing = self.labels_string[test_ind]
        labels_string_test = labels_string_testing[:-10]
        labels_string_val = labels_string_testing[-10:]
        train_x, labels_string_train = Tools.balance(train_x, labels_string_train)
        train_t = Tools.onehot_with_unique(labels_string_train, self.unique)
        test_t = Tools.onehot_with_unique(labels_string_test, self.unique)
        print("Train", train_x.shape, train_t.shape, labels_string_train.shape, "Test", test_x.shape, test_t.shape, labels_string_test.shape, "Val", val_x.shape, labels_string_val.shape)
        len_train, N, nr_joints, nr_coordinates = train_x.shape
        tr_x = train_x.reshape(len_train, N, nr_joints*nr_coordinates)
        te_x = test_x.reshape(len(test_x), N, nr_joints*nr_coordinates)
        val = val_x.reshape(len(val_x), N, nr_joints*nr_coordinates)

        out, net = RNN_network_tflearn(N, nr_joints*nr_coordinates, nr_classes,self.nr_layers, self.n_hidden, self.rate_dropout)
        m = DNN(net)
        m.fit(tr_x, train_t, validation_set=(te_x, test_t), show_metric=True, batch_size=self.BATCH_SZ, snapshot_step=1000, n_epoch=self.EPOCHS)
        # labels_string_test = labels_string_test[:40]
        out_test = m.predict(val)
        pitches_test = Tools.decode_one_hot(out_test, self.unique)
        print("Accuracy test: ", Tools.accuracy(pitches_test, labels_string_val))
        print("Accuracy test by class: ", Tools.accuracy_per_class(pitches_test, labels_string_val))
        print("True                   Test                 ", self.unique)
        # print(np.swapaxes(np.append([labels_string_test], [pitches_test], axis=0), 0,1))
        for i in range(len(labels_string_test)):
            print('{:20}'.format(labels_string_val[i]), '{:20}'.format(pitches_test[i]), ['%.2f        ' % elem for elem in out_test[i]])

        return pitches_test, out_test
