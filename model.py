import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn


class Model:
    #def __init__():

    def conv1d_with_parameters(self, x, nr_classes, training, rate_dropout, act, first_conv_filters, first_conv_kernel, second_conv_filter,
    second_conv_kernel, first_hidden_dense, second_hidden_dense):
        shape = x.get_shape().as_list()
        net = tf.reshape(x, (-1, shape[1], shape[2]*shape[3]))
        if first_conv_filters!=0:
            net = tf.layers.conv1d(net, filters=first_conv_filters, kernel_size=first_conv_kernel, activation=act, name="conv1")
            # tf.summary.histogram("conv_1_layer", net)
            net = tf.layers.dropout(net, rate=rate_dropout, training=training)
        if second_conv_filter!=0:
            net = tf.layers.conv1d(net, filters=second_conv_filter, kernel_size=second_conv_kernel, activation = act, name="conv4")
            net = tf.layers.dropout(net, rate=rate_dropout, training=training)
        shapes = net.get_shape().as_list()
        ff = tf.reshape(net, (-1, shapes[1]*shapes[2]))
        if first_hidden_dense!=0:
            ff = tf.layers.dense(ff, first_hidden_dense, activation = act)
        if second_hidden_dense!=0:
            ff = tf.layers.dense(ff, second_hidden_dense, activation = act)
        logits = tf.layers.dense(ff, nr_classes, activation = None, name = "ff3")
        out = tf.nn.softmax(logits)
        return out, logits

    def conv1stmove(self, x, nr_classes, training, rate_dropout, act, first_conv_filters, first_conv_kernel, second_conv_filter,
    second_conv_kernel, first_hidden_dense, second_hidden_dense):
        shape = x.get_shape().as_list()
        net = tf.reshape(x, (-1, shape[1], shape[2]*shape[3]))
        net = tf.layers.conv1d(net, filters=first_conv_filters, kernel_size=first_conv_kernel, activation=act, padding = "SAME", reuse = None)
        #print(net)
        # tf.summary.histogram("conv_1_layer", net)
        net = tf.layers.conv1d(net, filters=second_conv_filter, kernel_size=second_conv_kernel, activation = act, padding = "SAME",reuse = None)
        #print(net)
        net = tf.layers.conv1d(net, filters=1, kernel_size=second_conv_kernel, activation = None, padding = "SAME", reuse = None)
        #print(net)
        shapes = net.get_shape().as_list()
        print(shapes)
        logits = net #tf.reshape(net, (-1, shapes[1]*shapes[2]))
        out = tf.nn.softmax(logits)
        return out, logits

    def conv2d(self, net, nr_classes, training, rate_dropout, act, first_conv_filters, first_conv_kernel, second_conv_filter,
    second_conv_kernel, first_hidden_dense, second_hidden_dense):
        if first_conv_filters!=0:
            net = tf.layers.conv2d(net, filters=first_conv_filters, kernel_size=first_conv_kernel, strides=2, activation=act, name="conv1")
            # tf.summary.histogram("conv_1_layer", net)
            net = tf.layers.dropout(net, rate=rate_dropout, training=training)
        if second_conv_filter!=0:
            net = tf.layers.conv2d(net, filters=second_conv_filter, kernel_size=second_conv_kernel, activation = act, name="conv4")
            net = tf.layers.dropout(net, rate=rate_dropout, training=training)
        shapes = net.get_shape().as_list()
        ff = tf.reshape(net, (-1, shapes[1]*shapes[2]*shapes[3]))
        if first_hidden_dense!=0:
            ff = tf.layers.dense(ff, first_hidden_dense, activation = act)
        if second_hidden_dense!=0:
            ff = tf.layers.dense(ff, second_hidden_dense, activation = act)
        logits = tf.layers.dense(ff, nr_classes, activation = None, name = "ff3")
        out = tf.nn.softmax(logits)
        return out, logits

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
            net = input_data(shape=[None, frames, input_size])
            #net = embedding(net, input_dim=frames*input_size, output_dim=128)
            for i in range(nr_layers-1):
                net = tflearn.bidirectional_rnn(net, BasicLSTMCell(n_hidden), BasicLSTMCell(n_hidden), return_seq=True)
                net = dropout(net, 0.5)
            net = bidirectional_rnn(net, BasicLSTMCell(n_hidden), BasicLSTMCell(n_hidden))
            net = dropout(net, 0.5)
            out = fully_connected(net, num_classes, activation=act)
            trainer = regression(out, optimizer='adam', loss=loss)
            return out, trainer
        except ImportError:
            print("Tflearn not installed")

    def convshort(self, x, nr_classes, training, rate_dropout, act):
        #x = tf.nn.max_pool(x, ksize=[1,2,1,1], strides=[1,2,1,1], padding = "SAME")
        shape = x.get_shape().as_list()
        x_ = tf.reshape(x, (-1, shape[1], shape[2]*shape[3]))
        #shape_y = y.get_shape().as_list()
        net = tf.layers.conv1d(x_, filters=128, kernel_size=5, strides=2, activation=act, name="conv1")
        tf.summary.histogram("conv_1_layer", net)
        net= tf.layers.batch_normalization(net)
        #net = tf.nn.max_pool(net, ksize=[1,2,1,1], strides=[1,2,1,1], padding = "SAME")
        net = tf.layers.dropout(net, rate=rate_dropout, training=training)

        net = tf.layers.conv1d(net, filters=56, kernel_size=1, activation = act, name="conv4")
        net= tf.layers.batch_normalization(net)
        net = tf.layers.dropout(net, rate=rate_dropout, training=training)

        shapes = net.get_shape().as_list()
        ff = tf.reshape(net, (-1, shapes[1]*shapes[2]))
        logits = tf.layers.dense(ff, nr_classes, activation = None, name = "ff3")
        out = tf.nn.softmax(logits)
        return out, logits

    def conv1dnet(self, x, nr_classes, training, rate_dropout, act):   # conv1d(256,5,2)-conv1d(256,3)-conv1d(128,3)-conv1d(1,1)-dense(1024)-dense(128),dense(nr_classes)
        shape = x.get_shape().as_list()
        x_ = tf.reshape(x, (-1, shape[1], shape[2]*shape[3]))
        #shape_y = y.get_shape().as_list()

        net = tf.layers.conv1d(x_, filters=256, kernel_size=5, strides=2, activation=act, name="conv1")
        tf.summary.histogram("conv_1_layer", net)
        net = tf.layers.dropout(net, rate=rate_dropout, training=training)
        net = tf.layers.conv1d(net, filters=256, kernel_size=3, strides=1, activation=act , name="conv2")
        tf.summary.histogram("conv_2_layer", net)
        net = tf.layers.dropout(net, rate=rate_dropout, training=training)
        net = tf.layers.conv1d(net, filters=128, kernel_size=3, strides=1, activation=act, name="conv3")
        tf.summary.histogram("conv_3_layer", net)
        net = tf.layers.dropout(net, rate=rate_dropout, training=training)
        net = tf.layers.conv1d(net, filters=1, kernel_size=1, activation = act, name="conv4")
        shapes = net.get_shape().as_list()
        ff = tf.reshape(net, (-1, shapes[1]*shapes[2]))
        ff = tf.layers.dense(ff, 1024, activation = act, name="ff1")
        ff = tf.layers.dropout(ff, rate=rate_dropout, training=training)
        tf.summary.histogram("ff_1", net)
        ff = tf.layers.dense(ff, 128, activation = act, name="ff2")
        ff = tf.layers.dropout(ff, rate=rate_dropout, training=training)
        tf.summary.histogram("ff_2_layer", net)
        logits = tf.layers.dense(ff, nr_classes, activation = None, name = "ff3")
        out = tf.nn.softmax(logits)
        return out, logits


    def best_in_cluster_concat53(self, x, nr_classes, training, rate_dropout, act):
        shape = x.get_shape().as_list()
        x_ = tf.reshape(x, (-1, shape[1], shape[2]*shape[3]))
        net = tf.layers.conv1d(x_, filters=256, kernel_size=5, strides=2, activation=act, name="conv1")
        # tf.summary.histogram("conv_1_layer", net)
        net = tf.layers.dropout(net, rate=rate_dropout, training=training)
        net = tf.layers.conv1d(net, filters=128, kernel_size=3, activation = act, name="conv4")
        net = tf.layers.dropout(net, rate=rate_dropout, training=training)
        shapes = net.get_shape().as_list()
        ff = tf.reshape(net, (-1, shapes[1]*shapes[2]))
        logits = tf.layers.dense(ff, nr_classes, activation = None, name = "ff3")
        out = tf.nn.softmax(logits)
        return out, logits

    def best_in_cluster46(self, x, nr_classes, training, rate_dropout, act):
        shape = x.get_shape().as_list()
        x_ = tf.reshape(x, (-1, shape[1], shape[2], shape[3]))
        #shape_y = y.get_shape().as_list()

        net = tf.layers.conv2d(x_, filters=128, kernel_size=5, strides=2, activation=act, name="conv1")
        tf.summary.histogram("conv_1_layer", net)
        net = tf.layers.dropout(net, rate=rate_dropout, training=training)
        net = tf.layers.conv2d(net, filters=128, kernel_size=3, strides=1, activation=act, name="conv3")
        tf.summary.histogram("conv_3_layer", net)
        net = tf.layers.dropout(net, rate=rate_dropout, training=training)
        net = tf.layers.conv2d(net, filters=1, kernel_size=1, activation = act, name="conv4")
        shapes = net.get_shape().as_list()
        ff = tf.reshape(net, (-1, shapes[1]*shapes[2]*shapes[3]))
        ff = tf.layers.dense(ff, 1024, activation = act, name="ff1")
        tf.summary.histogram("ff_1", net)
        #ff = tf.layers.dense(ff, 128, activation = act, name="ff2")
        #tf.summary.histogram("ff_2_layer", net)
        logits = tf.layers.dense(ff, nr_classes, activation = None, name = "ff3")
        out = tf.nn.softmax(logits)

        return out, logits

    def conv2dnet(self, x, nr_classes, training, rate_dropout, act):  # conv2d(256,5,2)-conv2d(256,3)-conv2d(128,3)-conv2d(1,1)-dense(1024)-dense(128),dense(nr_classes)
        shape = x.get_shape().as_list()
        x_ = tf.reshape(x, (-1, shape[1], shape[2], shape[3]))
        #shape_y = y.get_shape().as_list()

        net = tf.layers.conv2d(x_, filters=256, kernel_size=5, strides=2, activation=act, name="conv1")
        tf.summary.histogram("conv_1_layer", net)
        net = tf.layers.dropout(net, rate=rate_dropout, training=training)
        net = tf.layers.conv2d(net, filters=256, kernel_size=3, strides=1, activation=act , name="conv2")
        tf.summary.histogram("conv_2_layer", net)
        net = tf.layers.dropout(net, rate=rate_dropout, training=training)
        net = tf.layers.conv2d(net, filters=128, kernel_size=3, strides=1, activation=act, name="conv3")
        tf.summary.histogram("conv_3_layer", net)
        net = tf.layers.dropout(net, rate=rate_dropout, training=training)
        net = tf.layers.conv2d(net, filters=1, kernel_size=1, activation = act, name="conv4")
        shapes = net.get_shape().as_list()
        ff = tf.reshape(net, (-1, shapes[1]*shapes[2]*shapes[3]))
        ff = tf.layers.dense(ff, 1024, activation = act, name="ff1")
        tf.summary.histogram("ff_1", net)
        ff = tf.layers.dense(ff, 128, activation = act, name="ff2")
        tf.summary.histogram("ff_2_layer", net)
        logits = tf.layers.dense(ff, nr_classes, activation = None, name = "ff3")
        out = tf.nn.softmax(logits)

        return out, logits

    def conv_RNN(self, x_in, nr_classes, n_hidden, nr_layers, nr_filters):
        shape = x_in.get_shape().as_list()
        net = tf.reshape(x_in, (-1, shape[1], shape[2], shape[3], 1))
        net = tf.layers.conv3d(net, filters=nr_filters, kernel_size = 5, activation = tf.nn.relu)
        net = tf.layers.conv3d(net, filters=1, kernel_size = 5, activation = tf.nn.relu)
        shape = net.get_shape().as_list()
        x = tf.reshape(net, (-1, shape[1], shape[2]*shape[3]))
        x = tf.unstack(x, shape[1], 1)
        def lstm_cell():
              return rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
        stacked_lstm = rnn.MultiRNNCell([lstm_cell() for _ in range(nr_layers)])
        outputs, states = rnn.static_rnn(stacked_lstm, x, dtype=tf.float32)
        # Linear activation, using rnn inner loop last output
        out_logits = tf.layers.dense(outputs[-1], nr_classes)   #tf.matmul(outputs[-1], weights['out']) + biases['out']
        out = tf.nn.softmax(out_logits)
        return out, out_logits

    def RNN(self, x_in, nr_classes, n_hidden, nr_layers):
        shape = x_in.get_shape().as_list()
        x = tf.reshape(x_in, (-1, shape[1], shape[2]*shape[3]))

        x = tf.unstack(x, shape[1], 1)

        def lstm_cell():
              return rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

        stacked_lstm = rnn.MultiRNNCell([lstm_cell() for _ in range(nr_layers)])
        outputs, states = rnn.static_rnn(stacked_lstm, x, dtype=tf.float32)
        # Linear activation, using rnn inner loop last output
        out_logits = tf.layers.dense(outputs[-1], nr_classes)   #tf.matmul(outputs[-1], weights['out']) + biases['out']
        out = tf.nn.softmax(out_logits)
        return out, out_logits

    def only_conv(self, x, training, rate_dropout, act):
        shape = x.get_shape().as_list()
        x_ = tf.reshape(x, (-1, shape[1], shape[2]*shape[3]))
        net = tf.layers.conv1d(x_, filters=256, kernel_size=5, strides=2, activation=act)
        net = tf.layers.dropout(net, rate=rate_dropout, training=training)
        net = tf.layers.conv1d(net, filters=256, kernel_size=3, strides=1, activation=act)
        net = tf.layers.conv1d(net, filters=128, kernel_size=3, strides=1, activation=act)
        net = tf.layers.dropout(net, rate=rate_dropout, training=training)
        net = tf.layers.conv1d(net, filters=1, kernel_size=1, activation = act)
        return net

    def only_ff(self, conv_tensor, nr_classes, act):
        shapes = conv_tensor.get_shape().as_list()
        ff = tf.reshape(conv_tensor, (-1, shapes[1]*shapes[2]))
        ff = tf.layers.dense(ff, 1024, activation = act)
        ff = tf.layers.dense(ff, 128, activation = act)
        logits = tf.layers.dense(ff, nr_classes, activation = None)
        out = tf.nn.softmax(logits)

        return out, logits
