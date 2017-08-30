import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

import tflearn
from tflearn import DNN

class Model:
    #def __init__():
    def RNN_network_tflearn(self, frames, input_size, num_classes):
        """Create a one-layer LSTM"""
        net = tflearn.input_data(shape=[None, frames, input_size])
        net = tflearn.lstm(net, 256, dropout=0.2)
        out = tflearn.fully_connected(net, num_classes, activation='softmax')
        trainer = tflearn.regression(out, optimizer='adam', loss='categorical_crossentropy', name='output1')
        return out, trainer

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

        net = tf.layers.conv1d(net, filters=1, kernel_size=1, activation = act, name="conv4")
        net= tf.layers.batch_normalization(net)
        net = tf.layers.dropout(net, rate=rate_dropout, training=training)

        shapes = net.get_shape().as_list()
        ff = tf.reshape(net, (-1, shapes[1]*shapes[2]))
        logits = tf.layers.dense(ff, nr_classes, activation = None, name = "ff3")
        out = tf.nn.softmax(logits)
        return out, logits

    def conv1dnet(self, x, nr_classes, training, rate_dropout, act):
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
        tf.summary.histogram("conv_1_layer", net)
        # net = tf.layers.dropout(net, rate=rate_dropout, training=training)
        net = tf.layers.conv1d(net, filters=128, kernel_size=3, activation = act, name="conv4")
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

    def conv2dnet(self, x, nr_classes, training, rate_dropout, act):
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
