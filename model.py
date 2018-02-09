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
    second_conv_kernel, first_hidden_dense, second_hidden_dense, out_filters = 1):
        shape = x.get_shape().as_list()
        net = tf.reshape(x, (-1, shape[1], shape[2]*shape[3]))
        net = tf.layers.conv1d(net, filters=first_conv_filters, kernel_size=first_conv_kernel, activation=act, padding = "SAME", reuse = None)
        net = tf.layers.batch_normalization(net, training = training)
        #print(net)
        # tf.summary.histogram("conv_1_layer", net)
        net = tf.layers.conv1d(net, filters=second_conv_filter, kernel_size=second_conv_kernel, activation = act, padding = "SAME",reuse = None)
        net = tf.layers.batch_normalization(net, training = training)
        #print(net)
        net = tf.layers.conv1d(net, filters=out_filters, kernel_size=second_conv_kernel, activation = None, padding = "SAME", reuse = None)
        net = tf.layers.batch_normalization(net, training = training)
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


    def conv1d_big(self, x, nr_classes, training, rate_dropout, act):   # conv1d(256,5,2)-conv1d(256,3)-conv1d(128,3)-conv1d(1,1)-dense(1024)-dense(128),dense(nr_classes)
        shape = x.get_shape().as_list()
        x_ = tf.reshape(x, (-1, shape[1], shape[2]*shape[3]))
        #shape_y = y.get_shape().as_list()
        net = tf.layers.conv1d(x_, filters=56, kernel_size=5, strides=2, activation=act, name="conv1")
        net = tf.layers.max_pooling1d(net, 2, 2, padding="same", name = "maxpool1")
        net = tf.layers.batch_normalization(net, training=training)
        print(net)
        #tf.summary.histogram("conv_1_layer", net)
        #net = tf.layers.dropout(net, rate=rate_dropout, training=training)
        net = tf.layers.conv1d(net, filters=128, kernel_size=5, strides=2, activation=act , name="conv2")
        # net = tf.layers.max_pooling1d(net, 2, 1, padding="same", name = "maxpool2")
        net = tf.layers.batch_normalization(net, training=training)
        print(net)
        #tf.summary.histogram("conv_2_layer", net)
        net = tf.layers.dropout(net, rate=rate_dropout, training=training)
        net = tf.layers.conv1d(net, filters=128, kernel_size=5, strides=1, activation=act, name="conv3")
        net = tf.layers.max_pooling1d(net, 2, 1, padding="same", name = "maxpool3")
        net = tf.layers.batch_normalization(net, training=training)
        print(net)
        #tf.summary.histogram("conv_3_layer", net)
        #net = tf.layers.dropout(net, rate=rate_dropout, training=training)
        net = tf.layers.conv1d(net, filters=256, kernel_size=3, activation = act, name="conv4")
        #net = tf.layers.max_pooling1d(net, 2, 1, padding="same", name = "maxpool4")
        net = tf.layers.batch_normalization(net)
        print(net)
        net = tf.layers.conv1d(net, filters=256, kernel_size=3, activation = act, name="conv5")
        net = tf.layers.max_pooling1d(net, 2, 1, padding="same", name = "maxpool5")
        net = tf.layers.batch_normalization(net, training=training)
        print(net)

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

    def RNN(self, x_in, nr_classes, n_hidden, nr_layers):
        shape = x_in.get_shape().as_list()
        x_in = tf.reshape(x_in, (-1, shape[1], shape[2]*shape[3]))

        x = tf.unstack(x_in, shape[1], 1)

        def lstm_cell():
              return rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

        stacked_lstm = rnn.MultiRNNCell([lstm_cell() for _ in range(nr_layers)])
        outputs, states = rnn.static_rnn(stacked_lstm, x, dtype=tf.float32)
        # Linear activation, using rnn inner loop last output
        out_logits = tf.layers.dense(outputs[-1], nr_classes)   #tf.matmul(outputs[-1], weights['out']) + biases['out']
        out = tf.nn.softmax(out_logits)
        return out, out_logits

    def only_conv(self, x, nr_classes, training, rate_dropout, act, first_conv_filters, first_conv_kernel, second_conv_filter,
    second_conv_kernel, first_hidden_dense, second_hidden_dense):
        with tf.variable_scope("convolution"):
            shape = x.get_shape().as_list()
            net = tf.reshape(x, (-1, shape[1], shape[2]*shape[3]))
            if first_conv_filters!=0:
                net = tf.layers.conv1d(net, filters=first_conv_filters, kernel_size=first_conv_kernel, activation=act, name="conv1")
                # tf.summary.histogram("conv_1_layer", net)
                net = tf.layers.dropout(net, rate=rate_dropout, training=training)
            if second_conv_filter!=0:
                net = tf.layers.conv1d(net, filters=second_conv_filter, kernel_size=second_conv_kernel, activation = act, name="conv4")
                net = tf.layers.dropout(net, rate=rate_dropout, training=training)
        return net

### OLD MODELS:

# def only_conv(self, x, training, rate_dropout, act):
#     shape = x.get_shape().as_list()
#     x_ = tf.reshape(x, (-1, shape[1], shape[2]*shape[3]))
#     net = tf.layers.conv1d(x_, filters=256, kernel_size=5, strides=2, activation=act)
#     net = tf.layers.dropout(net, rate=rate_dropout, training=training)
#     net = tf.layers.conv1d(net, filters=256, kernel_size=3, strides=1, activation=act)
#     net = tf.layers.conv1d(net, filters=128, kernel_size=3, strides=1, activation=act)
#     net = tf.layers.dropout(net, rate=rate_dropout, training=training)
#     net = tf.layers.conv1d(net, filters=1, kernel_size=1, activation = act)
#     return net
#
# def only_ff(self, conv_tensor, nr_classes, act):
#     shapes = conv_tensor.get_shape().as_list()
#     ff = tf.reshape(conv_tensor, (-1, shapes[1]*shapes[2]))
#     ff = tf.layers.dense(ff, 1024, activation = act)
#     ff = tf.layers.dense(ff, 128, activation = act)
#     logits = tf.layers.dense(ff, nr_classes, activation = None)
#     out = tf.nn.softmax(logits)
#
#     return out, logits

# def best_in_cluster_concat53(self, x, nr_classes, training, rate_dropout, act):
#     shape = x.get_shape().as_list()
#     x_ = tf.reshape(x, (-1, shape[1], shape[2]*shape[3]))
#     net = tf.layers.conv1d(x_, filters=256, kernel_size=5, strides=2, activation=act, name="conv1")
#     # tf.summary.histogram("conv_1_layer", net)
#     net = tf.layers.dropout(net, rate=rate_dropout, training=training)
#     net = tf.layers.conv1d(net, filters=128, kernel_size=3, activation = act, name="conv4")
#     net = tf.layers.dropout(net, rate=rate_dropout, training=training)
#     shapes = net.get_shape().as_list()
#     ff = tf.reshape(net, (-1, shapes[1]*shapes[2]))
#     logits = tf.layers.dense(ff, nr_classes, activation = None, name = "ff3")
#     out = tf.nn.softmax(logits)
#     return out, logits
#
# def best_in_cluster46(self, x, nr_classes, training, rate_dropout, act):
#     shape = x.get_shape().as_list()
#     x_ = tf.reshape(x, (-1, shape[1], shape[2], shape[3]))
#     #shape_y = y.get_shape().as_list()
#
#     net = tf.layers.conv2d(x_, filters=128, kernel_size=5, strides=2, activation=act, name="conv1")
#     tf.summary.histogram("conv_1_layer", net)
#     net = tf.layers.dropout(net, rate=rate_dropout, training=training)
#     net = tf.layers.conv2d(net, filters=128, kernel_size=3, strides=1, activation=act, name="conv3")
#     tf.summary.histogram("conv_3_layer", net)
#     net = tf.layers.dropout(net, rate=rate_dropout, training=training)
#     net = tf.layers.conv2d(net, filters=1, kernel_size=1, activation = act, name="conv4")
#     shapes = net.get_shape().as_list()
#     ff = tf.reshape(net, (-1, shapes[1]*shapes[2]*shapes[3]))
#     ff = tf.layers.dense(ff, 1024, activation = act, name="ff1")
#     tf.summary.histogram("ff_1", net)
#     #ff = tf.layers.dense(ff, 128, activation = act, name="ff2")
#     #tf.summary.histogram("ff_2_layer", net)
#     logits = tf.layers.dense(ff, nr_classes, activation = None, name = "ff3")
#     out = tf.nn.softmax(logits)
#
#     return out, logits
#
# def conv_RNN(self, x_in, nr_classes, n_hidden, nr_layers, nr_filters):
#     shape = x_in.get_shape().as_list()
#     net = tf.reshape(x_in, (-1, shape[1], shape[2], shape[3], 1))
#     net = tf.layers.conv3d(net, filters=nr_filters, kernel_size = 5, activation = tf.nn.relu)
#     net = tf.layers.conv3d(net, filters=1, kernel_size = 5, activation = tf.nn.relu)
#     shape = net.get_shape().as_list()
#     x = tf.reshape(net, (-1, shape[1], shape[2]*shape[3]))
#     x = tf.unstack(x, shape[1], 1)
#     def lstm_cell():
#           return rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
#     stacked_lstm = rnn.MultiRNNCell([lstm_cell() for _ in range(nr_layers)])
#     outputs, states = rnn.static_rnn(stacked_lstm, x, dtype=tf.float32)
#     # Linear activation, using rnn inner loop last output
#     out_logits = tf.layers.dense(outputs[-1], nr_classes)   #tf.matmul(outputs[-1], weights['out']) + biases['out']
#     out = tf.nn.softmax(out_logits)
#     return out, out_logits
#
# elif self.network=="conv1d(256,5,2)-conv1d(256,3)-conv1d(128,3)-conv1d(1,1)-dense(1024)-dense(128),dense(nr_classes)":
#     out, logits = model.conv1dnet(x, nr_classes, training, self.rate_dropout, self.act)
#
# elif self.network=="conv2d(256,5,2)-conv2d(256,3)-conv2d(128,3)-conv2d(1,1)-dense(1024)-dense(128),dense(nr_classes)":
#     out, logits = model.conv2dnet(x, nr_classes, training, self.rate_dropout, self.act)
#
# def conv2dnet(self, x, nr_classes, training, rate_dropout, act):  # conv2d(256,5,2)-conv2d(256,3)-conv2d(128,3)-conv2d(1,1)-dense(1024)-dense(128),dense(nr_classes)
#     shape = x.get_shape().as_list()
#     x_ = tf.reshape(x, (-1, shape[1], shape[2], shape[3]))
#     #shape_y = y.get_shape().as_list()
#
#     net = tf.layers.conv2d(x_, filters=256, kernel_size=5, strides=2, activation=act, name="conv1")
#     tf.summary.histogram("conv_1_layer", net)
#     net = tf.layers.dropout(net, rate=rate_dropout, training=training)
#     net = tf.layers.conv2d(net, filters=256, kernel_size=3, strides=1, activation=act , name="conv2")
#     tf.summary.histogram("conv_2_layer", net)
#     net = tf.layers.dropout(net, rate=rate_dropout, training=training)
#     net = tf.layers.conv2d(net, filters=128, kernel_size=3, strides=1, activation=act, name="conv3")
#     tf.summary.histogram("conv_3_layer", net)
#     net = tf.layers.dropout(net, rate=rate_dropout, training=training)
#     net = tf.layers.conv2d(net, filters=1, kernel_size=1, activation = act, name="conv4")
#     shapes = net.get_shape().as_list()
#     ff = tf.reshape(net, (-1, shapes[1]*shapes[2]*shapes[3]))
#     ff = tf.layers.dense(ff, 1024, activation = act, name="ff1")
#     tf.summary.histogram("ff_1", net)
#     ff = tf.layers.dense(ff, 128, activation = act, name="ff2")
#     tf.summary.histogram("ff_2_layer", net)
#     logits = tf.layers.dense(ff, nr_classes, activation = None, name = "ff3")
#     out = tf.nn.softmax(logits)
#     return out, logits
