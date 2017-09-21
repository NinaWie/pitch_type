import pandas as pd
import numpy as np
import tensorflow as tf
import scipy as sp
import scipy.stats

#from sklearn.preprocessing import StandardScaler
from data_preprocess import Preprocessor
from tools import Tools
from model import Model

import shutil
import threading

PATH = "cf"
CUT_OFF_Classes = 50
leaky_relu = lambda x: tf.maximum(0.2*x, x)

# PREPROCESS DATA
if PATH is "cf" or PATH is "concat":
    prepro = Preprocessor("cf_data.csv")
else:
    prepro = Preprocessor("sv_data.csv")

# ONE PLAYER
#players, _ = prepro.get_list_with_most("Pitcher")
#prepro.select_movement("Windup")

# prepro.cut_file_to_pitcher(players[player])  # change_restore


prepro.remove_small_classes(CUT_OFF_Classes)

if PATH is not "concat":
    # data_raw = prepro.get_coord_arr(PATH+"_all_coord.npy")
    data_raw = np.load("cf_all_coord.npy")
    print("data loaded")
else:
    data_raw = prepro.concat_with_second("sv_data.csv", PATH+"_all_coord.npy")

labels = prepro.get_labels()


def runscript(lock, data_in, labels_string, EPOCHS = 80, BATCH_SZ = 40, batch_nr_in_epoch = 100, align = False, act = tf.nn.relu, rate_dropout = 0,
        learning_rate = 0.0005, nr_layers = 4, n_hidden = 128, optimizer_type="adam", regularization=0, first_conv_filters=256, first_conv_kernel=5, second_conv_filter=128,
        second_conv_kernel=3, first_hidden_dense=0, second_hidden_dense=0, network = "adjustable conv1d"):

    tf.reset_default_graph()
    sess = tf.InteractiveSession()


    data_without_head = data_in[:, :, :12, :]

    if align:
        data_without_head = Tools.align_frames(data_without_head, prepro.get_release_frame(60, 120), 60, 40)
    data = Tools.normalize(data_without_head)

    # ONE PITCH TYPE
    # pitchi, _ = prepro.get_list_with_most("Pitch Type")
    # print("classe trained on:", pitchi[0])
    # prepro.set_labels(pitchi[0])  # change_restore

    # CONCAT
    # prepro.remove_small_classes(CUT_OFF_Classes)
    # data_raw = prepro.concat_with_second(SECOND) #prepro.get_coord_arr()  #np.load("coord_sv.npy")
    # data = Tools.normalize(data_raw)
    #data = np.load("coord_concat.npy")

    M,N,nr_joints,nr_coordinates = data.shape
    SEP = int(M*0.9)

    labels, unique = Tools.onehot_encoding(labels_string)

    nr_classes = len(np.unique(labels_string)) # hier
    ex_per_class = BATCH_SZ//nr_classes
    BATCHSIZE = nr_classes*ex_per_class
    # print("nr classes", nr_classes, "Batchsize", BATCHSIZE)
    # print("classes: ", unique)
    # print("data shape:", data.shape, "label_shape", labels.shape, labels_string.shape)

    # NET

    ind = np.random.permutation(len(data))
    train_ind = ind[:SEP]
    test_ind = ind[SEP:]

    train_x = data[train_ind]
    test_x = data[test_ind]
    train_t= labels[train_ind]
    test_t = labels[test_ind]
    labels_string_train = labels_string[train_ind]
    labels_string_test = labels_string[test_ind]


    index_liste = []
    for pitches in unique:
        index_liste.append(np.where(labels_string_train==pitches))

    len_test = len(test_x)
    len_train = len(train_x)
    # print("Test set size: ", len_test, " train set size: ", len_train)
    # print("Shapes of train_x", train_x.shape, "shape of test_x", test_x.shape)

    model = Model()

    # RNN tflearn
    # if network = "RNN tflearn, one lstm with n_hidden layers followed by dense":
    #     tr_x = train_x.reshape(len_train, N, nr_joints*nr_coordinates)
    #     te_x = test_x.reshape(len_test, N, nr_joints*nr_coordinates)
    #
    #     out, network = model.RNN_network_tflearn(N, nr_joints*nr_coordinates, nr_classes)
    #     m = DNN(network)
    #
    #     m.fit(tr_x, train_t, validation_set=(te_x, test_t), show_metric=True, batch_size=BATCHSIZE, snapshot_step=100, n_epoch=10)
    #
    #     out_test = m.predict(te_x)
    #     pitches_test = Tools.decode_one_hot(out_test, unique)
    #     print("Accuracy test: ", Tools.accuracy(pitches_test, labels_string_test))
    #     print("Accuracy test by class: ", Tools.accuracy_per_class(pitches_test, labels_string_test))
    #     print("True                   Test                 ", unique)
    #     # print(np.swapaxes(np.append([labels_string_test], [pitches_test], axis=0), 0,1))
    #     for i in range(len(labels_string_test)):
    #         print('{:20}'.format(labels_string_test[i]), '{:20}'.format(pitches_test[i]), ['%.2f        ' % elem for elem in out_test[i]])
    #
    #     return 0

    x = tf.placeholder(tf.float32, (None, N, nr_joints, nr_coordinates), name = "input")

    y = tf.placeholder(tf.float32, (None, nr_classes))

    training = tf.placeholder_with_default(False, None)


    if network == "conv1d (256, 5) - conv1d(128, 3) - dense(nr_classes) - softmax":
        out, logits = model.best_in_cluster_concat53(x, nr_classes, training, rate_dropout, act)
    elif network == "adjustable conv1d":
        out, logits = model.conv1d_with_parameters(x, nr_classes, training, rate_dropout, act, first_conv_filters, first_conv_kernel, second_conv_filter,
        second_conv_kernel, first_hidden_dense, second_hidden_dense)
    elif network == "rnn with lstm_units and lstm_hidden_layers + 1 dense(nr_classes)":
        out, logits = model.RNN(x, nr_classes, n_hidden, nr_layers)
    elif network=="conv1d(256,5,2)-conv1d(256,3)-conv1d(128,3)-conv1d(1,1)-dense(1024)-dense(128),dense(nr_classes)":
        out, logits = model.conv1dnet(x, nr_classes, training, rate_dropout, act)
    elif network=="conv2d(256,5,2)-conv2d(256,3)-conv2d(128,3)-conv2d(1,1)-dense(1024)-dense(128),dense(nr_classes)":
        out, logits = model.conv2dnet(x, nr_classes, training, rate_dropout, act)
    else:
        print("ERROR, WRONG NETWORK INPUT")

    tv = tf.trainable_variables()


    loss_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
    loss_regularization = regularization * tf.reduce_sum([ tf.nn.l2_loss(v) for v in tv ])
    loss_maximum = tf.reduce_mean(tf.reduce_max(tf.nn.relu(y-out), axis = 1))
    loss = loss_entropy + loss_regularization #+  loss_maximum #0.001  loss_entropy +
    # loss = tf.reduce_mean(tf.pow(out-y, 2))
    # loss = tf.reduce_sum(tf.pow(out - y, 2)) + 0.5*regularization_cost
    if optimizer_type=="sgd":
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    elif optimizer_type=="adam":
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    else:
        print("WRONG OPTIMIZER")

    # TENSORBOARD comment all in
    # tf.summary.scalar("loss_entropy", loss_entropy)
    # tf.summary.scalar("loss_regularization", loss_regularization)
    # tf.summary.scalar("loss_maximum", loss_maximum)
    # tf.summary.scalar("loss", loss)
    #
    # merged = tf.summary.merge_all()
    # train_writer = tf.summary.FileWriter("./logs/nn_logs" + '/train', sess.graph)
    #
    # # TRAINING
    # saver = tf.train.Saver(tv)
    # # sess = tf.Session()
    # if RESTORE is None:
    #     sess.run(tf.global_variables_initializer())
    # else:
    #     saver.restore(sess, RESTORE)
    #     print("Restored from file")

    sess.run(tf.global_variables_initializer())

    def balanced_batches(x, y, nr_classes):
        #print("balanced function: ", nr_classes)
        for j in range(batch_nr_in_epoch):
            liste=np.zeros((nr_classes, ex_per_class))
            for i in range(nr_classes):
                # print(j, i, np.random.choice(index_liste[i][0], ex_per_class))
                liste[i] = np.random.choice(index_liste[i][0], ex_per_class, replace=True)
            liste = liste.flatten().astype(int)
            yield j, x[liste], y[liste]

    acc_test  = []
    acc_train = []
    losses = []
    # Run session for EPOCH epochs
    for epoch in range(EPOCHS + 1):
        for i, batch_x, batch_t in balanced_batches(train_x, train_t, nr_classes):
            opt = sess.run(optimizer, {x: batch_x, y: batch_t, training: True})


        loss_test, out_test = sess.run([loss,out], {x: test_x, y: test_t, training: False})
        #print("Loss test", loss_test)
        pitches_test = Tools.decode_one_hot(out_test, unique)
        #print("Accuracy test: ", Tools.accuracy(pitches_test, labels_string_test))
        acc_test.append(Tools.accuracy(pitches_test, labels_string_test))
        losses.append(loss_test)
        #Train Accuracy
        out_train = sess.run(out, {x: train_x, y: train_t, training: False})
        pitches_train = Tools.decode_one_hot(out_train, unique)
        acc_train.append(Tools.accuracy(pitches_train, labels_string_train))
        #print("Balanced test accuracy: ", Tools.balanced_accuracy(pitches_test, labels_string_test))
        # print("Balanced train accuracy: ", Tools.balanced_accuracy(pitches_train, labels_string_train))
        # print(loss_test, acc_train[-1], acc_test[-1])

    # AUSGABE AM ENDE
    losses = np.around(np.array(losses).astype(float), 2)
    acc_test = np.around(acc_test, 2)
    acc_train = np.around(acc_train, 2)
    print("\n\n\n---------------------------------------------------------------------")
    print("\n\nNEW PARAMETERS: ", BATCHSIZE, CUT_OFF_Classes , EPOCHS, act, align, batch_nr_in_epoch, rate_dropout
                         , PATH, learning_rate, len_train, n_hidden, nr_layers, network, nr_classes, nr_joints, optimizer_type,
                         regularization, first_conv_filters, first_conv_kernel, second_conv_filter,
                         second_conv_kernel, first_hidden_dense, second_hidden_dense)
    #Test Accuracy
    print("Losses", losses)
    print("Accuracys test: ", acc_test)
    print("Accuracys train: ", acc_train)
    print("\n\nMAXIMUM ACCURACY TEST: ", max(acc_test))
    print("MAXIMUM ACCURACY TRAIN: ", max(acc_train))

    #print("Accuracy test by class: ", Tools.accuracy_per_class(pitches_test, labels_string_test))
    print("Balanced test accuracy: ", Tools.balanced_accuracy(pitches_test, labels_string_test))
    print("Balanced train accuracy: ", Tools.balanced_accuracy(pitches_train, labels_string_train))
    print("True                Test                 ", unique)
    # print(np.swapaxes(np.append([labels_string_test], [pitches_test], axis=0), 0,1))
    for i in range(20):
        print('{:20}'.format(labels_string_test[i]), '{:20}'.format(pitches_test[i]), ['%.2f        ' % elem for elem in out_test[i]])

    lock.acquire()
    new = pd.read_csv("test_parameters.csv")

    #new.drop(new.columns[[0]], axis=1, inplace=True)
    columns = new.columns.values.tolist()
    #print(columns)
    # print(len(columns))
    # print("0, BATCHSIZE, CUT_OFF_Classes , EPOCHS, act, align, batch_nr_in_epoch, rate_dropout, PATH, max(acc_test), learning_rate, len_train, losses, n_hidden, nr_layers, network, nr_classes,                         nr_joints, acc_train, acc_test, optimizer_type, regularization, first_conv_filters, first_conv_kernel, second_conv_filter, second_conv_kernel, first_hidden_dense, second_hidden_dense")
    # print(len(columns))
    # print("Written to csv: ", BATCHSIZE, CUT_OFF_Classes , EPOCHS, act, align, batch_nr_in_epoch, rate_dropout
    #                      , PATH, max(acc_test), learning_rate, len_train, losses, n_hidden, nr_layers, network, nr_classes,
    #                      nr_joints, acc_train, acc_test)

    add = pd.DataFrame([[0, BATCHSIZE, CUT_OFF_Classes , EPOCHS, act, align, batch_nr_in_epoch, rate_dropout
                         , PATH, max(acc_test), learning_rate, len_train, losses, n_hidden, nr_layers, network, nr_classes,
                         nr_joints, acc_train, acc_test, optimizer_type, regularization, first_conv_filters, first_conv_kernel, second_conv_filter,
                         second_conv_kernel, first_hidden_dense, second_hidden_dense]], columns=columns)
    concat = new.append(add, ignore_index = True)
    concat.drop(concat.columns[[0]], axis=1, inplace=True)
    concat.to_csv("test_parameters.csv")
    lock.release()

# a = ["conv1d(256,5,2)-conv1d(256,3)-conv1d(128,3)-conv1d(1,1)-dense(1024)-dense(128),dense(nr_classes)",
# "conv2d(256,5,2)-conv2d(256,3)-conv2d(128,3)-conv2d(1,1)-dense(1024)-dense(128),dense(nr_classes)"]
# b = [0, 0.3, 0.6]
# c = [tf.nn.relu, tf.nn.softmax]
# optim = ["sgd, adam"]
# regular = [0.0005, 0.001]
# hidden_ff = [0, 128, 256, 1024]
#
# for hi1 in hidden_ff:
#     for hi2 in hidden_ff:
#         for re in regular:
#             for opt in optim:

def start_thread(lock, data, labels):
    batches = [40, 80, 120]
    align = [True, False]
    second_conv_kern = [5, 7, 9, 11, 13]
    learning = [0.0005, 0.001]
    second_h_ff = [0, 256]

    for  ba in batches:
        for co in second_conv_kern:
            for le in learning:
                for ff in second_h_ff:
                    runscript(lock, data, labels, BATCH_SZ = ba, learning_rate = le, second_conv_kernel = co, second_hidden_dense = ff)
# runscript(data_raw, labels, network="adjustable conv1d", act=tf.nn.relu, rate_dropout=0)

lock = threading.Lock()
t1 = threading.Thread(target = start_thread, name = " 1", args=(lock, data_raw, labels))
t1.start()
