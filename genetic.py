import pandas as pd
import numpy as np
import tensorflow as tf
import scipy as sp
import scipy.stats

#from sklearn.preprocessing import StandardScaler
from data_preprocess import Preprocessor
from tools import Tools
from model import Model
from random import random, randint
import time

import shutil

PATH = "unpro"
CUT_OFF_Classes = 50
leaky_relu = lambda x: tf.maximum(0.2*x, x)

# PREPROCESS DATA
if PATH is "cf" or PATH is "concat":
    prepro = Preprocessor("cf_data.csv")
elif PATH is "unpro":
    prepro = Preprocessor("unprocessed_data.csv")
    print("unpro eingelesen")
else:
    prepro = Preprocessor("sv_data.csv")

# ONE PLAYER
#players, _ = prepro.get_list_with_most("Pitcher")
#prepro.select_movement("Windup")

# prepro.cut_file_to_pitcher(players[player])  # change_restore


prepro.remove_small_classes(CUT_OFF_Classes)

if PATH is not "concat":
    # data_raw = prepro.get_coord_arr(PATH+"_all_coord.npy")
    data_raw = np.load("interpolated.npy")
    print("data loaded")
else:
    data_raw = prepro.concat_with_second("sv_data.csv", PATH+"_all_coord.npy")

data_without_head = data_raw[:, :, :12, :]
data_new = Tools.normalize(data_without_head)

labels = prepro.get_labels()


def runscript(data_in, labels_string, BATCH_SZ=40, EPOCHS = 3, batch_nr_in_epoch = 100, align = False, act = tf.nn.relu, rate_dropout = 0,
        learning_rate = 0.0005, nr_layers = 4, n_hidden = 128, optimizer_type="adam", regularization=0, first_conv_filters=128, first_conv_kernel=9, second_conv_filter=128,
        second_conv_kernel=5, first_hidden_dense=128, second_hidden_dense=0, network = "adjustable conv1d"):

    #tic = time.time()

    tf.reset_default_graph()
    sess = tf.InteractiveSession()


    if align:
        data = Tools.align_frames(data_in, prepro.get_release_frame(60, 120), 60, 40)
    else:
        data= data_in

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

    #toc = time.time()
    #print("Time before training", toc-tic)

    model = Model()


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
    # loss_maximum = tf.reduce_mean(tf.reduce_max(tf.nn.relu(y-out), axis = 1))
    loss = loss_entropy + loss_regularization #+  loss_maximum #0.001  loss_entropy +
    # loss = tf.reduce_mean(tf.pow(out-y, 2))
    # loss = tf.reduce_sum(tf.pow(out - y, 2)) + 0.5*regularization_cost
    if optimizer_type=="sgd":
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    elif optimizer_type=="adam":
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    else:
        print("WRONG OPTIMIZER")


    sess.run(tf.global_variables_initializer())

    def balanced_batches(x, y, nr_classes):
        #print("balanced function: ", nr_classes)

        for j in range(batch_nr_in_epoch):
            # tic = time.time()
            liste=np.zeros((nr_classes, ex_per_class))
            for i in range(nr_classes):
                # print(j, i, np.random.choice(index_liste[i][0], ex_per_class))
                liste[i] = np.random.choice(index_liste[i][0], ex_per_class, replace=True)
            liste = liste.flatten().astype(int)
            #toc = time.time()
            #print("time to get batch ", toc-tic)
            yield j, x[liste], y[liste]

    acc_test  = []
    acc_balanced = []
    losses = []
    # Run session for EPOCH epochs
    for epoch in range(EPOCHS + 1):
        # tic2 = time.time()
        for i, batch_x, batch_t in balanced_batches(train_x, train_t, nr_classes):
            opt = sess.run(optimizer, {x: batch_x, y: batch_t, training: True})
        # toc2 = time.time()
        #print("one epoch train", toc2-tic2)

        # tic = time.time()

        loss_test, out_test = sess.run([loss,out], {x: test_x, y: test_t, training: False})
        #print("Loss test", loss_test)
        pitches_test = Tools.decode_one_hot(out_test, unique)
        #print("Accuracy test: ", Tools.accuracy(pitches_test, labels_string_test))
        acc_test.append(Tools.accuracy(pitches_test, labels_string_test))
        losses.append(loss_test)
        acc_balanced.append(Tools.balanced_accuracy(pitches_test, labels_string_test))

        # toc = time.time()
        # print("Time for test accuracy ", toc-tic)
        #Train Accuracy
        # out_train = sess.run(out, {x: train_x, y: train_t, training: False})
        # pitches_train = Tools.decode_one_hot(out_train, unique)
        # acc_train.append(Tools.accuracy(pitches_train, labels_string_train))
        # print(loss_test, acc_test[-1], acc_balanced[-1])

    # AUSGABE AM ENDE
    # losses = np.around(np.array(losses).astype(float), 2)
    # acc_test = np.around(acc_test, 2)
    # acc_train = np.around(acc_train, 2)
    # print("\n\n\n---------------------------------------------------------------------")
    # print("\n\nNEW PARAMETERS: ", BATCHSIZE, CUT_OFF_Classes , EPOCHS, act, align, batch_nr_in_epoch, rate_dropout
    #                      , PATH, learning_rate, len_train, n_hidden, nr_layers, network, nr_classes, nr_joints, optimizer_type, regularization, first_conv_filters, first_conv_kernel, second_conv_filter,
    #                      second_conv_kernel, first_hidden_dense, second_hidden_dense)
    # #Test Accuracy
    # print("Losses", losses)
    # print("Accuracys test: ", acc_test)
    # print("Accuracys train: ", acc_train)
    # print("\n\nMAXIMUM ACCURACY TEST: ", max(acc_test))
    # print("MAXIMUM ACCURACY TRAIN: ", max(acc_train))
    # print("Balanced test accuracy: ", Tools.balanced_accuracy(pitches_test, labels_string_test))
    # print("Balanced train accuracy: ", Tools.balanced_accuracy(pitches_train, labels_string_train))
    #
    # #print("Accuracy test by class: ", Tools.accuracy_per_class(pitches_test, labels_string_test))
    # print("True                Test                 ", unique)
    # # print(np.swapaxes(np.append([labels_string_test], [pitches_test], axis=0), 0,1))
    # for i in range(20):
    #     print('{:20}'.format(labels_string_test[i]), '{:20}'.format(pitches_test[i]), ['%.2f        ' % elem for elem in out_test[i]])


    # new = pd.read_csv("test_parameters.csv")
    #
    # #new.drop(new.columns[[0]], axis=1, inplace=True)
    # columns = new.columns.values.tolist()
    # #print(columns)
    # # print(len(columns))
    # # print("0, BATCHSIZE, CUT_OFF_Classes , EPOCHS, act, align, batch_nr_in_epoch, rate_dropout, PATH, max(acc_test), learning_rate, len_train, losses, n_hidden, nr_layers, network, nr_classes,                         nr_joints, acc_train, acc_test, optimizer_type, regularization, first_conv_filters, first_conv_kernel, second_conv_filter, second_conv_kernel, first_hidden_dense, second_hidden_dense")
    # # print(len(columns))
    # # print("Written to csv: ", BATCHSIZE, CUT_OFF_Classes , EPOCHS, act, align, batch_nr_in_epoch, rate_dropout
    # #                      , PATH, max(acc_test), learning_rate, len_train, losses, n_hidden, nr_layers, network, nr_classes,
    # #                      nr_joints, acc_train, acc_test)
    #
    # add = pd.DataFrame([[0, BATCHSIZE, CUT_OFF_Classes , EPOCHS, act, align, batch_nr_in_epoch, rate_dropout
    #                      , PATH, max(acc_test), learning_rate, len_train, losses, n_hidden, nr_layers, network, nr_classes,
    #                      nr_joints, acc_train, acc_test, optimizer_type, regularization, first_conv_filters, first_conv_kernel, second_conv_filter,
    #                      second_conv_kernel, first_hidden_dense, second_hidden_dense]], columns=columns)
    # concat = new.append(add, ignore_index = True)
    # concat.drop(concat.columns[[0]], axis=1, inplace=True)
    # concat.to_csv("test_parameters.csv")
    print(max(acc_test), max(acc_balanced))
    return max(acc_test), max(acc_balanced)


def individual():
    'Create a member of the population.'
    f = lambda x: np.random.choice(x)
    kernel_sizes = [3,5,7,9,11]
    filter_sizes = [0, 128, 256, 512, 1024]
    rates = [0.0001, 0.00025, 0.0005, 0.001]
    alignd = [True, False]
    dropout = [0, 0.6]
    act = [tf.nn.relu, leaky_relu]
    regularize = [0, 0.0005, 0.001, 0.005]
    return [f(act), f(dropout), f(alignd), f(rates), f(regularize), f(filter_sizes), f(kernel_sizes), f(filter_sizes), f(kernel_sizes), f(filter_sizes), f(filter_sizes)]

def population(count):
    """
    Create a number of individuals (i.e. a population).

    count: the number of individuals in the population
    length: the number of values per individual
    min: the minimum possible value in an individual's list of values
    max: the maximum possible value in an individual's list of values

    """
    return [ individual() for x in range(count) ]

def fitness(ind, target):
    acc, bala = runscript(data_new, labels, EPOCHS = 30, act = ind[0], rate_dropout=ind[1], align= ind[2], learning_rate=ind[3],
    regularization=ind[4], first_conv_filters=ind[5], first_conv_kernel=int(ind[6]), second_conv_filter=ind[7],
    second_conv_kernel=int(ind[8]), first_hidden_dense=ind[9], second_hidden_dense=ind[10], network = "adjustable conv1d") # np.sum(ind[6:10])
    return 2*target- acc - bala

def grade(pop):
    'Find average fitness for a population.'
    return np.array([fitness(x, target) for x in pop])

def evolve(pop, grades, target, retain=0.3, random_select=0.2, mutate=0.1): # 0.2, 0.05, 0.01
    #grades = np.array([fitness(x, target) for x in pop])
    print("\n unsorted")
    for g in grades:
        print(g)
    graded = [ (grades[i], pop[i]) for i in range(len(pop))]
    #print("\n fitness", graded)
    #grade_mean = np.mean(grades)
    sort = sorted(graded, key=lambda x: x[0])
    print("\n sorted")
    for i in sort:
        print(i)
    graded = [ x[1] for x in sorted(graded, key=lambda x: x[0])]
    retain_length = int(len(graded)*retain)
    parents = graded[:retain_length]
    # randomly add other individuals to
    # promote genetic diversity
    for ind in graded[retain_length:]:
        if random_select > random():
            parents.append(ind)
    # mutate some individuals
    for indi in parents:
        if mutate > random():
            pos_to_mutate = randint(0, len(indi)-1)
            lender = individual()
            # this mutation is not ideal, because it
            # restricts the range of possible values,
            # but the function is unaware of the min/max
            # values used to create the individuals,
            indi[pos_to_mutate] = lender[pos_to_mutate]
    # crossover parents to create children
    parents_length = len(parents)
    desired_length = len(pop) - parents_length
    children = []
    while len(children) < desired_length:
        male = randint(0, parents_length-1)
        female = randint(0, parents_length-1)
        if male != female:
            child = parents[male]
            #female = parents[female]
            half = int(len(parents[male]) / 2)
            indize = np.random.permutation(len(parents[male]))[:half]
            for i in indize:
                child[int(i)] = parents[female][int(i)]
            children.append(child)
    parents.extend(children)
    return parents



# from genetic import *
target = 1.0
p_count = 30
# i_length = 6

p = population(p_count)
for i in range(p_count):
    print(p[i])
grades = grade(p)
fitness_history = [np.mean(grades)]

for i in range(20):
    p = evolve(p, grades, target)
    print("\n new population \n ")
    for i in range(p_count):
        print(p[i])
    grades = grade(p)
    print("\n mean of grades", np.mean(grades), "Grades", grades)
    fitness_history.append(np.mean(grades))

print(fitness_history)
#runscript(data_raw, labels, network="adjustable conv1d", act=tf.nn.relu, rate_dropout=0)
