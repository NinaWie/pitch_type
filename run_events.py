import pandas as pd
import numpy as np
import tensorflow as tf
import scipy as sp
import scipy.stats
import threading


#from sklearn.preprocessing import StandardScaler
#from self.data_preprocess import Preprocessor
from tools import Tools
from model import Model

def test(data, restore_file):
    """
    Runs model of restore_file for the data
    data must be normalized before and aligned if desired
    returns labels
    """
    tf.reset_default_graph()
    #sess = tf.InteractiveSession()

    saver = tf.train.import_meta_graph(restore_file+'.meta')
    graph = tf.get_default_graph()

    sess = tf.Session()
    saver.restore(sess, restore_file)
    out = tf.get_collection("out")[0]
    unique = tf.get_collection("unique")[0]
    out_test = sess.run(out, {"input:0":  data, "training:0": False})

    # Evaluation
    pitches_test = Tools.decode_one_hot(out_test,  unique.eval(session = sess))
    try:
        pitches = [elem.decode("utf-8") for elem in pitches_test]
    except AttributeError:
        pitches = pitches_test
    return pitches, out_test

class Runner(threading.Thread):

    def __init__(self, data, labels_string, SAVE = None, BATCH_SZ=40, EPOCHS = 60, batch_nr_in_epoch = 100,
            act = tf.nn.relu, rate_dropout = 0,
            learning_rate = 0.0005, nr_layers = 4, n_hidden = 128, optimizer_type="adam", regularization=0,
            first_conv_filters=128, first_conv_kernel=9, second_conv_filter=128,
            second_conv_kernel=5, first_hidden_dense=128, second_hidden_dense=0,
            network = "adjustable conv1d"):
        threading.Thread.__init__(self)
        self.data = data
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

        labels = Tools.onehot_with_unique(self.labels_string, self.unique)
        ex_per_class = self.BATCH_SZ//nr_classes
        BATCHSIZE = nr_classes*ex_per_class

        self.data = Tools.normalize01(self.data)
        train_x = self.data[train_ind]
        test_x = self.data[test_ind]

        # print(train_x[:10])
        # print(np.any(np.isnull(train_x)))
        # print(labels[:20])

        # train_x = self.data[train_ind]
        # me = np.mean(train_x)
        # std = np.std(train_x)
        # test_x = self.data[test_ind]
        #
        # train_x = (train_x - me)/std
        # test_x = (test_x -me)/std
        #
        # print("mean of train", np.mean(train_x))
        # print("mean of test", np.mean(test_x))

        train_t= labels[train_ind]
        test_t = labels[test_ind]
        labels_string_train = self.labels_string[train_ind]
        labels_string_test = self.labels_string[test_ind]

        print(train_x.shape, train_t.shape, labels_string_train.shape, test_x.shape, test_t.shape, labels_string_test.shape)
        #train_x, labels_string_train = Tools.balance(train_x, labels_string_train)
        index_liste = []
        for pitches in self.unique:
            index_liste.append(np.where(labels_string_train==pitches)[0])
        #print(index_liste)
        len_test = len(test_x)
        len_train = len(train_x)

        x = tf.placeholder(tf.float32, (None, N, nr_joints, nr_coordinates), name = "input")
        y = tf.placeholder(tf.float32, (None, nr_classes))
        training = tf.placeholder_with_default(False, None, name = "training")


        if self.network == "conv1d_big":
            out, logits = model.conv1d_big(x, nr_classes, training, self.rate_dropout, self.act)
        elif self.network == "adjustable conv1d":
            out, logits = model.conv1d_with_parameters(x, nr_classes, training, self.rate_dropout, self.act, self.first_conv_filters, self.first_conv_kernel, self.second_conv_filter,
            self.second_conv_kernel, self.first_hidden_dense, self.second_hidden_dense)
        elif self.network == "rnn":
            out, logits = model.RNN(x, nr_classes, self.n_hidden, self.nr_layers)
        elif self.network=="adjustable conv2d":
            out, logits = model.conv2d(x, nr_classes, training, self.rate_dropout, self.act, self.first_conv_filters, self.first_conv_kernel, self.second_conv_filter,
            self.second_conv_kernel, self.first_hidden_dense, self.second_hidden_dense)
        elif self.network == "conv 1st move":
            out, logits = model.conv1stmove(x, nr_classes, training, self.rate_dropout, self.act, self.first_conv_filters, self.first_conv_kernel, self.second_conv_filter,
            self.second_conv_kernel, self.first_hidden_dense, self.second_hidden_dense)
        elif self.network == "combined":
            out_normal, logits = model.conv1stmove(x, nr_classes, training, self.rate_dropout, self.act, self.first_conv_filters, self.first_conv_kernel, self.second_conv_filter,
                                                    self.second_conv_kernel, self.first_hidden_dense, self.second_hidden_dense)
            out_normal = tf.reshape(out_normal, (-1, self.unique[0], 1, 1))
            wrist_ellbow_right = tf.reduce_mean(x[:, :, 1:3, 1], 2) # y coordinate of ellbow and wrist
            print(wrist_ellbow_right)
            wrist_ellbow_left = tf.reduce_mean(x[:, :, 4:6, 1], 2)
            print(wrist_ellbow_left)
            shoulder_left = tf.reshape(x[:, :, 0, 1], (-1,  self.unique[0], 1))
            shoulder_right = tf.reshape(x[:, :, 3, 1], (-1,  self.unique[0], 1))
            print(shoulder_right)
            shoulder_both = tf.concat([shoulder_left, shoulder_right],2)
            print(shoulder_both)
            shoulders = tf.reduce_mean(shoulder_both, 2)  # y coordinate of shoulders
            print(shoulders)
            new_x = tf.reshape(tf.concat([tf.reshape(wrist_ellbow_right-shoulders, (-1, self.unique[0], 1)), tf.reshape(wrist_ellbow_left-shoulders, (-1, self.unique[0], 1))], 2), (-1, self.unique[0], 2, 1))
            print(new_x)
            #out_wrist_ellbow, logits = model.conv1stmove(x, nr_classes, training, self.rate_dropout, self.act, self.first_conv_filters, self.first_conv_kernel, self.second_conv_filter,
            #                                            self.second_conv_kernel, self.first_hidden_dense, self.second_hidden_dense)
            combined_x = tf.concat([out_normal, new_x], 2)
            out, logits = model.conv1d_with_parameters(combined_x, nr_classes, training, self.rate_dropout, self.act, self.first_conv_filters, self.first_conv_kernel, self.second_conv_filter,
                                                        self.second_conv_kernel, self.first_hidden_dense, self.second_hidden_dense)
        elif self.network == "sv+cf":
            out_cf, logits_cf = model.conv1stmove(x[:,:,:,:2], nr_classes, training, self.rate_dropout, self.act, self.first_conv_filters, self.first_conv_kernel, self.second_conv_filter,
                                                    self.second_conv_kernel, self.first_hidden_dense, self.second_hidden_dense)

            out_sv, logits_sv = model.conv1stmove(x[:,:,:,2:], nr_classes, training, self.rate_dropout, self.act, self.first_conv_filters, self.first_conv_kernel, self.second_conv_filter,
                                                    self.second_conv_kernel, self.first_hidden_dense, self.second_hidden_dense)
            combined_x = tf.reshape(tf.concat([out_cf, out_sv], 2), (-1, N, 2,1))
            out, logits = model.conv1d_with_parameters(combined_x, nr_classes, training, self.rate_dropout, self.act, self.first_conv_filters, self.first_conv_kernel, self.second_conv_filter,
                                                        self.second_conv_kernel, self.first_hidden_dense, self.second_hidden_dense)
        elif self.network == "conv+rnn":
            first_out, _ =  model.conv1stmove(x, nr_classes, training, self.rate_dropout, self.act, self.first_conv_filters, self.first_conv_kernel, self.second_conv_filter,
                                                    self.second_conv_kernel, self.first_hidden_dense, self.second_hidden_dense, out_filters=128)
            print(first_out)
            out, logits = model.RNN(first_out, nr_classes, self.n_hidden, self.nr_layers)
        else:
            print("ERROR, WRONG", self.network, "INPUT")

        tv = tf.trainable_variables()

        out = tf.identity(out, "out")
        uni = tf.constant(self.unique, name = "uni")

        if len(self.unique)==1:
            out = tf.sigmoid(logits)
            loss = tf.reduce_mean(tf.square(y - out))
        else:
            loss_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
            loss_regularization = self.regularization * tf.reduce_sum([ tf.nn.l2_loss(v) for v in tv ])
            loss = loss_entropy + loss_regularization #+  loss_maximum #0.001  loss_entropy +



        # max_out = tf.argmax(out, axis = 1)
        # max_lab = tf.argmax(y, axis = 1)
        # diff = tf.cast(max_out-max_lab, tf.float32)
        # loss = tf.reduce_mean(tf.square(diff))

        optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

        # TENSORBOARD comment all in
        # tf.summary.scalar("loss_entropy", loss_entropy)
        # tf.summary.scalar("loss_regularization", loss_regularization)
        # tf.summary.scalar("loss_maximum", loss_maximum)
        # tf.summary.scalar("loss", loss)
        #
        # merged = tf.summary.merge_all()
        # train_writer = tf.summary.FileWriter("./logs/nn_logs" + '/train', sess.graph)

        saver = tf.train.Saver(tf.global_variables())

        #tf.summary.scalar("loss_entropy", loss_entropy)
        #tf.summary.scalar("loss_regularization", loss_regularization)
        # tf.summary.scalar("loss_maximum", loss_maximum)
        tf.summary.scalar("loss", loss)

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter("./logs/nn_logs" + '/train', sess.graph)

        # TRAINING

        sess.run(tf.global_variables_initializer())

        def batches(x, y, nr_classes, batchsize=40):
            permute = np.random.permutation(len(x))
            for i in range(0, len(x)-batchsize, batchsize):
                indices = permute[i:i+batchsize]
                yield i, x[indices], y[indices]

        acc_test  = []
        acc_train  = []
        acc_balanced = []
        losses = []
        print("Loss", "Acc test", "Acc balanced")
        # Run session for self.EPOCHS
        for epoch in range(self.EPOCHS + 1):
            for i, batch_x, batch_t in batches(train_x, train_t, nr_classes):
                summary, _ = sess.run([merged, optimizer], {x: batch_x, y: batch_t, training: True})
                train_writer.add_summary(summary, i+self.batch_nr_in_epoch*epoch)

            tf.add_to_collection("out", out)
            tf.add_to_collection("unique", uni)

            loss_test, out_test = sess.run([loss,out], {x: test_x, y: test_t, training: False})
            pitches_test = Tools.decode_one_hot(out_test, self.unique)
            acc_test.append(np.around(Tools.accuracy(pitches_test, labels_string_test), 2))
            losses.append(np.around(loss_test, 2))
            acc_balanced.append(np.around(Tools.balanced_accuracy(pitches_test, labels_string_test),2))
            #Train Accuracy
            out_train = sess.run(out, {x: train_x, y: train_t, training: False})
            pitches_train = Tools.decode_one_hot(out_train, self.unique)
            acc_train.append(np.around(Tools.accuracy(pitches_train, labels_string_train), 2))
            print(loss_test, acc_test[-1], acc_balanced[-1])
            if acc_train!=[]:
                print("acc_train: ", acc_train[-1])
            # if acc_test[-1]>=0.8 and acc_balanced[-1]>=0.8 and self.SAVE is not None:
            #     saver.save(sess, self.SAVE)
            #     print("model saved with name", self.SAVE)
            #     return pitches_test, max(acc_test)
        # AUSGABE AM ENDE
        print("\n\n\n---------------------------------------------------------------------")
        #print("NEW PARAMETERS: ", BATCHSIZE, self.EPOCHS, self.act, self.align, self.batch_nr_in_epoch, self.rate_dropout, self.learning_rate, len_train, self.n_hidden, self.nr_layers, self.network, nr_classes, nr_joints)
        #Test Accuracy
        #print("Losses", losses)
        #print("Accuracys test: ", acc_test)
        #print("Accuracys train: ", acc_train)
        print("\nMAXIMUM ACCURACY TEST: ", max(acc_test))
        #print("MAXIMUM ACCURACY TRAIN: ", max(acc_train))

        #print("Accuracy test by class: ", Tools.accuracy_per_class(pitches_test, labels_string_test))
        print("True                Test                 ", self.unique)
        if len(self.unique)==1:
            for i in range(10): #len(labels_string_test)):
                print(labels_string_test[i], pitches_test[i])
        else:
        # print(np.swapaxes(np.append([labels_string_test], [pitches_test], axis=0), 0,1))
            for i in range(len(labels_string_test)):
                print('{:20}'.format(labels_string_test[i]), '{:20}'.format(pitches_test[i])) #, ['%.2f        ' % elem for elem in out_test[i]])

        if self.SAVE!=None:
            saver.save(sess, self.SAVE)

        pitches = np.append(pitches_test, pitches_train, axis = 0)
        labs = np.append(labels_string_test, labels_string_train, axis = 0)
        print("ACCURACY IN RANGE 2", Tools.accuracy_in_range(pitches.flatten(), labs.flatten(), 2))
        return test_ind, pitches_test, labels_string_test[i]

#runner = Runner()
#pitches, accuracies = runner.runscript(self.data_raw[20:30], labels[20:30], np.self.unique(labels).tolist(), self.RESTORE="./model")
