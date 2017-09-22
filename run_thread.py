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
    sess = tf.InteractiveSession()

    saver = tf.train.import_meta_graph(restore_file+'.meta')
    graph = tf.get_default_graph()

    saver.restore(sess, restore_file)
    out = tf.get_collection("out")[0]
    unique = tf.get_collection("unique")[0]
    out_test = sess.run(out, {"input:0":  data, "training:0": False})

    # Evaluation
    pitches_test = Tools.decode_one_hot(out_test,  unique.eval())
    pitches = [elem.decode("utf-8") for elem in pitches_test]

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
        self.labels_string = labels_string
        self.unique  = np.unique(labels_string)
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
            unique = unique.tolist()
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

        # RNN tflearn
        if self.network=="tflearn":
            import tflearn
            from tflearn import DNN
            train_x = self.data[train_ind]
            testing = self.data[test_ind]
            test_x = testing[:-40]
            val_x = testing[-40:]
            labels_string_train = self.labels_string[train_ind]
            labels_string_testing = self.labels_string[test_ind]
            labels_string_test = labels_string_testing[:-40]
            labels_string_val = labels_string_testing[-40:]
            train_x, labels_string_train = Tools.balance(train_x, labels_string_train)
            train_t = Tools.onehot_with_unique(labels_string_train, self.unique.tolist())
            test_t = Tools.onehot_with_unique(labels_string_test, self.unique.tolist())
            print("Train", train_x.shape, train_t.shape, labels_string_train.shape, "Test", test_x.shape, test_t.shape, labels_string_test.shape, "Val", val_x.shape, labels_string_val.shape)
            len_train, N, nr_joints, nr_coordinates = train_x.shape
            tr_x = train_x.reshape(len_train, N, nr_joints*nr_coordinates)
            te_x = test_x.reshape(len(test_x), N, nr_joints*nr_coordinates)
            val = val_x.reshape(len(val_x), N, nr_joints*nr_coordinates)

            out, self.network = model.RNN_network_tflearn(N, nr_joints*nr_coordinates, nr_classes,self.nr_layers, self.n_hidden, self.rate_dropout)
            m = DNN(self.network)
            m.fit(tr_x, train_t, validation_set=(te_x, test_t), show_metric=True, batch_size=self.BATCH_SZ, snapshot_step=1000, n_epoch=self.EPOCHS)
            labels_string_test = labels_string_test[:40]
            out_test = m.predict(val)
            pitches_test = Tools.decode_one_hot(out_test, self.unique)
            print("Accuracy test: ", Tools.accuracy(pitches_test, labels_string_val))
            print("Accuracy test by class: ", Tools.accuracy_per_class(pitches_test, labels_string_val))
            print("True                   Test                 ", self.unique)
            # print(np.swapaxes(np.append([labels_string_test], [pitches_test], axis=0), 0,1))
            for i in range(len(labels_string_test)):
                print('{:20}'.format(labels_string_val[i]), '{:20}'.format(pitches_test[i]), ['%.2f        ' % elem for elem in out_test[i]])

            return pitches_test, out_test


        labels, _ = Tools.onehot_encoding(self.labels_string)
        ex_per_class = self.BATCH_SZ//nr_classes
        BATCHSIZE = nr_classes*ex_per_class

        train_x = self.data[train_ind]
        test_x = self.data[test_ind]
        train_t= labels[train_ind]
        test_t = labels[test_ind]
        labels_string_train = self.labels_string[train_ind]
        labels_string_test = self.labels_string[test_ind]

        print(train_x.shape, train_t.shape, labels_string_train.shape, test_x.shape, test_t.shape, labels_string_test.shape)
        #train_x, labels_string_train = Tools.balance(train_x, labels_string_train)
        index_liste = []
        for pitches in self.unique:
            index_liste.append(np.where(labels_string_train==pitches))
        len_test = len(test_x)
        len_train = len(train_x)

        x = tf.placeholder(tf.float32, (None, N, nr_joints, nr_coordinates), name = "input")
        y = tf.placeholder(tf.float32, (None, nr_classes))
        training = tf.placeholder_with_default(False, None, name = "training")


        if self.network == "conv1d (256, 5) - conv1d(128, 3) - dense(nr_classes) - softmax":
            out, logits = model.best_in_cluster_concat53(x, nr_classes, training, self.rate_dropout, self.act)
        elif self.network == "adjustable conv1d":
            out, logits = model.conv1d_with_parameters(x, nr_classes, training, self.rate_dropout, self.act, self.first_conv_filters, self.first_conv_kernel, self.second_conv_filter,
            self.second_conv_kernel, self.first_hidden_dense, self.second_hidden_dense)
        elif self.network == "rnn":
            out, logits = model.RNN(x, nr_classes, self.n_hidden, self.nr_layers)
        elif self.network=="conv1d(256,5,2)-conv1d(256,3)-conv1d(128,3)-conv1d(1,1)-dense(1024)-dense(128),dense(nr_classes)":
            out, logits = model.conv1dnet(x, nr_classes, training, self.rate_dropout, self.act)
        elif self.network=="conv2d(256,5,2)-conv2d(256,3)-conv2d(128,3)-conv2d(1,1)-dense(1024)-dense(128),dense(nr_classes)":
            out, logits = model.conv2dnet(x, nr_classes, training, self.rate_dropout, self.act)
        else:
            print("ERROR, WRONG self.network INPUT")

        tv = tf.trainable_variables()

        out = tf.identity(out, "out")
        uni = tf.constant(self.unique, name = "uni")

        loss_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
        loss_regularization = self.regularization * tf.reduce_sum([ tf.nn.l2_loss(v) for v in tv ])
        loss_maximum = tf.reduce_mean(tf.reduce_max(tf.nn.relu(y-out), axis = 1))
        loss = loss_entropy + loss_regularization #+  loss_maximum #0.001  loss_entropy +

        optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

        # TENSORBOARD comment all in
        # tf.summary.scalar("loss_entropy", loss_entropy)
        # tf.summary.scalar("loss_regularization", loss_regularization)
        # tf.summary.scalar("loss_maximum", loss_maximum)
        # tf.summary.scalar("loss", loss)
        #
        # merged = tf.summary.merge_all()
        # train_writer = tf.summary.FileWriter("./logs/nn_logs" + '/train', sess.graph)

        saver = tf.train.Saver(tv)

        tf.summary.scalar("loss_entropy", loss_entropy)
        tf.summary.scalar("loss_regularization", loss_regularization)
        tf.summary.scalar("loss_maximum", loss_maximum)
        tf.summary.scalar("loss", loss)

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter("./logs/nn_logs" + '/train', sess.graph)

        # TRAINING

        sess.run(tf.global_variables_initializer())

        def balanced_batches(x, y, nr_classes):
            #print("balanced function: ", nr_classes)
            for j in range(self.batch_nr_in_epoch):
                liste=np.zeros((nr_classes, ex_per_class))
                for i in range(nr_classes):
                    # print(j, i, np.random.choice(index_liste[i][0], ex_per_class))
                    liste[i] = np.random.choice(index_liste[i][0], ex_per_class, replace=True)
                liste = liste.flatten().astype(int)
                yield j, x[liste], y[liste]

        acc_test  = []
        acc_train  = []
        acc_balanced = []
        losses = []
        print("Loss", "Acc test", "Acc balanced")
        # Run session for self.EPOCHS
        for epoch in range(self.EPOCHS + 1):
            for i, batch_x, batch_t in balanced_batches(train_x, train_t, nr_classes):
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
        print("Losses", losses)
        print("Accuracys test: ", acc_test)
        #print("Accuracys train: ", acc_train)
        print("\nMAXIMUM ACCURACY TEST: ", max(acc_test))
        #print("MAXIMUM ACCURACY TRAIN: ", max(acc_train))

        print("Accuracy test by class: ", Tools.accuracy_per_class(pitches_test, labels_string_test))
        print("True                Test                 ", self.unique)
        # print(np.swapaxes(np.append([labels_string_test], [pitches_test], axis=0), 0,1))
        for i in range(len(labels_string_test)):
            print('{:20}'.format(labels_string_test[i]), '{:20}'.format(pitches_test[i]), ['%.2f        ' % elem for elem in out_test[i]])

        if self.SAVE!=None:
            saver.save(sess, self.SAVE)
        return pitches_test, max(acc_test)

#runner = Runner()
#pitches, accuracies = runner.runscript(self.data_raw[20:30], labels[20:30], np.self.unique(labels).tolist(), self.RESTORE="./model")
