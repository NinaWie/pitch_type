import pandas as pd
import numpy as np
import tensorflow as tf
import scipy as sp
import scipy.stats



#from sklearn.preprocessing import StandardScaler
#from data_preprocess import Preprocessor
from tools import Tools
from model import Model


class Runner:

    def run(self, data, labels_string, unique, normalize = False, RESTORE = None, BATCH_SZ=40, EPOCHS = 60, batch_nr_in_epoch = 100, align = False,
            act = tf.nn.relu, rate_dropout = 0,
            learning_rate = 0.0005, nr_layers = 4, n_hidden = 128, optimizer_type="adam", regularization=0,
            first_conv_filters=128, first_conv_kernel=9, second_conv_filter=128,
            second_conv_kernel=5, first_hidden_dense=128, second_hidden_dense=0,
            network = "adjustable conv1d"):

        try:
            shutil.rmtree("/Users/ninawiedemann/Desktop/UNI/Praktikum/logs")
            print("logs removed")
        except:
            print("logs could not be removed")

        tf.reset_default_graph()
        sess = tf.InteractiveSession()


        if align:
            data = Tools.align_frames(data, prepro.get_release_frame(60, 120), 60, 40)

        if normalize:
            data = Tools.normalize(data)

        M,N,nr_joints,nr_coordinates = data.shape

        SEP = int(M*0.9)

        nr_classes = len(unique)
        print("classes", unique)

        model = Model()

        if RESTORE is not None:
            labels = Tools.onehot_with_unique(labels_string, unique)
            print(labels.shape)
            print("Restore block")
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

            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

            tv = tf.trainable_variables()
            saver = tf.train.Saver(tv)
            saver.restore(sess, RESTORE)
            loss_test, out_test = sess.run([loss,out], {x: data, y: labels, training: False})
            pitches_test = Tools.decode_one_hot(out_test, unique)
            acc = Tools.accuracy(pitches_test, labels_string)
            print("True                Test                 ", unique)
            # print(np.swapaxes(np.append([labels_string_test], [pitches_test], axis=0), 0,1))
            for i in range(len(pitches_test)):
                print('{:20}'.format(labels_string[i]), '{:20}'.format(pitches_test[i]), ['%.2f        ' % elem for elem in out_test[i]])
            return pitches_test, acc
        # NET
        else:
            labels, _ = Tools.onehot_encoding(labels_string)
            ex_per_class = 40//nr_classes
            BATCHSIZE = nr_classes*ex_per_class
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


            # RNN tflearn
            if network=="tflearn":
                import tflearn
                from tflearn import DNN
                tr_x = train_x.reshape(len_train, N, nr_joints*nr_coordinates)
                te_x = test_x.reshape(len_test, N, nr_joints*nr_coordinates)

                out, network = model.RNN_network_tflearn(N, nr_joints*nr_coordinates, nr_classes, n_hidden)
                m = DNN(network)

                m.fit(tr_x, train_t, validation_set=(te_x, test_t), show_metric=True, batch_size=BATCHSIZE, snapshot_step=100, n_epoch=EPOCHS)

                out_test = m.predict(te_x)
                pitches_test = Tools.decode_one_hot(out_test, unique)
                print("Accuracy test: ", Tools.accuracy(pitches_test, labels_string_test))
                print("Accuracy test by class: ", Tools.accuracy_per_class(pitches_test, labels_string_test))
                print("True                   Test                 ", unique)
                # print(np.swapaxes(np.append([labels_string_test], [pitches_test], axis=0), 0,1))
                for i in range(len(labels_string_test)):
                    print('{:20}'.format(labels_string_test[i]), '{:20}'.format(pitches_test[i]), ['%.2f        ' % elem for elem in out_test[i]])

                return pitches_test, Tools.accuracy(pitches_test, labels_string_test)

            x = tf.placeholder(tf.float32, (None, N, nr_joints, nr_coordinates), name = "input")

            y = tf.placeholder(tf.float32, (None, nr_classes))

            training = tf.placeholder_with_default(False, None)


            if network == "conv1d (256, 5) - conv1d(128, 3) - dense(nr_classes) - softmax":
                out, logits = model.best_in_cluster_concat53(x, nr_classes, training, rate_dropout, act)
            elif network == "adjustable conv1d":
                out, logits = model.conv1d_with_parameters(x, nr_classes, training, rate_dropout, act, first_conv_filters, first_conv_kernel, second_conv_filter,
                second_conv_kernel, first_hidden_dense, second_hidden_dense)
            elif network == "rnn":
                out, logits = model.RNN(x, nr_classes, n_hidden, nr_layers)
            elif network=="conv1d(256,5,2)-conv1d(256,3)-conv1d(128,3)-conv1d(1,1)-dense(1024)-dense(128),dense(nr_classes)":
                out, logits = model.conv1dnet(x, nr_classes, training, rate_dropout, act)
            elif network=="conv2d(256,5,2)-conv2d(256,3)-conv2d(128,3)-conv2d(1,1)-dense(1024)-dense(128),dense(nr_classes)":
                out, logits = model.conv2dnet(x, nr_classes, training, rate_dropout, act)
            else:
                print("ERROR, WRONG NETWORK INPUT")
            #

            tv = tf.trainable_variables()


            loss_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
            loss_regularization = 0.0005 * tf.reduce_sum([ tf.nn.l2_loss(v) for v in tv ])
            loss_maximum = tf.reduce_mean(tf.reduce_max(tf.nn.relu(y-out), axis = 1))
            loss = loss_entropy # loss_regularization+  loss_maximum #0.001  loss_entropy +
            # loss = tf.reduce_mean(tf.pow(out-y, 2))
            # loss = tf.reduce_sum(tf.pow(out - y, 2)) + 0.5*regularization_cost

            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

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

            tf.summary.scalar("loss_entropy", loss_entropy)
            tf.summary.scalar("loss_regularization", loss_regularization)
            tf.summary.scalar("loss_maximum", loss_maximum)
            tf.summary.scalar("loss", loss)

            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter("./logs/nn_logs" + '/train', sess.graph)

            # TRAINING

            # sess = tf.Session()
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
            acc_balanced = []
            losses = []
            # Run session for EPOCH epochs
            for epoch in range(EPOCHS + 1):
                for i, batch_x, batch_t in balanced_batches(train_x, train_t, nr_classes):
                    summary, _ = sess.run([merged, optimizer], {x: batch_x, y: batch_t, training: True})
                    train_writer.add_summary(summary, i+batch_nr_in_epoch*epoch)

                loss_test, out_test = sess.run([loss,out], {x: test_x, y: test_t, training: False})
                pitches_test = Tools.decode_one_hot(out_test, unique)
                acc_test.append(np.around(Tools.accuracy(pitches_test, labels_string_test), 2))
                losses.append(np.around(loss_test, 2))
                acc_balanced.append(np.around(Tools.balanced_accuracy(pitches_test, labels_string_test),2))
                #Train Accuracy
                # out_train = sess.run(out, {x: train_x, y: train_t, training: False})
                # pitches_train = Tools.decode_one_hot(out_train, unique)
                # acc_train.append(Tools.accuracy(pitches_train, labels_string_train))
                print(loss_test, acc_test[-1], acc_balanced[-1])

            # AUSGABE AM ENDE
            print("\n\n\n---------------------------------------------------------------------")
            print("NEW PARAMETERS: ", BATCHSIZE, CUT_OFF_Classes , EPOCHS, act, align, batch_nr_in_epoch, rate_dropout
                                 , PATH, learning_rate, len_train, n_hidden, nr_layers, network, nr_classes, nr_joints)
            #Test Accuracy
            print("Losses", losses)
            print("Accuracys test: ", acc_test)
            #print("Accuracys train: ", acc_train)
            print("\nMAXIMUM ACCURACY TEST: ", max(acc_test))
            #print("MAXIMUM ACCURACY TRAIN: ", max(acc_train))

            #print("Accuracy test by class: ", Tools.accuracy_per_class(pitches_test, labels_string_test))
            print("True                Test                 ", unique)
            # print(np.swapaxes(np.append([labels_string_test], [pitches_test], axis=0), 0,1))
            for i in range(20):
                print('{:20}'.format(labels_string_test[i]), '{:20}'.format(pitches_test[i]), ['%.2f        ' % elem for elem in out_test[i]])


            # new = pd.read_csv("test_parameters.csv")
            # columns = new.columns.values.tolist()
            # # print(len(columns))
            # # print("Written to csv: ", BATCHSIZE, CUT_OFF_Classes , EPOCHS, act, align, batch_nr_in_epoch, rate_dropout
            # #                      , PATH, max(acc_test), learning_rate, len_train, losses, n_hidden, nr_layers, network, nr_classes,
            # #                      nr_joints, acc_train, acc_test)
            #
            # add = pd.DataFrame([[0, BATCHSIZE, CUT_OFF_Classes , EPOCHS, act, align, batch_nr_in_epoch, rate_dropout
            #                      , PATH, max(acc_test), learning_rate, len_train, losses, n_hidden, nr_layers, network, nr_classes,
            #                      nr_joints, acc_train, acc_test]], columns=columns)
            # concat = new.append(add, ignore_index = True)
            # concat.drop(concat.columns[[0]], axis=1, inplace=True)
            # concat.to_csv("test_parameters.csv")
            saver.save(sess, "model")
            return pitches_test, max(acc_test)

# a = ["conv1d (256, 5) - conv1d(128, 3) - dense(nr_classes) - softmax", "rnn with lstm_units and lstm_hidden_layers + 1 dense(nr_classes)"]
# b = [0, 0.3, 0.6]
# c = [tf.nn.relu, leaky_relu]
#
# for al in a:
#     for le in b:
#         for ac in c:
#runscript(data_raw, labels, network="RNN_network_tflearn")

#runner = Runner()
#pitches, accuracies = runner.runscript(data_raw[20:30], labels[20:30], np.unique(labels).tolist(), RESTORE="./model")
