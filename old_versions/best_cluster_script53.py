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
try:
    shutil.rmtree("logs")
    print("logs removed")
except:
    print("logs could not be removed")

tf.reset_default_graph()
sess = tf.InteractiveSession()

# META PARAMTETERS
RESTORE = None #"./model"  # change_restore
leaky_relu = lambda x: tf.maximum(0.2*x, x)
ex_per_class = 4
EPOCHS = 60
batch_nr_in_epoch = 100
PATH = "cf_data.csv"
SECOND = "sv_data.csv"
LABELS = "Pitch Type"
act = tf.nn.relu
CUT_OFF_Classes = 50
rate_dropout = 0.6
learning_rate = 0.0001
# nr_coordinates = 2

# FOR LSTM
nr_layers = 4
n_hidden = 128

# PREPROCESS DATA
prepro = Preprocessor(PATH)

# ONE PLAYER
# players, _ = prepro.get_list_with_most("Pitcher")
# prepro.select_movement("Stretch")

# prepro.cut_file_to_pitcher(players[1])  # change_restore


prepro.remove_small_classes(CUT_OFF_Classes)

data_raw = prepro.concat_with_second(SECOND) #prepro.get_coord_arr()  #np.load("coord_sv.npy")
#data_raw = prepro.get_coord_arr()
data_without_head = data_raw[:, :, :12, :]
#print("release frames:", min(np.absolute(prepro.get_release_frame())), max(np.absolute(prepro.get_release_frame())))
# data_aligned = Tools.align_frames(data_without_head, np.absolute(prepro.get_release_frame()), int(min(np.absolute(prepro.get_release_frame()))-1), int(160-max(np.absolute(prepro.get_release_frame()))))
data = Tools.normalize(data_without_head, None)
# data = np.load("coord_pl1_p1.npy")

# ONE PITCH TYPE
#pitchi, _ = prepro.get_list_with_most("Pitch Type")
# print("classe trained on:", pitchi[0])
# prepro.set_labels(pitchi[1])  # change_restore

# CONCAT
# prepro.remove_small_classes(CUT_OFF_Classes)
# data_raw = prepro.concat_with_second(SECOND) #prepro.get_coord_arr()  #np.load("coord_sv.npy")
# data = Tools.normalize(data_raw, None)
#data = np.load("coord_concat.npy")

M,N,nr_joints,nr_coordinates = data.shape

SEP = int(M*0.9)

labels_string = prepro.get_labels()
labels, unique = Tools.onehot_encoding(labels_string)


nr_classes = len(np.unique(labels_string)) # hier
BATCHSIZE = nr_classes*ex_per_class
print("nr classes", nr_classes, "Batchsize", BATCHSIZE)
print("classes: ", unique)
print("data shape:", data.shape, "label_shape", labels.shape, labels_string.shape)
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


# DATA TESTING:
# indiuh = np.where(ind==2000)
# print("Labels nach preprocc von 2000", labels_string[2000])
# print("new Index of 2000", indiuh, "test ob where funkt: ", ind[indiuh])
# print("train coord of 2000 u 140", train_x[indiuh, 140])
# print("labels_string von 2000", labels_string_train[indiuh])
# print("one hot von 2000", train_t[indiuh])


index_liste = []
for pitches in unique:
    index_liste.append(np.where(labels_string_train==pitches))

len_test = len(test_x)
len_train = len(train_x)
print("Test set size: ", len_test, " train set size: ", len_train)
print("Shapes of train_x", train_x.shape, "shape of test_x", test_x.shape)

model = Model()

x = tf.placeholder(tf.float32, (None, N, nr_joints, nr_coordinates), name = "input")

y = tf.placeholder(tf.float32, (None, nr_classes))
training = tf.placeholder_with_default(False, None)

#out, logits = model.conv1dnet(x, nr_classes, training, rate_dropout, act)
out, logits = model.convshort(x, nr_classes, training, rate_dropout, act)
# out, logits = model.RNN(x, nr_classes, n_hidden, nr_layers)
tv = tf.trainable_variables()


# a = tf.Print(out, [out])
# a = tf.Print(tf.reduce_max(y, axis = 1), [tf.reduce_max(y, axis = 1), "This is y"])
# b = tf.Print(tf.nn.relu(y-out), [tf.nn.relu(y-out)])
# diff = tf.reduce_max(tf.nn.relu(y-out), axis = 1)
# c = tf.Print(diff, [diff])

loss_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
loss_regularization = 0.0001* tf.reduce_sum([ tf.nn.l2_loss(v) for v in tv ])
loss_maximum = tf.reduce_mean(tf.reduce_max(tf.nn.relu(y-out), axis = 1))
loss = loss_entropy # + loss_regularization + loss_maximum #0.001
# loss = tf.reduce_mean(tf.pow(out-y, 2))
# loss = tf.reduce_sum(tf.pow(out - y, 2)) + 0.5*regularization_cost

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
# TENSORBOARD
tf.summary.scalar("loss_entropy", loss_entropy)
tf.summary.scalar("loss_regularization", loss_regularization)
tf.summary.scalar("loss_maximum", loss_maximum)
tf.summary.scalar("loss", loss)

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter("./logs/nn_logs" + '/train', sess.graph)

# TRAINING
saver = tf.train.Saver(tv)
# sess = tf.Session()
if RESTORE is None:
    sess.run(tf.global_variables_initializer())
else:
    saver.restore(sess, RESTORE)
    print("Restored from file")
#sess.run(tf.global_variables_initializer())


def balanced_batches(x, y, nr_classes):
    #print("balanced function: ", nr_classes)
    for j in range(batch_nr_in_epoch):
        liste=np.zeros((nr_classes, ex_per_class))
        for i in range(nr_classes):
            # print(j, i, np.random.choice(index_liste[i][0], ex_per_class))
            liste[i] = np.random.choice(index_liste[i][0], ex_per_class, replace=True)
        liste = liste.flatten().astype(int)
        yield j, x[liste], y[liste]

# Run session for EPOCH epochs
for epoch in range(EPOCHS + 1):
    for i, batch_x, batch_t in balanced_batches(train_x, train_t, nr_classes):
        summary, _ = sess.run([merged, optimizer], {x: batch_x, y: batch_t, training: True})
        train_writer.add_summary(summary, i+batch_nr_in_epoch*epoch)

    #Test Accuracy

    loss_test, out_test = sess.run([loss,out], {x: test_x, y: test_t, training: False})
    print("Loss test", loss_test)
    pitches_test = Tools.decode_one_hot(out_test, unique)
    print("Accuracy test: ", Tools.accuracy(pitches_test, labels_string_test))

    # print(a.eval(session = sess, feed_dict={x: test_x, y: test_t, training: False}))
    # print(b.eval(session = sess, feed_dict={x: test_x, y: test_t, training: False}))
    # print(c.eval(session = sess, feed_dict={x: test_x, y: test_t, training: False}))

    #Train Accuracy
    out_train = sess.run(out, {x: train_x, y: train_t, training: False})
    pitches_train = Tools.decode_one_hot(out_train, unique)
    print("Accuracy train: ", Tools.accuracy(pitches_train, labels_string_train))

    if epoch%10 ==0 :
        #print("Accuracy test by class: ", Tools.accuracy_per_class(pitches_test, labels_string_test))
        print("True                   Test                 ", unique)
        # print(np.swapaxes(np.append([labels_string_test], [pitches_test], axis=0), 0,1))
        for i in range(len(labels_string_test)):
            print('{:20}'.format(labels_string_test[i]), '{:20}'.format(pitches_test[i]), ['%.2f        ' % elem for elem in out_test[i]])
    #print("Test:", np.array(pitches_test))
    #print("True:", np.array(labels_string_test)) #.astype(int))

saver.save(sess, "model")
