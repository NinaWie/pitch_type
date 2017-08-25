import pandas as pd
import numpy as np
import tensorflow as tf
import scipy as sp
import scipy.stats

#from sklearn.preprocessing import StandardScaler
from data_preprocess import Preprocessor
from tools import Tools
from model import Model


tf.reset_default_graph()
sess = tf.InteractiveSession()

# META PARAMTETERS
ex_per_class = 4
EPOCHS = 10
batch_nr_in_epoch = 100
PATH = "cf_data.csv"
SECOND = "sv_data.csv"
LABELS = "Pitch Type"
act = tf.nn.relu
CUT_OFF_Classes = 60
rate_dropout = 0.6
learning_rate = 0.0001
nr_coordinates = 2
# FOR LSTM
nr_layers = 4
n_hidden = 128
leaky_relu = lambda x: tf.maximum(0.2*x, x)


# PREPROCESS DATA
prepro = Preprocessor(PATH)
players = prepro.get_list_with_most("Pitcher")
print(players[0])
prepro.cut_file_to_pitcher(players[0])
pitchi = prepro.get_list_with_most("Pitch Type")
print(pitchi[0])
prepro.set_labels(pitchi[0])

data_raw = prepro.get_coord_arr()
data = Tools.normalize(data_raw, "coord_pl1_p1")
# data = np.load("coord_pl1_p1.npy")
print(data.shape)

# CONCAT
# prepro.remove_small_classes(CUT_OFF_Classes)
# data_raw = prepro.concat_with_second(SECOND) #prepro.get_coord_arr()  #np.load("coord_sv.npy")
# data = Tools.normalize(data_raw, None)
#data = np.load("coord_concat.npy")



M,N,nr_joints,_ = data.shape
SEP = int(M*0.9)

labels_string = prepro.get_labels()
labels, unique = Tools.onehot_encoding(labels_string)
# labels = []
# for i in range(len(labels_string)):
#     labels.append([labels_string[i]])
# labels = np.array(labels) # hier
#
# unique = [0, 1] #hier

#labels_test = Tools.decode_one_hot(labels[SEP:, :], unique)

nr_classes = len(np.unique(labels_string)) # hier
BATCHSIZE = nr_classes*ex_per_class
print("nr classes", nr_classes, "Batchsize", BATCHSIZE)

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

#start
# x_ = tf.reshape(x, (-1, N, nr_joints*2))
#
# net = tf.layers.conv1d(x_, filters=256, kernel_size=5, strides=2, activation=act)
# net = tf.layers.dropout(net, rate=.5, training=training)
# net = tf.layers.conv1d(net, filters=256, kernel_size=3, strides=1, activation=act)
# net = tf.layers.conv1d(net, filters=128, kernel_size=3, strides=1, activation=act)
# net = tf.layers.dropout(net, rate=.5, training=training)
# net = tf.layers.conv1d(net, filters=1, kernel_size=1, activation = tf.nn.sigmoid)
# shapes = net.get_shape().as_list()
# ff = tf.reshape(net, (-1, shapes[1]*shapes[2]))
# ff = tf.layers.dense(ff, 1024, activation = act)
# ff = tf.layers.dense(ff, 128, activation = act)
# logits = tf.layers.dense(ff, len(labels[0]), activation = None)
# out = tf.nn.softmax(logits)
#end

out, logits = model.conv1dnet(x, nr_classes, training, rate_dropout, act)
# out, logits = model.RNN(x, nr_classes, n_hidden, nr_layers)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)) # tf.reduce_mean(tf.square(y-ff))
# loss = tf.reduce_mean(tf.pow(out-y, 2))
# tv = tf.trainable_variables()
# regularization_cost = tf.reduce_sum([ tf.nn.l2_loss(v) for v in tv ])
# loss = tf.reduce_sum(tf.pow(out - y, 2)) + 0.5*regularization_cost

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# TENSORBOARD
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
            liste[i] = np.random.choice(index_liste[i][0], ex_per_class, replace=False)
        liste = liste.flatten().astype(int)
        yield j, x[liste], y[liste]

# Run session for EPOCH epochs
for epoch in range(EPOCHS + 1):
    for i, batch_x, batch_t in balanced_batches(train_x, train_t, nr_classes):
        summary, _ = sess.run([merged, optimizer], {x: batch_x, y: batch_t, training: True})
        train_writer.add_summary(summary, i+batch_nr_in_epoch*epoch)

    # print("Loss test: ", sess.run(loss, {x: test_x, y: test_t}))
    #print("Loss train: ", sess.run(loss, {x: train_x, y: train_t}))
        #Test Accuracy
    loss_test, out_test = sess.run([loss,out], {x: test_x, y: test_t, training: False})
    print("Loss test", loss_test)
    pitches_test = Tools.decode_one_hot(out_test, unique)
    print("Accuracy test: ", Tools.accuracy(pitches_test, labels_string_test))
    #print("Accuracy test by class: ", Tools.accuracy_per_class(pitches_test, labels_string_test))

    #Train Accuracy
    out_train = sess.run(out, {x: train_x, y: train_t, training: False})
    pitches_train = Tools.decode_one_hot(out_train, unique)
    print("Accuracy train: ", Tools.accuracy(pitches_train, labels_string_train))



    if (epoch)%5==0:
        print("Test:", np.array(pitches_test).astype(int))
        print("True:", np.array(labels_string_test).astype(int))
