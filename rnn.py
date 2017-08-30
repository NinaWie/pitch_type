import pandas as pd
import numpy as np
import tensorflow as tf
import scipy as sp
import scipy.stats

import tflearn
from tflearn import DNN
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
# ex_per_class = 5
EPOCHS = 30
batch_nr_in_epoch = 100
PATH = "cf_data.csv"
SECOND = "sv_data.csv"
LABELS = "Pitch Type"
act = tf.nn.relu
CUT_OFF_Classes = 10
rate_dropout = 0.6
learning_rate = 0.0001
#nr_coordinates = 2
# FOR LSTM
nr_layers = 4
n_hidden = 128

# PREPROCESS DATA
prepro = Preprocessor(PATH)

# ONE PLAYER
# players, _ = prepro.get_list_with_most("Pitcher")
#prepro.select_movement("Windup")

# prepro.cut_file_to_pitcher(players[player])  # change_restore


prepro.remove_small_classes(CUT_OFF_Classes)

data_raw = prepro.get_coord_arr()
data_without_head = data_raw[:, :, :12, :]
#print("release frames:", min(np.absolute(prepro.get_release_frame())), max(np.absolute(prepro.get_release_frame())))
# data_aligned = Tools.align_frames(data_without_head, np.absolute(prepro.get_release_frame()), min(np.absolute(prepro.get_release_frame()))-1, 160-max(np.absolute(prepro.get_release_frame())))
data = Tools.normalize(data_without_head, None)
# data = np.load("coord_pl1_p1.npy")

# ONE PITCH TYPE
# pitchi, _ = prepro.get_list_with_most("Pitch Type")
# print("classe trained on:", pitchi[0])
# prepro.set_labels(pitchi[0])  # change_restore

# CONCAT
# prepro.remove_small_classes(CUT_OFF_Classes)
# data_raw = prepro.concat_with_second(SECOND) #prepro.get_coord_arr()  #np.load("coord_sv.npy")
# data = Tools.normalize(data_raw, None)
#data = np.load("coord_concat.npy")

M,N,nr_joints,nr_coordinates = data.shape
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
ex_per_class = 40//nr_classes
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


len_test = len(test_x)
len_train = len(train_x)
print("Test set size: ", len_test, " train set size: ", len_train)
print("Shapes of train_x", train_x.shape, "shape of test_x", test_x.shape)

model = Model()

# RNN tflearn

tr_x = train_x.reshape(len_train, N, nr_joints*nr_coordinates)
te_x = test_x.reshape(len_test, N, nr_joints*nr_coordinates)

out, network = model.RNN_network_tflearn(N, nr_joints*nr_coordinates, nr_classes)
m = DNN(network)

m.fit(tr_x, train_t, validation_set=(te_x, test_t), show_metric=True, batch_size=BATCHSIZE, snapshot_step=100, n_epoch=10)

out_test = m.predict(te_x)
pitches_test = Tools.decode_one_hot(out_test, unique)
print("Accuracy test: ", Tools.accuracy(pitches_test, labels_string_test))
print("Accuracy test by class: ", Tools.accuracy_per_class(pitches_test, labels_string_test))
print("True                   Test                 ", unique)
# print(np.swapaxes(np.append([labels_string_test], [pitches_test], axis=0), 0,1))
for i in range(len(labels_string_test)):
    print('{:20}'.format(labels_string_test[i]), '{:20}'.format(pitches_test[i]), ['%.2f        ' % elem for elem in out_test[i]])
