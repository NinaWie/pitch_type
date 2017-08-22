import pandas as pd
import numpy as np
import tensorflow as tf
#from sklearn.preprocessing import StandardScaler
from data_preprocess import Preprocessor

ex_per_class = 4
EPOCHS = 10
PATH = "/Users/ninawiedemann/Desktop/UNI/Praktikum/sv_data.csv"
LABELS = "Pitch Type"



def decode_one_hot(results, unique):
    """takes the maximum value and gets the corresponding pitch type
    input: array of size trials * pitchTypesNr
    returns: array of size trials containing the pitch type as a string
    """
    #unique  = np.unique(cf["Pitch Type"].values)
    p = []
    for _, pitch in enumerate(results):
        ind = np.argmax(pitch)
        if pitch[ind]>0.5:
            p.append(unique[ind])
        else:
            p.append("Too small")
    return p

prepro = Preprocessor(PATH, 30)
data =  prepro.get_coord_arr("coord_sv.npy") # np.load("coord_array.npy")

M,N,nr_joints,_ = data.shape
SEP = int(M*0.9)

labels, unique = prepro.get_labels_onehot(LABELS)
labels_string = prepro.get_labels(LABELS)
#labels_test = decode_one_hot(labels[SEP:, :], unique)

nr_classes = len(np.unique(labels_string))
BATCHSIZE = nr_classes*ex_per_class
print("nr classes", nr_classes, "Batchsize", BATCHSIZE)



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

"""
DATA TESTING:
indiuh = np.where(ind==2000)
print("Labels nach preprocc von 2000", labels_string[2000])
print("new Index of 2000", indiuh, "test ob where funkt: ", ind[indiuh])
print("train coord of 2000 u 140", train_x[indiuh, 140])
print("labels_string von 2000", labels_string_train[indiuh])
print("one hot von 2000", train_t[indiuh])
"""

index_liste = []
for pitches in unique:
    index_liste.append(np.where(labels_string_train==pitches))

len_test = len(test_x)
len_train = len(train_x)
print("Test set size: ", len_test, " train set size: ", len_train)
print("Shapes of train_x", train_x.shape, "shape of test_x", test_x.shape)

x = tf.placeholder(tf.float32, (None, N, nr_joints, 2), name = "input")
x_ = tf.reshape(x, (-1, N, nr_joints*2))
y = tf.placeholder(tf.float32, (None, len(labels[0])))

net = tf.layers.conv1d(x_, filters=256, kernel_size=5, strides=2, activation=tf.nn.relu)
net = tf.layers.conv1d(net, filters=256, kernel_size=3, strides=1, activation=tf.nn.relu)
net = tf.layers.conv1d(net, filters=128, kernel_size=3, strides=1, activation=tf.nn.relu)
net = tf.layers.conv1d(net, filters=1, kernel_size=1)
shapes = net.get_shape().as_list()
ff = tf.reshape(net, (-1, shapes[1]*shapes[2]))
ff = tf.layers.dense(ff, 1024, activation = tf.nn.relu)
ff = tf.layers.dense(ff, 128, activation = tf.nn.relu)
logits = tf.layers.dense(ff, len(labels[0]), activation = None)
out = tf.nn.softmax(logits)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)) # tf.reduce_mean(tf.square(y-ff))
optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

def batches(x, y, batchsize=32):
    permute = np.random.permutation(len(x))
    for i in range(0, len(x)-batchsize, batchsize):
        indices = permute[i:i+batchsize]
        yield x[indices], y[indices]

def balanced_batches(x, y, batchsize=32):
    for j in range(200):
        liste=np.zeros((nr_classes, ex_per_class))
        for i in range(nr_classes):
            # print(j, i, np.random.choice(index_liste[i][0], ex_per_class))
            liste[i] = np.random.choice(index_liste[i][0], ex_per_class, replace=False)
        liste = liste.flatten().astype(int)
        yield x[liste], y[liste]

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Run session for EPOCH epochs
for epoch in range(EPOCHS + 1):
    for batch_x, batch_t in balanced_batches(train_x, train_t, BATCHSIZE):
        sess.run(optimizer, {x: batch_x, y: batch_t})
    print("Loss test: ", sess.run(loss, {x: test_x, y: test_t}))
    print("Loss train: ", sess.run(loss, {x: train_x, y: train_t}))
    #Train Accuracy
    out_train = sess.run(out, {x: train_x, y: train_t})
    pitches_train = decode_one_hot(out_train, unique)
    print("Accuracy train: ", np.sum(np.asarray(labels_string_train)==pitches_train)/SEP)
    #Test Accuracy
    out_test = sess.run(out, {x: test_x, y: test_t})
    pitches_test = decode_one_hot(out_test, unique)
    print("Accuracy test: ", np.sum(np.asarray(labels_string_test)==pitches_test)/len_test)

print(pitches_test)
