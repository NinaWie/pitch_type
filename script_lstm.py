import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.contrib import rnn
#from sklearn.preprocessing import StandardScaler
from data_preprocess import Preprocessor

def decode_one_hot(results, unique):
    """takes the maximum value and gets the corresponding pitch type
    input: array of size trials * pitchTypesNr
    returns: array of size trials containing the pitch type as a string
    """
    #unique  = np.unique(cf["Pitch Type"].values)
    p = []
    for _, pitch in enumerate(results):
        ind = np.argmax(pitch)
        if pitch[ind]>0.2:
            p.append(unique[ind])
        else:
            p.append("Too small")
    return p

leaky_relu = lambda x: tf.maximum(0.2*x, x)


ex_per_class = 4
EPOCHS = 10
PATH = "cf_only_pitcher.csv"
LABELS = "Pitch Type"
act = leaky_relu
CUT_OFF_Classes = 30

prepro = Preprocessor(PATH, CUT_OFF_Classes)
data = prepro.get_coord_arr(None)

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


learning_rate = 0.001
training_iters = 100000
batch_size = BATCHSIZE
display_step = 10
nr_layers = 4

tf.reset_default_graph()

# Network Parameters
n_input = nr_joints*2 # MNIST data input (img shape: 28*28)
n_steps = N # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 12 # MNIST total classes (0-9 digits)

# tf Graph input
x_ = tf.placeholder("float", [None, n_steps, nr_joints, 2])
x = tf.reshape(x_, (-1, N, nr_joints*2))
y = tf.placeholder("float", [None, n_classes])

"""# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}"""


def RNN(x):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, n_steps, 1)

    # Define a lstm cell with tensorflow
    #lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

    def lstm_cell():
          return rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

    stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(nr_layers)])


    outputs, states = rnn.static_rnn(stacked_lstm, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.layers.dense(outputs[-1], n_classes)   #tf.matmul(outputs[-1], weights['out']) + biases['out']

out_logits = RNN(x)
out = tf.nn.softmax(out_logits)

# Define loss and optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_logits, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


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
        sess.run(optimizer, {x_: batch_x, y: batch_t})
    # print("Loss test: ", sess.run(loss, {x: test_x, y: test_t}))
    #print("Loss train: ", sess.run(loss, {x: train_x, y: train_t}))

    #Test Accuracy
    loss_test, out_test = sess.run([loss,out], {x_: test_x, y: test_t})
    print("Loss test", loss_test)
    pitches_test = decode_one_hot(out_test, unique)
    print("Accuracy test: ", np.sum(np.asarray(labels_string_test)==pitches_test)/len_test)
    print(pitches_test)

    #Train Accuracy
    """
    out_train = sess.run(out, {x_: train_x, y: train_t})
    pitches_train = decode_one_hot(out_train, unique)
    print("Accuracy train: ", np.sum(np.asarray(labels_string_train)==pitches_train)/SEP)
    """

print(pitches_test)
