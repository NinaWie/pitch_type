import pandas as pd
import numpy as np
import tensorflow as tf
from tools import Tools
import time

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
    try:
        sess = tf.InteractiveSession()
    except:
    	sess = tf.Session()

    # restore
    saver.restore(sess, restore_file)
    out = tf.get_collection("out")[0]
    unique = tf.get_collection("unique")[0]
    tic = time.time()

    # run for data
    out_test = sess.run(out, {"input:0":  data, "training:0": False})

    # Decode one hot vectors
    toc = time.time()
    print("time for nr labels", toc-tic)
    pitches_test = Tools.decode_one_hot(out_test,  unique.eval(session = sess))
    try:
        pitches = [elem.decode("utf-8") for elem in pitches_test]
    except AttributeError:
        pitches = pitches_test

    sess.close()
    return pitches, out_test

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test neural network model for with any input data')
    parser.add_argument('data_path', default= "data_test.npy", type=str, help='path to input data, should be saved as a numy array')
    parser.add_argument('-label', default="labels_test.npy", type=str, help=' path to numpy array with labels if accuracy should be calculated')
    parser.add_argument('model_path', default="saved_models/validation_test", type=str, help='either cf (center field) or sv (side view)')
    args = parser.parse_args()

    data = np.load(args.data_path)

    # Data needs to be normalized to be fed into the neural network
    data = Tools.normalize(data)

    print(data.shape, labels.shape, np.mean(data))
    tic = time.time()
    labs, out = test(data, "saved_models/validation_test")
    toc = time.time()
    print("time for nr labels", len(labs), toc-tic)
    for i in range(len(labs)):
        print(labs[i], out[i])

    ##  To compare with labels
    # labels = np.load(args.label)
    # print(Tools.accuracy(np.asarray(labs), labels))
    # print(Tools.balanced_accuracy(np.asarray(labs), labels))
    # for i in range(len(labels)): #len(labels_string_test)):
    #     print('{:20}'.format(labels[i]), '{:20}'.format(labs[i])) #, ['%.2f        ' % elem for elem in out_test[i]])
