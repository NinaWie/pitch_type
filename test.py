import pandas as pd
import numpy as np
import tensorflow as tf
from tools import Tools

def test(data, restore_file):
    """
    Runs model of restore_file for the data
    data must be normalized before and aligned if desired
    returns labels
    """
    #tf.reset_default_graph()
    #sess = tf.InteractiveSession()

    saver = tf.train.import_meta_graph(restore_file+'.meta')
    graph = tf.get_default_graph()
    try:
	sess = tf.InteractiveSession()
    except:
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


def test_old(data, restore_file):
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
