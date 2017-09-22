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
    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    saver = tf.train.import_meta_graph(restore_file+'.meta')
    graph = tf.get_default_graph()

    saver.restore(sess, restore_file)
    out = tf.get_collection("out")[0]
    unique = tf.get_collection("unique")[0]
    out_test = sess.run(out, {"input:0":  data, "training:0": False})

    # Evaluation
    pitches_test = Tools.decode_one_hot(out_test,  unique)

    return pitches_test



from data_preprocess import Preprocessor
from tools import Tools
from model import Model
# from run_thread import Runner

PATH = "cf"
CUT_OFF_Classes = 10
leaky_relu = lambda x: tf.maximum(0.2*x, x)
align = False
normalize = True

# PREPROCESS DATA
if PATH is "cf" or PATH is "concat":
    prepro = Preprocessor("cf_data.csv")
else:
    prepro = Preprocessor("sv_data.csv")

prepro.remove_small_classes(CUT_OFF_Classes)
# ONE PLAYER
players, _ = prepro.get_list_with_most("Pitcher")
prepro.select_movement("Windup")

prepro.cut_file_to_listof_pitcher(players)  # change_restore

# prepro.set_labels_toWindup()

if PATH is not "concat":
    data_raw = prepro.get_coord_arr(None) #PATH+"_all_coord.npy")
    # data_raw = np.load("cf_all_coord.npy")
    print("data loaded")
else:
    data_raw = prepro.concat_with_second("sv_data.csv", PATH+"_all_coord.npy")

data = data_raw[:,:,:12,:]


labels_string = prepro.get_labels()

labels_string = Tools.labels_to_classes(labels_string)

if align:
    data = Tools.align_frames(data, prepro.get_release_frame(60, 120), 60, 40)

if  normalize:
     data = Tools.normalize( data)

pitches = test(data, "/Users/ninawiedemann/Desktop/UNI/Praktikum/saved_models/modelCarlos2")
