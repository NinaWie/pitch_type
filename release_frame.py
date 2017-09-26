import pandas as pd
import numpy as np
import tensorflow as tf
import scipy as sp
import scipy.stats
import json
from os import listdir

path_input = "videos/atl"
path_output = "arrays/"
#path_input_dat = "/Volumes/Nina Backup/videos/atl/2017-04-14/center field/490251-0f308987-60b4-480c-89b7-60421ab39106.mp4.dat"

from run_thread import Runner


dates = ["2017-04-15", "2017-04-19", "2017-05-03", "2017-05-07", "2017-05-20", "2017-05-24", "2017-06-07",
 "2017-06-11", "2017-06-19", "2017-06-23", "2017-07-05", "2017-07-17", "2017-04-14", "2017-04-18",
  "2017-05-02", "2017-05-06", "2017-05-19", "2017-05-23", "2017-06-06", "2017-06-10", "2017-06-18", "2017-06-22", "2017-07-04", "2017-07-16"]

cut_off_min = 80
cut_off_max= 110
pos = []
neg = []
for date in dates:
    arr = np.load(path_output+"array_videos_"+date+".npy")[:, cut_off_min:cut_off_max, :, :]
    lab =  np.load(path_output+"labels_release_"+date+".npy")
    for i, pitch in enumerate(arr):
        rel = lab[i]
        if rel>cut_off_min and rel<cut_off_max and not np.isnan(rel):
            ind = int(rel)-cut_off_min
            pos.append(pitch[ind])
            rest = np.delete(pitch, int(rel)-cut_off_min, axis = 0)
            indizes = np.random.choice(len(rest), 3, replace = False)
            for elem in rest[indizes]:
                neg.append(elem)
        else:
            print("false", rel)
            #print(pitch.shape, pos[-1].shape, neg[-1].shape)
print(np.array(pos).shape, np.array(neg).shape)
 
pos = pos[93:]

labels = np.ones((len(pos)), dtype=np.int).tolist()+np.zeros((len(neg)), dtype=np.int).tolist()
print(len(pos), len(neg), len(labels))
data = np.append(np.array(pos), np.array(neg), axis = 0)
np.save("positive.npy", np.array(pos))
print("positive saved")
#np.save("negative.npy", np.array(neg)[:10])
examples, width, height = data.shape
data = np.reshape(data, (examples, width, height, 1))
print(data.shape)

runner = Runner(data, np.array(labels), SAVE = None, BATCH_SZ=40, EPOCHS = 40, batch_nr_in_epoch = 100,
        act = tf.nn.relu, rate_dropout = 0,
        learning_rate = 0.0005, nr_layers = 4, n_hidden = 128, optimizer_type="adam", regularization=0,
        first_conv_filters=128, first_conv_kernel=9, second_conv_filter=128,
        second_conv_kernel=9, first_hidden_dense=128, second_hidden_dense=0,
        network = "adjustable conv2d")
runner.start()


def save_release_labels(path_input, dates):
    df = pd.read_csv("cf_data.csv")
    df = df[df["Player"]=="Pitcher"]
    for date in dates:
        output = []
        labels = []
        # args = parser.parse_args()
        input_dir= path_input+"/"+date+"/center field/"
        list_files = listdir(input_dir)
        #print(list_files[0:3])
        # a  = np.array(range(len(list_files)))
        print(date)
        for f in list_files:
            if f[-4:]==".mp4":
                game_id = f[:-4]
                line = df[df["Game"]==game_id]
                #print(labels)
                if len(line["pitch_frame_index"].values)!=1:
                    print("PROBLEM: NO LABEL/ TOO MANY")
                    ##print(line["Pitch Type"].values)
                else:
                    labels.append(line["pitch_frame_index"].values[0])
                    print(labels[-1])

        np.save(path_output+"labels_release_"+date+".npy", np.array(labels))
