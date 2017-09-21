import numpy as np
from run import Runner
import pandas as pd
import pandas as pd
import argparse
import time
import json, codecs
from tools import Tools
from os import listdir

parser = argparse.ArgumentParser(description='Pose Estimation Baseball')
parser.add_argument('input_dir', metavar='DIR', help='folder where joints.json are')

args = parser.parse_args()
path_input_vid = args.input_dir
files = []
for ff in listdir(path_input_vid):
    files.append(str(ff))




tic = time.time()

# LOAD JSON WITH JOINTS
pitcher_array = []
for f in files:
    obj_text = codecs.open(path_input_vid+f, encoding='utf-8').read()
    b_new = json.loads(obj_text)
    pitcher_array.append(np.array(b_new)[:,:12,:])
pitcher_array = np.array(pitcher_array)

# RUN --> OUTPUTS PITCHES
all_classes = ['Changeup', 'Curveball', 'Fastball (2-seam)', 'Fastball (4-seam)', 'Fastball (Cut)', 'Knuckle curve', 'Knuckleball', 'Sinker', 'Slider']
runner = Runner()
out_test  = runner.run(pitcher_array, None, all_classes, RESTORE="/scratch/nvw224/good_results/model", normalize = True,
	BATCH_SZ=40, EPOCHS = 60, batch_nr_in_epoch = 100, align = False,
        rate_dropout = 0,
        learning_rate = 0.0005, nr_layers = 4, n_hidden = 128, optimizer_type="adam", regularization=0,
        first_conv_filters=128, first_conv_kernel=9, second_conv_filter=128,
        second_conv_kernel=9, first_hidden_dense=128, second_hidden_dense=0,
        network = "adjustable conv1d")
pitches_test = Tools.decode_one_hot(out_test, all_classes)
with open(path_input_vid+'pitchtypes.json', 'w') as fout:
    json.dump(pitches_test, fout)
print("saved: ", pitches_test)
# reload with with open('pitchtype.json', 'r') as fout:
# 	out = json.load(fout)
toc = time.time()
print("Time for array load and pitch type: ", toc-tic)

# EVALUATE
cf = pd.read_csv("/scratch/nvw224/cf_data.csv")

labels_string = []
for f in files:
    location_play = (cf[cf["Game"]==(f[:f.rfind('_')])].index.values)[0]
    labels_string.append((cf["Pitch Type"].values)[location_play])
labels_string= np.array(labels_string)

acc = Tools.accuracy(pitches_test, labels_string)
print("Accuracy: ", acc)
print("True                Test                 ", all_classes)
for i in range(len(pitches_test)):
	print('{:20}'.format(labels_string[i]), '{:20}'.format(pitches_test[i]), ['%.2f        ' % elem for elem in out_test[i]])
