from model import Model
import tflearn
from tflearn import DNN
from tools import Tools
import numpy as np

data = np.load("unpro_all_coord.npy")
norm = Tools.normalize(data)
FRAME = 96
DATA_EX = 795
RANGE = range(FRAME-19:FRAME+1)
sequ = np.delete(norm, FRAME, axis=1)[DATA_EX, RANGE[:-1], :12, :]
# sequ = Tools.normalize(sequ)[DATA_EX]
print(sequ[:, 0])
sequ_around = np.reshape(sequ, (1, len(sequ), 24))

ml = norm[DATA_EX, :, :12,:]
print(ml[RANGE,0,:])

# def get_ML_coord(sequ_around):
nr_layers = 4
n_hidden = 512
BATCHSIZE = 128
EPOCHS = 50
SEQU_LEN = 20
dropout = 0
_, fr, input_size = sequ_around.shape
model = Model()
out, network = model.RNN_network_tflearn(fr, input_size, input_size, nr_layers, n_hidden, dropout, loss = "mean_square", act = "tanh")
m = DNN(network, checkpoint_path='sequ/model.tfl.ckpt')
m.load("./sequ/model.tfl")
print("loaded")
out = m.predict(sequ_around)
#return out

print(np.reshape(out, (12,2)))
ml[FRAME] = np.reshape(out, (12,2))
#print(np.reshape(get_ML_coord(sequ_around), (12,2)))
# ml[FRAME] = np.reshape(get_ML_coord(sequ_around), (12,2))
print(ml[RANGE, 0, :])
np.save("ml_predict", ml)
