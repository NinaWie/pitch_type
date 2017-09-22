import tflearn
from tflearn import DNN
from tools import Tools
from model import Model
import numpy as np

BEFORE = 10
AFTER = 10
nr_layers = 4
n_hidden = 512
BATCHSIZE = 128
EPOCHS = 50
dropout = 0

data = np.load("unpro_all_coord.npy")
norm = Tools.normalize(data)[:,:,:12,:]
M,N, joints, co = norm.shape
index = []
for i in range(M):
    for j in range(N):
        if (data[i,j]==0).all():
            if j>BEFORE and j<(N-AFTER):
                index.append((i,j))
            else:
                norm[i,j] = norm[i,j-1]

sequ_around = np.zeros((len(index), BEFORE+AFTER-1, joints*co))
for i, (data_ex, frame) in enumerate(index):
    RANGE = range(frame-BEFORE,frame+AFTER)
    sequ = np.delete(norm, frame, axis=1)[data_ex, RANGE[:-1]]
    # sequ = Tools.normalize(sequ)[data_ex]
    # print("sequ after deleted", sequ[:, 0])
    sequ_around[i] = np.reshape(sequ, (len(sequ), joints*co))


_, fr, input_size = sequ_around.shape
model = Model()
out, network = model.RNN_network_tflearn(fr, input_size, input_size, nr_layers, n_hidden, dropout, loss = "mean_square", act = "tanh")
m = DNN(network, checkpoint_path='sequ/model.tfl.ckpt')
m.load("./sequ/model.tfl")
print("loaded model")
out = m.predict(sequ_around)

for i, (data_ex, frame) in enumerate(index):
    norm[data_ex, frame] = np.reshape(out[i], (12,2))

np.save("ml_predict_all.npy", norm)
print("saved output")
