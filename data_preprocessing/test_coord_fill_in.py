import tflearn
from tflearn import DNN
import sys
sys.path.append("/Users/ninawiedemann/Desktop/UNI/Praktikum/ALL")
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

data = np.load("/Users/ninawiedemann/Desktop/UNI/Praktikum/numpy arrays/unpro_all_coord.npy")
norm = Tools.normalize(data)[:,:,:12,:]
M,N, joints, co = norm.shape
index = []
for i in range(M):
    for j in range(N):
        if (data[i,j]==0).all():
            if j>BEFORE and j<(N-AFTER):
                print(1)
                index.append((i,j))
            else:
                print(i,j)
                norm[i,j] = norm[i,j-1]

print(index)

print(len(index))

sequ_around = np.zeros((len(index), BEFORE+AFTER-1, joints*co))
for i, (data_ex, frame) in enumerate(index):
    RANGE = range(frame-BEFORE,frame+AFTER)
    sequ = np.delete(norm, frame, axis=1)[data_ex, RANGE[:-1]]
    # sequ = Tools.normalize(sequ)[data_ex]
    # print("sequ after deleted", sequ[:, 0])
    sequ_around[i] = np.reshape(sequ, (len(sequ), joints*co))

print(sequ_around)

#ml = norm[data_ex, :, :12,:]
#print("output bevore deleted", ml[RANGE,0,:])

# def get_ML_coord(sequ_around):

_, fr, input_size = sequ_around.shape
model = Model()
out, network = model.RNN_network_tflearn(fr, input_size, input_size, nr_layers, n_hidden, dropout, loss = "mean_square", act = "tanh")
m = DNN(network, checkpoint_path='sequ/model.tfl.ckpt')
m.load("./sequ/model.tfl")
print("loaded")
out = m.predict(sequ_around)
#return out

print("output(alle)", np.reshape(out, (12,2)))
ml[frame] = np.reshape(out, (12,2))
#print(np.reshape(get_ML_coord(sequ_around), (12,2)))
# ml[frame] = np.reshape(get_ML_coord(sequ_around), (12,2))
print(ml[RANGE, 0, :])
np.save("ml_predict", ml)
