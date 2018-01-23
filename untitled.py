import numpy as np
from tools import Tools


pitches_test = np.load("bsp_outs.npy")
labels_string_test = np.load("bsp_labels.npy")

Tools.confused_classes(pitches_test, labels_string_test)
