import pandas as pd
import numpy as np
import tensorflow as tf
import scipy as sp
import scipy.stats

class Tools:

    @staticmethod
    def normalize(data, save_name = None):
        M,N, nr_joints,_ = data.shape
        means = np.mean(data, axis = 1)
        std = np.std(data, axis = 1)
        res = np.asarray([(data[:,i]-means)/(std+0.000001) for i in range(len(data[0]))])
        data_new = np.swapaxes(res, 0,1)

        # save
        if save_name!=None:
            np.save(save_name, data_new)
            print("Saved with name ", save_name)

        return data_new

    @staticmethod
    def onehot_encoding(labels):
        #dataframe = self.cf
        unique  = np.unique(labels).tolist()
        one_hot_labels = np.zeros((len(labels), len(unique)))
        for i in range(len(labels)):
            #print(cf.iloc[i,loc])
            pitch = labels[i]
            ind = unique.index(pitch)
            one_hot_labels[i, ind] = 1
        return one_hot_labels, unique

    @staticmethod
    def decode_one_hot(results, unique):
        """takes the maximum value and gets the corresponding pitch type
        input: array of size trials * pitchTypesNr
        returns: array of size trials containing the pitch type as a string
        """
        #unique  = np.unique(cf["Pitch Type"].values)
        p = []
        for _, pitch in enumerate(results):
            ind = np.argmax(pitch)
            if pitch[ind]>0.1:
                p.append(unique[ind])
            else:
                p.append("Too small")
        return p

    @staticmethod
    def batches(x, y, batchsize=32):
        permute = np.random.permutation(len(x))
        for i in range(0, len(x)-batchsize, batchsize):
            indices = permute[i:i+batchsize]
            yield x[indices], y[indices]

    @staticmethod
    def accuracy_per_class(out_list, ground_truth_list):
        out = np.array(out_list)
        ground_truth = np.array(ground_truth_list)

        same = out[np.where(out==ground_truth)[0]]

        right_frequency = sp.stats.itemfreq(same)
        total_frequency = sp.stats.itemfreq(ground_truth)
        right_dict = dict(zip(right_frequency[:,0], right_frequency[:,1]))
        total_dict = dict(zip(total_frequency[:,0], total_frequency[:,1]))

        acc= right_dict
        for types in right_dict.keys():
            acc[types] = (int(right_dict[types])/float(total_dict[types]))

        return acc
        
    @staticmethod
    def accuracy(out, ground_truth):
        return np.sum(np.array(ground_truth)==np.array(out))/float(len(out))
