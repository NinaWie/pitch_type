import pandas as pd
import numpy as np
import tensorflow as tf
import scipy as sp
#import scipy.stats

class Tools:

    @staticmethod
    def normalize(data):
        """
        normalizes across frames - axix to zero mean and standard deviation
        """
        M,N, nr_joints,_ = data.shape
        means = np.mean(data, axis = 1)
        std = np.std(data, axis = 1)
        res = np.asarray([(data[:,i]-means)/(std+0.000001) for i in range(len(data[0]))])
        data_new = np.swapaxes(res, 0,1)
        return data_new

    @staticmethod
    def renormalize(data, means, std, one_pitch=None):
        if one_pitch is None:
            res = np.asarray([data[:,i]*std+means for i in range(len(data[0]))])
            #print(res.shape)
            data_new = np.swapaxes(res, 0,1)
        else:
            print(data.shape)
            data_new = np.asarray([data[i]*std[one_pitch]+means[one_pitch] for i in range(len(data))])
        return data_new

    @staticmethod
    def labels_to_classes(labels):
        print(labels[:20])
        classes = {"Fastball (4-seam)":"Fastball", "Fastball (2-seam)": "Fastball", 'Fastball (Cut)': "Fastball", 'Fastball (Split-finger)': "Fastball", "Sinker": "Fastball",
        'Curveball': "Breaking Ball", "Slider": "Breaking Ball", 'Knuckle curve':"Breaking Ball", 'Knuckleball': "Breaking Ball", "Changeup": "Changeup"}
        for uni in np.unique(labels):
            labels[labels==uni] = classes[uni]
        print(labels[:20])
        return labels

    def get_paths_from_games(game_ids, view):
        dic = get_filesystem_dic("/scratch/nvw224/videos/atl/", view)
        #print(dic)
        dates_belonging = []
        for g in game_ids:
            if view == "side view":
                new = g+".m4v"
            else:
                new = g+".mp4"
            for key in dic.keys():
                if new in dic[key]:
                    dates_belonging.append(key)
        assert(len(game_ids)==len(dates_belonging))
        np.save("outs/dates.npy", dates_belonging)


    @staticmethod
    def onehot_encoding(labels):
        """ one hot encoding for labels"""
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
    def onehot_with_unique(labels, unique):
        """ one hot encoding if unique is already known
        use this one if restore, to avoid different nr of CUT_OFF_Classes"""
        one_hot_labels = np.zeros((len(labels), len(unique)))
        for i in range(len(labels)):
            #print(cf.iloc[i,loc])
            pitch = labels[i]
            ind = unique.index(pitch)
            one_hot_labels[i, ind] = 1
        return one_hot_labels

    @staticmethod
    def align_frames(data, release_frame, fr_before, fr_after):
        """
        data: four dim array with dims (data_length, #frames,
        nr_joints, nr_coordinates)
        release frame: must have same data_length as data and contain the
        frame of ball release for each corresponding sample
        fr_befoer: #frames considered before the ball release
        fr_after: # frames considered after ball release

        aligns the data around the ball release frame and cuts of other frames
        returns: data cut to shape (data_length, fr_after+fr_before,
        nr_joints, nr_coordinates)
        """
        M, _, nr_joints, nr_coord = data.shape
        new = np.zeros((M, fr_after+fr_before, nr_joints, nr_coord))
        for i, row in enumerate(data):
            ind = release_frame[i]
            start = int(ind-fr_before)
            end = int(ind+ fr_after)
            #print(start, end)
            new[i] = data[i, start:end, :,:]
        return new


    @staticmethod
    def decode_one_hot(results, unique):
        """takes the maximum value of the one hot vector and yields the corresponding pitch type
        input: array of size data_length * pitchTypesNr
        returns: array of size data_length containing the pitch type as a string
        """
        #unique  = np.unique(cf["Pitch Type"].values)
        p = []
        for _, pitch in enumerate(results):
            ind = np.argmax(pitch)
            if pitch[ind]>0:
                p.append(unique[ind])
            else:
                p.append("Too small")
        return p

    @staticmethod
    def missing_interpolate(data):
        """
        interpolates the values for all missing values/ whole frames in data
        """
        M,N, nr_j, _ = data.shape
        #flat = np.flatten()
        #print(np.where(data[:,:,:, 0].flatten()==0 and data[:,:,:, 1]!=0))
        print(data.shape)
        data = data[:,:,:12,:]

        for i in range(M):
            for j in range(N):
                for k in range(12):
                    if k!=16 and data[i,j,k,0]==0:
                        # print(i,j,k)
                        ind  = np.where(data[i, :, k, 0]>0)[0]
                        if ind[-1]>j and ind[0]<j: #ind!=np.array([]) and
                            #print(ind)
                            ind_before = ind[ind<j][-1]
                            ind_after = ind[ind>j][0]
                            inter0 = np.linspace(data[i, ind_before, k, 0], data[i, ind_after, k, 0], ind_after-ind_before-1)
                            inter1 = np.linspace(data[i, ind_before, k, 1], data[i, ind_after, k, 1], ind_after-ind_before-1)
                            #print(ind)
                            #print(i,j,k)
                            #print(ind)
                            #print(data[i,j])
                            #print(data[i, j:ind_after, k, 0])
                            data[i,j:ind_after,k,0]= inter0
                            data[i,j:ind_after,k,1]= inter1
                            #print(data[i,j])
                        elif ind[-1]>j:
                            ind_before = ind[ind>j][0]
                            data[i,j,k,0] = data[i,ind_before,k,0]
                            data[i,j,k,1] = data[i,ind_before,k,1]
                        elif ind[0]<j:
                            ind_after = ind[ind<j][-1]
                            data[i,j,k,0] = data[i,ind_after,k,0]
                            data[i,j,k,1] = data[i,ind_after,k,1]
            return data

    @staticmethod
    def balance(data, labels):
        uni = np.unique(labels)
        index_liste = []
        for pitches in uni:
            index_liste.append(np.where(labels==pitches)[0])
        M, N, w, h = data.shape
        nr_examples = [len(liste) for liste in index_liste]
        max_length = max(nr_examples)
        nr_classes = len(index_liste)
        new_data = np.zeros((max_length*nr_classes, N, w, h))
        new_labels = list(labels)
        new_data[:M] = data
        aktuell = M
        for clas in range(len(index_liste)):
            nr = nr_examples[clas]
            while nr<max_length:
                add = data[index_liste[clas]]
                add_lab = labels[index_liste[clas]]
                if nr+len(add)<max_length:
                    new_data[aktuell:aktuell+len(add)] = add
                    new_labels[aktuell:aktuell+len(add)] = add_lab
                    nr+=len(add)
                    aktuell+=len(add)
                else:
                    missing = max_length-nr
                    new_data[aktuell:aktuell+missing] = add[:missing]
                    new_labels[aktuell:aktuell+missing] = add_lab[:missing]
                    nr+=missing
                    aktuell+=missing
        print("New Shapes: ", new_data.shape, np.array(new_labels).shape)
        return new_data, np.array(new_labels)

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

        acc= total_dict
        for types in total_dict.keys():
            try:
                acc[types] = (int(right_dict[types])/float(total_dict[types]))
            except KeyError:
                acc[types] = 0

        return acc

    @staticmethod
    def accuracy_in_range(out, ground_truth, r):
        assert(len(out)==len(ground_truth))
        res = 0
        for i in range(len(out)):
            if abs(out[i]-ground_truth[i])<r:
                res+=1
        return res/float(len(out))

    @staticmethod
    def balanced_accuracy(out_list, ground_truth_list):
        acc = Tools.accuracy_per_class(out_list, ground_truth_list)
        frequ = np.array(list(acc.values()))
        return np.sum(frequ/float(len(frequ)))

    @staticmethod
    def accuracy(out, ground_truth):
        return np.sum(np.array(ground_truth)==np.array(out))/float(len(out))
