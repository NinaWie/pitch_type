import pandas as pd
import numpy as np
import tensorflow as tf
import scipy as sp
from scipy import stats
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
    def normalize01(data):
        maxi = np.amax(data, axis=1)
        mini = np.amin(data, axis = 1)
        res = np.asarray([(data[:,i]-mini)/(maxi-mini+0.000001) for i in range(len(data[0]))])
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
        #classes = {"Fastball (4-seam)":"Fastball", "Fastball (2-seam)": "Fastball", 'Fastball (Cut)': "Fastball", 'Fastball (Split-finger)': "Fastball", "Sinker": "Fastball",
        #'Curveball': "Breaking Ball", "Slider": "Breaking Ball", 'Knuckle curve':"Breaking Ball", 'Knuckleball': "Breaking Ball", "Changeup": "Changeup"}
        classes = {"Fastball (4-seam)":"Fastball", "Fastball (2-seam)": "Fastball", 'Fastball (Cut)': "Fastball", 'Fastball (Split-finger)': "Fastball", "Sinker": "Sinker",
        'Curveball': "Change_curve", "Slider": "Slider", 'Knuckle curve':"Knuckle curve", 'Knuckleball': "Knuckleball", "Changeup": "Change_curve"}
        for uni in np.unique(labels):
            labels[labels==uni] = classes[uni]
        print(labels[:20])
        return labels

    @staticmethod
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
        if len(unique)==1:
            range_nr = unique[0]
            return labels/float(range_nr)
        else:
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
        if len(unique)==1:
            return np.asarray(results*unique[0], dtype = int)
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
        right_frequency = stats.itemfreq(same)
        total_frequency = stats.itemfreq(ground_truth)
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
    def confused_classes(out, labels):
        import operator
        for pitch in np.unique(labels):
            inds = np.where(labels==pitch)[0]
            outs_type = out[inds]
            frequ = stats.itemfreq(outs_type)
            values = np.asarray(frequ[:,1]).astype(int)
            dictionary = dict(zip(frequ[:,0], (values/float(np.sum(values))*100).astype(int)))
            sorted_dic = sorted(dictionary.items(), key=operator.itemgetter(1), reverse=True)
            print(pitch,"percentage predicted as", sorted_dic)

    @staticmethod
    def confusion_matrix(out,labels):
        uni = np.unique(out) if len(np.unique(out))> len(np.unique(labels)) else np.unique(labels)
        data = []
        for i in uni:
            inds = np.where(labels==i)[0]
            outs_type = out[inds]
            # print(i, [np.sum(outs_type==j) for j in uni])
            data.append([np.sum(outs_type==j) for j in uni])
        row_format ="{:>15}" * (len(uni) + 1)
        print(row_format.format("", *uni))
        for lab, row in zip(uni, data):
            print(row_format.format(lab, *row))


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

    @staticmethod
    def do_pca(data, nr_components):
        from sklearn.decomposition import PCA
        M, frames, joi, xy = data.shape
        res = data.reshape(M,frames,joi*xy)
        res = res.reshape(M*frames, joi*xy)
        #print(data[1,3])
        #print(res[170])
        print(res.shape)
        pca = PCA(n_components=nr_components)
        new_data = pca.fit_transform(res)
        # new_data = pca.transform(res)
        print(new_data.shape)
        new = new_data.reshape(M,frames, nr_components, 1)
        return new

    @staticmethod
    def shift_data(data, labels, shift_labels= True, max_shift=30):
        new_data=[]
        for i in range(len(data)):
            if shift_labels:
                bound_shift = min(max_shift, len(data[0])-max_shift-labels[i])
            else:
                bound_shift = max_shift
            # print(bound_shift)
            shift = np.random.randint(-max_shift, max_shift)
            new = np.roll(data[i], shift, axis=0)
            if shift_labels:
                labels[i] = labels[i]+shift-max_shift
            new_data.append(new[max_shift:len(new)-max_shift])
        return np.array(new_data), labels

    @staticmethod
    def flip_x_data(data, x = 0):
        for i in range(len(data)):
            mean = np.mean(data[i, :, :, x])
            flipped = (data[i, :,:,x]-mean)*(-1)
            flipped+=mean
            data[i,:,:,x] = flipped
        return data

    @staticmethod
    def squish_data(data, factor, required_length=100):
        """
        takes data of shape nr_examples, nr_frame, nr_joints, nr_coords and returns the same sequence with every factor frame deleted
        bsp: takes data of length 160 and returns length 100 with every 3rd frame deleted and then some more deleted before and in the end
        """
        nr_ex, l, j, co = data.shape
        del_inds = np.arange(0, l, factor)
        new = np.delete(data, del_inds, axis = 1)
        if len(new[0])>l:
            cut = (len(new[0])-l)//2
            if len(new[0,cut:-cut])!=l:
                return new[:, cut-1:-cut]
            else:
                return new[:,cut:-cut]
        else:
            i = 0
            while len(new[0])<l:
                if i==0:
                    new = np.append(np.reshape(new[:,i], (nr_ex, 1, j, co)), new, axis=1)
                    i=-1
                elif i==-1:
                    new = np.append(new, np.reshape(new[:,i], (nr_ex, 1, j, co)), axis=1)
                    i=0
            return new

    @staticmethod
    def stretch_data(data, factor):
        nr_ex, l, j, co = data.shape
        indices = []
        c = 0
        while len(indices)<l:
            if c%factor!=0:
                indices.append(c)
            c+=1
        inds_new = np.arange(indices[-1]+1)

        # print(indices, inds_new)
        new = np.zeros((nr_ex, len(inds_new), j, co))
        for ex in range(1): #nr_ex):
            for limb in range(j):
                for xy in [0, 1]: # x and y coord dimension
                    values = data[ex, :, limb, xy]
                    new[ex, :, limb, xy] = np.round(np.interp(inds_new, indices, values), 1)
        cut = (len(inds_new)-len(indices))//2
        if len(new[0,cut:-cut])!=l:
            return new[:, cut-1:-cut]
        else:
            return new[:,cut:-cut]

    @staticmethod
    def extend_data_old(joints_array_batter, labels):
        """
        normalizes data between 0 and 1, then varies them by shifting and stretching (exponentiation) data
        forms variations array of possible data changes, then randomly selects 5 of these
        returns an array of 5 times the size of joints_array_batter
        """
        def shift_trajectory(play, label, shift):
            new = np.roll(play, shift, axis = 0)
            for i in range(abs(shift)):
                if shift>0:
                    new[i]=new[shift]
                else:
                    new[-i-1]=new[shift-1]
            return new, label+shift

        shifts = [-15, -12, -10, -7, -5, 5]
        stretch = [0.5, 0.75, 1.25, 1.5]
        variations = []
        for s in shifts:
            for st in stretch:
                variations.append((s, st))

        norm = Tools.normalize01(joints_array_batter).tolist()
        #labels = labels.tolist()
        M, N, j, xy = joints_array_batter.shape
        more_data = np.zeros((M*6, N, j, xy))
        more_data[:M]=np.array(norm)
        ind = M
        for i in range(len(norm)):
            var = np.array(variations)[np.random.permutation(len(variations))[:5]]
            for j in var:
                #print(i,j)
                new_data, new_label = shift_trajectory(np.array(norm[i]**j[1]), labels[i], int(j[0]))
                more_data[ind] = new_data
                labels.append(new_label)
                ind+=1
        return more_data, labels
