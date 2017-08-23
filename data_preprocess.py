import pandas as pd
import numpy as np
import tensorflow as tf
import scipy as sp
import scipy.stats

class Preprocessor:
# COORDINATES
    def __init__(self, path, min_class_members):
        self.cf = pd.read_csv(path)
        print("csv eingelesen with length ", len(self.cf.values))

        self.cf = self.cf.loc[self.cf["Player"] == "Pitcher"]
        print("Only Pitcher rows")

        types = self.cf["Pitch Type"].values
        note_frequency = sp.stats.itemfreq(types)
        print(note_frequency)
        smaller_min = (note_frequency[np.where(note_frequency[:,1]<min_class_members)])[:,0].flatten()

        for typ in smaller_min:
            self.cf = self.cf.drop(self.cf[self.cf["Pitch Type"]==typ].index)

        #print("csv file pitch von 2000", (self.cf["Pitch Type"].values)[2000])
        #print("csv file coord frame 140 von 200", (self.cf["140"].values)[2000])

        print("Removed because not enought class members: ", smaller_min)


    def get_coord_arr(self, save_name = None):
        # get all columns of frames because sometimes 140, sometimes more than 160
        pointer = self.cf.columns.get_loc("0")
        columns = self.cf.columns.tolist()
        start = pointer
        while(True):
            try:
                zahl = int(columns[pointer])
                pointer+=1
            except ValueError:
                break

        coordinates = self.cf.iloc[:, start:pointer]

        data_array = coordinates.values
        M, N = data_array.shape

        nr_joints = len(eval(data_array[0,0]))
        print("Shape: ", M, N, nr_joints, 2)
        data = np.zeros((M,N,nr_joints,2))

        # get rid of strings and evaluate to lists
        for i in range(M):
            for j in range(N):
                if not pd.isnull(data_array[i,j]):
                    data[i,j]=np.array(eval(data_array[i,j]))
                else:
                    data[i,j] = data[i,j-1]

        # normalization along frame axis
        M,N, nr_joints,_ = data.shape
        means = np.mean(data, axis = 1)
        std = np.std(data, axis = 1)
        res = np.asarray([(data[:,i]-means)/(std+0.000001) for i in range(len(data[0]))])
        data_new = np.swapaxes(res, 0,1)

        # save
        if save_name!=None:
            np.save(save_name, data_new)

        return data_new

    def balance(self):
        weights = compute_class_weight("auto", np.unique(self.cf["Pitch Type"].values),self.cf["Pitch Type"].values )


    def get_labels_onehot(self, column):
        dataframe = self.cf
        unique  = np.unique(dataframe[column].values)
        l = len(dataframe.index)
        loc = dataframe.columns.get_loc(column)
        labels = np.zeros((l, len(unique)))
        for i in range(l):
            #print(cf.iloc[i,loc])
            pitch = dataframe.iloc[i,loc]
            ind = unique.tolist().index(pitch)
            labels[i, ind] = 1
        return labels, unique

    def get_labels(self, column):
        return self.cf[column].values
