import pandas as pd
import numpy as np
import tensorflow as tf
import scipy as sp
import scipy.stats

class Preprocessor:
# COORDINATES
    def __init__(self, path):
        self.cf = pd.read_csv(path)
        print("csv eingelesen with length ", len(self.cf.values))

        self.cf = self.cf.loc[self.cf["Player"] == "Pitcher"]
        print("Only Pitcher rows")

        # lange an frames herausfinden
        pointer = self.cf.columns.get_loc("0")
        columns = self.cf.columns.tolist()
        start = pointer
        while(True):
            try:
                zahl = int(columns[pointer])
                pointer+=1
            except ValueError:
                break
        self.nr_frames = pointer-start
        #print("csv file pitch von 2000", (self.cf["Pitch Type"].values)[2000])
        #print("csv file coord frame 140 von 200", (self.cf["140"].values)[2000])
        self.label = self.cf["Pitch Type"].values

        print("1",self.cf.values.shape)
        print("2", self.label.shape)

    def remove_small_classes(self, min_class_members):
        types = self.cf["Pitch Type"].values
        note_frequency = sp.stats.itemfreq(types)
        print(note_frequency)
        smaller_min = (note_frequency[np.where(note_frequency[:,1]<min_class_members)])[:,0].flatten()
        for typ in smaller_min:
            self.cf = self.cf.drop(self.cf[self.cf["Pitch Type"]==typ].index)
        print("Removed because not enought class members: ", smaller_min)
        self.label = self.cf["Pitch Type"].values

        print("3",self.cf.values.shape)
        print("4", self.label.shape)

    def get_coord_arr(self):
        # get all columns of frames because sometimes 140, sometimes more than 160

        begin_cf = self.cf.columns.get_loc("0")
        data_array = self.cf.iloc[:, begin_cf:begin_cf+self.nr_frames].values
        M, N = data_array.shape

        nr_joints = len(eval(data_array[0,0]))
        print("Shape: ", M, N, nr_joints, 2)
        data = np.zeros((M,N,nr_joints,2))

        # get rid of strings and evaluate to lists
        c = 0
        g = 0
        for i in range(M):
            for j in range(N):
                if not pd.isnull(data_array[i,j]):
                    data[i,j]=np.array(eval(data_array[i,j]))
                    g+=1
                else:
                    data[i,j] = data[i,j-1]
                    c+=1
        print("percent of missing values:", c/float(g+c))

        # normalization along frame axi
        return data

    def balance(self):
        weights = compute_class_weight("auto", np.unique(self.cf["Pitch Type"].values),self.cf["Pitch Type"].values )

    def get_labels(self):
        #print("9",self.cf.values.shape)
        #print("10", self.label.shape)
        return self.label

    def concat_with_second(self, file2):
        sv = pd.read_csv(file2)
        sv = sv[sv["Player"]=="Pitcher"]
        #for typ in self.smaller_min:
        #    sv = sv.drop(sv[sv["Pitch Type"]==typ].index)

        cf_plays = self.cf['play_id'].values
        sv_plays = sv["play_id"].values

        redundant = []
        begin_sv = sv.columns.get_loc("0")
        begin_cf = self.cf.columns.get_loc("0")
        data_cf = self.cf.iloc[:, begin_cf:begin_cf+self.nr_frames].values
        data_sv = sv.iloc[:, begin_sv:begin_sv+self.nr_frames].values
        nr_joints = len(eval(data_cf[0,0]))
        M,N = data_cf.shape
        data = np.zeros((M,N,nr_joints,4))
        for i in range(M):
            if cf_plays[i] in sv_plays:
                ind_sv = np.where(sv_plays==cf_plays[i])[0][0]
                for j in range(N):
                    if not pd.isnull(data_cf[i,j]):
                        data[i,j,:,:2]=np.array(eval(data_cf[i,j]))
                    else:
                        data[i,j,:,:2] = data[i,j-1, :,:2]
                    if not pd.isnull(data_sv[ind_sv,j]):
                        data[i,j,:,2:]=np.array(eval(data_sv[ind_sv, j]))
                    else:
                        data[i,j,:,2:] = data[i,j-1, :,2:]
            else:
                redundant.append(i)
        # print(data.shape)
        # print(len(redundant))
        print("5",self.data.shape)
        print("6", self.label.shape)

        new = np.delete(data, redundant, axis = 0)
        self.label = np.delete(self.cf["Pitch Type"].values, redundant, axis = 0)
        #print(new.shape)
        print("7",self.data.shape)
        print("8", self.label.shape)

        return new

    def get_list_with_most(self, column):
        pitcher = self.cf[column].values #.astype(int)
        statistic = sp.stats.itemfreq(pitcher) #.sort(axis = 0)
        if column == "Pitch Type":
            print(statistic)
        number = np.array(statistic[:,1])
        a = b = []
        for i in range(5):
            maxi = np.argmax(number)
            a.append(statistic[maxi,0])
            number[maxi]=0
        return a

    def cut_file_to_pitcher(self, player):
        #print(np.any(self.cf["Pitcher"].values ==))
        self.cf = self.cf[self.cf["Pitcher"].values==player]

    def set_labels(self, pitchType):
        self.label = (self.cf["Pitch Type"]==pitchType).values.astype(float)
