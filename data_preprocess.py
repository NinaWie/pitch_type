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
        self.release_frame = self.cf['pitch_frame_index'].values

        # print("1",self.cf.values.shape)
        # print("2", self.label.shape)

    def remove_small_classes(self, min_class_members):
        types = self.cf["Pitch Type"].values
        note_frequency = sp.stats.itemfreq(types)
        print(note_frequency)
        smaller_min = (note_frequency[np.where(note_frequency[:,1]<min_class_members)])[:,0].flatten() # changed to smaller so fastball are cut off
        self.cf = self.cf.drop(self.cf[self.cf["Pitch Type"]=="Unknown Pitch Type"].index)
        for typ in smaller_min:
            self.cf = self.cf.drop(self.cf[self.cf["Pitch Type"]==typ].index)
        print("Removed because not enought class members: ", smaller_min, "Unknown Pitch Type")
        self.label = self.cf["Pitch Type"].values
        self.release_frame = self.cf['pitch_frame_index'].values

        # print("3",self.cf.values.shape)
        # print("4", self.label.shape)

    def select_movement(self, pitching_position):
        self.cf = self.cf[self.cf["Pitching Position (P)"]==pitching_position]
        self.label = self.cf["Pitch Type"].values
        self.release_frame = self.cf['pitch_frame_index'].values
        print("Selected all rows with Pitching position ", pitching_position)

    def get_coord_arr(self, save_name=None):
        # get all columns of frames because sometimes 140, sometimes more than 160

        begin_cf = self.cf.columns.get_loc("0")
        data_array = self.cf.iloc[:, begin_cf:begin_cf+self.nr_frames].values
        M, N = data_array.shape

        nr_joints = len(eval(data_array[0,0]))
        # print("Shape: ", M, N, nr_joints, 2)
        data = np.zeros((M,N,nr_joints,2))

        # get rid of strings and evaluate to lists
        #c = 0
        #g = 0
        for i in range(M):
            for j in range(N):
                if not pd.isnull(data_array[i,j]):
                    data[i,j]=np.array(eval(data_array[i,j]))
        #            g+=1
                else:
                    data[i,j] = data[i,j-1]
        #            c+=1
        #print("percent of missing values:", c/float(g+c))
        # self.label = self.cf["Pitch Type"].values

        # self.release_frame = self.cf['pitch_frame_index'].values

        if save_name!=None:
            np.save(save_name, data)
            print("Saved with name ", save_name)

        return data

    def get_release_frame(self, mini, maxi):
        releases = self.release_frame
        over_min = releases[np.where(releases>mini)]
        below_max = over_min[np.where(over_min<maxi)]
        mean = np.mean(below_max)
        releases[np.where(np.isnan(releases))] = mean
        releases[np.where(releases<=mini)] = mean
        releases[np.where(releases>=maxi)] = mean
        return releases


    def balance(self):
        weights = compute_class_weight("auto", np.unique(self.cf["Pitch Type"].values),self.cf["Pitch Type"].values )

    def get_labels(self):
        #print("9",self.cf.values.shape)
        #print("10", self.label.shape)
        return self.label

    def set_labels_toWindup(self):
        arr = self.cf["Pitching Position (P)"].values
        new = []
        c=0
        for i, lab in enumerate(arr):
            if lab=="Windup" or lab=="Stretch":
                new.append(lab)
            else:
                self.cf.drop(self.cf.index[[i-c]], inplace = True)
                c+=1
        self.label = np.array(new)

    def concat_with_second(self, file2, save_name=None):
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
        # print("5",self.data.shape)
        # print("6", self.label.shape)

        new = np.delete(data, redundant, axis = 0)
        # self.label = np.delete(self.cf["Pitch Type"].values, redundant, axis = 0)
        # self.release_frame = self.cf['pitch_frame_index'].values
        #print(new.shape)
        # print("7",self.data.shape)
        # print("8", self.label.shape)
        if save_name!=None:
            np.save(save_name, new)
            print("Saved with name ", save_name)

        return new

    def get_list_with_most(self, column):
        pitcher = self.cf[column].values #.astype(int)
        statistic = sp.stats.itemfreq(pitcher) #.sort(axis = 0)
        #if column == "Pitch Type":
        #    print(statistic)
        number = np.array(statistic[:,1])
        a = b = []
        for i in range(5):
            maxi = np.argmax(number)
            a.append(statistic[maxi,0])
            number[maxi]=0
        return a, statistic

    def cut_file_to_pitcher(self, player):
        #print(np.any(self.cf["Pitcher"].values ==))
        self.cf = self.cf[self.cf["Pitcher"].values==player]

    def cut_file_to_listof_pitcher(self, player):
        #print(np.any(self.cf["Pitcher"].values ==))
        ind = []
        for p in player:
            ind += (np.where(self.cf["Pitcher"].values==p)[0]).tolist()
        self.cf = self.cf.iloc[ind]
        self.label = self.cf["Pitch Type"].values
        self.release_frame = self.cf['pitch_frame_index'].values
        # print(self.cf.values.shape)

    def set_labels_toOnePitchtype(self, pitchType):
        self.label = (self.cf["Pitch Type"]==pitchType).values.astype(float)
