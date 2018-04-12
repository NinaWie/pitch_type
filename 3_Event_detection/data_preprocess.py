import pandas as pd
import numpy as np
import tensorflow as tf
import scipy as sp
import scipy.stats
from os import listdir
import json

def get_data_from_csv(cf, label, min_length=160):
    print("number of different games", len(np.unique(cf["Game"].values)), "length of csv file", len(cf.index))
    data = []
    labels = []
    for i in cf.index:
        d = np.array(eval(cf.loc[i]["Data"]))
        if len(d)<min_length or pd.isnull(cf.loc[i][label]):
            # print("too short or wrong label", cf.loc[i][label])
            continue
        data.append(d[:min_length])
        labels.append(cf.loc[i][label])
    data = np.array(data)
    labels = np.array(labels)
    unique = np.unique(labels)
    print(data.shape, labels.shape, unique)
    return data, labels

def cut_csv_to_pitchers(cf):
    def get_list_with_most(cf, column):
        pitcher = cf[column].values #.astype(int)
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

    player, _ = get_list_with_most(cf, "Pitcher")
    print(player)
    ind = []
    for p in player:
        ind += (np.where(cf["Pitcher"].values==p)[0]).tolist()
    return cf.iloc[ind]

class JsonProcessor:
    #def __init__(self):

    def from_json(self, file):
        coordinates = ["x", "y"]
        joints_list = ["right_shoulder", "right_elbow", "right_wrist", "left_shoulder","left_elbow", "left_wrist",
                "right_hip", "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle",
                "right_eye", "right_ear","left_eye", "left_ear", "nose ", "neck"]

        with open(file, 'r') as inf:
            out = json.load(inf)

        liste = []
        for fr in out["frames"]:
            l_joints = []
            for j in joints_list[:12]:
                l_coo = []
                for xy in coordinates:
                    l_coo.append(fr[j][xy])
                l_joints.append(l_coo)
            liste.append(l_joints)

        return np.array(liste)

    def get_label_release(self, play_list, csv_path, label_name, cut_off_min=70, cut_off_max=110):

        df = pd.read_csv(csv_path)
        game = df["play_id"].values.tolist()
        event = df[label_name].values.tolist()
        label = []

        # Get release frame data for new videos (saved in dictionary instead of single dat files)
        with open("Pose_Estimation/release_frame_new_cf.json", "r") as infile:
            release_frame_json = json.load(infile)
        with open("Pose_Estimation/release_frame_boston.json", "r") as infile:
            release_frame_boston = json.load(infile)
        for play in play_list:
            try:
                label_event = release_frame_json[play]
            except KeyError:
                try:
                    label_event = release_frame_boston[play]
                    print("boston file funktioniert")
                except KeyError:
                # print("play not in release frame dic")
                    try:
                        game_index = game.index(play)
                        if event[game_index]<cut_off_min or event[game_index]>cut_off_max or np.isnan(event[game_index]):
                            print("wrong label")
                            label_event=0
                        else:
                            label_event = event[game_index]
                    except ValueError:
                        print("play neither in csv not in json dic", play)
                        label_event = 0
            label.append(label_event)
        return np.array(label)

    def lab_from_csv(self, out, label_name):
        if pd.isnull(out):
            return "Unknown"
        if label_name=="Pitch Type" or label_name=="Pitching Position (P)":
            if out=="Unknown Pitch Type" or out=="Eephus":
                return "Unknown"
            else:
                return out
        elif label_name=="Play Outcome":
            if "Foul" in out or "Swinging strike" in out:
                return "hit"
            elif "Ball/Pitcher" in out or "Called strike" in out:
                return "nothing"
            elif "Hit into play" in out:
                return "run"
            else:
                print("invalid label: ", out)
                return "Unknown"
        else: return "Unknown"

    def get_label(self, play_list, csvs, label_name):
        game=[]
        event=[]
        for file_ in csvs:
            try:
                df = pd.read_csv(file_)
                # df = df[df["Player"]=="Pitcher"]
            except pd.io.common.CParserError:
                df = pd.read_csv(file_, delimiter=";")
            game.append(df["play_id"].values.tolist())
            event.append(df[label_name].values.tolist())
        label = []
        #print(game[0][:20], game[1][:20])
        #print(event[0][:20], event[1][:20])
        for play in play_list:
            for i, g in enumerate(game):
                #print(play)
                try:
                    game_index = g.index(play)
                    lab = event[i][game_index]
                    label.append(self.lab_from_csv(lab, label_name))
                    #print(label[-1])
                    break
                except ValueError:
                    if i== (len(game)-1):
                        print("label not in any csv")
                        label.append("Unknown")
                    else: continue
        return label

    def get_data_concat(self, dir_cf, dir_sv, sequ_len=160, player="pitcher"):
        dir_sv_list = listdir(dir_sv)
        dir_cf_list = listdir(dir_cf)
        play_list=[]
        data=[]
        cut = len(list("_"+player+".json"))
        for f in dir_cf_list:
            #print(player in f, f[-4:]=="json", f in dir_sv_list)
            if player in f and f[-4:]=="json" and f in dir_sv_list:
                data_cf = self.from_json(dir_cf+f)[:sequ_len, :12]
                data_sv = self.from_json(dir_sv+f)[:sequ_len, :12]
                if len(data_sv)<sequ_len or len(data_cf)<sequ_len:
                    print("too short")
                    continue
                data.append(np.append(data_cf, data_sv, axis=2))
                play_list.append(f[7:-cut])
        return np.array(data), play_list

    def get_test_data(self, inp_dir, test_data_path,  sequ_len, start, shift = 60, labels=None):
        #sequ_len=150
        data = []

        filenames = []
        for files in listdir(inp_dir):
            name = files.split(".")[0]
            if name+".mp4" in listdir(test_data_path): # and ("Ryan" in name or "Chavez" in name):
                array = self.from_json(inp_dir+files)
                if labels is None:
                    if len(array)>start+sequ_len:
                        data.append(array[start:start+sequ_len])
                        filenames.append(name)
                else:
                    real = labels[name]
                    if len(array)>real+sequ_len-shift and real>shift:
                        data.append(array[real-shift:real+sequ_len-shift])
                        filenames.append(name)
        return np.array(data), filenames

    def get_data(self, inp_dir, sequ_len, player="pitcher"):
        data = []
        play_list = []
        for view in range(len(inp_dir)):
            files_list =[]
            for path in inp_dir[view]:
                processed = listdir(path)
                for files in processed:
                    if player in files:

                        if "new" in path:
                            play = files.split("_")[1] #files.split("_")[0]+ "_" +
                        else:
                            play = files.split("_")[0][7:]
                        if play in files_list:
                            continue

                        array = self.from_json(path+files)[:sequ_len, :12]
                        if len(array)<sequ_len:
                            print("too short", files)
                            continue
                        files_list.append(play)
                        play_list.append(play)
                        data.append(array)
                print("after path:", len(data))
        print(np.array(data).shape, len(play_list))
        return np.array(data), play_list


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
                        #print(data_sv[ind_sv, j])
                        try:
                            data[i,j,:,2:]=np.array(eval(data_sv[ind_sv, j]))
                        except:
                            print(data_sv[ind_sv, j])
                            data[i,j,:,2:] = data[i,j-1, :,2:]
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

    def get_release_frame(self, mini, maxi):
        releases = self.release_frame
        over_min = releases[np.where(releases>mini)]
        below_max = over_min[np.where(over_min<maxi)]
        mean = np.mean(below_max)
        releases[np.where(np.isnan(releases))] = mean
        releases[np.where(releases<=mini)] = mean
        releases[np.where(releases>=maxi)] = mean
        return releases

    @staticmethod
    def clear_csv(file_path):
        df = pd.read_csv(file_path)
        for col in df.columns.tolist():
            zeros = pd.isnull(df[col].values)
            nr_null = np.sum(zeros)
            #print(nr_null)
            if nr_null>200:
                df.drop(col, 1, inplace = True)
        print(df.columns.tolist())
        df.to_csv(filepath[:-4]+"_cut.csv")


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
