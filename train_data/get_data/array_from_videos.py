import cv2
import ast

import numpy as np
import matplotlib.pylab as plt
from os import listdir
import pandas as pd

import scipy

class VideoProcessor:
    """
    Class to get frames as a numpy array from video files - different functions to use for different tasks (for example used for release_frame_from_images model)
    """

    def __init__(self, path_input="videos/atl", df_path = "cf_data.csv", resize_width = 55, resize_height = 55):
        self.path_input=path_input
        df = pd.read_csv(df_path)
        self.df = df[df["Player"]=="Pitcher"]
        self.resize_width = resize_width
        self.resize_height = resize_height
#e.g. path_input_dat = "/Volumes/Nina Backup/videos/atl/2017-04-14/center field/490251-0f308987-60b4-480c-89b7-60421ab39106.mp4.dat"

# dates = ["2017-04-14", "2017-04-18", "2017-05-02", "2017-05-06"] # , "2017-05-19", "2017-05-23", "2017-06-06", "2017-06-10", "2017-06-18", "2017-06-22", "2017-07-04", "2017-07-16",
# "2017-04-15", "2017-04-19", "2017-05-03", "2017-05-07", "2017-05-20", "2017-05-24", "2017-06-07", "2017-06-11", "2017-06-19", "2017-06-23", "2017-07-05", "2017-07-17"]
# only first two rows von den im cluster angezeigten
# output_folder=args.output_dir

    def get_arrays_for_dates(self, dates, save_path = None):
        for date in dates:
            video = []
            label_pitchtype = []
            label_release = []
            input_dir= self.path_input+"/"+date+"/center field/"
            list_files = listdir(input_dir)
            print(date)
            for f in list_files:
                if f[-4:]==".mp4":
                    pitchtype = get_labels(f, "Pitch Type", "cf_data")
                    release = get_labels(f, "pitch_frame_index", "cf_data")
                    if pitchtype is not None and release is not None:
                        label_pitchtype.append(pitchtype)
                        label_release.append(release)
                        video.append(get_pitcher_array(input_dir, f))
            if save_path is not None:
                np.save(save_path+"video"+date+".npy", np.array(video))
                np.save(save_path+"label_pitchtype"+date+".npy", np.array(label_pitchtype))
                np.save(save_path+"label_release"+date+".npy", np.array(label_release))
                print("arrays saved on path ", save_path, date)

    def get_labels(self, f, column):
        game_id = f[:-4]
        print(f, game_id)
        line = self.df[self.df["Game"]==game_id]
        #print(labels)
        if len(line[column].values)!=1:
            print("PROBLEM: NO LABEL/ TOO MANY")
            print(line[column].values)
            return None
        else:
            return line[column].values[0]

    def get_pitcher_array(self, filepath):
        video_capture = cv2.VideoCapture(filepath)
        for i in open(filepath+".dat").readlines():
            datContent=ast.literal_eval(i)
        bottom_p=datContent['Pitcher']['bottom']
        left_p=datContent['Pitcher']['left']
        right_p=datContent['Pitcher']['right']
        top_p=datContent['Pitcher']['top']
        frames = np.zeros((167, self.resize_width, self.resize_height))
        i = 0
        while True:
            ret, frame = video_capture.read()
            if frame is None:
                break
            pitcher = frame[top_p:bottom_p, left_p:right_p]
            pitcher = cv2.resize(np.mean(pitcher, axis = 2),(self.resize_width, self.resize_height), interpolation = cv2.INTER_LINEAR)/255
            frames[i]= pitcher
            i+=1
        return frames

    def get_pitcherAndBatter_array(self, filepath):
        video_capture = cv2.VideoCapture(filepath)
        for i in open(filepath+".dat").readlines():
            datContent=ast.literal_eval(i)
        bottom_p=datContent['Pitcher']['bottom']
        left_p=datContent['Pitcher']['left']
        right_p=datContent['Pitcher']['right']
        top_p=datContent['Pitcher']['top']
        bottom_b=datContent['Batter']['bottom']
        left_b=datContent['Batter']['left']
        right_b=datContent['Batter']['right']
        top_b=datContent['Batter']['top']
        #center_dic['Pitcher']=np.array([abs(top_p-bottom_p)/2., abs(left_p-right_p)/2.])
        #center_dic['Batter']=np.array([abs(top_b-bottom_b)/2., abs(left_b-right_b)/2.])
        frames_pitcher = np.zeros((167, self.resize_width, self.resize_height))
        frames_batter = np.zeros((167, self.resize_width, self.resize_height))
        i = 0
        for i in range(167):
            ret, frame = video_capture.read()
            if frame is None:
                frames_pitcher[i] = frames_pitcher[i-1]
                frames_batter[i] = frames_batter[i-1]
                continue
            pitcher = frame[top_p:bottom_p, left_p:right_p]
            frames_pitcher[i] = cv2.resize(np.mean(pitcher, axis = 2),(self.resize_width, self.resize_height), interpolation = cv2.INTER_LINEAR)/255 #scipy.misc.imresize(pitcher, (109, 143, 3))/255
            batter = frame[top_b:bottom_b, left_b:right_b]
            frames_batter[i] = cv2.resize(np.mean(batter, axis = 2),(self.resize_width, self.resize_height), interpolation = cv2.INTER_LINEAR)/255
            i+=1
        return frames_pitcher, frames_batter
