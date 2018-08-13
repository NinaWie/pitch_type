import pandas as pd
import numpy as np
import tensorflow as tf
import scipy as sp
import scipy.stats
import json
from os import listdir
import os
import cv2
import ast
import argparse
import matplotlib.pylab as plt

import sys
sys.path.append("..")

from run_thread import Runner
from test_script import test



def boxplots_testing(labels, results):
    """
    plots the labels, the output of my approach, and the error as boxplots
    :param labels: list of n labels for each video
    :param results: list of release frame outputs for each video
    """
    # Delete the rows for which there is no result (can only happen if threshold is used)
    inds = np.where(np.isnan(results))
    labels = np.delete(labels, inds)
    if len(inds[0])>0:
        print("For plotting the results, ", len(inds[0]), " data were deleted because no results were available (corresponds to", len(inds[0])*100/float(len(results)), "%)")
    results = np.delete(results, inds)

    deviation = labels-results
    error = np.absolute(deviation)

    plt.figure(figsize = (20,10))
    plt.subplot(131)
    plt.boxplot(labels, positions= [0], widths=(0.7))
    plt.xticks([0], ["Statcast"], fontsize=15)
    plt.yticks(fontsize=15)
    # plt.title("Ground truth labels")
    plt.ylabel("Frame index", fontsize=15)
    plt.ylim(60,160)
    # plt.show()

    plt.subplot(132)
    plt.boxplot(results, positions= [0], widths=(0.7))
    plt.xticks([0], ["2D image approach output"], fontsize=15)
    plt.yticks(fontsize=15)
    # plt.title("Error of detected release frame")
    plt.ylabel("Frame index", fontsize=15)
    plt.ylim(60,160)
    # plt.show()

    plt.subplot(133)
    plt.boxplot(deviation, positions= [0], widths=(0.7))
    plt.xticks([0], ["Distribution Error"], fontsize=15)
    plt.yticks(fontsize=15)
    # plt.title("Error of detected release frame")
    plt.ylabel("Error (in frames)", fontsize=15)
    plt.tight_layout()
    plt.show()

def get_test_data(vid_path, csv_path, start=0, end=160, image_resize_width = 55, image_resize_height = 55, max_release_frame = 120, min_release_frame = 60):
    csv = pd.read_csv(csv_path)
    labels = []
    data = []
    for files in listdir(vid_path):
        if files[-4:]!=".mp4" or files[0]==".":
            continue

        # get label
        csv_line = csv[csv["Game"]==files[:-4]]
        release_label = csv_line["pitch_frame_index"].values
        # check label
        if release_label<min_release_frame or release_label>max_release_frame or np.isnan(release_label):
            print("label not in realistic range/missing, label:", release_label)
            continue

        # get bounding box for pitcher
        if not os.path.exists(os.path.join(vid_path, files +".dat")):
            print("file skipped because no metadata available:", files)
            continue

        for i in open(os.path.join(vid_path, files +".dat")).readlines():
            datContent=ast.literal_eval(i)
        bottom_p=datContent['Pitcher']['bottom']
        left_p=datContent['Pitcher']['left']# +20
        right_p=datContent['Pitcher']['right']# -20
        top_p=datContent['Pitcher']['top']# +20

        cap = cv2.VideoCapture(os.path.join(vid_path, files))

        # set camera to start frame
        if start is not 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for i in range(end-start):
            # read all relevant frames
            ret,frame = cap.read()
            if frame is None:
                break
            # greyscale image resized (otherwise very high dimensional input)
            pitcher = frame[top_p:bottom_p, left_p:right_p]
            pitcher = cv2.resize(np.mean(pitcher, axis = 2),(image_resize_width, image_resize_height), interpolation = cv2.INTER_LINEAR)/255
            data.append(pitcher)
        # if video was too short, the array length is not a multiple of (end-start) anymore:
        if len(data)%(end-start) is not 0:
            print("video is too short!")
            number_processed= len(data)//(end-start)
            data = data[:number_processed*(end-start)]
            continue
        print(files, release_label, len(data), len(labels))
        # append label
        labels.append(release_label-start)
    return data, labels

def testing(test_dates, restore_path, start = 0, end=160):
    final_results = []
    final_labels = []

    # Because otherwise there is too much data, we process all videos of one folder at a time,
    # and save the results and labels in final_results and final_labels
    for date in test_dates:
        frames = []
        labels = []
        input_dir= os.path.join(path_input, date, "center field/")
        list_files = listdir(input_dir)
        print("start processing videos of date", date)

        # Get data (array of length nr_videos*(end-start))
        frames, labels = get_test_data(input_dir, csv_path, start=start, end = end)
        for lab in labels:
            final_labels.append(lab)

        # make data array
        leng = end-start
        frames = np.array(frames)
        labels = np.array(labels)
        examples, width, height = frames.shape
        data = np.reshape(frames, (examples, width, height, 1))
        print("Data:", data.shape, "number videos", len(data)/leng, "number labels (must be same as number of videos)", len(labels), "labels",labels.tolist())

        # restore model and predict labels for each frame
        lab, out = test(data, restore_path)

        # TWO POSSIBILITIES:
        # - TAKE FRAME WITH HIGHEST OUTPUT
        # - TAKE FIRST FRAME FOR WHICH THE OUTPUT EXCEEDS A THRESHOLD
        def highest_prob(outputs):
            return np.argmax(outputs)
        def first_over_threshold(outputs, thresh=0.8):
            over_thresh = np.where(outputs>thresh)[0]
            if len(over_thresh)==0:
                return np.nan
            else:
                return over_thresh[0]

        for i in range(int(len(data)/leng)):
            highest = highest_prob(out[leng*i:leng*(i+1), 1]) # get always sequence of (end-start) frames
            # highest = first_over_threshold(out[leng*i:leng*(i+1), 1]) # SECOND POSSIBILITY (see above)
            print("real label:", labels[i], "frame index predicted: ", highest)
            final_results.append(highest)

        print("----------------------------")
        print("finished processing for date", date, "now results:", final_results)
    # Evaluation
    boxplots_testing(final_labels, final_results)


def get_train_data(train_dates, video_path, csv_path, image_resize_width = 55, image_resize_height = 55, max_release_frame = 120, min_release_frame = 60):
    """
    Read the videos from the folders train_dates in video_path, and get the label for each first.
    Get the release frame (frame index corresponding to the label) and append to the positive data
    Get three frames between min_release_frame and max_release_frame randomly and append to negative data
    resize all frames and turn into grayscale (otherwise too much data)
    Return the data and corresponding labels (1 for pos and 0 for neg)
    """
    csv = pd.read_csv(csv_path)
    pos_data = []
    neg_data = []
    for date in train_dates:
        print("start loading data for date", date)
        vid_path = os.path.join(video_path, date, "center field")
        # iterate through videos
        for files in listdir(vid_path):
            if files[-4:]!=".mp4" or files[0]==".":
                continue

            # get label
            csv_line = csv[csv["Game"]==files[:-4]]
            release_label = csv_line["pitch_frame_index"].values
            # check label (smaller than min_release_frame or higer than max_release_frame is unrealistic)
            if release_label<min_release_frame or release_label>max_release_frame or np.isnan(release_label):
                continue

            # get bounding box for pitcher
            # skip if no metadata is available:
            if not os.path.exists(os.path.join(vid_path, files +".dat")):
                print("file skipped because no metadata available:", files)
                continue

            for i in open(os.path.join(vid_path, files +".dat")).readlines():
                datContent=ast.literal_eval(i)
            bottom_p=datContent['Pitcher']['bottom'] # optional: make bounding box even smaller
            left_p=datContent['Pitcher']['left'] # +20
            right_p=datContent['Pitcher']['right']# -20
            top_p=datContent['Pitcher']['top']# +20

            cap = cv2.VideoCapture(os.path.join(vid_path, files))

            # positive example: release frame label
            cap.set(cv2.CAP_PROP_POS_FRAMES, release_label)
            ret,frame = cap.read()
            # get ROI
            pitcher = frame[top_p:bottom_p, left_p:right_p]
            # greyscale image resized (otherwise very high dimensional input)
            pitcher = cv2.resize(np.mean(pitcher, axis = 2),(image_resize_width, image_resize_height), interpolation = cv2.INTER_LINEAR)/255
            pos_data.append(pitcher)

            # negative examples: randomly selected 3 frames in a range of frames around ball release
            negative_range = np.delete(np.arange(min_release_frame,max_release_frame,1), release_label-min_release_frame)
            random_frames = np.random.choice(negative_range, 3, replace=False)
            print(files, "release frame:", release_label, "negative example frames:", random_frames)
            for r in random_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, r)
                ret,frame = cap.read()
                pitcher = frame[top_p:bottom_p, left_p:right_p]
                pitcher = cv2.resize(np.mean(pitcher, axis = 2),(image_resize_width, image_resize_height), interpolation = cv2.INTER_LINEAR)/255
                neg_data.append(pitcher)

    # concatenate positive and negative data and create labels
    labels = np.ones((len(pos_data)), dtype=np.int).tolist()+np.zeros((len(neg_data)), dtype=np.int).tolist()
    print("pos", len(pos_data), "neg", len(neg_data), "labels", len(labels))
    data = np.append(np.array(pos_data), np.array(neg_data), axis = 0)
    print("data: ", data.shape)
    examples, width, height = data.shape
    data = np.reshape(data, (examples, width, height, 1))
    return data, np.array(labels)


def training(dates, save_path, video_path, csv_path):
    data, labels = get_train_data(dates, video_path, csv_path)
    runner = Runner(data, labels, SAVE = save_path, BATCH_SZ=40, EPOCHS = 40, batch_nr_in_epoch = 100,
            act = tf.nn.relu, rate_dropout = 0,
            learning_rate = 0.0005, nr_layers = 4, n_hidden = 128, optimizer_type="adam", regularization=0,
            first_conv_filters=128, first_conv_kernel=9, second_conv_filter=128,
            second_conv_kernel=9, first_hidden_dense=128, second_hidden_dense=0,
            network = "adjustable conv2d")
    runner.start()


# HYPER PARAMETERS

## for cluster
# path_input = "/scratch/nvw224/videos/atl" # TEST DATA: Path to input data (videos and region of interest dat files)
# cf_data_path = "/scratch/nvw224/cf_pitcher.csv" # Statcast labels, download from google drive

path_input = os.path.join("..", "train_data", "ATL") # TEST DATA: Path to input data (videos and region of interest dat files)
csv_path = os.path.join("..","train_data", "cf_pitcher.csv")


# THESE DATES WERE USED FOR TRAINING THE SAVED MODEL
train_dates = ["2017-07-16", "2017-05-02", "2017-05-03", "2017-05-07", "2017-05-24", "2017-06-07"] # "2017-05-20", , , "2017-04-15"]# letztes falsch
 # "2017-06-11", "2017-06-19", "2017-06-23", "2017-07-05", "2017-07-17", "2017-04-14", "2017-04-18",
 # "2017-05-02", "2017-05-06", "2017-05-19", "2017-05-23", "2017-06-06", "2017-06-10", "2017-06-18", "2017-06-22", "2017-07-04", ]

test_dates = ['2017-05-22', '2017-05-04'] #, '2017-07-15', '2017-05-01'], '2017-06-09', '2017-07-19',
 #'2017-06-16', '2017-04-16', '2017-05-05', '2017-04-20', '2017-05-18', '2017-06-24', '2017-06-20', '2017-05-25',
 # '2017-05-17',  '2017-06-05', '2017-06-06', '2017-04-17', , '2017-07-18', '2017-07-14','2017-06-08']
# ERLEDIGT: '2017-06-17', '2017-05-21', '2017-06-21'

if __name__ == "__main__":
    def boolean_string(s):
        if s not in {'False', 'True'}:
            raise ValueError('Not a valid boolean string')
        return s == 'True'
    parser = argparse.ArgumentParser(description='Train neural network to classify whether a frame is a ball release frame')
    parser.add_argument('-training', default= "True", type=boolean_string, help='if training, set True, if testing, set False')
    parser.add_argument('-model_save_path', default="../saved_models/release_model", type=str, help='if training, path to save model, it testing, path to restore model')
    args = parser.parse_args()

    train = args.training
    save_path = args.model_save_path

    if train:
        training(train_dates, save_path, path_input, csv_path)
    else:
        print("testing")
        testing(test_dates, save_path)

# OLD VERSIONS:

# np.save("../outputs/release_160_frames_2/each_frame_release_from_images", np.array(each_frame))
# np.save("../outputs/release_160_frames_2/results_release_from_images", np.array(results))
# np.save("../outputs/release_160_frames_2/labels_release_from_images", np.array(final_labels))
# print("saved intermediate", len(final_results), len(final_labels))

# einfuegen in testing:
# for f in list_files:
#     if f[-4:]==".mp4":
#         data, label = get_test_data(input_dir, f)
#         if (data is None) or (label is None):
#             continue
#         for elem in data[start:end]:
#             frames.append(elem)
#         labels.append(label-start)

#
# for i in range(int(len(data)/30)):
#     print("real label:", labels[i])
#     print([round(elem,2) for elem in out[30*i:30*(i+1), 1]])
#     highest = np.argmax(out[30*i:30*(i+1), 1])
#     print("frame index predicted: ", highest)
#     if abs(labels[i]-highest)<2:
#         right+=1
#     results.append(highest)

# print(results)
# np.save("results_release_from_images", np.array(results))
# np.save("labels_release_from_images", np.array(labels))
# print("Accuracy (only 1 frame later or earlier): ", right/float(len(data)/leng))

#np.save("predicted_frame", data[highest])
#np.save("all_frames", data)

# def get_test_data_old(input_dir, f):
#     process = VideoProcessor(path_input=path_input, df_path = cf_data_path)
#     label = process.get_labels(f, "pitch_frame_index")
#     print(input_dir, f, label)
#     if (label is not None) and not np.isnan(label):
#         data = process.get_pitcher_array(input_dir+f)
#         return data[:160], label#[cut_off_min:cut_off_max], label-cut_off_min
#     else:
#         return None, None

# def get_train_data(dates):
#     pos = []
#     neg = []
#     for date in dates:
#         arr = np.load(path_output+"array_videos_"+date+".npy")[:, cut_off_min:cut_off_max, :, :]
#         lab =  np.load(path_output+"labels_release_"+date+".npy")
#         for i, pitch in enumerate(arr):
#             rel = lab[i]
#             if rel>cut_off_min and rel<cut_off_max and not np.isnan(rel):
#                 ind = int(rel)-cut_off_min
#                 pos.append(pitch[ind])
#                 rest = np.delete(pitch, int(rel)-cut_off_min, axis = 0)
#                 indizes = np.random.choice(len(rest), 3, replace = False)
#                 for elem in rest[indizes]:
#                     neg.append(elem)
#             else:
#                 print("false", rel)
#                 #print(pitch.shape, pos[-1].shape, neg[-1].shape)
#     print(np.array(pos).shape, np.array(neg).shape)
#
#     pos = pos[93:]
#
#     labels = np.ones((len(pos)), dtype=np.int).tolist()+np.zeros((len(neg)), dtype=np.int).tolist()
#     print(len(pos), len(neg), len(labels))
#     data = np.append(np.array(pos), np.array(neg), axis = 0)
#     # np.save("positive.npy", np.array(pos))
#     # print("positive saved")
#     #np.save("negative.npy", np.array(neg)[:10])
#     examples, width, height = data.shape
#     data = np.reshape(data, (examples, width, height, 1))
#     print(data.shape)
#     return data, np.array(labels)
