import pandas as pd
import numpy as np
import tensorflow as tf
import scipy as sp
import scipy.stats
import json
from os import listdir
import cv2
import ast
from run_events import Runner
from test import test
# import matplotlib.pylab as plt
#from notebooks.code_to_json import from_json

from data_preprocess import JsonProcessor
from tools import Tools
from test import test
# path_outputs = "/Volumes/Nina Backup/finished_outputs/"
# test_json_files = "/Volumes/Nina Backup/high_quality_outputs/"
# test_data_path = "/Users/ninawiedemann/Desktop/UNI/Praktikum/high_quality_testing/pitcher/"
# save =  "/Users/ninawiedemann/Desktop/UNI/Praktikum/ALL/saved_models/pitch_type_svcf"

### BATTER/PITCHER FIRST MOVEMENT
def first_move_gradient(frames_joints_array, relevant_joints_list = range(0,12,1), cutoff=4, relevant_coordinate=0, minimum_sequ_len=1):
    """
    Returns frame index of batter's first step
    By looking where the mean gradient of the joints in relevant_joints_list is >cutoff
    Works if camera is positioned such that the batter is running to the left or right

    frames_joint_array must have size (nr_frames * nr_joints* nr_coordinates))
    relevant_joints_list: the joints which should be taken into account for the movement
    cutoff: threshold for gradient
    relevant_coordinate: set to the index corresponding to the x coordinate (left-right)
    minimum_sequ_len: if several possible points, take the beginning of a sequence with a defined minimum length
    """
    while True:
        gradients = np.array([np.gradient(frames_joints_array[:,j, relevant_coordinate], edge_order = 2)
                              for j in relevant_joints_list])
        mean_gradient = np.median(gradients, axis=0)
        move = np.where(np.absolute(mean_gradient)>cutoff)[0]
        if cutoff< 0.4: # cannot be found at all
            return None
            break
        if len(move)!=0: # found one
            break
        cutoff-= 0.3 # if nothing found, lower the threshold

    if len(move)==1 or minimum_sequ_len==0:
        return move[0]
    while len(move)>minimum_sequ_len: # find sequence of length> minimum_sequ_len
        for i in range(minimum_sequ_len):
            if move[i]!=move[i+1]-1:
                move = np.delete(move, 0)
                break
            elif i==minimum_sequ_len-1:
                return move[0]
    return move[0]

### BATTER MOVEMENT
def foot_to_ground(batter, release = 90, start_run = None, relevant_joints = [7,10, 8,11], relevant_coordinate = 1):
    """
    returns frame the batter's leg is highest before the swing, and then the moment the foot is back on the ground
    (back to ground is a maximum of 10 frames later than the leg highest)
    for this analysis, an estimaet for the release frame is required, plus the moment the batter starts to run is helpful
    (if not given, release_frame+30 is taken)

    batter: array of nr_frames*nr_joints*nr_coordinates
    release:
    relevant_joints_list: the joints which should be taken into account for the movement
    relevant_coordinate: set to the index corresponding to the x coordinate (left-right)

    OTHER VERSION:
    idea: take only left or right foot, depending which one is lifted more (problem: position of foot in beginning)
    in arguments: relevant_joints = [[7,10],[8,11]]

    mean_right = np.mean(batter[:, relevant_joints[0], 1], axis=1)
    mean_left = np.mean(batter[:, relevant_joints[1], 1], axis=1)
    print(mean_right.shape)
    means = np.array([mean_right, mean_left])
    print(means.shape)
    means_means = np.mean(means[:, :release-20], axis = 1)
    start = release-20
    left_or_right = np.argmin(np.amin(means[:, start:first_move] - np.swapaxes(np.array([means_means for _ in range(50)]), 0,1), axis = 1))
    print("left_right", left_or_right)
    print(np.amin(means, axis = 1).shape)
    print("minimums", np.amin(means, axis = 1))
    foot_up = start + np.argmin(means[left_or_right, start:first_move])
    print("foot_up", foot_up)
    mean_ground = np.mean(means[left_or_right, :foot_up-10])
    print("mean", mean_ground)

    #foot_down_gradient = first_move_batter_gradient(batter[:first_move],  relevant_joints_list=relevant_joints, relevant_coordinate=1, cutoff=4, minimum_sequ_len=1)
    foot_down_gradient = foot_up + np.argmin(np.absolute(means[left_or_right, foot_up:foot_up+10]-mean_ground))# foot_up - 2+ np.where(np.gradient(target_sequence)[foot_up-70+2:]<0.1)[0][0]

    plt.plot(means[left_or_right, foot_up:foot_up+10])
    plt.plot([mean_ground for _ in range(10)])
    plt.show()
    """
    if start_run is None:
        start_run = release+30
    leg_sequence = np.mean(batter[:, relevant_joints, relevant_coordinate], axis=1)
    start = release-20
    foot_up = start + np.argmin(leg_sequence[start:start_run])
    if foot_up>start_run-5:
        print("Too close to first step, not possible")
        return None, None
    mean_ground = np.mean(leg_sequence[:foot_up-10])
    foot_down_gradient = foot_up + np.argmin(np.absolute(leg_sequence[foot_up:foot_up+10]-mean_ground))
    print("in function: first step", start_run, "foot highest", foot_up, "foot down", foot_down_gradient)
    return foot_up, foot_down_gradient

def first_move_batter_NN(joints_array_batter, release_frames, model = "saved_models/batter_first_rnn_10_40"):
    """
    Neural network method: takes an array of some joint trajectories data,
    cuts it to length 32, starting from 10 frames after the relase frame,
    returns predicted first movement frame index

    joints_array_batter: list or array of size nr_data, nr_frames, nr_joints, nr_cordinates
    (should be smoothed and interpolated) - can be list because different data can have different nr_frames
    release frames: array of size nr_data, required to cut array at the right spot
    """
    start_after_release = int(model.split("_")[-2])
    sequence_length = int(model.split("_")[-1])
    print(start_after_release, sequence_length)
    data = []
    for i, d in enumerate(joints_array_batter):
        cutoff_min = release_frames[i]+ start_after_release
        cutoff_max = cutoff_min+sequence_length
        data.append(d[cutoff_min:cutoff_max, 6:12])
    data = Tools.normalize01(np.array(data))
    lab, out = test(data, model)
    labels = np.asarray(lab.reshape(-1)) + np.asarray(release_frames) + start_after_release
    return labels

# PITCHER BALL RELEASE FRAME

def release_frame_conv_net(joints_array_pitcher, model = "saved_models/release_frame_minmax"):
    """
    returns the release frame of a pitcher's joint trajectory
    joints_array_batter: array of size 100(nr_frames)*nr_joints*nr_coordinates
    (conv network can only handle input of certain size, therefore the joints trajectory array must be cut to length 100, release frame should be expected between frame 40 and 80)
    """
    data = Tools.normalize01(joints_array_pitcher)
    lab, out = test(data, model)
    return lab

def release_frame_2Dfrom_video(video_path, bbox=None, model = "saved_models/release_model"):
    """
    Returns release frame index for a video of any length
     - takes 2D image as input and decides if the position is likely to be a ball release position
     - only works with center field camera
    bbox: [left_p, right_p, top_p, bottom_p] box defining region of interest, if None, whole frame is selected
    """

    tf.reset_default_graph()
    # import tensorflow graph and start session
    saver = tf.train.import_meta_graph(model+'.meta')
    graph = tf.get_default_graph()
    try:
        sess = tf.InteractiveSession()
    except:
        sess = tf.Session()
    print("session started")
    saver.restore(sess, model)
    print("session restored")
    out = tf.get_collection("out")[0]
    unique = tf.get_collection("unique")[0]

    video_capture = cv2.VideoCapture(video_path)
    ret, frame = video_capture.read()
    if bbox is None:
        bbox = [0, len(frame[0]), 0, len(frame)]

    found = False
    # start reading video
    p=0
    release_probability=[0] # [0] because one frame already read
    while True:
        ret, frame = video_capture.read()
        if frame is None:
            print("end of video capture")
            break
        pitcher = frame[bbox[2]:bbox[3], bbox[0]:bbox[1]]
        # resize to input in network
        input_release_frame = cv2.resize(np.mean(pitcher, axis = 2),(55, 55), interpolation = cv2.INTER_LINEAR)/255
        data = np.reshape(input_release_frame, (1, 55, 55, 1))
        if not found:
            out_release_frame = sess.run(out, {"input:0":  data, "training:0": False})
            release_probability.append(out_release_frame[0,1])
            ## Version 2: stop when release frame is found (no waste of power for the rest of the video)
            #if out_release_frame[0,1]>0.1:
                # found = True
                # return p
        p+= 1
    return np.argmax(release_probability), release_probability
