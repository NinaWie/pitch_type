import numpy as np
import os
import sys
from os import listdir
import time
import cv2
import json

import math

# Load ball_detection code from the parent directory (because it is used for object detection and event detection)
import sys
sys.path.append("..")
from fmo_detection import detect_ball, plot_trajectory, from_json

# batter_videos = "/Volumes/Nina Backup/high_quality_testing/batter/"
batter_videos = os.path.join("..", "train_data", "high_quality_videos", "batter")
# OUT_PATH = "/Volumes/Nina Backup/outputs/"
OUT_PATH = "outputs"

def areOverlapping(recA, recB, OVERLAPPING_THRESH):
    """
    tests if two FMO candidates are overlapping
    """
    if (recA[1][0]+OVERLAPPING_THRESH<recB[0][0] or recB[1][0]+OVERLAPPING_THRESH<recA[0][0] or
        recA[1][1]+OVERLAPPING_THRESH<recB[0][1] or recB[1][1]+OVERLAPPING_THRESH<recA[0][1]):
        return False
    else:
        return True

def combineOverlapping(cands, OVERLAPPING_THRESH):
    """
    merge two candidates if they are overlapping
    """
    for i in range(len(cands)):
        recA = cands[i]
        for j in range(len(cands)):
            recB = cands[j]
            if i != j:
                if areOverlapping(recA,recB, OVERLAPPING_THRESH):
                    recComb = ((min(recA[0][0],recB[0][0]), min(recA[0][1],recB[0][1])),(max(recA[1][0],recB[1][0]),max(recA[1][1],recB[1][1])))
                    cands[i] = recComb
                    cands[j] = recComb
    return cands

def closestBox(cands, batLoc, threshold_max_dist):
    """
    find closest candidate dependent on the last bat detection batLoc
    """
    minInd = -1
    for i in range(len(cands)):
        candBat = np.array(cands[i][:2])
        candDist = np.linalg.norm((batLoc[0]+batLoc[1])/2 - (candBat[0]+candBat[1])/2)
        # print("distance", candDist)
        if candDist < threshold_max_dist:
            # candidate closer to threshold found, keep on searching for closer ones
            threshold_max_dist = candDist
            minInd = i
    if minInd != -1:
        return np.array(cands[minInd][:2])
    else:
        return np.zeros([2,2])   # bat is set as missing values if no candidate is sufficiently close

# Lists for saving the detection rates
glove_results = []
bat_swing_results = []
bat_before_swing_results = []
fmoc_swing_results = []

for vid in os.listdir(batter_videos):
    if vid[0]==".":
        continue

    NAME = vid[:-4]
    print("-------", NAME, "-------------------")


    # PATHS
    INPUT_VIDEO_PATH = os.path.join(batter_videos, NAME+".mp4")
    # Specify swing frames to get the detection rate of faster R-CNN and FMO-C for the relevant frames
    with open("swing_frames_bat.json", "r") as infile:
        swing_frames = json.load(infile)
    SWING_START = swing_frames[NAME][0]
    SWING_END = swing_frames[NAME][1]
    print("Start swing:", SWING_START, ", End swing", SWING_END)
    # Get number of total frames
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    TOTAL_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Total numeber of frames in video:", TOTAL_FRAMES)
    # Joints needed for wrist etc
    PATH_JOINTS = os.path.join("..","train_data","batter_hq_joints", NAME+".json")


    # ### Load json file with joints of target player

    coords = from_json(PATH_JOINTS)
    wristTracker = np.mean(coords[:, [2,5]], axis=1)
    print("Loaded wrist coordinates (shape: number of frames times number of coordinates (x and y)):", wristTracker.shape)

    # ### Load json file with FMO outputs

    with open(os.path.join(OUT_PATH, NAME+ "_fmoc.json"), "r") as infile:
        candidates_per_frame = json.load(infile)

    print("Loaded FMO-C output")

    # ### Load json file with Faster R-CNN outputs

    with open(os.path.join(OUT_PATH, NAME+ "_fasterrcnn.json"), "r") as infile:
        new_dic = json.load(infile)
    batBox_new = []
    gloveBox_new=[]
    keys = sorted(new_dic.keys())
    # last frame in which glove or bat were detected
    highest_key = int(keys[-1])

    # make arrays from bounding boxes for each frame
    for i in range(highest_key):
        dic = new_dic[str(i+1).zfill(4)]
        try:
            glove = dic["glove"]
            aabb = np.array(glove["box"]).astype(int)
            gloveBox_new.append([np.array([[aabb[1], aabb[0]], [aabb[3], aabb[2]]]), i])
        except KeyError:
            pass
        try:
            bat = dic["bat"]
            aabb = np.array(bat["box"]).astype(int)
            batBox_new.append([np.array([[aabb[1], aabb[0]], [aabb[3], aabb[2]]]), i]) # auf i ändern!!!
        except KeyError:
            pass

    batBox = np.array(batBox_new)
    gloveBox = np.array(gloveBox_new)

    print("Loaded Faster R-CNN output")

    # EVALUATION

    # Evaluate glove on all frames
    if len(gloveBox)!=0:
        frames_detected = gloveBox[:,1]
        detection_rate_glove = len(frames_detected)/float(TOTAL_FRAMES)
        print("Faster R-CNN detected the GLOVE in ", detection_rate_glove*100, "% of ALL frames")
    else:
        detection_rate_glove = 0
        print("Glove detected in 0%")

    glove_results.append(detection_rate_glove)

    frames_detected = batBox[:,1]
    frames_before_end = frames_detected[frames_detected<SWING_END]
    frames_after_start = frames_before_end[frames_before_end>SWING_START]
    detection_rate_bat = len(frames_after_start)/float(SWING_END- SWING_START) # //every_x_frame)
    print("Faster R-CNN detected the BAT in ", detection_rate_bat*100, "% of the swing frames")
    frames_before_start = frames_detected[frames_detected<SWING_START]
    detection_rate_bat_before_swing = len(frames_before_start)/float(SWING_START) # //every_x_frame
    print("Faster R-CNN detected the BAT in ", detection_rate_bat_before_swing*100, "% of the frames before the swing")

    bat_swing_results.append(detection_rate_bat)
    bat_before_swing_results.append(detection_rate_bat_before_swing)

    ### THRESHOLDS for merging: 2 times the bat length of the first bat detection ###

    # Take candidate as the new bat if it is less than MAX_DIST_THRESH away from the previous bat detection
    # --> to have a resolution independent threshold, two times the length of a bat serves as the threshold
    MAX_DIST_THRESH = 2*max(np.linalg.norm(batBox[i,0][0]-batBox[i,0][1]) for i in range(len(batBox)))

    # Merge two candidates if they are very close to each other, because sometimes the bat is split into two parts
    # Here, also a threshold dependent on the bat length is taken
    OVERLAPPING_THRESH = 0.05*MAX_DIST_THRESH

    # MERGE FMO-C AND FASTER R-CNN

    frame_count = 0
    currentBat = batBox[0,0]
    onlyBat = batBox.tolist()
    frames_detected_fmo = []
    for i, cand_list in enumerate(candidates_per_frame):
        # check if bat was already detected by Faster R-CNN
        inds_bat = np.where(batBox[:,1]==i)[0]
        if len(inds_bat)>0:
            currentBat = batBox[inds_bat[0], 0]
        # if not detected by faster R-CNN, find closest motion candidates
        else:
            if len(cand_list)>1:
                cand_list = combineOverlapping(cand_list, OVERLAPPING_THRESH)

            if len(cand_list)>0:
                # print("candidates", cand_list, "currentBat", currentBat)
                newBox = closestBox(cand_list, currentBat, MAX_DIST_THRESH)
                # print(newBox)
                if newBox.any():
                    frames_detected_fmo.append(i)
                    # print("found fmoc detection", newBox)
                    onlyBat.append([np.array(newBox), int(frame_count)])
                    currentBat = newBox

        frame_count+=1

    onlyBat = np.array(onlyBat)
    combBat = onlyBat[onlyBat[:,1].argsort()] # sorted frames in combined array


    # EVALUATION FMO-C:

    fasterRcnn=0
    fmoc=0
    undetected_frames=[]
    for i in range(SWING_START, SWING_END,1):
        if i in frames_after_start:
            fasterRcnn+=1 # already detected by faster RCNN
        if i in frames_detected_fmo:
            fmoc+=1
        else:
            undetected_frames.append(i) # not detected at all

    print("Bat not detected at all (during swing):", undetected_frames)
    detection_rate_bat_fmo = fmoc/float(SWING_END-SWING_START-fasterRcnn)# //every_x_frame)
    # print("frames during swing detected by FMO-C:", frames_only_fmo)
    print("FMO-C detected the BAT in ", detection_rate_bat_fmo*100, "% of the frames during the swing in which faster R-CNN did not detect the bat")
    fmoc_swing_results.append(detection_rate_bat_fmo)

print("\n \n \n")
print("-------------------- MEAN DETECTION RATES -----------------------")
print("GLOVE DETECTION:", np.mean(glove_results))
print("BAT DETECTION DURING SWING", np.mean(bat_swing_results))
print("BAT DETECTION BEFORE SWING", np.mean(bat_before_swing_results))
print("FMOC SWING", np.mean(fmoc_swing_results))
print("TOGETHER DETECTION RATE DURING SWING:", np.mean(fmoc_swing_results)+  np.mean(bat_swing_results))
