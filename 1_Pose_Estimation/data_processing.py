"""
to run pose estimation for one video

performance: ca 60 sec runtime, of which 2 seconds are the processing, file saving and also including the first frame, and the rest is ALL runtime required by the handle_one function for pose Estimation

For reducing the multiplier (different scales) from 4 to just one scale (resolution very low), the runtime for everything is around 8.5, just for handle one at 7 sec (slighly more than claimed in the paper)
"""

import numpy as np
import time
import cv2
import math
from scipy.signal import butter, lfilter, freqz, group_delay, filtfilt
from scipy.ndimage.filters import gaussian_filter
from scipy.spatial.distance import cdist
import json

coordinates = ["x", "y"]
joints_list = ["right_shoulder", "right_elbow", "right_wrist", "left_shoulder","left_elbow", "left_wrist",
        "right_hip", "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle",
        "right_eye", "right_ear","left_eye", "left_ear", "nose ", "neck"]

index_shoulder=[0,3]
index_elbow=[1,4]
index_wrist=[2,5]
index_hip=[6,9]
index_knee=[7,10]
index_ankle=[8,11]
index_eye=[12,14]
index_ear=[13,15]

index_list=[index_shoulder,index_elbow,index_wrist,index_hip,index_knee,index_ankle,index_eye,index_ear]

important_joints = [0,3,6,7,8,9,10,11]
print(important_joints)
limb_list=['shoulder','elbow','wrist','hip','knee','ankle','Neck','eye','ear']

# find connection in the specified sequence, center 29 is in the position 15
limbSeq = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10], \
           [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17], \
           [1,16], [16,18], [3,17], [6,18]]

# the middle joints heatmap correpondence
mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44], [19,20], [21,22], \
          [23,24], [25,26], [27,28], [29,30], [47,48], [49,50], [53,54], [51,52], \
          [55,56], [37,38], [45,46]]

# visualize
colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

def overlap(A, B):
    """
    returns IoU of two rectangles A and B (if no overlap, returns 0)
    """
    # if no overlap
    if (A[0] > B[1]) or (A[1] < B[0]):
        return 0
    if (A[2] > B[3]) or (A[3] < B[2]):
        return 0
    # else calculate areas
    I = [max(A[0], B[0]), min(A[1], B[1]), max(A[2], B[2]), min(A[3], B[3])]
    Aarea = abs((A[0]-A[1])*(A[2]-A[3]))
    Barea = abs((B[0]-B[1])*(B[2]-B[3]))
    Iarea = abs((I[0]-I[1])*(I[2]-I[3]))
    return Iarea/(Aarea+Barea-Iarea)

def player_localization(handle_one ,old_array, low_thresh=0.1, higher_tresh = 0.5):
    """
    returns an array containing only the coordinates of the target person, based on the previous location in old_array, and
    the possible detections in handle_one
    handle_one: current detections - array of shape number_people_detected * number_joints * number_coordinates
    old_array: located detection in previous frame, array of shape number_joints * number_coordinates
    low_thresh: minimum IoU value
    higher_thresh: if more than one person has a IoU>higher_tresh with the previous detection, the people seem to be confused - missing frame
    """
    handle_one = np.asarray(handle_one)
    zerrow2=np.where(old_array[:,0]!=0)[0]
    joints_for_bbox = np.intersect1d(zerrow2, important_joints) # non missing important values of previous frame
    old_arr_bbox = [np.min(old_array[joints_for_bbox, 0]), np.max(old_array[joints_for_bbox, 0]),
                   np.min(old_array[joints_for_bbox, 1]), np.max(old_array[joints_for_bbox, 1])] # bbox of previous frame detecten of target person
    intersections = []
    dist = []

    # check all detected persons
    for i in range(np.asarray(handle_one).shape[0]):
        player_array = handle_one[i]

        zerrow1=np.where(np.asarray(handle_one)[i,:,0]!=0)[0] # nonzero current person
        zerrow_all =np.intersect1d(zerrow1,zerrow2) # nonzero current and previous
        zerrow = np.intersect1d(zerrow_all, important_joints) # important joint and not missing in previous and current person
        joints_for_bbox = np.intersect1d(zerrow1, important_joints) # intersection of important non-missing values of current person

        if len(zerrow)<3: # not enough joints detected
            intersections.append(0)
            dist.append(np.inf)
            continue
        dist.append(np.linalg.norm(handle_one[i,zerrow,:] - old_array[zerrow,:])/len(zerrow)) # eucledian distance between all joints, normalized to mean distance per joint
        player_arr_bbox = [np.min(player_array[joints_for_bbox, 0]), np.max(player_array[joints_for_bbox, 0]), # bbbox aound current person
                        np.min(player_array[joints_for_bbox, 1]), np.max(player_array[joints_for_bbox, 1])]
        intersections.append(overlap(player_arr_bbox, old_arr_bbox)) # IoU of bbox of current and previous detetcon

    # print("dist", dist, "inter", intersections)
    # cases where we set frame to zeros (missing values):
    # if no person detected at all
    # if no person is intersection more than IoU = low_thresh
    # if two in intersections are bigger than higher_tresh (because if two guys overlaping that much, it is better to set it as missing value than to take the guy with most)
    if len(intersections)==0 or np.all(np.array(intersections)<low_thresh) or np.sum(np.array(intersections)>higher_tresh)>1: # player seem to be significantly overlapping
        print("missing frame because of not enough or too many intersections", "with intersections:", intersections) #, "with players", df[player][frame])
        res = np.array([[0,0] for i in range(18)])
    else:
        # --- FIRST VERSION: simply take the one with highest intersection
        target = np.argmax(intersections)
        # print("taken index from distances:", target)
        res = handle_one[target]
        # --- SECOND VERSION: minimum of distance of joints is taken as target, if IoU>0.1 (to prevent it from picking up simply the closest person in a missing frame ----
        #np.argmin(dist)
        # --- THIRD VERSION: minimum of distance of joints is taken as target, but only if it corresponds to the maximum of IoU ----
        # if np.argmax(intersections) == target:
        #     res = handle_one[target]
        # else:
        #     print("missing frame: maximum von intesections is different from minimum of distances", intersections, dist)
        #     res = np.array([[0,0] for i in range(18)])
    return res

def mix_right_left_old(df, index_list):
    """
    ---- OLD VERSION: only swaps if both are wrong, and df needs to be interpolated (cannot contain missing values) ----
    returns an array of same size as df, with the cleaned up joint trajectories, i.e. when left and right are swapped, they are swapped back,
    and if just one joint is suddenly in the spot of the left one, it is set as a missing value
    df: an array of shape number_frames * number_joints * number_coordinates
    index_list: containing the indizes of the corresponding joints: [[index_left_knee, index_right_knee], [index_left_ankle, index_right_ankle], ...]
    """
    tic = time.time()
    player=player+'_player'
    for i in range(1, len(df)-1):
        if abs(np.asarray(df[player][i])[index[1]][1]-np.asarray(df[player][i-1])[index[1]][1])+abs(np.asarray(df[player][i])[index[1]][0]-np.asarray(df[player][i-1])[index[1]][0])>abs(np.asarray(df[player][i])[index[0]][0]-np.asarray(df[player][i-1])[index[1]][0])+abs(np.asarray(df[player][i])[index[0]][1]-np.asarray(df[player][i-1])[index[1]][1]) and abs(np.asarray(df[player][i])[index[0]][1]-np.asarray(df[player][i-1])[index[0]][1])+abs(np.asarray(df[player][i])[index[0]][0]-np.asarray(df[player][i-1])[index[0]][0])>abs(np.asarray(df[player][i])[index[1]][0]-np.asarray(df[player][i-1])[index[0]][0])+abs(np.asarray(df[player][i])[index[0]][1]-np.asarray(df[player][i-1])[index[0]][1]):

            left=df[player][i][index[1]]
            right=df[player][i][index[0]]
            df[player][i][index[1]]=right
            df[player][i][index[0]]=left
    toc = time.time()
    #print("Time for mix right left", toc-tic)
    return df

def mix_right_left(df, index_list, factor = 3):
    """
    returns an array of same size as df, with the cleaned up joint trajectories, i.e. when left and right are swapped, they are swapped back,
    and if just one joint is suddenly in the spot of the left one, it is set as a missing value
    df: an array of shape number_frames * number_joints * number_coordinates, without interpolation or other preprocessing
    index_list: containing the indizes of the corresponding joints: [[index_left_knee, index_right_knee], [index_left_ankle, index_right_ankle], ...]
    """
    df = np.asarray(df)
    zeros_filled = df.copy()
    for index in index_list:
        r = index[0] # right
        l = index[1] # left
        r_cond = True # r_cond and l_cond indicate if previous detection was 0
        l_cond = True
        for i in range(1, len(df)-1):

            # if first detection is missing, nothing can be done - continue until first detection found
            if zeros_filled[i-1, r,0]==0 or zeros_filled[i-1, l,0]==0:
                # print("first one 0", i, index)
                # print("zf i-1", zeros_filled[i-1, index].tolist(), "zf i", zeros_filled[i,index].tolist(), "df i-1",df[i-1,index].tolist() ,"df i", df[i,index].tolist())
                continue

            # zeros filled is an array with all missing values filled in, not interpolated, but always with the last detected value
            if df[i,r,0]==0:
                zeros_filled[i,r] = zeros_filled[i-1, r]
            if df[i,l,0]==0:
                zeros_filled[i,l] = zeros_filled[i-1, l]

            ## cond1: if distance between right joint last frame and this frame is bigger than right joint current frame and left joint last frame
            ## cond2: if distance between left joint last frame and this frame is bigger than left joint current frame and right joint last frame

            ## CHESSBOARD DISTANCE
            # cond1 = abs(zeros_filled[i, r, 1] - zeros_filled[i-1, r, 1]) + abs(zeros_filled[i, r, 0] - zeros_filled[i-1, r, 0]) > factor * abs(zeros_filled[i, r, 1] - zeros_filled[i-1, l, 1])+ abs(zeros_filled[i, r, 0] - zeros_filled[i-1, l, 0])
            # cond2 = abs(zeros_filled[i, l, 1] - zeros_filled[i-1, l, 1]) + abs(zeros_filled[i, l, 0] - zeros_filled[i-1, l, 0]) > 3* abs(zeros_filled[i-1, r, 1] - zeros_filled[i, l, 1])+ abs(zeros_filled[i-1 , r, 0] - zeros_filled[i, l, 0])

            ## EUCLEDIAN DISTANCE
            cond1 = np.linalg.norm(zeros_filled[i, r] - zeros_filled[i-1, r]) >  factor* np.linalg.norm(zeros_filled[i, r] - zeros_filled[i-1, l])
            cond2 = np.linalg.norm(zeros_filled[i, l] - zeros_filled[i-1, l]) >  factor* np.linalg.norm(zeros_filled[i, l] - zeros_filled[i-1, r])
            if cond1 and cond2 and r_cond and l_cond: # both swapped, and previous frame not zero
                left=df[i, l].copy()
                right=df[i,r].copy()
                df[i,l]=right
                zeros_filled[i,r] = left
                df[i,r]=left
                zeros_filled[i,l] = right
            elif cond1 and r_cond: # only right is wrong, and previous right detection not zero: set as missing value (real position not known)
                df[i,r, 0] = 0
                df[i,r, 1] = 0
                zeros_filled[i,r] = zeros_filled[i-1, r]
            elif cond2 and l_cond: # only left is wrong, and previous left detection not zero: set as missing value
                df[i,l, 0] = 0
                df[i,l, 1] = 0
                zeros_filled[i,l] = zeros_filled[i-1, l]
            else: # set r_cond and l_cond
                if df[i,r,0]==0:
                    r_cond = False
                else:
                    r_cond = True
                if df[i,l,0]==0:
                    l_cond = False
                else:
                    l_cond = True
    return df



def outlier_removal(mat, eval_range = 10):
    """
    evaluates set of 2*eval_range values and removes the once above 3*std
    """
    frames, num_joints, xy_len = mat.shape
    for limb in range(num_joints):
        for xy in [0, 1]:
            values = mat[:, limb, xy]
            new = []
            for i in range(-eval_range, eval_range):
                new.append(np.roll(values, i))
            new = np.array(new)
            diff = np.absolute(np.median(new, axis = 0)-values) # difference from median over 2*eval_range values
            replace_values = np.median(new, axis = 0) # replace these values with the median

            inds = diff> np.mean(diff)+3*np.std(diff) # all indices with outlier higher 2*std
            values[inds] = replace_values[inds] # replacement

            mat[:, limb, xy] = values
    return mat

def interpolate(mat, erosion = False):
    """
    returns interpolated version of mat, with all 0 removed (linear interpolation)
    mat: array of size frames * num_joints * number_coordinates
    erosion: if it is likely that there is noise with single misdetections, erode such that isolated points between missing values are removed
    """
    frames, num_joints, xy_len = mat.shape

    for limb in range(num_joints):
        for xy in range(xy_len): # x and y coord dimension
            # TODO: Examine any performance degradation from calling ':' on primary dimension.
            values = mat[:, limb, xy]
            #print(values)

            if erosion:
            # FOR EROSION TO AVOID NOISE:
                not_zer = ndimage.morphology.binary_erosion(values)
                not_zer[0] = values[0]
            else:
                not_zer = np.logical_not(values == 0)

            # indices for interpolation - possibly reduce to leave edges at zero
            indices = np.arange(len(values))

            if not any(not_zer): # everything is zero, so can't interpolate
                mat[:, limb, xy] = 0
                print("whole joint is zero")
            else:
                mat[:, limb, xy] = np.round(
                    ## for other types of interpolation (e.g. cubic)
                    # interp1d(indices[not_zer], values[not_zer], kind="cubic")(indices) ,1)
                np.interp(indices, indices[not_zer], values[not_zer]), 1) # linear interpolation
    ## for removal of outliers:
    # mat = outlier_removal(mat)
    return mat

def lowpass(sequence, cutoff = 1, fs = 15, order=5):
    """
    simple lowpass filtering of 1D data: returns sequence of same length
    sequence: sequence of k frames of just on coordinate of one joint
    fs is the frame rate
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / float(nyq)
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b,a,sequence) # lfilter(b, a, data)
    return y


def df_coordinates(new_df, right_left = True, do_interpolate = True, smooth = True, fps = 20):
    """
    data processing: right left, interpolation and smoothing
    """
    if right_left:
        new_df = mix_right_left(new_df, index_list)
    if do_interpolate:
        new_df = interpolate(new_df)
    if smooth:
        for k in range(len(new_df[0])):
            for j in range(2):
                new_df[:,k,j] = lowpass(new_df[:,k,j]-new_df[0,k,j], cutoff = 1, fs = fps)+new_df[0,k,j]
    return new_df


def to_json(play, events_dic, save_path, position = None, pitchtype = None, frames_per_sec = 30):
    """
    Saves data in the standard json format
    play: array of size nr_frames*nr_joints*nr_coordinates
    events_dic: dictionary containing meta information about the game (time, pitchers first movement...)
    pitchtype and position can also be saved in json file, if processed in real time
    """
    frames, joints, xy = play.shape
    start_time = int(round(events_dic["start_time"] * 1000))
    dic = {}
    dic["time_start"] = start_time
    dic["bbox_pitcher"] = events_dic["bbox_pitcher"]
    dic["bbox_batter"] = events_dic["bbox_batter"]
    dic["video_directory"] = events_dic["video_directory"]
    dic["Pitching position"]= position
    dic["Pitch Type"] = pitchtype
    dic["device"] = "?"
    dic["deployment"] = "?"
    dic["frames"] = []
    for i in range(frames):
        dic_joints = {}
        dic_joints["timestamp"] = int(round(start_time + (1000*i)/float(frames_per_sec)))
        for j in range(18): #joints):
            dic_xy = {}
            for k in range(xy):
                dic_xy[coordinates[k]] = play[i,j,k]
            dic_joints[joints_list[j]] = dic_xy
        dic_joints["events"]=[]
        for j in events_dic.keys():
            if i==events_dic[j]:
                dic_joints["events"].append({"timestamp": int(round(time.time() * 1000)), "name": j,"code": 1,
                                    "target_name": "Pitcher", "target_id": 1})
        dic["frames"].append(dic_joints)

    with open(save_path+".json", 'w') as outfile:
        json.dump(dic, outfile, indent=10)

def color_and_save_image(ori_img, all_peaks, save_name):
    for x in range(len(all_peaks)):
        for j in range(len(all_peaks[x])):
            cv2.circle(ori_img, (int(all_peaks[x,j,0]),int(all_peaks[x,j,1])) , 2, colors[x], thickness=-1)
    cv2.imwrite(save_name, ori_img)

def define_bbox(res, boundaries, min_width=30):
    """
    return bbox from last detection, extended by factor* boxwidth
    bbox: [left bound, right bound, upper bound, lower bound]
    input is ouput of pose estimation and boundaries of frame
    """
    joints_for_bbox = np.where(res[:,0]!=0)[0]
    # takes minima and maxima of the pose as edges of bounding box b
    bbox = np.array([np.min(res[joints_for_bbox, 0]), np.max(res[joints_for_bbox, 0]),
           np.min(res[joints_for_bbox, 1]), np.max(res[joints_for_bbox, 1])]).astype(int)
    # extends bounding box by adding the width of the box on all sides (up to boundaries)
    width = max(0.5*(bbox[1]-bbox[0]), min_width) # a minimum width is set
    for i in range(len(bbox)): # every second of the box must subtract the width, the other parts add the width
        bbox[i]-=width
        width*=(-1)
        if (bbox[i]<boundaries[i] and width<0) or (bbox[i]>boundaries[i] and width>0):
            bbox[i]=boundaries[i] # set to boundary if over edge
    return bbox

def save_inbetween(arr, fi, out_dir, events_dic):
    """
    takes intermediate result array arr and saves it with the video name fi, and the output directory out_dir
    events_dic is a dictionary containing meta information
    """
    pitcher_array = np.array(arr)
    print("shape pitcher_array", pitcher_array.shape)
    # NEW: JSON FORMAT
    game_id = fi[:-4]
    file_path_pitcher = out_dir+game_id
    print(events_dic, file_path_pitcher)
    to_json(pitcher_array, events_dic, file_path_pitcher)

## OLD FUNCTIONS
# def mix_right_left(df,index,player):
#     tic = time.time()
#     player=player+'_player'
#     for i in range(1, len(df)-1):
#         if abs(np.asarray(df[player][i])[index[1]][1]-np.asarray(df[player][i-1])[index[1]][1])+abs(np.asarray(df[player][i])[index[1]][0]-np.asarray(df[player][i-1])[index[1]][0])>abs(np.asarray(df[player][i])[index[0]][0]-np.asarray(df[player][i-1])[index[1]][0])+abs(np.asarray(df[player][i])[index[0]][1]-np.asarray(df[player][i-1])[index[1]][1]) and abs(np.asarray(df[player][i])[index[0]][1]-np.asarray(df[player][i-1])[index[0]][1])+abs(np.asarray(df[player][i])[index[0]][0]-np.asarray(df[player][i-1])[index[0]][0])>abs(np.asarray(df[player][i])[index[1]][0]-np.asarray(df[player][i-1])[index[0]][0])+abs(np.asarray(df[player][i])[index[0]][1]-np.asarray(df[player][i-1])[index[0]][1]):
#
#             left=df[player][i][index[1]]
#             right=df[player][i][index[0]]
#             #print i,player,'left is',left,'right is',right
#             df[player][i][index[1]]=right
#             df[player][i][index[0]]=left
#
#     toc = time.time()
#     #print("Time for mix right left", toc-tic)
#     return df
#
# def continuity(df_res, player, num_joints=18):
#     mat = np.array(df_res[player+'_player'].values)
#     mat = np.stack(mat) # seems necessary because DataFrame does not return a pure np matrix
#     for limb in range(num_joints):
#         for xy in [0, 1]: # x and y coord dimension
#             # TODO: Examine any performance degradation from calling ':' on primary dimension.
#             values = mat[:, limb, xy]
#             not_zer = np.logical_not(values == 0)
#             indices = np.arange(len(values))
#
#             if not any(not_zer): # everything is zero, so can't interpolate
#                 mat[:, limb, xy] = 0
#                 print("whole joint is zero")
#             else:
#                 mat[:, limb, xy] = np.round(
#                     np.interp(indices, indices[not_zer], values[not_zer]),
#                     1)
#     for frame_ii in range(mat.shape[0]):
#         df_res[player+'_player'][frame_ii] = mat[frame_ii, :, :].tolist()
#
#     return df_res
#
# def player_localization_ratio(df,frame,player,old_array, body_dist):
#     tic = time.time()
#     player2=player+'_player'
#     dist=[]
#     ratios = []
#     for i in range(np.asarray(df[player][frame]).shape[0]):
#         zerrow1=np.where(np.asarray(df[player][frame])[i,:,0]!=0)
#         zerrow2=np.where(old_array[:,0]!=0)
#         zerrow=np.intersect1d(zerrow1,zerrow2)
#
#         if len(zerrow)<2:
#             zerrow=zerrow2
#         dist.append(np.linalg.norm(np.asarray(df[player][frame])[i,zerrow[0],:]-old_array[zerrow[0],:])/len(zerrow))
#
#         p = df[player][frame][i][joints_for_cdist]
#         player_dist = cdist(p,p)
#         ratios.append(np.linalg.norm(body_dist-player_dist))
#     #print dist
#     #print df[player][frame]
#     if len(dist)==0 or np.min(ratios)>400:
#         df[player2][frame]=[[0,0] for i in range(18)]
#     else:
#         df[player2][frame]=df[player][frame][np.argmin(dist)]
#     array_stored=np.asarray(df[player2][frame])
#     array_stored[np.where(array_stored==0)]=old_array[np.where(array_stored==0)]
#
#     joint_arr_cdist = np.array(array_stored)[joints_for_cdist]
#     new_body_dist = cdist(joint_arr_cdist, joint_arr_cdist)
#     old_array=array_stored
#     toc = time.time()
#     #print("Time for player_localization: ", toc-tic)
#     return df, old_array, new_body_dist
#

#
# def df_coordinates(df,centerd, player_list, interpolate = True):
#     df.sort_values(by='Frame',ascending=1,inplace=True)
#     df.reset_index(inplace=True,drop=True)
#     for player in player_list:
#         df[player+'_player']=df[player].copy()
#         player2=player+'_player'
#         center=centerd[player]
#         old_norm=10000
#         indices=[6,9]
#         #print df[player][0][0]
#         i=0
#         found = False
#         while not found:
#             #len(df[player][i])==0:
#
#             for person in range(len(df[player][i])):
#                 hips=np.asarray(df[player][i][person])[indices]
#
#                 hips=hips[np.sum(hips,axis=1)!=0]
#                 mean_hips=np.mean(hips,axis=0)
#
#
#                 norm= abs(mean_hips[0]-center[0])+abs(mean_hips[1]-center[1]) #6 hip
#                 if norm<old_norm:
#                     found = True
#                     loc=person
#                     old_norm=norm
#             if found:
#                 break
#             df[player2][i]=[[0,0] for j in range(18)]
#             print("no person detected in frame", i)
#             i+=1
#
#         df[player2][i]=df[player][i][loc]
#         globals()['old_array_%s'%player]=np.asarray(df[player][i][loc])
#         # joint_arr_cdist = np.array(df[player][0][loc])[joints_for_cdist]
#         # print("joints_arr_cdist", np.array(joint_arr_cdist).shape)
#         # globals()['cdist_%s'%player] = cdist(joint_arr_cdist, joint_arr_cdist)
#
#         for frame in df['Frame'][(i+1):len(df)]:
#             df,globals()['old_array_%s'%player] = player_localization(df,frame,player,globals()['old_array_%s'%player])
#
#         for index in index_list:
#             df=mix_right_left(df,index,player)
#
#         if interpolate:
#             df=continuity(df,player)
#
#     return df #[['Frame','Pitcher_player','Batter_player']]
