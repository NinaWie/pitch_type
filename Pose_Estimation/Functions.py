#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 16:02:39 2017

@author: estelleaflalo
"""

import numpy as np
import time
import cv2
import math
from torch import np
import util
import torch
import torch as T
import torch.nn as nn
from torch.autograd import Variable
from config_reader import config_reader
from scipy.signal import butter, lfilter, freqz, group_delay, filtfilt
from scipy.ndimage.filters import gaussian_filter
from scipy.spatial.distance import cdist
import json
# from time_probe import tic, toc, time_summary

param_, model_ = config_reader()
# USE_MODEL = model_['use_model']
USE_GPU = param_['use_gpu']
TORCH_CUDA = lambda x: x.cuda() if USE_GPU else x

head=0
weight_name = './model/pose_model.pth'


from os import listdir

print(listdir("./model"))

torch.set_num_threads(torch.get_num_threads())
blocks = {}

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

block0  = [{'conv1_1':[3,64,3,1,1]},{'conv1_2':[64,64,3,1,1]},{'pool1_stage1':[2,2,0]},{'conv2_1':[64,128,3,1,1]},{'conv2_2':[128,128,3,1,1]},{'pool2_stage1':[2,2,0]},{'conv3_1':[128,256,3,1,1]},{'conv3_2':[256,256,3,1,1]},{'conv3_3':[256,256,3,1,1]},{'conv3_4':[256,256,3,1,1]},{'pool3_stage1':[2,2,0]},{'conv4_1':[256,512,3,1,1]},{'conv4_2':[512,512,3,1,1]},{'conv4_3_CPM':[512,256,3,1,1]},{'conv4_4_CPM':[256,128,3,1,1]}]

blocks['block1_1']  = [{'conv5_1_CPM_L1':[128,128,3,1,1]},{'conv5_2_CPM_L1':[128,128,3,1,1]},{'conv5_3_CPM_L1':[128,128,3,1,1]},{'conv5_4_CPM_L1':[128,512,1,1,0]},{'conv5_5_CPM_L1':[512,38,1,1,0]}]

blocks['block1_2']  = [{'conv5_1_CPM_L2':[128,128,3,1,1]},{'conv5_2_CPM_L2':[128,128,3,1,1]},{'conv5_3_CPM_L2':[128,128,3,1,1]},{'conv5_4_CPM_L2':[128,512,1,1,0]},{'conv5_5_CPM_L2':[512,19,1,1,0]}]

for i in range(2,7):
    blocks['block%d_1'%i]  = [{'Mconv1_stage%d_L1'%i:[185,128,7,1,3]},{'Mconv2_stage%d_L1'%i:[128,128,7,1,3]},{'Mconv3_stage%d_L1'%i:[128,128,7,1,3]},{'Mconv4_stage%d_L1'%i:[128,128,7,1,3]},
{'Mconv5_stage%d_L1'%i:[128,128,7,1,3]},{'Mconv6_stage%d_L1'%i:[128,128,1,1,0]},{'Mconv7_stage%d_L1'%i:[128,38,1,1,0]}]
    blocks['block%d_2'%i]  = [{'Mconv1_stage%d_L2'%i:[185,128,7,1,3]},{'Mconv2_stage%d_L2'%i:[128,128,7,1,3]},{'Mconv3_stage%d_L2'%i:[128,128,7,1,3]},{'Mconv4_stage%d_L2'%i:[128,128,7,1,3]},
{'Mconv5_stage%d_L2'%i:[128,128,7,1,3]},{'Mconv6_stage%d_L2'%i:[128,128,1,1,0]},{'Mconv7_stage%d_L2'%i:[128,19,1,1,0]}]

def make_layers(cfg_dict):
    layers = []
    for i in range(len(cfg_dict)-1):
        one_ = cfg_dict[i]
        for k,v in one_.items():
            if 'pool' in k:
                layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2] )]
            else:
                conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride = v[3], padding=v[4])
                layers += [conv2d, nn.ReLU(inplace=True)]
    one_ = list(cfg_dict[-1].keys())
    k = one_[0]
    v = cfg_dict[-1][k]
    conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride = v[3], padding=v[4])
    layers += [conv2d]
    return nn.Sequential(*layers)

layers = []
for i in range(len(block0)):
    one_ = block0[i]
    for k,v in one_.items():
        if 'pool' in k:
            layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2] )]
        else:
            conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride = v[3], padding=v[4])
            layers += [conv2d, nn.ReLU(inplace=True)]

models = {}
models['block0']=nn.Sequential(*layers)

for k,v in blocks.items():
    models[k] = make_layers(v)

class pose_model(nn.Module):
    def __init__(self,model_dict,transform_input=False):
        super(pose_model, self).__init__()
        self.model0   = model_dict['block0']
        self.model1_1 = model_dict['block1_1']
        self.model2_1 = model_dict['block2_1']
        self.model3_1 = model_dict['block3_1']
        self.model4_1 = model_dict['block4_1']
        self.model5_1 = model_dict['block5_1']
        self.model6_1 = model_dict['block6_1']

        self.model1_2 = model_dict['block1_2']
        self.model2_2 = model_dict['block2_2']
        self.model3_2 = model_dict['block3_2']
        self.model4_2 = model_dict['block4_2']
        self.model5_2 = model_dict['block5_2']
        self.model6_2 = model_dict['block6_2']

    def forward(self, x):
        out1 = self.model0(x)

        out1_1 = self.model1_1(out1)
        out1_2 = self.model1_2(out1)
        out2  = torch.cat([out1_1,out1_2,out1],1)

        out2_1 = self.model2_1(out2)
        out2_2 = self.model2_2(out2)
        out3   = torch.cat([out2_1,out2_2,out1],1)

        out3_1 = self.model3_1(out3)
        out3_2 = self.model3_2(out3)
        out4   = torch.cat([out3_1,out3_2,out1],1)

        out4_1 = self.model4_1(out4)
        out4_2 = self.model4_2(out4)
        out5   = torch.cat([out4_1,out4_2,out1],1)

        out5_1 = self.model5_1(out5)
        out5_2 = self.model5_2(out5)
        out6   = torch.cat([out5_1,out5_2,out1],1)

        out6_1 = self.model6_1(out6)
        out6_2 = self.model6_2(out6)

        return out6_1,out6_2


model = pose_model(models)
model.load_state_dict(torch.load(weight_name))
TORCH_CUDA(model)
model.float()
model.eval()

def handle_one(oriImg):
    tic = time.time()
 #   print 1

    # for visualize
#canvas = np.copy(oriImg)
    multiplier = [x * model_['boxsize'] / oriImg.shape[0] for x in param_['scale_search']]

    scale = model_['boxsize'] / float(oriImg.shape[0])

    ind = np.argmin(np.absolute(np.array(multiplier)-1))
    scale = multiplier[ind]

    #print("handle one1" ,time.time()-tic)
    tic=time.time()
 #   print 3,time.time()-tic
    tic=time.time()
    e=1
    b=0
    len_mul=e-b
    multiplier=multiplier[b:e]
    heatmap_avg = TORCH_CUDA(torch.zeros((len(multiplier),19,oriImg.shape[0], oriImg.shape[1])))
    paf_avg = TORCH_CUDA(torch.zeros((len(multiplier),38,oriImg.shape[0], oriImg.shape[1])))
    toc =time.time()
    #print("handle one 2",toc-tic)

    tic = time.time()


    for m in range(len(multiplier)):
        # print multiplier[m]
        tictic= time.time()
        scale = multiplier[m]
        print(scale)
        imageToTest = cv2.resize(oriImg, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model_['stride'], model_['padValue'])
        imageToTest_padded = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,2,0,1))/256 - 0.5
        print imageToTest_padded.shape
        feed = TORCH_CUDA(Variable(T.from_numpy(imageToTest_padded)))
        output1,output2 = model(feed)
 #       print time.time()-tictic,"first part"
        tictic=time.time()
        heatmap = TORCH_CUDA(nn.UpsamplingBilinear2d((oriImg.shape[0], oriImg.shape[1])))(output2)
        #nearest neighbors
        paf = TORCH_CUDA(nn.UpsamplingBilinear2d((oriImg.shape[0], oriImg.shape[1])))(output1)

        globals()['heatmap_avg_%s'%m] = heatmap[0].data
        globals()['paf_avg_%s'%m] = paf[0].data
        #heatmap_avg[m] = heatmap[0].data
        #paf_avg[m] = paf[0].data
  #      print 'loop', m ,' ',time.time()-tictic, "second part"

    #print("handle one 3" , time.time()-tic)
    toc = time.time()
    #print 'time is %.5f'%(toc-tic)
    temp1=(heatmap_avg_0)#+heatmap_avg_1)/float(len_mul)
    temp2=(paf_avg_0)#+paf_avg_1)/float(len_mul)
    heatmap_avg = TORCH_CUDA(T.transpose(T.transpose(T.squeeze(temp1),0,1),1,2))
    paf_avg     = TORCH_CUDA(T.transpose(T.transpose(T.squeeze(temp2),0,1),1,2))
    #heatmap_avg = T.transpose(T.transpose(T.squeeze(T.mean(heatmap_avg, 0)),0,1),1,2).cuda()
    #paf_avg     = T.transpose(T.transpose(T.squeeze(T.mean(paf_avg, 0)),0,1),1,2).cuda()
    heatmap_avg=heatmap_avg.cpu().numpy()
    paf_avg    = paf_avg.cpu().numpy()
    all_peaks = []
    peak_counter = 0
    #print '5bis', time.time()-toc
    #maps =
#    s= heatmap_avg[:,:,0].shape
#    map_ori=heatmap_avg[:,:,12].cuda()
#    map = gaussian_filter(map_ori, sigma=3)
#
#    map_left = np.ones(map.shape)
#    map_left[1:,:] = map[:-1,:]
#    map_right = np.zeros(map.shape)
#    map_right[:-1,:] = map[1:,:]
#    map_up = np.zeros(map.shape)
#    map_up[:,1:] = map[:,:-1]
#    map_down = np.zeros(map.shape)
#    map_down[:,:-1] = map[:,1:]
#
#    peaks_binary = np.logical_and.reduce((map>=map_left, map>=map_right, map>=map_up, map>=map_down, map > param_['thre1']))
#
#    peaks0 = zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]) # note reverse
#
#    peaks_with_score = [x + (map_ori[x[1],x[0]],) for x in peaks0]
#    peaks_with_score_and_id0 = [peaks_with_score[i] + (id[i],) for i in range(len(id))]
    for part in range(18):
        if part<18-head:
            map_ori = heatmap_avg[:,:,part]
            map = gaussian_filter(map_ori, sigma=3)

            map_left = np.zeros(map.shape)
            map_left[1:,:] = map[:-1,:]
            map_right = np.zeros(map.shape)
            map_right[:-1,:] = map[1:,:]
            map_up = np.zeros(map.shape)
            map_up[:,1:] = map[:,:-1]
            map_down = np.zeros(map.shape)
            map_down[:,:-1] = map[:,1:]

            peaks_binary = np.logical_and.reduce((map>=map_left, map>=map_right, map>=map_up, map>=map_down, map > param_['thre1']))

            peaks = zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]) # note reverse
            globals()['peaks%s'%part]=peaks
            peaks_with_score = [x + (map_ori[x[1],x[0]],) for x in peaks]
            id = range(peak_counter, peak_counter + len(peaks))
            peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]
            globals()['peaks_with_score_and_id_%s'%part]=  peaks_with_score_and_id
            all_peaks.append(peaks_with_score_and_id)
            peak_counter += len(peaks)
        else :
            all_peaks.append(peaks_with_score_and_id_0)
            peak_counter += len(peaks0)


    #print("handle one 4",time.time()-toc)
    tic=time.time()
    connection_all = []
    special_k = []
    mid_num = 10

    for k in range(len(mapIdx)-head):
        score_mid = paf_avg[:,:,[x-19-head for x in mapIdx[k]]]
        candA = all_peaks[limbSeq[k][0]-1]
        candB = all_peaks[limbSeq[k][1]-1]
        nA = len(candA)
        nB = len(candB)
        indexA, indexB = limbSeq[k]
        if(nA != 0 and nB != 0):
            connection_candidate = []
            for i in range(nA):
                for j in range(nB):
                    vec = np.subtract(candB[j][:2], candA[i][:2])
                    norm = math.sqrt(vec[0]*vec[0] + vec[1]*vec[1])
                    vec = np.divide(vec, norm)

                    startend = zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
                                   np.linspace(candA[i][1], candB[j][1], num=mid_num))

                    vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
                                      for I in range(len(startend))])
                    vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
                                      for I in range(len(startend))])

                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                    if norm==0:
                        score_with_dist_prior = sum(score_midpts)/len(score_midpts)
                    else:
                        score_with_dist_prior = sum(score_midpts)/len(score_midpts) + min(0.5*oriImg.shape[0]/norm-1, 0)
                    criterion1 = len(np.nonzero(score_midpts > param_['thre2'])[0]) > 0.8 * len(score_midpts)
                    criterion2 = score_with_dist_prior > 0
                    if criterion1 and criterion2:
                        connection_candidate.append([i, j, score_with_dist_prior, score_with_dist_prior+candA[i][2]+candB[j][2]])

            connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
            connection = np.zeros((0,5))
            for c in range(len(connection_candidate)):
                i,j,s = connection_candidate[c][0:3]
                if(i not in connection[:,3] and j not in connection[:,4]):
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                    if(len(connection) >= min(nA, nB)):
                        break

            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])
    #print("handle one 5", time.time()-tic)
    tic=time.time()
    # last number in each row is the total parts number of that person
    # the second last number in each row is the score of the overall configuration
    subset = -1 * np.ones((0, 20))
    candidate = np.array([item for sublist in all_peaks for item in sublist])

    #print 'candidate is ', candidate
    for k in range(len(mapIdx)-head):

        if k not in special_k:
            partAs = connection_all[k][:,0]
            partBs = connection_all[k][:,1]
            indexA, indexB = np.array(limbSeq[k]) - 1

            for i in range(len(connection_all[k])): #= 1:size(temp,1)
                found = 0
                subset_idx = [-1, -1]
                for j in range(len(subset)): #1:size(subset,1):
                    if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                        subset_idx[found] = j
                        found += 1

                if found == 1:
                    j = subset_idx[0]
                    if(subset[j][indexB] != partBs[i]):
                        subset[j][indexB] = partBs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                elif found == 2: # if found 2 and disjoint, merge them
                    j1, j2 = subset_idx
                    #print "found = 2"
                    membership = ((subset[j1]>=0).astype(int) + (subset[j2]>=0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0: #merge
                        subset[j1][:-2] += (subset[j2][:-2] + 1)
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else: # as like found == 1
                        subset[j1][indexB] = partBs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(20)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = 2
                    row[-2] = sum(candidate[connection_all[k][i,:2].astype(int), 2]) + connection_all[k][i][2]
                    subset = np.vstack([subset, row])

    """
    inserted in original Code from github because we want the function to return the raw data and not an image
    """
    output_coordinates=np.zeros((len(subset),18,2))
    for n in range(len(subset)):
        later = []
        for i in range(18):
            j = int(subset[n][i])
            if j==-1:
                continue
            output_coordinates[n,i-2,0]= candidate[j, 0]
            output_coordinates[n,i-2,1]= candidate[j, 1]
    return output_coordinates

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

    print("dist", dist, "inter", intersections)
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
        print("taken index from distances:", target)
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
                print("first one 0", i, index)
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
