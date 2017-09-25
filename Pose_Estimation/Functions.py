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
from scipy.ndimage.filters import gaussian_filter
from PoseModels import AvailableModels

limb_list=['shoulder','elbow','wrist','hip','knee','ankle','Neck','eye','ear']
player_list=['Batter','Pitcher']
col_pitcher=['Pitcher_Right_shoulder','Pitcher_Left_shoulder','Pitcher_Right_elbow','Pitcher_Right_wrist','Pitcher_Left_elbow','Pitcher_Left_wrist','Pitcher_Right_hip','Pitcher_Right_knee','Pitcher_Right_ankle','Pitcher_Left_hip','Pitcher_Left_knee','Pitcher_Left_ankle','Pitcher_Neck','Pitcher_Right_eye','Pitcher_Right_ear','Pitcher_Left_eye','Pitcher_Left_ear']
col_batter=['Batter_Right_shoulder','Batter_Left_shoulder','Batter_Right_elbow','Batter_Right_wrist','Batter_Left_elbow','Batter_Left_wrist','Batter_Right_hip','Batter_Right_knee','Batter_Right_ankle','Batter_Left_hip','Batter_Left_knee','Batter_Left_ankle','Batter_Neck','Batter_Right_eye','Batter_Right_ear','Batter_Left_eye','Batter_Left_ear']

index_shoulder=[0,1]
index_elbow=[2,4]
index_wrist=[3,5]
index_hip=[6,9]
index_knee=[7,10]
index_ankle=[8,11]
index_eye=[13,15]
index_ear=[14,16]
index_list=[index_shoulder,index_elbow,index_wrist,index_hip,index_knee,index_ankle,index_eye,index_ear]
head=0

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


param_, model_ = config_reader()

USE_MODEL = model_['use_model']
USE_GPU = param_['use_gpu']
TORCH_CUDA = lambda x: x.cuda() if USE_GPU else x

model = AvailableModels[USE_MODEL]()

def handle_one(oriImg):
    tic = time.time()
 #   print 1

    # for visualize
#canvas = np.copy(oriImg)
    multiplier = [x * model_['boxsize'] / oriImg.shape[0] for x in param_['scale_search']]

    scale = model_['boxsize'] / float(oriImg.shape[0])

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


    for m in range(1): #len(multiplier)):
        tictic= time.time()

        (output1, output2), (heatmap, paf) = model.evaluate(oriImg, scale=multiplier[m])

 #       print time.time()-tictic,"first part"
        tictic=time.time()


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
    #print("handle one 6", time.time()-tic)
    # delete some rows of subset which has few parts occur
    #deleteIdx = [];
    #for i in range(len(subset)):
    #    if subset[i][-1] < 4 or subset[i][-2]/subset[i][-1] < 0.4:
    #        deleteIdx.append(i)
    #temp = np.delete(subset, deleteIdx, axis=0)


#    canvas = cv2.imread(test_image) # B,G,R order



    output_coordinates=np.zeros((len(subset),18,2))



    for i in range(18):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i])-1]
            if -1 in index:
                continue
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            output_coordinates[n,i,0]=mX
            output_coordinates[n,i,1]=mY
    return output_coordinates




def player_localization(df,frame,player,old_array):
    tic = time.time()
    player2=player+'_player'
    dist=[]
    for i in range(np.asarray(df[player][frame]).shape[0]):
        zerrow1=np.where(np.asarray(df[player][frame])[i,:,0]<>0)
        zerrow2=np.where(old_array[:,0]<>0)
        zerrow=np.intersect1d(zerrow1,zerrow2)

        if len(zerrow)<2:
            zerrow=zerrow2
        dist.append(np.linalg.norm(np.asarray(df[player][frame])[i,zerrow[0],:]-old_array[zerrow[0],:])/len(zerrow))
    #print dist
    #print df[player][frame]
    if len(dist)==0 or np.min(dist)>3:
        df[player2][frame]=[[0,0] for i in range(18)]
    else:
        df[player2][frame]=df[player][frame][np.argmin(dist)]
    array_stored=np.asarray(df[player2][frame])
    array_stored[np.where(array_stored==0)]=old_array[np.where(array_stored==0)]

    old_array=array_stored
    toc = time.time()
    #print("Time for player_localization: ", toc-tic)
    return df, old_array



def continuity(df_res,player):
    tic = time.time()
    temp=df_res[player+'_player']
    mat=np.zeros([18,2,len(temp)])

    for i in range(len(temp)):
        mat[:,:,i]=temp.iloc[i]#.tolist()
    for limb in range(17):
        for xy in range(2):
            for i in range(mat.shape[2]):
                not_zer = np.logical_not(mat[limb,xy,:]==0)
                indices = np.arange(len(mat[limb,xy,:]))
                try :
                    mat[limb,xy,:]=np.round(np.interp(indices, indices[not_zer], mat[limb,xy,:][not_zer]),1)
                    # from scipy.interpolate import interp1d
                    # f = interpld(indices[not_zer], mat[limb,xy,:][not_zer])
                    # mat[limb, xy, :]  = np.round(f(indices), 1)
                except ValueError:
                    continue

    for i in range(mat.shape[2]):
        df_res[player+'_player'][i]=mat[:,:,i].tolist()
    toc = time.time()
    print("Time for continuity ", toc-tic)
    return df_res



def mix_right_left(df,index,player):
    tic = time.time()
    player=player+'_player'
    for i in range(len(df)-1):
        if i==0: continue
        else:
            if abs(np.asarray(df[player][i])[index[1]][1]-np.asarray(df[player][i-1])[index[1]][1])+abs(np.asarray(df[player][i])[index[1]][0]-np.asarray(df[player][i-1])[index[1]][0])>abs(np.asarray(df[player][i])[index[0]][0]-np.asarray(df[player][i-1])[index[1]][0])+abs(np.asarray(df[player][i])[index[0]][1]-np.asarray(df[player][i-1])[index[1]][1]) and abs(np.asarray(df[player][i])[index[0]][1]-np.asarray(df[player][i-1])[index[0]][1])+abs(np.asarray(df[player][i])[index[0]][0]-np.asarray(df[player][i-1])[index[0]][0])>abs(np.asarray(df[player][i])[index[1]][0]-np.asarray(df[player][i-1])[index[0]][0])+abs(np.asarray(df[player][i])[index[0]][1]-np.asarray(df[player][i-1])[index[0]][1]):

                left=df[player][i][index[1]]
                right=df[player][i][index[0]]
                #print i,player,'left is',left,'right is',right
                df[player][i][index[1]]=right
                df[player][i][index[0]]=left

    toc = time.time()
    #print("Time for mix right left", toc-tic)
    return df


def df_coordinates(df,centerd):
    df.sort_values(by='Frame',ascending=1,inplace=True)
    df.reset_index(inplace=True,drop=True)
    df['Batter_player']=df['Batter'].copy()
    df['Pitcher_player']=df['Pitcher'].copy()
    for player in ['Batter','Pitcher']:
        player2=player+'_player'
        center=centerd[player]
        old_norm=10000
        indices=[6,9]
        #print df[player][0]
        for person in range(len(df[player][0])):
            hips=np.asarray(df[player][0][person])[indices]

            hips=hips[np.sum(hips,axis=1)<>0]
            mean_hips=np.mean(hips,axis=0)


            norm= abs(mean_hips[0]-center[0])+abs(mean_hips[1]-center[1]) #6 hip
            if norm<old_norm:
                loc=person
                old_norm=norm

        df[player2][0]=df[player][0][loc]
        globals()['old_array_%s'%player]=np.asarray(df[player][0][loc])

    for frame in df['Frame'][1:len(df)]:
        for player in ['Batter','Pitcher']:
            df,globals()['old_array_%s'%player]=player_localization(df,frame,player,globals()['old_array_%s'%player])

    for player in player_list:
        for index in index_list:
            df=mix_right_left(df,index,player)
    df=continuity(df,'Pitcher')
    df=continuity(df,'Batter')

    return df[['Frame','Pitcher_player','Batter_player']]
