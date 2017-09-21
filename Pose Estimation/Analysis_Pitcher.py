#!/usr/bin/env python2datContent = [i.strip().split() for i in open(path_input_dat).readlines()]
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 15:17:48 2017

@author: estelleaflalo
"""

import argparse
import pandas as pd
import numpy as np
#WITHOUT /

merge_path="./conc15-16.csv"
#parser = argparse.ArgumentParser(description='Pose Estimation Baseball')
#parser.add_argument('input_dir15', metavar='DIR',
#                    help='folder where merge.csv are')
#parser.add_argument('input_dir16', metavar='DIR',
#                    help='folder where merge.csv are')
#
#args = parser.parse_args()
#input_folder15=args.input_dir15
#merge11_csv=input_folder15+'/merge_200.csv'
#merge22_csv=input_folder15+'/merge_400.csv'
#merge33_csv=input_folder15+'/merge_600.csv'
#merge11=pd.read_csv(merge11_csv)
#merge22=pd.read_csv(merge22_csv)
#merge33=pd.read_csv(merge33_csv)
#
#
#
#
#
#input_folder16=args.input_dir16
#merge1_csv=input_folder16+'/merge_200.csv'
#merge2_csv=input_folder16+'/merge_400.csv'
#merge3_csv=input_folder16+'/merge_600.csv'
#merge4_csv=input_folder16+'/merge_800.csv'
#merge5_csv=input_folder16+'/merge_1000.csv'
#merge6_csv=input_folder16+'/merge_1200.csv'
#merge1=pd.read_csv(merge1_csv)
#merge2=pd.read_csv(merge2_csv)
#merge3=pd.read_csv(merge3_csv)
#merge4=pd.read_csv(merge4_csv)
#merge5=pd.read_csv(merge5_csv)
#merge6=pd.read_csv(merge6_csv)
#
#
#

def smooth(x,window_len=6,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """ 
     
    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."
        

    if window_len<3:
        return x
    
    
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
    

    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
    
    y=np.convolve(w/w.sum(),s,mode='valid')
    return y    

from numpy import array, zeros, argmin, inf, equal, ndim
from scipy.spatial.distance import cdist

def _traceback(D):
    i, j = array(D.shape) - 2
    p, q = [i], [j]
    while ((i > 0) or (j > 0)):
        tb = argmin((D[i, j], D[i, j+1], D[i+1, j]))
        if (tb == 0):
            i -= 1
            j -= 1
        elif (tb == 1):
            i -= 1
        else: # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return array(p), array(q)
    
def dtw(x, y, dist):
    """
    Computes Dynamic Time Warping (DTW) of two sequences.
    :param array x: N1*M array
    :param array y: N2*M array
    :param func dist: distance used as cost measure
    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    assert len(x)
    assert len(y)
    r, c = len(x), len(y)
    D0 = zeros((r + 1, c + 1))
    D0[0, 1:] = inf
    D0[1:, 0] = inf
    D1 = D0[1:, 1:] # view
    for i in range(r):
        for j in range(c):
            D1[i, j] = dist(x[i], y[j])
    C = D1.copy()
    for i in range(r):
        for j in range(c):
            D1[i, j] += min(D0[i, j], D0[i, j+1], D0[i+1, j])
    if len(x)==1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = _traceback(D0)
    return D1[-1, -1] / sum(D1.shape), C, D1, path

def fastdtw(x, y, dist):
    """
    Computes Dynamic Time Warping (DTW) of two sequences in a faster way.
    Instead of iterating through each element and calculating each distance,
    this uses the cdist function from scipy (https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html)
    :param array x: N1*M array
    :param array y: N2*M array
    :param string or func dist: distance parameter for cdist. When string is given, cdist uses optimized functions for the distance metrics.
    If a string is passed, the distance function can be 'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'wminkowski', 'yule'.
    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    assert len(x)
    assert len(y)
    if ndim(x)==1:
        x = x.reshape(-1,1)
    if ndim(y)==1:
        y = y.reshape(-1,1)
    r, c = len(x), len(y)
    D0 = zeros((r + 1, c + 1))
    D0[0, 1:] = inf
    D0[1:, 0] = inf
    D1 = D0[1:, 1:]
    D0[1:,1:] = cdist(x,y,dist)
    C = D1.copy()
    for i in range(r):
        for j in range(c):
            D1[i, j] += min(D0[i, j], D0[i, j+1], D0[i+1, j])
    if len(x)==1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = _traceback(D0)
    return D1[-1, -1] / sum(D1.shape), C, D1, path


#merge1=pd.read_csv("/Users/estelleaflalo/Desktop/Stage_Fin_Etudes/NYU/Internship/test/16/merge_200.csv")
#merge2=pd.read_csv("/Users/estelleaflalo/Desktop/Stage_Fin_Etudes/NYU/Internship/test/16/merge_400.csv")
#merge3=pd.read_csv("/Users/estelleaflalo/Desktop/Stage_Fin_Etudes/NYU/Internship/test/16/merge_600.csv")
#merge4=pd.read_csv("/Users/estelleaflalo/Desktop/Stage_Fin_Etudes/NYU/Internship/test/16/merge_800.csv")
#merge5=pd.read_csv("/Users/estelleaflalo/Desktop/Stage_Fin_Etudes/NYU/Internship/test/16/merge_1000.csv")
#merge6=pd.read_csv("/Users/estelleaflalo/Desktop/Stage_Fin_Etudes/NYU/Internship/test/16/merge_1200.csv")

#for i in range(170):
#    for merge in [merge11,merge22,merge33,merge1,merge2,merge3,merge4,merge5,merge6]:
#        if i>159:
#            if str(i) in merge.columns:
#                del merge[str(i)]
                     
#merge=pd.concat((merge11,merge22,merge33,merge1,merge2,merge3,merge4,merge5,merge6),axis=0)
#merge.to_csv("/Users/estelleaflalo/Desktop/Stage_Fin_Etudes/NYU/Internship/test/16/merge_16.csv")

merge=pd.read_csv(merge_path)


for i in range(170):
    if i>159:
        if str(i) in merge.columns:
            del merge[str(i)]
mergedrop=merge.dropna(axis=1, how='all')

#PITCH TYPE

from numpy.linalg import norm
import operator
import itertools

pitch_type_label=mergedrop[mergedrop['Player']=='Pitcher']['Pitch Type']
pitch_type_label=pd.get_dummies(pitch_type_label)

pitcher_label=mergedrop[mergedrop['Player']=='Pitcher']['Pitcher']
pitcher_label=pd.get_dummies(pitcher_label)


pitch_data=mergedrop[mergedrop['Player']=='Pitcher'].copy()#.iloc[:,-162:]

#del pitch_type_data['Game'],pitch_type_data['Player']
#data=mergedrop[mergedrop['Player']=='Pitcher'].copy()

    
pitch_data=pitch_data.dropna(thresh=5, axis=1)

coord=[]
for i in range(160):
    coord.append(str(i)+'.0')

label_type_dic={}
for i,col in enumerate(pitch_type_label.columns.tolist()):
    label_type_dic[i]=col

label_pitcher_dic={}
for i,col in enumerate(pitcher_label.columns.tolist()):
    label_pitcher_dic[i]=col

p=0
#df=pd.DataFrame(index=range(6*len(pitch_data)),columns=['key i','key j','pitch type i','pitch type j','pitcher i','pitcher j','dist'])   
df=pd.DataFrame(columns=['key i','key j','pitch type i','pitch type j','pitcher i','pitcher j','dist'])   

for i,j in  list(itertools.combinations(range(len(pitch_data)), r=2)):
#for i in range(len(pitch_data)):
    #if i%100==0:
 #   for j in range(len(pitch_data)):     
    x=np.array([np.array(eval(c)) for c in pitch_data.iloc[i][coord]])[:,:-5,:]#-5 not to take into account head's coord
    y=np.array([np.array(eval(c)) for c in pitch_data.iloc[j][coord]])[:,:-5,:]
    dist, cost, acc, path = dtw(x, y, dist=lambda x, y: norm(x - y))
    df.loc[p]=[pitch_data.iloc[i]['key'],pitch_data.iloc[j]['key'],pitch_data.iloc[i]['Pitch Type'],pitch_data.iloc[j]['Pitch Type'],pitch_data.iloc[i]['Pitcher'],pitch_data.iloc[j]['Pitcher'],dist]
#            df['key i'].loc[p]=pitch_data.iloc[i]['key']
#            df['pitch type i'].loc[p]=label_type_dic[np.where(pitch_type_label.iloc[i]<>0)[0][0]]
#            df['pitcher i'].loc[p]=pitch_data.iloc[i]['Pitcher']
#            df['key j'].loc[p]=pitch_data.iloc[j]['key']
#            df['pitch type j'].loc[p]=label_type_dic[np.where(pitch_type_label.iloc[j]<>0)[0][0]]
#            df['pitcher j'].loc[p]=pitch_data.iloc[j]['Pitcher']
#            df['dist'].loc[p]=dist
    p+=1
    print dist,i,j,pitch_data.iloc[j][coord]# pitch_type_label[pitch_type_label==1].iloc[i],pitch_type_label[pitch_type_label==1].iloc[j]`
    df.sort('dist',inplace=True)

    df.to_csv('./analysis-15-16.csv')



##-------------------------------
##df=pd.read_csv("/Users/estelleaflalo/Desktop/Stage_Fin_Etudes/NYU/Internship/test/analysis_every100.csv")
##491456-13b6f482-06be-41f2-a320-36bcbbab69ab  close to himpd.read_csv("/Users/estelleaflalo/Desktop/Stage_Fin_Etudes/NYU/Internship/test/16/merge_16.csv")

#
##491456-4a96724e-3bdf-4ca3-aa2d-e12a79b22043  close
# #491463-a4737f07-80a4-4ccb-977b-759a4c1c4bf2  not close
#x={}
#y={} 
#x["key"]='491456-4a96724e-3bdf-4ca3-aa2d-e12a79b22043'
#x["coord"]=np.array([np.array(eval(c)) for c in pitch_data[pitch_data['Game']==x["key"]][coord].iloc[0]])[:,:-5,:]
##x["result"]=
#y=np.array([np.array(eval(c)) for c in pitch_data[pitch_data['Game']=='491456-645eb2ce-4b4d-4c96-bcb1-87de18832083'][coord].iloc[0]])[:,:-5,:]
#col_pitcher=['Pitcher_Right_shoulder','Pitcher_Left_shoulder','Pitcher_Right_elbow','Pitcher_Right_wrist','Pitcher_Left_elbow','Pitcher_Left_wrist','Pitcher_Right_hip','Pitcher_Right_knee','Pitcher_Right_ankle','Pitcher_Left_hip','Pitcher_Left_knee','Pitcher_Left_ankle','Pitcher_Neck','Pitcher_Right_eye','Pitcher_Right_ear','Pitcher_Left_eye','Pitcher_Left_ear']
#
#for i in range(12):
#    x_knee=x[:,i,:]
#    y_knee=y[:,i,:]
#
#
#    plot(smooth(x_knee[:,0]),smooth(x_knee[:,1]))
#    plot(smooth(y_knee[:,0]),smooth(y_knee[:,1]))
#    plt.title('Part'+col_pitcher[i])
#    plt.show()
#df=pd.read_csv("/Users/estelleaflalo/Desktop/Stage_Fin_Etudes/NYU/Internship/test/analysis-15.csv")
#
#feature='Play Outcome'
#feature="Batter"
#feature='Umpire Call'
#
#dic_diff={}
#dic_all={}
#old=10000000
#column=merge.columns.tolist()[1:-169]
#for i,feature in enumerate(column):
#    print feature
#    print i, len(column)
#    test=df.merge(merge[['Game',feature]], left_on='key i', right_on='Game', how='inner')
#    test=test.rename(columns = {feature:feature + ' i'})
#    del test['Game']#,test['Unnamed: 0']
#    
#    test=test.merge(merge[['Game',feature]], left_on='key j', right_on='Game', how='inner')
#    test=test.rename(columns = {feature:feature+ ' j'})
#    del test['Game']
#    test.sort('dist',inplace=True)
#    
#
#    
#    length1=len(test[test['pitcher i']<> test['pitcher j']][:100][test[feature + ' i']==test[feature + ' j']])
#    dic_diff[feature]=length1
#    
#    length2=len(test[:100][test[feature + ' i']==test[feature + ' j']])
#    dic_all[feature]=length2
#    
#    
#    
#    test[:10]
#    test[[feature + ' i',feature + ' j','dist']][test['pitcher i']<> test['pitcher j']][:100]
#
#for feature in column:
#    if feature in dic_all.keys():
#        dic_all[feature+ '(' +str(len(np.unique(merge[feature])))+')'] = dic_all.pop(feature)
#    if feature in dic_diff.keys():
#        dic_diff[feature+ '(' +str(len(np.unique(merge[feature])))+')'] = dic_diff.pop(feature)  
#    
#dic_all2=dic_all.copy()
#dic_diff2=dic_diff.copy()
#
##dic_all=sorted(dic_all.items(), key=operator.itemgetter(1),reverse=True)
#for k,v in dic_all.items():
#    if v == 0:
#       del dic_all[k]
#pd.DataFrame.from_dict(dic_all, orient='index').sort(0,ascending=True).plot(kind='barh',title='From all pitches 16')
#
##dic_diff=sorted(dic_diff.items(), key=operator.itemgetter(1),reverse=True)
#for k,v in dic_diff.items():
#    if v == 0:
#       del dic_diff[k]
#pd.DataFrame.from_dict(dic_diff, orient='index').sort(0,ascending=True).plot(kind='barh',title='For different pitchers 16')
#
