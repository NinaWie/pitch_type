#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 14:51:41 2017

@author: estelleaflalo
"""

from pycocotools.coco import COCO
import numpy as np

test1=COCO("/Users/estelleaflalo/Downloads/annotations-3/person_keypoints_train2014.json")
#only cat=1 are in there (relative to persons)

test2=COCO("/Users/estelleaflalo/Downloads/annotations-2/instances_train2014.json")

for cat in test2.dataset['categories']:
    print cat
    
    
baseball=[]
baseball.extend(test2.getImgIds(catIds=[1,37]))  #sport
baseball.extend(test2.getImgIds(catIds=[1,39]))    #glove
baseball.extend(test2.getImgIds(catIds=[1,40]))    #bat

baseball_im_id=np.unique(baseball)

baseball=[]
baseball.extend(test2.getAnnIds(catIds=[1,37]))  #sport
baseball.extend(test2.getAnnIds(catIds=[1,39]))    #glove
baseball.extend(test2.getAnnIds(catIds=[1,40]))    #bat

baseball_ann_id=np.unique(baseball)



#4463 images


annotations=COCO("/Users/estelleaflalo/Downloads/annotations-3/person_keypoints_train2014.json").dataset["annotations"]
images=COCO("/Users/estelleaflalo/Downloads/annotations-3/person_keypoints_train2014.json").dataset["images"]
inter=[ann for ann in annotations if ann['id'] in baseball_ann_id]

import json
import copy

with open("/Users/estelleaflalo/Downloads/annotations-3/person_keypoints_train2014.json") as json_data:
    #json_data=open("/Users/estelleaflalo/Downloads/annotations-3/person_keypoints_train2014.json") 
    data = json.load(json_data)
    data_cop=copy.deepcopy(data)
    j=0
    for i in range(len(data['annotations'])):
        if data['annotations'][i]['image_id'] not in baseball_im_id:
            del data_cop['annotations'][i-j]
            print j
            j+=1   
    j=0
    for i in range(len(data['images'])):
        if data['images'][i]['id'] not in baseball_im_id:
            del data_cop['images'][i-j]
            j+=1   
        else: 
            print data['images'][i]['flickr_url']


with open('/Users/estelleaflalo/Downloads/annotations-3/person_keypoints_sport_train2014.json', 'w') as outfile:
    json.dump(data_cop, outfile)


with open("/Users/estelleaflalo/Downloads/annotations-3/person_keypoints_val2014.json") as json_data:
    data = json.load(json_data)
    data_cop=copy.deepcopy(data)
    j=0
    for i in range(len(data['annotations'])):
        if data['annotations'][i]['image_id'] not in baseball_im_id:
            del data_cop['annotations'][i-j]
            print j
            j+=1   
    j=0
    for i in range(len(data['images'])):
        if data['images'][i]['id'] not in baseball_im_id:
            del data_cop['images'][i-j]
            j+=1   
        else: 
            print data['images'][i]['flickr_url']

with open('/Users/estelleaflalo/Downloads/annotations-3/person_keypoints_sport_val2014.json', 'w') as outfile:
    json.dump(data_cop, outfile)

    
    #inter=len([ann for ann in data["annotations"] if ann['id'] in baseball_ann_id]) 
#inter_im=len([im for im in data["images"] if im['id'] in baseball_im_id])

#sport images ==  4 327
#not sport = 78 455
#len(data['images'])
#len(data_cop['annotations'])

#for im in data_cop['images']:
#    print im['flickr_url']


#for ann in data['annotations']:
#    if ann['id'] in baseball_ann_id:
#        print ann
        
        #print evthg

 #k=0
#for ann in data['annotations']:
#    if ann['image_id'] in baseball_im_id:
#        print ann
#        k+=1
#        if k==10:
#            break