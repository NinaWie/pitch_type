#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 14:26:20 2017

@author: estelleaflalo
"""
import os
from os.path import isfile, join
from os import listdir
import argparse
import pandas as pd
parser = argparse.ArgumentParser(description='Pose Estimation Baseball')
parser.add_argument('dir1', metavar='DIR',
                    help='folder where merge.csv are')
parser.add_argument('dir2', metavar='DIR',
                    help='folder where merge.csv are')


args = parser.parse_args()
dir1=args.dir1 #example /scratch/ea1921/atl
dir2=args.dir2 #example /scratch/ea1921/pyt.../2017
dates=os.listdir(dir1)
all_csv=[]

for view in ['cf','sv']:
    all_frames=[]
    i=0             
    for d in dates:
        i+=1
        csv_folder=dir2+'/'+d+'/'+view
        frames=[]    
        if len(listdir(csv_folder))==0:
            continue
        else:
            metadata_path=dir1+'/'+d+'/csv_gameplay.csv' 
            for csv_path in [csv_folder+'/'+f for f in listdir(csv_folder) if isfile(join(csv_folder, f))]:
 
                df=pd.read_csv(csv_path)

                df=df.transpose() 
                df.columns=df[df.columns[:-9]].loc['Unnamed: 0'].tolist()+df[df.columns[-9:]].loc['Frame'].tolist()
                df=df.drop(['Frame'])
                df=df.drop(['Unnamed: 0'])
                frames.append(df) 
            df_final = pd.concat(frames).reset_index(drop=True) 

            metadata=pd.read_csv(metadata_path,sep=';')
            metadata["key"] = metadata["game_primary_key"].map(str) +'-'+ metadata["play_id"]
            merge=metadata.merge(df_final, left_on='key', right_on='Game', how='inner')  
            merge['View']=view
            merge.to_csv(dir2+'/'+d+'/'+view+'_conc.csv')
            print i, dir2+'/'+d+'/'+view+'_conc.csv'
            all_frames.append(merge)

    DATA=pd.concat(all_frames,axis=0,ignore_index=True)
    DATA.to_csv(dir2+'/' +view+'_data.csv')

#python Conc_all "/scratch/ea1921/atl" "/scratch/ea1921/pytorch_Realtime_Multi-Person_Pose_Estimation/2017"

