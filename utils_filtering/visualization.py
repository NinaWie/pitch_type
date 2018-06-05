import cv2
import numpy as np
import matplotlib
import matplotlib.pylab as plt
import json
from scipy.signal import argrelextrema
from scipy import ndimage
import ast
from scipy.spatial.distance import cdist
from scipy.signal import butter, lfilter, freqz, group_delay, filtfilt
from skvideo import io
from os import listdir
import pandas as pd
from scipy.interpolate import interp1d


def color_video(json_array, vid_file, start = 0, cut_frame = True, end = 300, point = 8, printing =None, plotting=True):
    colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255], [0, 170, 255],
              [0, 0, 0],
          [255, 0, 85], [0, 255, 170],  [0, 170, 255], [0, 85, 255],  [85, 0, 255],
          [170, 0, 255], [255, 0, 170], [255, 0, 85], [255, 0, 85], [0, 255, 170],  [0, 170, 255],
              [0, 85, 255],  [85, 0, 255],
          [170, 0, 255], [255, 0, 170], [255, 0, 85], [255, 0, 85], [0, 255, 170],  [0, 170, 255], [0, 85, 255],  [85, 0, 255], \
          [170, 0, 255], [255, 0, 170], [255, 0, 85]]
    colors_string = ["blue", "green", "red", "tuerkis", "pink", "yellow", "orange", "black", "purple"]
    nr_joints =12
    #print(json_array.shape)
    #writer = cv2.VideoWriter("/Users/ninawiedemann/Desktop/UNI/Praktikum/high_quality_testing/outputs_example/test.avi",cv2.VideoWriter_fourcc(*"XVID") , 20, (500,800))

    #writer = io.FFmpegWriter("/Users/ninawiedemann/Desktop/UNI/Praktikum/high_quality_testing/outputs_example/test.avi", (10,800,500,3))
    #writer.open()
    video_capture = cv2.VideoCapture(vid_file)
    print(vid_file)
    if start!=0:
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, 100)
    arr = [] #np.zeros((100,800,500,3))


    # fig = plt.figure(figsize=(5, 15)) # for subplots
    for k in range(start, end):
        #print(k)
        if printing!=None:
            #print("dist_min",  "ratio_min")
            #print(colors_string[printing[k][0]], colors_string[printing[k][1]])
            print(printing[k])
        ret, frame = video_capture.read()
        if frame is None:
            print("end", k)
            break
        if len(np.array(json_array[k]).shape)==2:
            all_peaks = np.reshape(np.array(json_array[k]), (12, 1,2))
        else:
            all_peaks = np.array(json_array[k])
        #print(all_peaks.shape)

        if cut_frame:
            canvas = frame[top_b:bottom_b, left_b:right_b] # cv2.imread(f) # B,G,R order
        else:
            canvas = frame
        oriImg = canvas.copy()

        for i in range(len(all_peaks)):
            #print("person", all_peaks[i])
            for j in range(len(all_peaks[i])):
                cv2.circle(canvas, (int(all_peaks[i,j,0]),int(all_peaks[i,j,1])) , point, colors[i], thickness=-1)

        to_plot = cv2.addWeighted(oriImg, 0.3, canvas, 0.7, 0)
        arr.append(to_plot[:,:,[2,1,0]])
        if plotting:
            plt.imshow(to_plot[:,:,[2,1,0]])
            fig = matplotlib.pyplot.gcf()
            fig.set_size_inches(12, 12)
            plt.show()
    arr = np.array(arr)
    return arr


def color_box(vid, bbox, ax, color = "red"):

    ax.add_patch(
    plt.Rectangle((int(bbox[0]), int(bbox[2])),
                  int(bbox[1]-bbox[0]), int(bbox[3]-bbox[2]), fill=False,
                  edgecolor=color, linewidth=3.5)
    )
