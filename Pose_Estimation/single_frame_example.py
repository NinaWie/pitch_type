import time
import argparse
import pandas as pd
import codecs, json
import ast
import cv2
from os.path import isfile, join
from os import listdir
import numpy as np
from config_reader import config_reader
from PoseModels import AvailableModels
from matplotlib import pyplot as plt
# from Functions import handle_one, df_coordinates, score_coordinates, define_model

param_, model_ = config_reader()
USE_MODEL = model_['use_model']

parser = argparse.ArgumentParser(description='Execute model on a single frame.')
parser.add_argument('input_file', metavar='DIR', help='video file to process')
args = parser.parse_args()
input_file = args.input_file


og_model = AvailableModels[USE_MODEL]()
fast_model = AvailableModels['fast']()

if __name__ == '__main__':
    j=0
    center_dic={}

    path_input_dat = input_file + '.dat'

    for i in open(path_input_dat).readlines():
        datContent=ast.literal_eval(i)
    bottom_p=datContent['Pitcher']['bottom']
    left_p=datContent['Pitcher']['left']
    right_p=datContent['Pitcher']['right']
    top_p=datContent['Pitcher']['top']
    center_dic['Pitcher']=np.array([abs(top_p-bottom_p)/2., abs(left_p-right_p)/2.])

    # video_capture = cv2.VideoCapture(input_file)
    # ret, frame = video_capture.read()
    frame = cv2.imread('./hi_pitcher_frame_1.png')
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    plt.imshow(frame)
    plt.ion()
    plt.show()
    raw_input('[INPUT FRAME]:')

    if frame is None:
        print '| Video stream ended prematurely!'
        exit(1)

    # pitcher_img = frame[top_p:bottom_p, left_p:right_p]
    pitcher_img = frame
    multiplier = [x * model_['boxsize'] / pitcher_img.shape[0] for x in param_['scale_search']]
    scale = multiplier[0]

    (output1, output2), (heatmap, paf) = og_model.evaluate(pitcher_img, scale=scale)
    print '| Starting evaluation...'
    frames_t0 = time.time()
    _, (heatmap2, _) = fast_model.evaluate(pitcher_img, scale=scale)
    print '| Time to process entire video:', time.time() - frames_t0
    fig, axes = plt.subplots(2, 4, figsize=(15, 10))

    def double_img(in_img):
        return cv2.resize(in_img, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    show_imgs = 4
    for ii, ax in enumerate(axes.flat):
        if ii < show_imgs:
            ax.imshow(double_img(pitcher_img))
            ax.imshow(double_img(heatmap[ii * 2]), alpha=0.5)
        else:
            ax.imshow(double_img(pitcher_img))
            ax.imshow(double_img(heatmap2[(ii - show_imgs) * 2]), alpha=0.5)

    plt.tight_layout()
    plt.ion()
    plt.show()

    raw_input('[END PROGRAM]:')
