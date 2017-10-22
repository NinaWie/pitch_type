import time
from os.path import isfile, join
from os import listdir
import numpy as np
import numpy.linalg as la
from config_reader import config_reader
from PoseModels import AvailableModels
from matplotlib import pyplot as plt
import cv2

param_, model_ = config_reader()

def define_models():
    param_, model_ = config_reader()
    USE_MODEL = model_['use_model']

    # parser = argparse.ArgumentParser(description='Execute model on a single frame.')
    # parser.add_argument('input_file', metavar='DIR', help='video file to process')
    # args = parser.parse_args()
    # input_file = ar/gs.input_file
    # input_file = ''

    og_model = AvailableModels[USE_MODEL]()
    fast_model = AvailableModels['fast']()
    return og_model, fast_model

if __name__ == '__main__':
    og_model, fast_model = define_models()

    j=0
    center_dic={}

    # path_input_dat = input_file + '.dat'

    # for i in open(path_input_dat).readlines():
    #     datContent=ast.literal_eval(i)
    # bottom_p=datContent['Pitcher']['bottom']
    # left_p=datContent['Pitcher']['left']
    # right_p=datContent['Pitcher']['right']
    # top_p=datContent['Pitcher']['top']
    # center_dic['Pitcher']=np.array([abs(top_p-bottom_p)/2., abs(left_p-right_p)/2.])

    # video_capture = cv2.VideoCapture(input_file)
    # ret, frame = video_capture.read()
    frame = cv2.imread('./hi_pitcher_frame_1.png')
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    plt.imshow(frame)
    plt.ion()
    plt.show()
    raw_input('[INPUT FRAME]:')

    # if frame is None:
    #     print '| Video stream ended prematurely!'
    #     exit(1)

    # pitcher_img = frame[top_p:bottom_p, left_p:right_p]
    pitcher_img = frame
    multiplier = [x * model_['boxsize'] / pitcher_img.shape[0] for x in param_['scale_search']]
    scale = multiplier[0]


    print '| Starting evaluation...'
    for ii in range(5):
        frames_t0 = time.time()
        (output1, output2), (heatmap, paf) = og_model.evaluate(pitcher_img, scale=scale)
        dt = time.time() - frames_t0
        frames_t0 = time.time()
        _, (heatmap2, _) = fast_model.evaluate(pitcher_img, scale=scale)
        dt2 = time.time() - frames_t0
        print '| Time difference: G_%.2f vs F_%.2f' % (dt, dt2)

    plt.ion()


    def double_img(in_img):
        return cv2.resize(in_img, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)


    show_imgs = 4
    iter = 0
    last_ind = len(heatmap)
    plt.figure()
    fig, axes = plt.subplots(1, 4, figsize=(15, 10))

    def stack_rgb_img(one, two, dims=3):
        if dims is 3:
            sy, sx, sc = one.shape
            out = np.zeros((sy * 2, sx, sc), dtype=np.uint8)
            out[:sy, :, :] = one
            out[sy:, :, :] = two
        else:
            sy, sx = one.shape
            out = np.zeros((sy * 2, sx))
            out[:sy, :] = one
            out[sy:, :] = two
        return out

    for ii in range(last_ind):
        top_plot = axes.flat[iter]
        # bot_plot = axes.flat[show_imgs + iter]

        top_plot.set_title('%d (%.2f)' % (ii+1, la.norm(heatmap[ii] - heatmap2[ii])))
        stacked_bg = stack_rgb_img(pitcher_img, pitcher_img)
        top_plot.imshow(stacked_bg)
        stacked_heatmap = stack_rgb_img(heatmap[ii], heatmap2[ii], dims=2)
        top_plot.imshow(stacked_heatmap, alpha=0.5)

        iter += 1

        if (ii + 1) % show_imgs is 0 or ii is last_ind - 1:
            plt.tight_layout()
            plt.show()
            if ii is last_ind - 1:
                raw_input('[END PROGRAM]:')
            else:
                raw_input('[NEXT PLOTS ~%d]:' % (ii + 1))
            # fig.cla()
            iter = 0
