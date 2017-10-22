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

HEATMAP_LABELS = [
    'Neck?',
    'Center',
    'Sho',
    'Elb',
    'Hand',
    'Sho',
    'Elb',
    'Hand',
    'Hip',
    'Knee',
    'Foot',
    'Hip',
    'Knee',
    'Foot',
    'Eye',
    'Eye',
    'Ear',
    'Ear',
    'All?'
]

HEATMAP_NONCRITICAL = [0, 18, 17, 16, 15, 14]

def define_models():
    param_, model_ = config_reader()
    USE_MODEL = model_['use_model']

    # parser = argparse.ArgumentParser(description='Execute model on a single frame.')
    # parser.add_argument('input_file', metavar='DIR', help='video file to process')
    # args = parser.parse_args()
    # input_file = ar/gs.input_file
    # input_file = ''

    og_model = AvailableModels[USE_MODEL]()
    # fast_model = AvailableModels['fast']()
    return og_model

def plot_baseline_compare(plot, heat1, heat2):
    fig, axes = plot
    staticplot, scaledplot = axes.flatten()

    staticplot.cla(), scaledplot.cla()
    staticplot.axis([-1, 20, 0, 5])
    staticplot.set_autoscale_on(False)

    losses = []
    for ii in range(19):
        # label = HEATMAP_LABELS[ii]
        loss = la.norm(heat1[ii] - heat2[ii])
        losses.append(loss)


    colors = ['red' if ii in HEATMAP_NONCRITICAL else 'blue' for ii in range(19)]
    staticplot.scatter(range(19), losses, color=colors)
    scaledplot.scatter(range(19), losses, color=colors)
    for ii, txt in enumerate(HEATMAP_LABELS):
        staticplot.annotate(txt, (range(19)[ii], losses[ii]))
        scaledplot.annotate(txt, (range(19)[ii], losses[ii]))
    plt.draw()
    plt.pause(0.01)
    # time.sleep(1)
    return losses

if __name__ == '__main__':
    og_model = define_models()

    frame = cv2.imread('./hi_pitcher_frame_1.png')
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    plt.ion()
    pitcher_img = frame

    # do one execution for baseline
    multiplier = [x * model_['boxsize'] / pitcher_img.shape[0] for x in param_['scale_search']]
    scale = multiplier[0]
    (output1, output2), (heatmap, paf) = og_model.evaluate(pitcher_img, scale=scale)

    # examine_layer = 'Mconv1_stage6_L2'

    loss_plot = plt.subplots(1, 2, figsize=(14,7))
    # plt.show()
    plot_baseline_compare(loss_plot, heatmap, heatmap)
    raw_input('[Baseline ground comparison]:')

    with open('prune_ok.txt', 'wb') as dump:
        dump.write('=======\n')

    all_layers = og_model.model.layers[47:]
    for layer_i, keraslayer in enumerate(all_layers):

        print '========= PRUNE SENSITIVITY (%d/%d) =========' % (layer_i, len(all_layers))
        print '| Layer name:', keraslayer.name
        print '| Layer shape:', keraslayer.get_weights()[0].shape
        print '| Output dim :', keraslayer.get_weights()[0].shape[-1]

        if len(keraslayer.get_weights()) is not 2:
            continue

        should_prune = []
        outDim = keraslayer.get_weights()[0].shape[-1]
        for dim in range(outDim):
            wmat, bmat = keraslayer.get_weights()
            wmat0 = wmat.copy()
            bmat0 = bmat.copy()

            bmat[dim] = 0
            wmat[:, :, :, dim] = 0
            keraslayer.set_weights([wmat0, bmat])

            _, (heatmap_mod, _) = og_model.evaluate(pitcher_img, scale=scale)
            losses = plot_baseline_compare(loss_plot, heatmap, heatmap_mod)

            maxind = np.argmax(losses[:-1])
            print dim + 1, '/', outDim, maxind
            if int(maxind) in HEATMAP_NONCRITICAL:
                # raw_input('[Continue]:')
                should_prune.append(dim)
            keraslayer.set_weights([wmat0, bmat0])
        with open('prune_ok.txt', 'wb') as dump:
            dump.write('%s:%s\n' % (keraslayer.name, ','.join(should_prune)))


    # plt.figure(figsize=(14, 10))
