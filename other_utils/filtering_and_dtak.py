import numpy as np
import json
import cv2
import pandas as pd
from scipy import ndimage
import matplotlib.pylab as plt
from numpy import array, zeros, argmin, inf, equal, ndim
from scipy.spatial.distance import cdist


def smooth_estelle(x,window_len=6,window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """
    #print(len(x))
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s = np.r_[x[window_len//2-1:0:-1],x,x[-1:-window_len//2:-1],x[-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
    #print(w)

    y=np.convolve(w/w.sum(),x,mode='same')
    #print(len(y))
    return y

def kalmann(sequence):
    # intial parameters
    n_iter = len(sequence)
    sz = (n_iter,) # size of array
    #x = -0.37727 # truth value (typo in example at top of p. 13 calls this z)
    z = sequence #np.random.normal(x,0.1,size=sz) # observations (normal about x, sigma=0.1)

    Q = 1e-5 # process variance

    # allocate space for arrays
    xhat=np.zeros(sz)      # a posteri estimate of x
    P=np.zeros(sz)         # a posteri error estimate
    xhatminus=np.zeros(sz) # a priori estimate of x
    Pminus=np.zeros(sz)    # a priori error estimate
    K=np.zeros(sz)         # gain or blending factor

    R = 0.1**3 # estimate of measurement variance, change to see effect

    # intial guesses
    xhat[0] = sequence[0]
    P[0] = 1.0

    for k in range(1,n_iter):
        # time update
        xhatminus[k] = xhat[k-1]
        Pminus[k] = P[k-1]+Q

        # measurement update
        K[k] = Pminus[k]/( Pminus[k]+R )
        xhat[k] = xhatminus[k]+K[k]*(z[k]-xhatminus[k])
        P[k] = (1-K[k])*Pminus[k]
    return xhat

def lowpass(sequence, cutoff = 1, fs = 15, order=5):
    """
    returns lowpass filtered sequence of same length
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / float(nyq)
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b,a,sequence) # lfilter(b, a, data)
    return y

# DTAK


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


def dtak_nn_single_joints(play, template, template_label, plot = False):
    result = []
    for joint in range(len(play[0])):
        # joint = 6
        xy = 1
        # distances = []
        test_sequ = play[:, joint]
        test_sequ = (test_sequ-np.mean(test_sequ))/np.std(test_sequ)
        # for i in range(2, 3): #len(labels_first_ten)):
        sequ = template[:,joint]
        sequ = (sequ-np.mean(sequ))/np.std(sequ)
            # plt.plot(sequ, label = i)
        x = sequ
        y = test_sequ
        dist, cost, acc, path = fastdtw(x, y, dist=lambda x, y: np.linalg.norm(x - y))
        # distances.append(np.sum(cost))
        # ind = np.argmin(distances)
        # nn = joints_array_batter[ind, :, joint, xy]
        # nn = (nn-np.mean(nn))/np.std(nn)
        nn_label = template_label #labels_first_ten[ind]
        # print("label nn", nn_label)
        # x = nn
        # y = test_sequ
        if plot:
            plotting(x,y, path, title = joints_list[joint])
        # dist, cost, acc, path = fastdtw(x, y, dist=lambda x, y: np.linalg.norm(x - y))

        loc = list(path[0]).index(nn_label)
        # print(loc)
        #print(path[0][loc])
        res = path[1][loc]
        # print("new label", res)
        result.append(res)
    print(result)
    return int(np.median(result))


def normalize(play):
    means = np.mean(play, axis = 0)
    stds = np.std(play, axis = 0)
    return np.array([(elem-means)/stds for elem in play])

def dtak_nn(play, template, template_label, plot = False):
    x = normalize(play[:, :])
    y = normalize(template[:,:])
    dist, cost, acc, path = fastdtw(x, y, 'mahalanobis')  #dist=lambda x, y: np.linalg.norm(x - y))
    if plot:
        plotting(x,y, path, title = "example")
    loc = list(path[0]).index(template_label)
    res = path[1][loc]
    return res


def find_null(joint_array):
    xn = normalize(joint_array)
    res = []
    for j in range(xn.shape[1]):
        res.append(np.argmin(np.absolute(xn[:,j])))
    return int(np.median(res))

def align_sequence_simple(joint_array, template, label):
    xn = normalize(template)
    yn = normalize(joint_array)
    distance_label = [np.linalg.norm((xn[label]-yn[j])[0]) for j in range(len(xn))]
    inds = np.argsort(distance_label)
    print(inds[:6])
    return int(np.mean(inds[:6]))

def simple_shift(joint_array, template, label):
    xn = normalize(template)# [90:160]
    yn = normalize(joint_array)

    gradx = np.gradient(xn, axis = 0)
    grady = np.gradient(yn, axis = 0)
    minx = np.argmin(np.mean(gradx, axis = 1))
    miny = np.argmin(np.mean(grady, axis = 1))
    grad_shift = minx-miny

    diff = []
    #plt.figure(figsize = (20,10))
    #plt.plot(np.mean(xn, axis = 1), linewidth = 4)

    for i in range(-20, 20):
        new = np.roll(yn, i, axis = 0)
        #plt.plot(np.mean(new, axis =1), label = i)
        diff.append(np.mean(np.absolute(new-xn)))
    #plt.legend()
    #plt.show()
    #print(diff)
    #print(np.argmin(diff))
    lab_rolls = np.argmin(diff)-20
    return label- int(np.mean([lab_rolls, grad_shift]))

"""def align_sequence_old(joint_array, template, label):
    xn = normalize(template)
    yn = normalize(joint_array)
    distances = []
    for i in range(len(xn)):
        distances.append([np.linalg.norm((yn[i]-xn[j])[0]) for j in range(len(xn))])
    distances = np.array(distances)
    indizes = []
    for i in range(len(xn)):
        indizes.append(np.argmin(distances[i]))
    res = []
    print(indizes, label)
    for ind in range(label-5, label+5):
        try:
            res.append(indizes.index(ind))
        except ValueError:
            continue
    return int(np.mean(res))"""
