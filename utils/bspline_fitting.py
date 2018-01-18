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


# BSPLINE FITTING instead of interpolation and lowpass

def get_bspline(knots):
    """ Takes an array-like parameter and returns a function implementing a B-spline. """

    def bspline(x, k, d):
        if d==1: # recursion stop
            if knots[k] <= x < knots[k+1]:
                y = 1
            else:
                y = 0
        else:
            factor1 = (x-knots[k])/(knots[k+d-1]-knots[k])
            factor2 = (knots[k+d]-x)/(knots[k+d]-knots[k+1])
            b1 = bspline(x, k, d-1)   # recursion!
            b2 = bspline(x, k+1, d-1) # recursion!
            y = factor1*b1 + factor2*b2
        return y

    return np.vectorize(bspline, excluded=['k','d'])

def bspline_regression(X, Y, knots, deg):
    """
    Performes a B-spline regression.

    Parameters
    ----------
    X : 1-D ndarray
        samples of the independent variable
    Y : 1-D ndarray
        samples of the dependent variable
    knots : 1-D array-like
        knots of the splines
    deg : integer
        order of splines

    Returns
    -------
    w : ndarray
        egression weights
    """

    bspline = get_bspline(knots)
    col_num = len(knots) - deg + 1

    # design matrix! nr_data_points X nr_bsplines
    ϕ = np.ones( (len(X), col_num))
    for i in range(1, col_num):
        ϕ[:, i] = bspline(X, i-1, deg) #offset for first val

    # std linear model: calculate weights
    w = np.dot( np.linalg.pinv(ϕ), Y)

    return w

def spline_curve(x, w, knots, deg):
    """Evaluates a function, which is a linear combination of B-splines.

    Parameters
    ----------
    x : 1-D ndarray
        evaluation points.
    w : 1-D ndarray
        regression weights
    knots : 1-D array-like
        knots of the splines
    deg : integer
        degree of splines

    Returns
    -------
    y : ndarray
        linear combination of splines evaluated at x

    """
    bspline = get_bspline(knots)
    M = [ bspline(x, i-1, deg) * w[i] for i in range (1,len(w)) ]
    y = np.sum(np.array(M), axis=0) + w[0]
    return y

def filter_bspline(new_df, deg = 3, knot_dist = 5):
    #new_df = interpolate(new_df)
    for k in range(len(new_df[0])):
        for j in range(2):
            #print(new_df[:,i,j].tolist())
            values = np.append(np.append(np.array([new_df[0,k,j] for _ in range(50)]), new_df[:,k,j]), np.array([new_df[-1,k,j] for _ in range(50)]))
            not_zer = np.logical_not(values == 0)
            X = np.arange(len(values))[not_zer]
            # new_df[:,k,j] = np.poly1d(np.polyfit(X, values[not_zer], 5))(np.arange(len(new_df)))
            # X = indices
            knots = []
            for knot in range(0, len(new_df)+100, knot_dist):
                if knot in X:
                    knots.append(knot)
            knots = np.array(knots)
            Y = values[not_zer]
            w = bspline_regression(X, Y, knots, deg)
            x = np.arange(len(values))
            y = spline_curve(x, w, knots, deg)
            #plt.figure(figsize=((12,9)))
            #plt.scatter(X,Y, c = "blue")
            #plt.plot(y, c="red")
            #plt.show()
            new_df[:,k,j] = y[50:-50]
    return new_df

if False:
    # new_df = interpolate(new_df)
    b_new_df = mix_right_left(new_df.copy())
    b_new_df = filter_bspline(b_new_df, knot_dist=10)

    #print(new_df[:3])
    plt.figure(figsize=(20,10))
    for i in range(len(new_df[0])):
        plt.plot(b_new_df[:,i,0], label = i)
    plt.legend()
    plt.title("curve fitting")
    plt.show()

    arr2 = color_video(b_new_df, f, start = 0, cut_frame=False, end =len(new_df), printing = None, plotting=False)
    io.vwrite(outputs+name+"_bsplinefit"+".mp4", arr2)

    # to_json(new_df, {}, outputs+name+"_smooth")
    # continue

    #arr2 = color_video(new_df[:,:12,:], f, start = 0, cut_frame=False, end =2, printing = None, plotting=True)
