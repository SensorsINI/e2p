"""
 @Time    : 23.06.22 20:37
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : events2frame.py
 @Function:
 
"""
import os
import cv2
import h5py
import numpy as np


def hist2d_numba_seq(tracks, bins, ranges):
    H = np.zeros((bins[0], bins[1]), dtype=np.float64)
    delta = 1/((ranges[:, 1] - ranges[:, 0]) / bins)

    for t in range(tracks.shape[1]):
        i = (tracks[0, t] - ranges[0, 0]) * delta[0]
        j = (tracks[1, t] - ranges[1, 0]) * delta[1]
        if 0 <= i < bins[0] and 0 <= j < bins[1]:
            H[int(i), int(j)] += 1

    return H


def accumulate_event_frame(events, histrange, currentFrame=None):
    """Accumulate event frame from an array of events.

    # Arguments
    events: np.ndarray
        an [N events x 4] array

    # Returns
    event_frame: np.ndarray
        an event frame
    """
    pol_on = (events[:, 3] == 1)
    pol_off = np.logical_not(pol_on)

    img_on = hist2d_numba_seq(
        np.array([events[pol_on, 2], events[pol_on, 1]],
                 dtype=np.float64),
        bins=np.asarray([260, 346], dtype=np.int64),
        ranges=histrange)
    img_off = hist2d_numba_seq(
        np.array([events[pol_off, 2], events[pol_off, 1]],
                 dtype=np.float64),
        bins=np.asarray([260, 346], dtype=np.int64),
        ranges=histrange)

    if currentFrame is None:
        currentFrame = np.zeros_like(img_on)

    # accumulate event histograms to the current frame,
    # clip values of zero-centered current frame with new events added
    currentFrame = np.clip(
        currentFrame + (img_on - img_off),
        -256, 256)

    cv2.imshow('asdf', currentFrame)
    cv2.waitKey()


step = 50000
root = '/home/mhy/data/beetles/22June-20220623T122254Z-001/22June_h5'
h5_list = os.listdir(root)
for h5_name in h5_list:
    h5_path = os.path.join(root, h5_name)
    h5 = h5py.File(h5_path, 'r')
    events = h5['/events']
    print(events.shape)
    print(events.dtype)

    histrange = np.asarray([(0, v) for v in (260, 346)],
                           dtype=np.int64)

    start = events[0, 0]
    end = start + step
    i = 0
    while end < events[-1, 0]:

        index = (events[:, 0] >= start) * (events[:, 0] <= end)
        events_step = events[:, :][index]

        accumulate_event_frame(events_step, histrange)

        i += events_step.shape[0]
        start = events[i, 0]
        end = start + step
        print(start, end)

    exit(0)


