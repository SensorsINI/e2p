"""
 @Time    : 17.10.22 16:47
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : my_merge_h5.py
 @Function:
 
"""
import os
import sys
sys.path.append('..')
import h5py
import numpy as np
import cv2
import math
from tqdm import tqdm
from train.events_contrast_maximization.utils.event_utils import binary_search_h5_dset
from easygui import diropenbox
from utils.prefs import MyPreferences
prefs=MyPreferences()
lastfolder=prefs.get('lastaedatfolder','./data/')
root = diropenbox(msg='select folder with aedat files',title='AEDAT folder',default=lastfolder)
prefs.put('lastaedatfolder',root)

print(f'creating h5 merged files from folder {root}')

list = sorted([x for x in os.listdir(root) if x.endswith('.aedat')])

for aedat_name in tqdm(list):
    name = aedat_name[:-6]
    events_path = os.path.join(root, name + '-events.txt')
    image_dir = os.path.join(root, name)
    timestamp_path = os.path.join(root, name + '-timecode.txt')
    h5_path = os.path.join(root, name + '.h5')
    frame_idx_path = os.path.join(root, name + '-frame_idx.txt')

    # read events
    print(f'loading events from {events_path} to create HDF5 file {h5_path}...',end='')
    events = np.loadtxt(events_path, dtype=np.int64)
    # events is an Nx4 array, ts,x,y,polarity
    print('checking range of polarities to ensure they are mapped to -1/OFF +1/ON values...')
    p_min=np.min(events[:,2]) # find min polarity value
    if p_min==0:
        events01=True
    elif p_min==-1:
        events01=False
    else:
        raise ValueError(f'the minimum polarity value {p_min} is not 0 or -1')
    print(f'done loading events from {events_path}, minimum polarity value detected as {p_min}')
    # print('removing nonmonotonic events at start')
    print('rescaling timestamps to seconds')
    events = events * [1e6, 1, 1, 1] # convert us to seconds
    if not events01:
        print('remapping polarity values from values (-1,+1) to values (0,1)')
        events[:,2]=(events[:,2]+1)/2
    events = (events - [0, 0, 259, 0]) * [1, 1, -1, 1] # flip y axis upside down since jAER uses LL corner for pixel 0,0
    events = events.astype(np.uint32)
    print('done')
    print('--------- events ------------')
    print(f'events.shape={events.shape} events.dtype={events.dtype}')

    # read frame
    print(f'loading frames from {image_dir}...',end='')
    image_list = sorted(os.listdir(image_dir))
    image_list = image_list[1:]
    i = 0
    for image_name in tqdm(image_list):
        image_path = os.path.join(image_dir, image_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
        if i == 0:
            frame = image[np.newaxis, :, :]
        else:
            frame = np.append(frame, image[np.newaxis, :, :], axis=0)
        i += 1
    print('--------- frame ------------')
    print(f'frame.shape={frame.shape} frame.dtype={events.dtype}')

    # read timestamp
    print(f'reading timestamps from {timestamp_path}')
    frame_ts = np.loadtxt(timestamp_path, dtype=np.uint32, skiprows=13)
    frame_ts = frame_ts[2::2, 1]
    print('--------- frame_ts ------------')
    print(frame_ts.shape)
    print(frame_ts.dtype)

    # find frame_idx
    frame_idx = []
    for ts in frame_ts:
        idx = binary_search_h5_dset(events[:, 0], ts)
        frame_idx.append(idx)
    frame_idx = np.array(frame_idx).astype(np.uint64)
    print('--------- frame_idx ------------')
    print(frame_idx.shape)
    print(frame_idx.dtype)
    with open(frame_idx_path, 'w') as f:
        for idx in frame_idx:
            f.write(f"{idx}\n")

    # compute polarization
    i90 = frame[:, 0::2, 0::2].astype(float)
    i45 = frame[:, 0::2, 1::2].astype(float)
    i135 = frame[:, 1::2, 0::2].astype(float)
    i0 = frame[:, 1::2, 1::2].astype(float)

    s0 = i0 + i90
    s1 = i0 - i90
    s2 = i45 - i135

    intensity = (s0 / 2).astype(np.uint8)

    print('--------- intensity ------------')
    print(intensity.shape)
    print(intensity.dtype)

    aolp = 0.5 * np.arctan2(s2, s1)
    aolp = aolp + 0.5 * math.pi
    aolp = (aolp * (255 / math.pi)).astype(np.uint8)

    print('--------- aolp ------------')
    print(aolp.shape)
    print(aolp.dtype)

    dolp = np.divide(np.sqrt(np.square(s1) + np.square(s2)), s0, out=np.zeros_like(s0).astype(float), where=s0 != 0)
    dolp = dolp.clip(0.0, 1.0)
    dolp = (dolp * 255).astype(np.uint8)

    print('--------- dolp ------------')
    print(dolp.shape)
    print(dolp.dtype)

    # write to hdf5 file
    output = h5py.File(h5_path, 'w')
    output.create_dataset('/events', data=events, chunks=True) # note that raw events are stored here, voxel grid DNN inputs are created on the fly by the DataLoader
    output.create_dataset('/frame', data=frame, chunks=True)
    output.create_dataset('/frame_idx', data=frame_idx, chunks=True)
    output.create_dataset('/frame_ts', data=frame_ts, chunks=True)

    output.create_dataset('/intensity', data=intensity, chunks=True)
    output.create_dataset('/aolp', data=aolp, chunks=True)
    output.create_dataset('/dolp', data=dolp, chunks=True)

    output.attrs['sensor_resolution'] = (frame.shape[2], frame.shape[1])

    output.close()
    print(h5_path + ' Conversion Succeeded!')

print('Nice!')
