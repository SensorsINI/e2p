"""
 @Time    : 16.05.22 09:38
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : h5gt2pfile.py
 @Function:
 
"""
"""
 @Time    : 08.04.22 15:33
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com

 @Project : firenet-pdavis
 @File    : h5gt2ph5.py
 @Function:

"""
import os
import sys

sys.path.append('..')
from misc import check_mkdir
import h5py
import numpy as np
import cv2
import math
from tqdm import tqdm

# input_path = '/home/mhy/firenet-pdavis/data/train/subject07_group2_time1.h5'
# output_path = '/home/mhy/firenet-pdavis/data/train_p/subject07_group2_time1_p.h5'
input_path = '/home/mhy/firenet-pdavis/data/test/subject09_group2_time1.h5'
output_path = '/home/mhy/firenet-pdavis/data/test_p/subject09_group2_time1_p.h5'

f_input = h5py.File(input_path, 'r')

raw = f_input['/frame']

print('--------- Frame ------------')
print(raw.shape)
print(raw.dtype)

i90 = raw[:, 0::2, 0::2]
i45 = raw[:, 0::2, 1::2]
i135 = raw[:, 1::2, 0::2]
i0 = raw[:, 1::2, 1::2]

s0 = i0.astype(float) + i90.astype(float)
s1 = i0.astype(float) - i90.astype(float)
s2 = i45.astype(float) - i135.astype(float)

intensity = (s0 / 2).astype(np.uint8)

print('--------- Intensity ------------')
print(intensity.shape)
print(intensity.dtype)

aolp = 0.5 * np.arctan2(s2, s1)
aolp[s2 < 0] += math.pi
aolp = (aolp * (255 / math.pi)).astype(np.uint8)

print('--------- AoLP ------------')
print(aolp.shape)
print(aolp.dtype)

dolp = np.divide(np.sqrt(np.square(s1) + np.square(s2)), s0, out=np.zeros_like(s0).astype(float), where=s0 != 0)
dolp = dolp.clip(0.0, 1.0)
dolp = (dolp * 255).astype(np.uint8)

print('--------- DoLP ------------')
print(dolp.shape)
print(dolp.dtype)

# mask operation
mask = np.where(dolp[:, :, :] >= 12.75, 255, 0).astype(np.uint8)
aolp_masked = np.where(mask[:, :, :] == 255, aolp, 0).astype(np.uint8)

print('--------- Mask ------------')
print(mask.shape)
print(mask.dtype)

print('--------- AoLP Masked ------------')
print(aolp_masked.shape)
print(aolp_masked.dtype)

f_output = h5py.File(output_path, 'w')
f_output.create_dataset('/events', data=f_input['/events'], chunks=True)
f_output.create_dataset('/frame', data=f_input['/frame'], chunks=True)
f_output.create_dataset('/frame_idx', data=f_input['/frame_idx'], chunks=True)
f_output.create_dataset('/frame_ts', data=f_input['/frame_ts'], chunks=True)

f_output.create_dataset('/intensity', data=intensity, chunks=True)
f_output.create_dataset('/aolp', data=aolp_masked, chunks=True)
f_output.create_dataset('/dolp', data=dolp, chunks=True)

f_output.attrs['sensor_resolution'] = (256, 256)

f_output.close()
print('Conversion Succeed!')

