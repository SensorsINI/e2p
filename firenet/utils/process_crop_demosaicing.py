"""
 @Time    : 18.05.22 17:10
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : process_crop_demosaicing.py
 @Function:
 
"""
import os
import sys

sys.path.append('..')
import cv2
from tqdm import tqdm
import numpy as np
import math
from misc import check_mkdir

crop_dir = '/home/mhy/data/movingcam/crop_davis640_demosaicing/00704'
output_dir = crop_dir + '_extracted'
check_mkdir(output_dir)

frame_list = os.listdir(crop_dir)
frame_list = sorted(frame_list)
for frame_name in tqdm(frame_list):
    frame_path = os.path.join(crop_dir, frame_name)
    frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)

    i90 = frame[0::2, 0::2]
    i45 = frame[0::2, 1::2]
    i135 = frame[1::2, 0::2]
    i0 = frame[1::2, 1::2]

    s0 = i0.astype(float) + i90.astype(float)
    s1 = i0.astype(float) - i90.astype(float)
    s2 = i45.astype(float) - i135.astype(float)

    intensity = (s0 / 2).astype(np.uint8)

    aolp = 0.5 * np.arctan2(s2, s1)
    aolp[s2 < 0] += math.pi
    aolp = (aolp * (255 / math.pi)).astype(np.uint8)

    dolp = np.divide(np.sqrt(np.square(s1) + np.square(s2)), s0, out=np.zeros_like(s0).astype(float), where=s0 != 0)
    dolp = dolp.clip(0.0, 1.0)
    dolp = (dolp * 255).astype(np.uint8)

    iad = np.hstack([intensity, aolp, dolp])

    iad_path = os.path.join(output_dir, frame_name)
    cv2.imwrite(iad_path, iad)

print('Succeed!')

