"""
 @Time    : 18.05.22 13:53
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : process_raw_demosaicing.py
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

raw_dir = '/home/mhy/data/movingcam/raw_demosaicing'
output_dir = raw_dir + '_extracted'

raw_list = os.listdir(raw_dir)
raw_list = sorted(raw_list)
for raw_name in tqdm(raw_list):
    raw_path = os.path.join(raw_dir, raw_name)
    raw = np.load(raw_path)

    iad_dir = os.path.join(output_dir, raw_name.split('.')[0])
    check_mkdir(iad_dir)

    for i in range(raw.shape[0]):
        i90 = raw[i, 0::2, 0::2]
        i45 = raw[i, 0::2, 1::2]
        i135 = raw[i, 1::2, 0::2]
        i0 = raw[i, 1::2, 1::2]

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

        iad_path = os.path.join(iad_dir, str(i + 1) + '.png')
        cv2.imwrite(iad_path, iad)

print('Succeed!')
