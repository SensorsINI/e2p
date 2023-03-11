"""
 @Time    : 09.06.22 22:42
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : raw2p.py
 @Function:
 
"""
import os
import sys
sys.path.append('..')
from misc import check_mkdir
import numpy as np
import cv2
import math
from tqdm import tqdm

# mask_aolp = False

input_root_dir = '/home/mhy/firenet-pdavis/output/firenet_raw'
output_root_dir = '/home/mhy/firenet-pdavis/output/firenet_raw_iad'
list = os.listdir(input_root_dir)
for name in tqdm(list):
    input_dir = os.path.join(input_root_dir, name)
    output_dir = os.path.join(output_root_dir, name)
    check_mkdir(output_dir)

    raw_list = [x for x in os.listdir(input_dir) if x.endswith('.png')]
    raw_list = sorted(raw_list)

    for raw_name in tqdm(raw_list, desc=name):
        raw_path = os.path.join(input_dir, raw_name)
        iad_path = os.path.join(output_dir, raw_name)

        raw = cv2.imread(raw_path, cv2.IMREAD_GRAYSCALE)

        i90 = raw[0::2, 0::2]
        i45 = raw[0::2, 1::2]
        i135 = raw[1::2, 0::2]
        i0 = raw[1::2, 1::2]

        s0 = i0.astype(float) + i90.astype(float)
        s1 = i0.astype(float) - i90.astype(float)
        s2 = i45.astype(float) - i135.astype(float)

        intensity = (s0 / 2).astype(np.uint8)

        aolp = 0.5 * np.arctan2(s2, s1)
        aolp = aolp + 0.5 * math.pi
        aolp = (aolp * (255 / math.pi)).astype(np.uint8)

        dolp = np.divide(np.sqrt(np.square(s1) + np.square(s2)), s0, out=np.zeros_like(s0).astype(float), where=s0 != 0)
        dolp = dolp.clip(0.0, 1.0)
        dolp = (dolp * 255).astype(np.uint8)

        # if mask_aolp:
        #     mask = np.where(dolp[:, :, :] >= 12.75, 255, 0).astype(np.uint8)
        #     aolp_masked = np.where(mask[:, :, :] == 255, aolp, 0).astype(np.uint8)

        iad = cv2.hconcat([intensity, aolp, dolp])
        cv2.imwrite(iad_path, iad)

print('Done!')
