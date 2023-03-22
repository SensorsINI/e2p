"""
 @Time    : 10.06.22 15:16
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : direction2p.py
 @Function:
 
"""
# from different dir
import os
import numpy as np
import cv2
import math
from tqdm import tqdm

def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

method = 'firenet'
i90_root = './output_synthetic/{}_90'.format(method)
i45_root = './output_synthetic/{}_45'.format(method)
i135_root = './output_synthetic/{}_135'.format(method)
i0_root = './output_synthetic/{}_0'.format(method)

iad_root = './output_synthetic/{}_direction_iad'.format(method)

dir_list = os.listdir(i90_root)
for dir_name in tqdm(dir_list):
    i90_dir = os.path.join(i90_root, dir_name)
    i45_dir = os.path.join(i45_root, dir_name)
    i135_dir = os.path.join(i135_root, dir_name)
    i0_dir = os.path.join(i0_root, dir_name)

    iad_dir = os.path.join(iad_root, dir_name)
    check_mkdir(iad_dir)

    i_list = [x for x in os.listdir(i90_dir) if x.endswith('.png')]
    for i_name in tqdm(i_list):
        i90_path = os.path.join(i90_dir, i_name)
        i45_path = os.path.join(i45_dir, i_name)
        i135_path = os.path.join(i135_dir, i_name)
        i0_path = os.path.join(i0_dir, i_name)

        iad_path = os.path.join(iad_dir, i_name)

        i90 = cv2.imread(i90_path, cv2.IMREAD_GRAYSCALE)
        i45 = cv2.imread(i45_path, cv2.IMREAD_GRAYSCALE)
        i135 = cv2.imread(i135_path, cv2.IMREAD_GRAYSCALE)
        i0 = cv2.imread(i0_path, cv2.IMREAD_GRAYSCALE)

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

        iad = cv2.hconcat([intensity, aolp, dolp])

        cv2.imwrite(iad_path, iad)

print('Succeed!')
