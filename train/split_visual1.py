"""
 @Time    : 14.10.22 12:24
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : split_visual1.py
 @Function:
 
"""
import os
import numpy as np
import sys

sys.path.append('..')
from misc import check_mkdir
from tqdm import tqdm
import cv2

method1 = 'v_test_gt'

input_dir = '/home/mhy/firenet-pdavis/visual1'
output_dir = '/home/mhy/firenet-pdavis/visual1_separate'
check_mkdir(output_dir)

list = sorted(os.listdir(input_dir))
for name in tqdm(list):
    input_path = os.path.join(input_dir, name)
    input = cv2.imread(input_path)

    intensity1, aolp1, dolp1 = np.hsplit(input, 3)

    intensity1_path = os.path.join(output_dir, name[:-4] + '_i_' + method1 + name[-4:])

    aolp1_path = os.path.join(output_dir, name[:-4] + '_a_' + method1 + name[-4:])

    dolp1_path = os.path.join(output_dir, name[:-4] + '_d_' + method1 + name[-4:])

    cv2.imwrite(intensity1_path, intensity1)
    cv2.imwrite(aolp1_path, aolp1)
    cv2.imwrite(dolp1_path, dolp1)

print('Succeed!')

