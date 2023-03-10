"""
 @Time    : 16.11.22 17:03
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : my_compareimage2video.py
 @Function:
 
"""
import os
import sys

sys.path.append('..')
from misc import check_mkdir
import cv2
import numpy as np
from tqdm import tqdm

# root = '/home/mhy/firenet-pdavis/output/compare_v_test_gt_firenet_direction_iad_v16_mix2'
# list = [
#     '007000',
#     '008107',
# ]

# root = '/home/mhy/firenet-pdavis/output_real/compare_test_gt_firenet_direction_iad_v16_mix2'
# list = [
#     'real-28',
#     'real-61',
# ]

root = '/home/mhy/firenet-pdavis/output_real_10ms/compare_test_gt_firenet_direction_iad_v16_mix2'
list = [
    'real-50',
    'real-64',
]

rate = 10
# size = (960, 720)
size = (519, 390)
fourcc = cv2.VideoWriter_fourcc(*'XVID')

for name in list:
    dir = os.path.join(root, name)
    output_path = dir + '.avi'
    video = cv2.VideoWriter(output_path, fourcc, rate, size, isColor=True)

    for image_name in tqdm(sorted(os.listdir(dir))):
        path = os.path.join(dir, image_name)
        image = cv2.imread(path)
        video.write(image)

print('Succeed!')
