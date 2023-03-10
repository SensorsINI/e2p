"""
 @Time    : 25.04.22 10:06
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : images2video.py
 @Function:
 
"""
import os
import sys
sys.path.append('..')
from misc import check_mkdir
import cv2
import numpy as np
from tqdm import tqdm

# dir = '/home/mhy/firenet-pdavis/data/test_p/flow_vis_subject09_group2_time1_p'
# output_path = '/home/mhy/firenet-pdavis/data/test_p/flow_vis_subject09_group2_time1_p.avi'
dir = '/home/mhy/v2e/output/raw_demosaicing/00704/00704_iad'
output_path = '/home/mhy/v2e/output/raw_demosaicing/00704/00704_iad.avi'

list = sorted(os.listdir(dir))

rate = 25
size = (960, 240)
# fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter(output_path, fourcc, rate, size, isColor=True)

for name in tqdm(list):
    path = os.path.join(dir, name)

    image = cv2.imread(path)

    video.write(image)

print('Succeed!')
