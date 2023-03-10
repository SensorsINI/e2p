"""
 @Time    : 16.10.22 20:58
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : pvideo2iad.py
 @Function:
 
"""
import os
import sys
sys.path.append('..')
from misc import check_mkdir
from tqdm import tqdm
import cv2
import numpy as np
import math

dir = '/home/mhy/aedat2pvideo/aedat/own'
video_list = sorted([x for x in os.listdir(dir) if x.endswith('.avi')])

for video_name in video_list:
    video_path = os.path.join(dir, video_name)
    output_dir = os.path.join(dir, video_name.split('.')[0])
    check_mkdir(output_dir)

    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in tqdm(range(length)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        _, raw = cap.read()
        raw = raw[:, :, 1]

        i90 = raw[0::2, 0::2].astype(float)
        i45 = raw[0::2, 1::2].astype(float)
        i135 = raw[1::2, 0::2].astype(float)
        i0 = raw[1::2, 1::2].astype(float)

        s0 = i0 + i90
        s1 = i0 - i90
        s2 = i45 - i135

        aolp = 0.5 * np.arctan2(s2, s1)
        aolp = aolp + 0.5 * math.pi
        aolp = (aolp * (255 / math.pi)).astype(np.uint8)

        dolp = np.divide(np.sqrt(np.square(s1) + np.square(s2)), s0, out=np.zeros_like(s0).astype(float), where=s0 != 0)
        dolp = dolp.clip(0.0, 1.0)
        dolp = (dolp * 255).astype(np.uint8)

        intensity = (s0 / 2).astype(np.uint8)

        iad = cv2.hconcat([intensity, aolp, dolp])

        iad_path = os.path.join(output_dir, str(i-1) + '.png')
        cv2.imwrite(iad_path, iad)

print('Nice!')
