"""
 @Time    : 23.05.22 09:26
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : align_video.py
 @Function:
 
"""
import os
import sys

import numpy as np

sys.path.append('..')
from tqdm import tqdm
import cv2

root_dir = '/home/mhy/v2e/output/for_visual'
dir_11 = os.path.join(root_dir, '11')
path_12 = os.path.join(root_dir, '12.avi')
path_21 = os.path.join(root_dir, '21.avi')
path_22 = os.path.join(root_dir, '22.avi')
path_31 = os.path.join(root_dir, '31.avi')
path_32 = os.path.join(root_dir, '32.avi')

rate = 25
size = (1600, 1440)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter(os.path.join(root_dir, 'align.avi'), fourcc, rate, size, isColor=True)

cap_12 = cv2.VideoCapture(path_12)
cap_21 = cv2.VideoCapture(path_21)
cap_22 = cv2.VideoCapture(path_22)
cap_31 = cv2.VideoCapture(path_31)
cap_32 = cv2.VideoCapture(path_32)

# n = 0
list_11 = sorted(os.listdir(dir_11))
# while(cap_22.isOpened()):
for i in tqdm(range(1500)):
    index = (i + 200) // 8
    frame_11 = cv2.imread(os.path.join(dir_11, list_11[index]))

    if i % 8 ==0:
        _, frame_12 = cap_12.read()

    frame_13 = np.ones_like(frame_12)
    # print(i)
    # print(index)
    # print(frame_12.shape)
    # print(frame_12.dtype)
    # print(frame_13.shape)
    # print(frame_13.dtype)
    frame_12_final = cv2.vconcat([frame_13, frame_12])

    frame_1 = cv2.hconcat([frame_11, frame_12_final])
    # cv2.imshow('1', frame_1)
    # cv2.waitKey()


    _, frame_21 = cap_21.read()

    _, frame_22 = cap_22.read()
    frame_23 = np.ones_like(frame_22)
    frame_22_final = cv2.vconcat([frame_23, frame_22])

    frame_2 = cv2.hconcat([frame_21, frame_22_final])


    _, frame_31 = cap_31.read()

    _, frame_32 = cap_32.read()
    frame_33 = np.ones_like(frame_32)
    frame_32_final = cv2.vconcat([frame_33, frame_32])

    frame_3 = cv2.hconcat([frame_31, frame_32_final])

    frame = cv2.vconcat([frame_1, frame_2, frame_3])

    # cv2.imshow('show', frame)
    # print(frame.shape)

    video.write(frame)


cap_12.release()
cap_21.release()
cap_22.release()
cap_31.release()
cap_32.release()

print('Done!')
