"""
 @Time    : 23.05.22 19:43
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : align_prediction1_and_gt.py
 @Function:
 
"""
import os
import sys
import numpy as np
sys.path.append('..')
from tqdm import tqdm
import cv2

list = [
    # '00702',
    '00704',
    # '00705',
    '01402',
    # '01404',
    # '01405',
]

prediction_dir = '/home/mhy/firenet-pdavis/output/m34_new'
gt_dir = '/home/mhy/firenet-pdavis/output/m_gt'

for name in list:
    print(name)

    prediction_path = os.path.join(prediction_dir, name + '_1599.avi')
    # prediction_path = os.path.join(prediction_dir, name + '_1600.avi')
    gt_path = os.path.join(gt_dir, name + '_1600.avi')

    rate = 25
    size = (960, 480)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(prediction_path[:-4] + '_compare.avi', fourcc, rate, size, isColor=True)

    cap_prediction = cv2.VideoCapture(prediction_path)
    cap_gt = cv2.VideoCapture(gt_path)

    length_prediction = int(cap_prediction.get(cv2.CAP_PROP_FRAME_COUNT))
    length_gt = int(cap_gt.get(cv2.CAP_PROP_FRAME_COUNT))
    # if length_prediction + 1 != length_gt:
    #     print('Inconsistency in frame numbers!')
    #     print('Prediction has %d frames while gt has %d frames.' % (length_prediction, length_gt))
    #     exit(0)
    # else:
    #     print('Prediction has %d frames.' % length_prediction)

    for i in tqdm(range(length_prediction)):
        cap_prediction.set(cv2.CAP_PROP_POS_FRAMES, i)
        cap_gt.set(cv2.CAP_PROP_POS_FRAMES, i+1)
        # cap_gt.set(cv2.CAP_PROP_POS_FRAMES, i)

        _, prediction = cap_prediction.read()
        _, gt = cap_gt.read()

        frame = cv2.vconcat([gt, prediction])

        # cv2.imshow('show', frame)
        # cv2.waitKey()
        # exit(0)

        video.write(frame)

    cap_prediction.release()
    cap_gt.release()

print('Done!')
