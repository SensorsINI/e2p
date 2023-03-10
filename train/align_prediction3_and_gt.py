"""
 @Time    : 30.05.22 09:19
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : align_prediction3_and_gt.py
 @Function:
 
"""
import os
import sys
import numpy as np
sys.path.append('..')
from misc import check_mkdir
from tqdm import tqdm
import cv2

list = [
    '00702',
    '00704',
    '00705',
    '01402',
    '01404',
    '01405',
]

method1 = 'firenet_raw_iad'
method2 = 'firenet_direction_iad'
method3 = 'm49'
# method1 = 'm27'
# method2 = 'm28'
# method3 = 'm29'

prediction1_dir = os.path.join('/home/mhy/firenet-pdavis/output', method1)
prediction2_dir = os.path.join('/home/mhy/firenet-pdavis/output', method2)
prediction3_dir = os.path.join('/home/mhy/firenet-pdavis/output', method3)
gt_dir = '/home/mhy/firenet-pdavis/output/m_gt'
compare_dir = '/home/mhy/firenet-pdavis/output/compare_{}_{}_{}'.format(method1, method2, method3)
check_mkdir(compare_dir)

for name in list:
    print(name)

    prediction1_path = os.path.join(prediction1_dir, name + '_1599.avi')
    prediction2_path = os.path.join(prediction2_dir, name + '_1599.avi')
    prediction3_path = os.path.join(prediction3_dir, name + '_1599.avi')
    gt_path = os.path.join(gt_dir, name + '_1600.avi')

    rate = 25
    size = (960, 960)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(os.path.join(compare_dir, name + '_compare_gt_{}_{}_{}_1599.avi'.format(method1, method2, method3)), fourcc, rate, size, isColor=True)

    cap_prediction1 = cv2.VideoCapture(prediction1_path)
    cap_prediction2 = cv2.VideoCapture(prediction2_path)
    cap_prediction3 = cv2.VideoCapture(prediction3_path)
    cap_gt = cv2.VideoCapture(gt_path)

    length_prediction = int(cap_prediction1.get(cv2.CAP_PROP_FRAME_COUNT))
    length_gt = int(cap_gt.get(cv2.CAP_PROP_FRAME_COUNT))
    if length_prediction + 1 != length_gt:
        print('Inconsistency in frame numbers!')
        print('Prediction has %d frames while gt has %d frames.' % (length_prediction, length_gt))
        exit(0)
    else:
        print('Prediction has %d frames.' % length_prediction)

    for i in tqdm(range(length_prediction)):
        cap_prediction1.set(cv2.CAP_PROP_POS_FRAMES, i)
        cap_prediction2.set(cv2.CAP_PROP_POS_FRAMES, i)
        cap_prediction3.set(cv2.CAP_PROP_POS_FRAMES, i)
        cap_gt.set(cv2.CAP_PROP_POS_FRAMES, i+1)

        _, prediction1 = cap_prediction1.read()
        _, prediction2 = cap_prediction2.read()
        _, prediction3 = cap_prediction3.read()
        _, gt = cap_gt.read()

        frame = cv2.vconcat([gt, prediction1, prediction2, prediction3])

        # cv2.imshow('show', frame)
        # cv2.waitKey()
        # exit(0)

        video.write(frame)

    cap_prediction1.release()
    cap_prediction2.release()
    cap_prediction3.release()
    cap_gt.release()

print('Done!')

