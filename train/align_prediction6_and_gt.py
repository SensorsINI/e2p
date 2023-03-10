"""
 @Time    : 04.07.22 10:25
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : align_prediction6_and_gt.py
 @Function:
 
"""
import os
import sys

sys.path.append('..')
from misc import check_mkdir
from tqdm import tqdm
import cv2

list = [
    '00702',
    '00704',
    '00705',
    '01402',
    '01405',
]

# method0 = 'm_gt'
# method1 = 'firenet_raw_iad'
# method2 = 'firenet_direction_iad'
# method3 = 'm44_new'
#
# method4 = 'm68'
# method5 = 'm68_2'
# method6 = 'm69'
# method7 = 'm70'

method0 = 'm_gt'
method1 = 'm88'
method2 = 'm88_new_pae_2'
method3 = 'm88_new_onlypae'

method4 = 'm_gt'
method5 = 'm88_new_w1248'
method6 = 'm88_new_w8421'
method7 = 'm88_new_onlyp'

method0_dir = os.path.join('/home/mhy/firenet-pdavis/output', method0)
method1_dir = os.path.join('/home/mhy/firenet-pdavis/output', method1)
method2_dir = os.path.join('/home/mhy/firenet-pdavis/output', method2)
method3_dir = os.path.join('/home/mhy/firenet-pdavis/output', method3)
method4_dir = os.path.join('/home/mhy/firenet-pdavis/output', method4)
method5_dir = os.path.join('/home/mhy/firenet-pdavis/output', method5)
method6_dir = os.path.join('/home/mhy/firenet-pdavis/output', method6)
method7_dir = os.path.join('/home/mhy/firenet-pdavis/output', method7)

compare_dir = '/home/mhy/firenet-pdavis/output/compare6'
check_mkdir(compare_dir)

for name in list:
    print(name)

    method0_path = os.path.join(method0_dir, name + '_1600.avi')
    method1_path = os.path.join(method1_dir, name + '_1599.avi')
    method2_path = os.path.join(method2_dir, name + '_1599.avi')
    method3_path = os.path.join(method3_dir, name + '_1599.avi')
    method4_path = os.path.join(method4_dir, name + '_1600.avi')
    method5_path = os.path.join(method5_dir, name + '_1599.avi')
    method6_path = os.path.join(method6_dir, name + '_1599.avi')
    method7_path = os.path.join(method7_dir, name + '_1599.avi')

    rate = 25
    size = (1920, 960)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_path = os.path.join(compare_dir,
                              name + '_{}_{}_{}_{}_{}_{}_{}_{}_1599.avi'.format(method0, method1, method2,
                                                                                      method3, method4, method5,
                                                                                      method6, method7))
    print(video_path)
    video = cv2.VideoWriter(video_path, fourcc, rate, size, isColor=True)

    cap_method0 = cv2.VideoCapture(method0_path)
    cap_method1 = cv2.VideoCapture(method1_path)
    cap_method2 = cv2.VideoCapture(method2_path)
    cap_method3 = cv2.VideoCapture(method3_path)
    cap_method4 = cv2.VideoCapture(method4_path)
    cap_method5 = cv2.VideoCapture(method5_path)
    cap_method6 = cv2.VideoCapture(method6_path)
    cap_method7 = cv2.VideoCapture(method7_path)

    length_method = int(cap_method1.get(cv2.CAP_PROP_FRAME_COUNT))
    length_gt = int(cap_method0.get(cv2.CAP_PROP_FRAME_COUNT))
    if length_method + 1 != length_gt:
        print('Inconsistency in frame numbers!')
        print('Prediction has %d frames while gt has %d frames.' % (length_method, length_gt))
        exit(0)
    else:
        print('Prediction has %d frames.' % length_method)

    for i in tqdm(range(length_method)):
        cap_method0.set(cv2.CAP_PROP_POS_FRAMES, i + 1)
        cap_method1.set(cv2.CAP_PROP_POS_FRAMES, i)
        cap_method2.set(cv2.CAP_PROP_POS_FRAMES, i)
        cap_method3.set(cv2.CAP_PROP_POS_FRAMES, i)
        cap_method4.set(cv2.CAP_PROP_POS_FRAMES, i + 1)
        cap_method5.set(cv2.CAP_PROP_POS_FRAMES, i)
        cap_method6.set(cv2.CAP_PROP_POS_FRAMES, i)
        cap_method7.set(cv2.CAP_PROP_POS_FRAMES, i)

        _, method0_frame = cap_method0.read()
        _, method1_frame = cap_method1.read()
        _, method2_frame = cap_method2.read()
        _, method3_frame = cap_method3.read()
        _, method4_frame = cap_method4.read()
        _, method5_frame = cap_method5.read()
        _, method6_frame = cap_method6.read()
        _, method7_frame = cap_method7.read()

        five_left = cv2.vconcat([method0_frame, method1_frame, method2_frame, method3_frame])
        five_right = cv2.vconcat([method4_frame, method5_frame, method6_frame, method7_frame])
        ten = cv2.hconcat([five_left, five_right])
        # print(ten.shape)

        video.write(ten)

    cap_method0.release()
    cap_method1.release()
    cap_method2.release()
    cap_method3.release()
    cap_method4.release()
    cap_method5.release()
    cap_method6.release()
    cap_method7.release()

print('Done!')

