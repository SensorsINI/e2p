"""
 @Time    : 21.07.22 15:24
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : align4.py
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

method1 = 'm_gt'
method2 = 'm88_new_pae_2'
method3 = 'm88_new_onlypae_2'
method4 = 'm88_new_pae_4'

method1_dir = os.path.join('/home/mhy/firenet-pdavis/output', method1)
method2_dir = os.path.join('/home/mhy/firenet-pdavis/output', method2)
method3_dir = os.path.join('/home/mhy/firenet-pdavis/output', method3)
method4_dir = os.path.join('/home/mhy/firenet-pdavis/output', method4)
compare_dir = '/home/mhy/firenet-pdavis/output/compare_{}_{}_{}_{}'.format(method1, method2, method3, method4)
check_mkdir(compare_dir)

for name in list:
    print(name)

    method1_path = os.path.join(method1_dir, name + '_1600.avi')
    method2_path = os.path.join(method2_dir, name + '_1599.avi')
    method3_path = os.path.join(method3_dir, name + '_1599.avi')
    method4_path = os.path.join(method4_dir, name + '_1599.avi')

    rate = 25
    size = (960, 960)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(
        os.path.join(compare_dir, name + '_{}_{}_{}_{}_1599.avi'.format(method1, method2, method3, method4)), fourcc,
        rate, size, isColor=True)

    cap_method1 = cv2.VideoCapture(method1_path)
    cap_method2 = cv2.VideoCapture(method2_path)
    cap_method3 = cv2.VideoCapture(method3_path)
    cap_method4 = cv2.VideoCapture(method4_path)

    length_method1 = int(cap_method1.get(cv2.CAP_PROP_FRAME_COUNT))
    length_method2 = int(cap_method2.get(cv2.CAP_PROP_FRAME_COUNT))
    length_method3 = int(cap_method3.get(cv2.CAP_PROP_FRAME_COUNT))
    length_method4 = int(cap_method4.get(cv2.CAP_PROP_FRAME_COUNT))
    if length_method1 != length_method2 + 1 or length_method1 != length_method3 + 1 or length_method1 != length_method4 + 1:
        # if length_method1 != length_method2 or length_method1 != length_method3:
        print('Inconsistency in frame numbers!')
        print(length_method1, length_method2, length_method3, length_method4)
        exit(0)
    else:
        print('Method has %d frames.' % length_method1)

    for i in tqdm(range(length_method2)):
        cap_method1.set(cv2.CAP_PROP_POS_FRAMES, i + 1)
        # cap_method1.set(cv2.CAP_PROP_POS_FRAMES, i)
        cap_method2.set(cv2.CAP_PROP_POS_FRAMES, i)
        cap_method3.set(cv2.CAP_PROP_POS_FRAMES, i)
        cap_method4.set(cv2.CAP_PROP_POS_FRAMES, i)

        _, frame_method1 = cap_method1.read()
        _, frame_method2 = cap_method2.read()
        _, frame_method3 = cap_method3.read()
        _, frame_method4 = cap_method4.read()

        frame = cv2.vconcat([frame_method1, frame_method2, frame_method3, frame_method4])

        video.write(frame)

    cap_method1.release()
    cap_method2.release()
    cap_method3.release()
    cap_method4.release()

print('Done!')

