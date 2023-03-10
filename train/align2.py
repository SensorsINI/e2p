"""
 @Time    : 06.09.22 17:00
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : align2.py
 @Function:
 
"""
import os
import sys
sys.path.append('..')
from misc import check_mkdir
from tqdm import tqdm
import cv2

# list = [
#     '007118',
#     # '00702',
#     # '00704',
#     # '00705',
#     # '01402',
#     # '01405',
# ]


method1 = 'm_gt'
# method2 = 'firenet_direction_iad'
method2 = 'v0_1'

method1_dir = os.path.join('/home/mhy/firenet-pdavis/output', method1)
method2_dir = os.path.join('/home/mhy/firenet-pdavis/output', method2)
compare_dir = '/home/mhy/firenet-pdavis/output/compare_{}_{}'.format(method1, method2)
check_mkdir(compare_dir)

list = [x for x in os.listdir(method2_dir) if x.endswith('.avi')]

for name in list:
    print(name)

    # method1_path = os.path.join(method1_dir, name + '_1600.avi')
    # method2_path = os.path.join(method2_dir, name + '_1599.avi')
    # method1_path = os.path.join(method1_dir, name + '_496.avi')
    # method2_path = os.path.join(method2_dir, name + '_495.avi')
    # method1_path = os.path.join(method1_dir, name.replace('_iad.', '.'))
    method1_path = os.path.join(method1_dir, name)
    method2_path = os.path.join(method2_dir, name)

    rate = 25
    size = (960, 480)
    # size = (519, 260)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(
        # os.path.join(compare_dir, name + '_{}_{}_1600.avi'.format(method1, method2)), fourcc,
        os.path.join(compare_dir, name + '_{}_{}.avi'.format(method1, method2)), fourcc,
        rate, size, isColor=True)

    cap_method1 = cv2.VideoCapture(method1_path)
    cap_method2 = cv2.VideoCapture(method2_path)

    length_method1 = int(cap_method1.get(cv2.CAP_PROP_FRAME_COUNT))
    length_method2 = int(cap_method2.get(cv2.CAP_PROP_FRAME_COUNT))
    # if length_method1 != length_method2 + 1:
    #     # if length_method1 != length_method2 or length_method1 != length_method3:
    #     print('Inconsistency in frame numbers!')
    #     print(length_method1, length_method2)
    #     exit(0)
    # else:
    #     print('Method has %d frames.' % length_method1)

    for i in tqdm(range(length_method2)):
        cap_method1.set(cv2.CAP_PROP_POS_FRAMES, i + 1)
        cap_method2.set(cv2.CAP_PROP_POS_FRAMES, i)

        _, frame_method1 = cap_method1.read()
        _, frame_method2 = cap_method2.read()

        frame = cv2.vconcat([frame_method1, frame_method2])

        video.write(frame)

    cap_method1.release()
    cap_method2.release()

print('Done!')
