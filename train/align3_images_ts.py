"""
 @Time    : 16.11.22 10:23
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : align3_images_ts.py
 @Function:
 
"""
import os
import sys
import numpy as np

sys.path.append('..')
from misc import check_mkdir
from tqdm import tqdm
import cv2

# for simulated data
# method1 = 'v_test_gt'
# method2 = 'firenet_direction_iad'
# method3 = 'v16_mix2'
#
# method1_dir = os.path.join('/home/mhy/firenet-pdavis/output', method1)
# method2_dir = os.path.join('/home/mhy/firenet-pdavis/output', method2)
# method3_dir = os.path.join('/home/mhy/firenet-pdavis/output', method3)
# output_dir = '/home/mhy/firenet-pdavis/output/compare_{}_{}_{}'.format(method1, method2, method3)
# check_mkdir(output_dir)

# for real data
method1 = 'test_gt'
method2 = 'firenet_direction_iad'
method3 = 'v16_mix2'

method1_dir = os.path.join('/home/mhy/firenet-pdavis/output_real_10ms', method1)
method2_dir = os.path.join('/home/mhy/firenet-pdavis/output_real_10ms', method2)
method3_dir = os.path.join('/home/mhy/firenet-pdavis/output_real_10ms', method3)
output_dir = '/home/mhy/firenet-pdavis/output_real_10ms/compare_{}_{}_{}'.format(method1, method2, method3)
check_mkdir(output_dir)

list = [x for x in os.listdir(method3_dir) if not x.endswith('.txt')]


def find_index_gt(list, ts):
    index = 0
    # print(list)
    # print(ts)

    for i in range(len(list)):
        if list[index] <= ts and index + 1 == len(list):
            return index
        if list[index] <= ts and list[index + 1] > ts:
            return index
        else:
            index += 1


for name in sorted(list):
    print(name)

    method1_path = os.path.join(method1_dir, name)
    method2_path = os.path.join(method2_dir, name)
    method3_path = os.path.join(method3_dir, name)

    i = 0
    for image_name in tqdm(sorted(os.listdir(method3_path))[:-1]):
        i += 1
        if i == 1:
            continue

        txt1_path = os.path.join(method1_path, 'timestamps.txt')
        txt3_path = os.path.join(method3_path, 'timestamps.txt')

        frame_name_list1 = []
        frame_ts_list1 = []
        with open(txt1_path, 'r') as f:
            lines = f.readlines()
            for j in range(len(lines)):
                line = lines[j]
                frame_name = line.split(' ')[0]
                frame_ts = float(line.split(' ')[1])
                frame_name_list1.append(frame_name)
                frame_ts_list1.append(frame_ts)

        frame_name_list3 = []
        frame_ts_list3 = []
        with open(txt3_path, 'r') as f:
            lines = f.readlines()
            for j in range(len(lines)):
                line = lines[j]
                frame_name = line.split(' ')[0]
                frame_ts = float(line.split(' ')[1])
                frame_name_list3.append(frame_name)
                frame_ts_list3.append(frame_ts)

        index = frame_name_list3.index(image_name)
        ts = frame_ts_list3[index]

        if ts < frame_ts_list1[0]:
            continue

        index_gt = find_index_gt(frame_ts_list1, ts)
        # print(index_gt)
        image_gt_name = frame_name_list1[index_gt]

        concat1 = cv2.imread(os.path.join(method1_path, image_gt_name), cv2.IMREAD_GRAYSCALE)
        intensity1, aolp1, dolp1 = np.hsplit(concat1, 3)
        intensity1 = np.repeat(intensity1[:, :, None], 3, axis=2)
        aolp1 = cv2.applyColorMap(aolp1, cv2.COLORMAP_HSV)
        dolp1 = cv2.applyColorMap(dolp1, cv2.COLORMAP_HOT)
        frame1 = cv2.hconcat([intensity1, aolp1, dolp1])

        concat2 = cv2.imread(os.path.join(method2_path, image_name), cv2.IMREAD_GRAYSCALE)
        intensity2, aolp2, dolp2 = np.hsplit(concat2, 3)
        intensity2 = np.repeat(intensity2[:, :, None], 3, axis=2)
        aolp2 = cv2.applyColorMap(aolp2, cv2.COLORMAP_HSV)
        dolp2 = cv2.applyColorMap(dolp2, cv2.COLORMAP_HOT)
        frame2 = cv2.hconcat([intensity2, aolp2, dolp2])

        concat3 = cv2.imread(os.path.join(method3_path, image_name), cv2.IMREAD_GRAYSCALE)
        intensity3, aolp3, dolp3 = np.hsplit(concat3, 3)
        intensity3 = np.repeat(intensity3[:, :, None], 3, axis=2)
        aolp3 = cv2.applyColorMap(aolp3, cv2.COLORMAP_HSV)
        dolp3 = cv2.applyColorMap(dolp3, cv2.COLORMAP_HOT)
        frame3 = cv2.hconcat([intensity3, aolp3, dolp3])

        output = cv2.vconcat([frame1, frame2, frame3])
        check_mkdir(os.path.join(output_dir, name))
        output_path = os.path.join(output_dir, name, name + '_' + image_name)
        cv2.imwrite(output_path, output)

print('Done!')

