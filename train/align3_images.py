"""
 @Time    : 13.10.22 10:28
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : align3_images.py
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
# method1 = 'test_gt'
# method2 = 'firenet_direction_iad'
# method3 = 'v16_mix2'
# method1 = 'v16_mix2'
# method2 = 'v16_so_1'
# method3 = 'v16_so_2'
method1 = 'test_gt'
method2 = 'v16_so_2'
method3 = 'v16_mix2'

method1_dir = os.path.join('/home/mhy/firenet-pdavis/output_real', method1)
method2_dir = os.path.join('/home/mhy/firenet-pdavis/output_real', method2)
method3_dir = os.path.join('/home/mhy/firenet-pdavis/output_real', method3)
output_dir = '/home/mhy/firenet-pdavis/output_real/compare_{}_{}_{}'.format(method1, method2, method3)
check_mkdir(output_dir)

list = [x for x in os.listdir(method2_dir) if not x.endswith('.avi')]

for name in sorted(list):
    print(name)

    method1_path = os.path.join(method1_dir, name)
    method2_path = os.path.join(method2_dir, name)
    method3_path = os.path.join(method3_dir, name)

    length_method1 = len(os.listdir(method1_path))
    length_method2 = len(os.listdir(method2_path))
    length_method3 = len(os.listdir(method3_path))
    # if length_method1 != length_method2 or length_method1 != length_method3 - 1:
    # if length_method1 != length_method2 or length_method1 != length_method3:
    if length_method1 != length_method2 - 1 or length_method1 != length_method3 - 1:
        # if length_method1 != length_method2 or length_method1 != length_method3:
        print('Inconsistency in frame numbers!')
        print(length_method1, length_method2, length_method3)
        exit(0)
    else:
        print('Method has %d frames.' % length_method1)

    i = 0
    # for image_name in tqdm(sorted(os.listdir(method1_path))):
    for image_name in tqdm(sorted(os.listdir(method2_path))[:-2]):
        i += 1
        if i == 1:
            continue

        concat1 = cv2.imread(os.path.join(method1_path, image_name), cv2.IMREAD_GRAYSCALE)
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
