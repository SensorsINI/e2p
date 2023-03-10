"""
 @Time    : 15.11.22 09:27
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : align4_ablation.py
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
#
# method1_dir = os.path.join('/home/mhy/firenet-pdavis/output_real', method1)
# method2_dir = os.path.join('/home/mhy/firenet-pdavis/output_real', method2)
# method3_dir = os.path.join('/home/mhy/firenet-pdavis/output_real', method3)
# output_dir = '/home/mhy/firenet-pdavis/output_real/compare_{}_{}_{}'.format(method1, method2, method3)
# check_mkdir(output_dir)

# for ablation study
method1 = 'v16_b_3_16'
method2 = 'v16_br_30'
method3 = 'v16_bc'
method4 = 'v16_mix2'
method5 = 'v_test_gt'

method1_dir = os.path.join('/home/mhy/firenet-pdavis/output_ablation', method1)
method2_dir = os.path.join('/home/mhy/firenet-pdavis/output_ablation', method2)
method3_dir = os.path.join('/home/mhy/firenet-pdavis/output_ablation', method3)
method4_dir = os.path.join('/home/mhy/firenet-pdavis/output_ablation', method4)
method5_dir = os.path.join('/home/mhy/firenet-pdavis/output_ablation', method5)
output_dir = '/home/mhy/firenet-pdavis/output_ablation/compare_{}_{}_{}'.format(method1, method2, method3, method4, method5)
check_mkdir(output_dir)

list = [x for x in os.listdir(method5_dir) if not x.endswith('.avi')]

for name in sorted(list):
    print(name)

    method1_path = os.path.join(method1_dir, name)
    method2_path = os.path.join(method2_dir, name)
    method3_path = os.path.join(method3_dir, name)
    method4_path = os.path.join(method4_dir, name)
    method5_path = os.path.join(method5_dir, name)

    length_method1 = len(os.listdir(method1_path))
    length_method2 = len(os.listdir(method2_path))
    length_method3 = len(os.listdir(method3_path))
    length_method4 = len(os.listdir(method4_path))
    length_method5 = len(os.listdir(method5_path))
    if length_method1 != length_method2 or length_method1 != length_method3 or length_method1 != length_method4 or length_method1 != length_method5 + 1:
        # if length_method1 != length_method2 or length_method1 != length_method3:
        print('Inconsistency in frame numbers!')
        print(length_method1, length_method2, length_method3, length_method4, length_method5)
        exit(0)
    else:
        print('Method has %d frames.' % length_method1)

    i = 0
    for image_name in tqdm(sorted(os.listdir(method5_path))):
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

        concat4 = cv2.imread(os.path.join(method4_path, image_name), cv2.IMREAD_GRAYSCALE)
        intensity4, aolp4, dolp4 = np.hsplit(concat4, 3)
        intensity4 = np.repeat(intensity4[:, :, None], 3, axis=2)
        aolp4 = cv2.applyColorMap(aolp4, cv2.COLORMAP_HSV)
        dolp4 = cv2.applyColorMap(dolp4, cv2.COLORMAP_HOT)
        frame4 = cv2.hconcat([intensity4, aolp4, dolp4])

        concat5 = cv2.imread(os.path.join(method5_path, image_name), cv2.IMREAD_GRAYSCALE)
        intensity5, aolp5, dolp5 = np.hsplit(concat5, 3)
        intensity5 = np.repeat(intensity5[:, :, None], 3, axis=2)
        aolp5 = cv2.applyColorMap(aolp5, cv2.COLORMAP_HSV)
        dolp5 = cv2.applyColorMap(dolp5, cv2.COLORMAP_HOT)
        frame5 = cv2.hconcat([intensity5, aolp5, dolp5])

        output = cv2.vconcat([frame1, frame2, frame3, frame4, frame5])
        check_mkdir(os.path.join(output_dir, name))
        output_path = os.path.join(output_dir, name, name + '_' + image_name)
        cv2.imwrite(output_path, output)

print('Done!')
