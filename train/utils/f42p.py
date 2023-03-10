"""
 @Time    : 2/15/22 20:45
 @Author  : Haiyang Mei
 @E-mail  : mhy666@mail.dlut.edu.cn
 
 @Project : firenet-pdavis
 @File    : f42p.py
 @Function:
 
"""
import os
import sys
sys.path.append('..')
from misc import check_mkdir
import numpy as np
import cv2
import math
from tqdm import tqdm

method = 'FireNet'
# method = 'E2VID'
dataset = 'subject09_group2_time1'

if_color = False

i_root = '/home/mhy/firenet-pdavis/output'
i0_dir = os.path.join(i_root, method, dataset + '_i0')
i45_dir = os.path.join(i_root, method, dataset + '_i45')
i90_dir = os.path.join(i_root, method, dataset + '_i90')
i135_dir = os.path.join(i_root, method, dataset + '_i135')

output_methodname = 'f42p'
intensity_dir = os.path.join(i_root, method, dataset + '_' + output_methodname, 'intensity')
aolp_dir = os.path.join(i_root, method, dataset + '_' + output_methodname, 'aolp')
dolp_dir = os.path.join(i_root, method, dataset + '_' + output_methodname, 'dolp')
check_mkdir(intensity_dir)
check_mkdir(aolp_dir)
check_mkdir(dolp_dir)

i0_list = [x for x in os.listdir(i0_dir) if x.endswith('.png')]
i45_list = [x for x in os.listdir(i45_dir) if x.endswith('.png')]
i90_list = [x for x in os.listdir(i90_dir) if x.endswith('.png')]
i135_list = [x for x in os.listdir(i135_dir) if x.endswith('.png')]

i0_list.sort(key=lambda x:int(x.split('_')[1].split('.')[0]))
i45_list.sort(key=lambda x:int(x.split('_')[1].split('.')[0]))
i90_list.sort(key=lambda x:int(x.split('_')[1].split('.')[0]))
i135_list.sort(key=lambda x:int(x.split('_')[1].split('.')[0]))

print('The number of channel 0 frame is: ', len(i0_list))
print('The number of channel 45 frame is: ', len(i45_list))
print('The number of channel 90 frame is: ', len(i90_list))
print('The number of channel 135 frame is: ', len(i135_list))
# assert len(i0_list) == len(i45_list), 'The number of channel 0 and 45 frame does not match!'
# assert len(i0_list) == len(i90_list), 'The number of channel 0 and 90 frame does not match!'
# assert len(i0_list) == len(i135_list), 'The number of channel 0 and 135 frame does not match!'

number_list = [len(i0_list), len(i45_list), len(i90_list), len(i135_list)]
number = min(number_list)
print(number)

# print(i0_list)
# print(i45_list)
# print(i90_list)
# print(i135_list)

# for i in tqdm(range(len(i0_list))):
for i in tqdm(range(number)):
    output_name = i0_list[i]

    i0_path = os.path.join(i0_dir, i0_list[i])
    i45_path = os.path.join(i45_dir, i45_list[i])
    i90_path = os.path.join(i90_dir, i90_list[i])
    i135_path = os.path.join(i135_dir, i135_list[i])

    # print(i0_path)
    # print(i45_path)
    # print(i90_path)
    # print(i135_path)
    # exit(0)

    i0 = cv2.imread(i0_path, cv2.IMREAD_GRAYSCALE)
    i45 = cv2.imread(i45_path, cv2.IMREAD_GRAYSCALE)
    i90 = cv2.imread(i90_path, cv2.IMREAD_GRAYSCALE)
    i135 = cv2.imread(i135_path, cv2.IMREAD_GRAYSCALE)

    # i0 = cv2.flip(i0, 0)
    # i45 = cv2.flip(i45, 0)
    # i90 = cv2.flip(i90, 0)
    # i135 = cv2.flip(i135, 0)

    s0 = i0.astype(float) + i90.astype(float)
    s1 = i0.astype(float) - i90.astype(float)
    s2 = i45.astype(float) - i135.astype(float)

    # print(np.max(s0))
    # intensity = s0.astype(np.uint8)
    intensity = (s0 / 2).astype(np.uint8)
    # print(np.max(intensity))
    # exit(0)

    aolp = 0.5 * np.arctan2(s2, s1)
    aolp[s2 < 0] += math.pi
    aolp = (aolp * (255 / math.pi)).astype(np.uint8)
    if if_color:
        aolp = cv2.applyColorMap(aolp, cv2.COLORMAP_JET)

    dolp = np.divide(np.sqrt(np.square(s1) + np.square(s2)), s0, out=np.zeros_like(s0).astype(float), where=s0 != 0)
    dolp = dolp.clip(0.0, 1.0)
    dolp = (dolp * 255).astype(np.uint8)
    if if_color:
        dolp = cv2.applyColorMap(dolp, cv2.COLORMAP_HOT)

    intensity_path = os.path.join(intensity_dir, output_name[:-4] + '_i.png')
    aolp_path = os.path.join(aolp_dir, output_name[:-4] + '_aolp.png')
    dolp_path = os.path.join(dolp_dir, output_name[:-4] + '_dolp.png')

    cv2.imwrite(intensity_path, intensity)
    cv2.imwrite(aolp_path, aolp)
    cv2.imwrite(dolp_path, dolp)

print('Succeed!')
