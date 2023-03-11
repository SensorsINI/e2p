"""
 @Time    : 2/14/22 16:06
 @Author  : Haiyang Mei
 @E-mail  : mhy666@mail.dlut.edu.cn
 
 @Project : firenet-pdavis
 @File    : f12p.py
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

method = 'FireNet_retrained'
dataset = 'mhy3'

if_color = False

frame_root = '/home/mhy/firenet-pdavis/output'
# i_dir = os.path.join(frame_root, method, dataset + '_i')
i_dir = os.path.join(frame_root, method, dataset)

output_methodname = 'f12p'
intensity_dir = os.path.join(frame_root, method, dataset + '_' + output_methodname, 'intensity')
aolp_dir = os.path.join(frame_root, method, dataset + '_' + output_methodname, 'aolp')
dolp_dir = os.path.join(frame_root, method, dataset + '_' + output_methodname, 'dolp')
check_mkdir(intensity_dir)
check_mkdir(aolp_dir)
check_mkdir(dolp_dir)

i_list = [x for x in os.listdir(i_dir) if x.endswith('.png')]
i_list.sort(key=lambda x:int(x.split('_')[1].split('.')[0]))

print('The number of frame is: ', len(i_list))

for j in tqdm(range(len(i_list))):
    output_name = i_list[j]

    i_path = os.path.join(i_dir, i_list[j])

    i = cv2.imread(i_path, cv2.IMREAD_GRAYSCALE)

    i90 = i[0::2, 0::2]
    i45 = i[0::2, 1::2]
    i135 = i[1::2, 0::2]
    i0 = i[1::2, 1::2]

    s0 = i0.astype(float) + i90.astype(float)
    s1 = i0.astype(float) - i90.astype(float)
    s2 = i45.astype(float) - i135.astype(float)

    intensity = (s0 / 2).astype(np.uint8)

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
