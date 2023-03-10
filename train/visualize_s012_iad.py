"""
 @Time    : 21.09.22 13:55
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : visualize_s012_iad.py
 @Function:
 
"""
import os
import cv2
import numpy as np
from tqdm import tqdm

root = '/home/mhy/firenet-pdavis/output/m_gt'
output_dir = root

# mask_aolp_flag = True
mask_aolp_flag = False

avi_name_list = os.listdir(root)
# avi_name_list = [
#     '007118',
#     # '01405'
# ]

for avi_name in avi_name_list:
    print(avi_name)

    dir = os.path.join(root, avi_name)

    if mask_aolp_flag:
        avi_name = avi_name + '_masked_aolp'

    list = [x for x in os.listdir(dir) if x.endswith('.png')]
    list.sort(key=lambda x: int(x.split('.')[0]))

    rate = 25
    size = (960, 480)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(os.path.join(output_dir, avi_name + '.avi'), fourcc, rate, size, isColor=True)

    for name in tqdm(list):
        path = os.path.join(dir, name)

        concat = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        s0_i, s1_a, s2_d = np.hsplit(concat, 3)
        s0, i = np.vsplit(s0_i, 2)
        s1, a = np.vsplit(s1_a, 2)
        s2, d = np.vsplit(s2_d, 2)

        intensity = np.repeat(i[:, :, None], 3, axis=2)

        aolp = cv2.applyColorMap(a, cv2.COLORMAP_HSV)

        if mask_aolp_flag:
            mask = np.where(d[:, :] >= 12.75, 255, 0).astype(np.uint8)
            for i in range(3):
                aolp[:, :, i] = np.where(mask == 255, aolp[:, :, i], 0).astype(np.uint8)

        dolp = cv2.applyColorMap(d, cv2.COLORMAP_HOT)

        s0 = np.repeat(s0[:, :, None], 3, axis=2)
        s1 = cv2.applyColorMap(s1, cv2.COLORMAP_HOT)
        s2 = cv2.applyColorMap(s2, cv2.COLORMAP_HOT)

        s012 = cv2.hconcat([s0, s1, s2])
        iad = cv2.hconcat([intensity, aolp, dolp])
        frame = cv2.vconcat([s012, iad])

        video.write(frame)

print('Succeed!')
