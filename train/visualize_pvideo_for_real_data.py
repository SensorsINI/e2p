"""
 @Time    : 15.09.22 17:37
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : visualize_pvideo_for_real_data.py
 @Function:
 
"""
import os
import sys

sys.path.append('')
import cv2
import numpy as np
from tqdm import tqdm

# root = '/home/mhy/aedat2pvideo/output_pdavis_hallway/firenet_direction_iad'
# root = '/home/mhy/firenet-pdavis/output_pdavis/m88_new_pae_2_5ms_hallway'
# root = '/home/mhy/firenet-pdavis/output_pdavis/m88_new_pae_2_5ms_rpm0430'
# root = '/home/mhy/firenet-pdavis/output_pdavis/v3_1'
root = '/home/mhy/firenet-pdavis/output_pdavis/v11_s/10*5ms'
output_dir = root

mask_aolp_flag = False
flip_flag = False

avi_name_list = [
    # 'Davis346B-2022-09-11T20-33-30+0200-00000000-0-Hallway',
    # 'Davis346B-2022-09-11T20-28-06+0200-00000000-0-Desktop',
    # 'UIUC_RPM_1000',
    # 'Davis346B-2022-09-11T20-33-30+0200-00000000-0-Hallway_004',
    # 'Davis346B-2022-10-16T11-38-07+0200-00000000-0_000',
    'Davis346B-2022-10-15T17-42-39+0200-00000000-0_000',
]

# avi_name_list = os.listdir(root)

check_list = os.listdir(root)
check_list = sorted(check_list)
print(check_list)

for avi_name in avi_name_list:
    print(avi_name)

    rate = 25
    # size = (519, 130)
    size = (519, 260)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(os.path.join(output_dir, avi_name + '.avi'), fourcc, rate, size, isColor=True)

    for check_name in check_list:
        if avi_name in check_name:

            dir = os.path.join(root, check_name)

            list = [x for x in os.listdir(dir) if x.endswith('.png')]
            list.sort(key=lambda x: int(x.split('.')[0]))

            for name in tqdm(list):
                path = os.path.join(dir, name)

                concat = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if flip_flag:
                    concat = np.flipud(concat)
                intensity, aolp, dolp = np.hsplit(concat, 3)
                if flip_flag:
                    intensity = np.fliplr(intensity)
                    aolp = np.fliplr(aolp)
                    dolp = np.fliplr(dolp)

                intensity = np.repeat(intensity[:, :, None], 3, axis=2)

                aolp = cv2.applyColorMap(aolp, cv2.COLORMAP_HSV)

                if mask_aolp_flag:
                    mask = np.where(dolp[:, :] >= 12.75, 255, 0).astype(np.uint8)
                    for i in range(3):
                        aolp[:, :, i] = np.where(mask == 255, aolp[:, :, i], 0).astype(np.uint8)

                dolp = cv2.applyColorMap(dolp, cv2.COLORMAP_HOT)

                frame = cv2.hconcat([intensity, aolp, dolp])

                video.write(frame)

print('Succeed!')
