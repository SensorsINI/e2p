"""
 @Time    : 17.03.22 16:33
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : visualize_pvideo.py
 @Function:
 
"""
import os
import cv2
import numpy as np
from tqdm import tqdm

# set root same as in [h5gt2iad.py](utils%2Fh5gt2iad.py)
root = 'train/output/test_output'
# root = '/home/mhy/firenet-pdavis/output/v_test_gt'
# root = '/home/mhy/firenet-pdavis/output/v5'
output_dir = root

# mask_aolp_flag = True
mask_aolp_flag = False


avi_name_list = os.listdir(root)


for avi_name in avi_name_list:
    print(avi_name)

    dir = os.path.join(root, avi_name)

    if mask_aolp_flag:
        avi_name = avi_name + '_masked_aolp'

    list = [x for x in os.listdir(dir) if x.endswith('.png')]
    list.sort(key=lambda x: int(x.split('.')[0]))

    rate = 25
    # rate = 20
    size = (960, 240)
    # size = (519, 130)
    # size = (3672, 1024)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # video = cv2.VideoWriter(os.path.join(output_dir, avi_name + '_' + str(len(list)) + '.avi'), fourcc, rate, size, isColor=True)
    video = cv2.VideoWriter(os.path.join(output_dir, avi_name + '.avi'), fourcc, rate, size, isColor=True)

    for name in tqdm(list):
        path = os.path.join(dir, name)

        concat = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        intensity, aolp, dolp = np.hsplit(concat, 3)

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
