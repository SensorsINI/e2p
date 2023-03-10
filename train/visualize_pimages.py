"""
 @Time    : 04.10.22 16:49
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : visualize_pimages.py
 @Function:
 
"""
import os
import cv2
import numpy as np
from tqdm import tqdm

# root = '/home/mhy/firenet-pdavis/output/v_test_gt'
# root = '/home/mhy/firenet-pdavis/output/firenet_direction_iad'
root = '/home/mhy/firenet-pdavis/output/e2vid_direction_iad'
output_dir = root + '_color'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# mask_aolp_flag = True
mask_aolp_flag = False

scene_name_list = os.listdir(root)

for scene_name in scene_name_list:
    print(scene_name)

    dir = os.path.join(root, scene_name)

    if mask_aolp_flag:
        scene_name = scene_name + '_masked_aolp'

    output_scene_dir = os.path.join(output_dir, scene_name)
    if not os.path.exists(output_scene_dir):
        os.makedirs(output_scene_dir)

    list = [x for x in os.listdir(dir) if x.endswith('.png')]
    list.sort(key=lambda x: int(x.split('.')[0]))

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

        frame_path = os.path.join(output_scene_dir, name[:-4] + '_color.png')

        cv2.imwrite(frame_path, frame)

print('Succeed!')
