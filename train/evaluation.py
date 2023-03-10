"""
 @Time    : 29.05.22 18:54
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : evaluation.py
 @Function:
 
"""
import os
import cv2
import torch
import numpy as np
from collections import OrderedDict
from tabulate import tabulate
from tqdm import tqdm
from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity
import lpips
learned_perceptual_image_patch_similarity = lpips.LPIPS(net='alex')
# learned_perceptual_image_patch_similarity = lpips.LPIPS(net='vgg')

lpips_flag = True
# lpips_flag = False

gt_root = '/home/mhy/firenet-pdavis/output/m_gt'

method = 'm88_new_pae_3'
prediction_root = '/home/mhy/firenet-pdavis/output/' + method
txt_path = '/home/mhy/firenet-pdavis/txt_evaluation/' + method + '.txt'

video_list = [x for x in os.listdir(prediction_root) if not x.endswith('.avi')]
video_list = sorted(video_list)
# video_list = ['00704']

results = OrderedDict()
for video_name in video_list:
    gt_dir = os.path.join(gt_root, video_name)
    prediction_dir = os.path.join(prediction_root, video_name)

    frame_list = [y for y in os.listdir(prediction_dir) if y.endswith('.png')]
    frame_list = sorted(frame_list)

    mse_i = []
    mse_a = []
    mse_d = []
    ssim_i = []
    ssim_a = []
    ssim_d = []
    lpips_i = []
    lpips_a = []
    lpips_d = []

    for frame_name in tqdm(frame_list):
        # for 2polarization
        gt_path = os.path.join(gt_dir, frame_name)
        # for 2raw
        # gt_path = os.path.join(gt_dir, '%05d.png'%(int(frame_name.split('.')[0])+1))
        prediction_path = os.path.join(prediction_dir, frame_name)

        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        prediction = cv2.imread(prediction_path, cv2.IMREAD_GRAYSCALE)

        gt_i, gt_a, gt_d = np.hsplit(gt / 255, 3)
        prediction_i, prediction_a, prediction_d = np.hsplit(prediction / 255, 3)

        mse_i.append(mean_squared_error(gt_i, prediction_i))
        mse_a.append(mean_squared_error(gt_a, prediction_a))
        mse_d.append(mean_squared_error(gt_d, prediction_d))

        ssim_i.append(structural_similarity(gt_i, prediction_i))
        ssim_a.append(structural_similarity(gt_a, prediction_a))
        ssim_d.append(structural_similarity(gt_d, prediction_d))

        if lpips_flag:
            lpips_i.append(learned_perceptual_image_patch_similarity(
                torch.from_numpy(np.repeat(gt_i[None, None, :, :], 3, axis=1) * 2 - 1).float(),
                torch.from_numpy(np.repeat(prediction_i[None, None, :, :], 3, axis=1) * 2 - 1).float(),
            ).data.squeeze().numpy().astype(float))
            lpips_a.append(learned_perceptual_image_patch_similarity(
                torch.from_numpy(np.repeat(gt_a[None, None, :, :], 3, axis=1) * 2 - 1).float(),
                torch.from_numpy(np.repeat(prediction_a[None, None, :, :], 3, axis=1) * 2 - 1).float(),
            ).data.squeeze().numpy().astype(float))
            lpips_d.append(learned_perceptual_image_patch_similarity(
                torch.from_numpy(np.repeat(gt_d[None, None, :, :], 3, axis=1) * 2 - 1).float(),
                torch.from_numpy(np.repeat(prediction_d[None, None, :, :], 3, axis=1) * 2 - 1).float(),
            ).data.squeeze().numpy().astype(float))
        else:
            lpips_i.append(0)
            lpips_a.append(0)
            lpips_d.append(0)

    results[video_name] = [
        np.mean(mse_i), np.mean(mse_a), np.mean(mse_d),
        np.mean(ssim_i), np.mean(ssim_a), np.mean(ssim_d),
        np.mean(lpips_i), np.mean(lpips_a), np.mean(lpips_d)
        ]

show_list_i = []
for m in range(3):
    for video_name in video_list:
        show_list_i.append(results[video_name][0 + 3 * m])
show_list_i = ['%.3f'%a for a in show_list_i]


show_list_a = []
for m in range(3):
    for video_name in video_list:
        show_list_a.append(results[video_name][1 + 3 * m])
show_list_a = ['%.3f'%a for a in show_list_a]

show_list_d = []
for m in range(3):
    for video_name in video_list:
        show_list_d.append(results[video_name][2 + 3 * m])
show_list_d = ['%.3f'%a for a in show_list_d]

head = []
for n in ['m', 's', 'l']:
    for video_name in video_list:
        head.append(video_name + n)

print(method)
print(tabulate([show_list_i, show_list_a, show_list_d], headers=head))

f = open(txt_path, 'w')
f.write(tabulate([show_list_i, show_list_a, show_list_d], headers=head))
