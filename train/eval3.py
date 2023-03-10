"""
 @Time    : 17.10.22 20:22
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : eval3.py
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

# learned_perceptual_image_patch_similarity = lpips.LPIPS(net='alex').cuda()
learned_perceptual_image_patch_similarity = lpips.LPIPS(net='vgg').cuda()

lpips_flag = True
# lpips_flag = False

root = '/home/mhy/firenet-pdavis/output/v11_s_ft'
# root = '/home/mhy/firenet-pdavis/output_real/v11_s_ft'

# method = root.split('/')[-1]
method = root.split('/')[-1] + '_ours'
txt_path = '/home/mhy/firenet-pdavis/txt_eval/' + method + '.txt'

video_list = [x for x in os.listdir(root) if not x.endswith('.avi')]
video_list = sorted(video_list)
print(video_list)

results = OrderedDict()
for video_name in video_list:
    dir = os.path.join(root, video_name)

    frame_list = [y for y in os.listdir(dir) if y.endswith('.png')]
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
        path = os.path.join(dir, frame_name)
        align3 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        intensity, aolp, dolp = np.hsplit(align3 / 255, 3)

        # firenet
        # gt_i, prediction_i, _ = np.vsplit(intensity, 3)
        # gt_a, prediction_a, _ = np.vsplit(aolp, 3)
        # gt_d, prediction_d, _ = np.vsplit(dolp, 3)

        # prnet
        gt_i, _, prediction_i = np.vsplit(intensity, 3)
        gt_a, _, prediction_a = np.vsplit(aolp, 3)
        gt_d, _, prediction_d = np.vsplit(dolp, 3)

        mse_i.append(mean_squared_error(gt_i, prediction_i))
        mse_a.append(mean_squared_error(gt_a, prediction_a))
        mse_d.append(mean_squared_error(gt_d, prediction_d))

        ssim_i.append(structural_similarity(gt_i, prediction_i))
        ssim_a.append(structural_similarity(gt_a, prediction_a))
        ssim_d.append(structural_similarity(gt_d, prediction_d))

        if lpips_flag:
            lpips_i.append(learned_perceptual_image_patch_similarity(
                torch.from_numpy(np.repeat(gt_i[None, None, :, :], 3, axis=1)).float().cuda(),
                torch.from_numpy(np.repeat(prediction_i[None, None, :, :], 3, axis=1)).float().cuda(), normalize=True
            ).cpu().data.squeeze().numpy().astype(float))
            lpips_a.append(learned_perceptual_image_patch_similarity(
                torch.from_numpy(np.repeat(gt_a[None, None, :, :], 3, axis=1)).float().cuda(),
                torch.from_numpy(np.repeat(prediction_a[None, None, :, :], 3, axis=1)).float().cuda(), normalize=True
            ).cpu().data.squeeze().numpy().astype(float))
            lpips_d.append(learned_perceptual_image_patch_similarity(
                torch.from_numpy(np.repeat(gt_d[None, None, :, :], 3, axis=1)).float().cuda(),
                torch.from_numpy(np.repeat(prediction_d[None, None, :, :], 3, axis=1)).float().cuda(), normalize=True
            ).cpu().data.squeeze().numpy().astype(float))
        else:
            lpips_i.append(0)
            lpips_a.append(0)
            lpips_d.append(0)

    results[video_name] = [
        np.mean(mse_i), np.mean(ssim_i), np.mean(lpips_i),
        np.mean(mse_a), np.mean(ssim_a), np.mean(lpips_a),
        np.mean(mse_d), np.mean(ssim_d), np.mean(lpips_d)
    ]

all_mse_i = []
all_ssim_i = []
all_lpips_i = []

all_mse_a = []
all_ssim_a = []
all_lpips_a = []

all_mse_d = []
all_ssim_d = []
all_lpips_d = []

for video_name in video_list:
    all_mse_i.append(results[video_name][0])
    all_ssim_i.append(results[video_name][1])
    all_lpips_i.append(results[video_name][2])

    all_mse_a.append(results[video_name][3])
    all_ssim_a.append(results[video_name][4])
    all_lpips_a.append(results[video_name][5])

    all_mse_d.append(results[video_name][6])
    all_ssim_d.append(results[video_name][7])
    all_lpips_d.append(results[video_name][8])

results['mean'] = [
    np.mean(all_mse_i), np.mean(all_ssim_i), np.mean(all_lpips_i),
    np.mean(all_mse_a), np.mean(all_ssim_a), np.mean(all_lpips_a),
    np.mean(all_mse_d), np.mean(all_ssim_d), np.mean(all_lpips_d)
]

show_list = [['Mean'] + results['mean']]
show_list.append([])
for video_name in video_list:
    show_list.append([video_name] + results[video_name])

head = ['Video', 'I_mse', 'I_ssim', 'I_lpips', 'A_mse', 'A_ssim', 'A_lpips', 'D_mse', 'D_ssim', 'D_lpips']
floatfmts = ('6s', '.4f', '.4f', '.4f', '.4f', '.4f', '.4f', '.4f', '.4f', '.4f')

print(method)
print(tabulate(show_list, floatfmt=floatfmts, tablefmt='fancy_grid', headers=head))

f = open(txt_path, 'w')
f.write(tabulate(show_list, floatfmt=floatfmts, tablefmt='fancy_grid', headers=head))

