"""
 @Time    : 06.10.22 19:04
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : eval.py
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

# gt_root = '/home/mhy/firenet-pdavis/output/v_test_gt'
gt_root = './output_synthetic/gt'

# method = 'firenet_direction_iad'
# method = 'e2vid_direction_iad'
# method = 'v16_mix2'
# method = 'v16_b_3_14'
# method = 'v16_b_4_10'
# method = 'v16_br_20'
# method = 'v16_br_2_30'
# method = 'v16_b_k4_3_60'
# method = 'v16_b_c16_s_73'
# method = 'v16_b_c16_2'
# method = 'v16_b_c16_i_30'
# method = 'v16_so_2'
method = 'e2p'
print(method)
# prediction_root = '/home/mhy/firenet-pdavis/output/' + method
prediction_root = './output_synthetic/' + method
# prediction_root = '/home/mhy/firenet-pdavis/output_ablation/' + method
# txt_path = '/home/mhy/firenet-pdavis/txt_eval/' + method + '.txt'
txt_path = './txt_eval/' + method + '.txt'

video_list = [x for x in os.listdir(gt_root) if not x.endswith('.avi')]
video_list = sorted(video_list)

results = OrderedDict()
for video_name in video_list:
    gt_dir = os.path.join(gt_root, video_name)
    prediction_dir = os.path.join(prediction_root, video_name)

    frame_list = [y for y in os.listdir(gt_dir) if y.endswith('.png')]
    frame_list = sorted(frame_list)
    frame_list = frame_list[1:]     # because frame 0 has no input events

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
