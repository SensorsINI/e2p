"""
 @Time    : 20.03.22 17:49
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : evaluation.py
 @Function:
 
"""
import os
import sys
sys.path.append('..')
from misc import check_mkdir
import cv2
import torch
import numpy as np
from tqdm import tqdm
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import lpips
learned_perceptual_image_patch_similarity = lpips.LPIPS(net='alex')

gt_root = '/home/mhy/firenet-pdavis/data/v2e/gt'
prediction_root = '/home/mhy/firenet-pdavis/output/FireNet'

manner = 'f12p'

# video_name = 'subject01_group1_time1'
# video_name = 'subject02_group3_time4'
# video_name = 'subject05_group3_time3'
# video_name = 'subject09_group2_time1'
# video_name = 'subject09_group2_time3'
video_name = 'subject09_group2_time4'

gt_dir = os.path.join(gt_root, video_name)
prediction_dir = os.path.join(prediction_root, video_name + '_' + manner, 'concat')

output_dir = '../evaluation_txt'
check_mkdir(output_dir)

name_list = []
intensity_mse = []
intensity_psnr = []
intensity_ssim = []
intensity_lpips = []
aolp_mse = []
aolp_psnr = []
aolp_ssim = []
aolp_lpips = []
dolp_mse = []
dolp_psnr = []
dolp_ssim = []
dolp_lpips = []

prediction_list = os.listdir(prediction_dir)
prediction_list.sort(key=lambda x: int(x.split('.')[0]))
print('GT has {} frames.'.format(len(os.listdir(gt_dir))))
print('PR has {} frames.'.format(len(prediction_list)))
for name in tqdm(prediction_list):
    name_list.append(name)

    prediction_path = os.path.join(prediction_dir, name)
    gt_path = os.path.join(gt_dir, name)

    prediction = cv2.imread(prediction_path, cv2.IMREAD_GRAYSCALE)
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

    gt_intensity, gt_aolp, gt_dolp = np.hsplit(gt / 255.0, 3)
    prediction_intensity, prediction_aolp, prediction_dolp = np.hsplit(prediction / 255.0, 3)

    # mask operation
    # mask = np.where(gt_dolp[:, :] >= 0.15, 255, 0).astype(np.uint8)
    # gt_aolp = np.where(mask == 255, gt_aolp, 0).astype(np.float64)
    # prediction_aolp = np.where(mask == 255, prediction_aolp, 0).astype(np.float64)

    # print(np.max(gt_intensity))
    # print(np.min(gt_intensity))
    # print((np.repeat(gt_intensity[None, None, :, :], 3, axis=1) * 2 - 1).shape)
    # print(np.max((np.repeat(gt_intensity[None, None, :, :], 3, axis=1) * 2 - 1)))
    # print(np.min((np.repeat(gt_intensity[None, None, :, :], 3, axis=1) * 2 - 1)))
    # print(learned_perceptual_image_patch_similarity(
    #     torch.from_numpy(np.repeat(gt_intensity[None, None, :, :], 3, axis=1) * 2 - 1).float(),
    #     torch.from_numpy(np.repeat(prediction_intensity[None, None, :, :], 3, axis=1) * 2 - 1).float(),
    # ).data.squeeze().numpy().astype(float).dtype)
    # print(mean_squared_error(gt_intensity, prediction_intensity).dtype)
    # exit(0)

    # intensity
    intensity_mse.append(mean_squared_error(gt_intensity, prediction_intensity))
    intensity_psnr.append(peak_signal_noise_ratio(gt_intensity, prediction_intensity))
    intensity_ssim.append(structural_similarity(gt_intensity, prediction_intensity))
    intensity_lpips.append(learned_perceptual_image_patch_similarity(
        torch.from_numpy(np.repeat(gt_intensity[None, None, :, :], 3, axis=1) * 2 - 1).float(),
        torch.from_numpy(np.repeat(prediction_intensity[None, None, :, :], 3, axis=1) * 2 - 1).float(),
    ).data.squeeze().numpy().astype(float))

    # aolp
    aolp_mse.append(mean_squared_error(gt_aolp, prediction_aolp))
    aolp_psnr.append(peak_signal_noise_ratio(gt_aolp, prediction_aolp))
    aolp_ssim.append(structural_similarity(gt_aolp, prediction_aolp))
    aolp_lpips.append(learned_perceptual_image_patch_similarity(
        torch.from_numpy(np.repeat(gt_aolp[None, None, :, :], 3, axis=1) * 2 - 1).float(),
        torch.from_numpy(np.repeat(prediction_aolp[None, None, :, :], 3, axis=1) * 2 - 1).float(),
    ).data.squeeze().numpy().astype(float))

    # dolp
    dolp_mse.append(mean_squared_error(gt_dolp, prediction_dolp))
    dolp_psnr.append(peak_signal_noise_ratio(gt_dolp, prediction_dolp))
    dolp_ssim.append(structural_similarity(gt_dolp, prediction_dolp))
    dolp_lpips.append(learned_perceptual_image_patch_similarity(
        torch.from_numpy(np.repeat(gt_dolp[None, None, :, :], 3, axis=1) * 2 - 1).float(),
        torch.from_numpy(np.repeat(prediction_dolp[None, None, :, :], 3, axis=1) * 2 - 1).float(),
    ).data.squeeze().numpy().astype(float))

print("---------------------")
print("{}".format(video_name + '_' + manner))
print("Overall Performance:")
print("Overall Intensity:  MSE: {:.4f}  PSNR: {:>7.4f}  SSIM: {:.4f}  LPIPS: {:.4f}".format(np.mean(intensity_mse), np.mean(intensity_psnr), np.mean(intensity_ssim), np.mean(intensity_lpips)))
print("Overall AoLP     :  MSE: {:.4f}  PSNR: {:>7.4f}  SSIM: {:.4f}  LPIPS: {:.4f}".format(np.mean(aolp_mse), np.mean(aolp_psnr), np.mean(aolp_ssim), np.mean(aolp_lpips)))
print("Overall DoLP     :  MSE: {:.4f}  PSNR: {:>7.4f}  SSIM: {:.4f}  LPIPS: {:.4f}".format(np.mean(dolp_mse), np.mean(dolp_psnr), np.mean(dolp_ssim), np.mean(dolp_lpips)))
print("---------------------")

print('Writing to txt file.')
output_path = os.path.join(output_dir, video_name + '_' + manner + '.txt')
f = open(output_path, 'w')
f.write("{}\n\n".format(video_name + '_' + manner))
f.write("Overall Performance:\n")
f.write("Overall Intensity:  MSE: {:.4f}  PSNR: {:>7.4f}  SSIM: {:.4f}  LPIPS: {:.4f}\n".format(np.mean(intensity_mse), np.mean(intensity_psnr), np.mean(intensity_ssim), np.mean(intensity_lpips)))
f.write("Overall AoLP     :  MSE: {:.4f}  PSNR: {:>7.4f}  SSIM: {:.4f}  LPIPS: {:.4f}\n".format(np.mean(aolp_mse), np.mean(aolp_psnr), np.mean(aolp_ssim), np.mean(aolp_lpips)))
f.write("Overall DoLP     :  MSE: {:.4f}  PSNR: {:>7.4f}  SSIM: {:.4f}  LPIPS: {:.4f}\n".format(np.mean(dolp_mse), np.mean(dolp_psnr), np.mean(dolp_ssim), np.mean(dolp_lpips)))

f.write("\nDetailed Performance:\n")
i = 0
for name in tqdm(prediction_list):
    f.write("{} --- Intensity:  MSE: {:.4f}  PSNR: {:>7.4f}  SSIM: {:.4f}  LPIPS: {:.4f}\n".format(name.rjust(9, ' '), intensity_mse[i], intensity_psnr[i], intensity_ssim[i], intensity_lpips[i]))
    f.write("{} --- AoLP     :  MSE: {:.4f}  PSNR: {:>7.4f}  SSIM: {:.4f}  LPIPS: {:.4f}\n".format(name.rjust(9, ' '), aolp_mse[i], aolp_psnr[i], aolp_ssim[i], aolp_lpips[i]))
    f.write("{} --- DoLP     :  MSE: {:.4f}  PSNR: {:>7.4f}  SSIM: {:.4f}  LPIPS: {:.4f}\n".format(name.rjust(9, ' '), dolp_mse[i], dolp_psnr[i], dolp_ssim[i], dolp_lpips[i]))
    f.write('-------------------------------\n')
    i += 1
print('Succeed!')
