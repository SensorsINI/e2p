"""
 @Time    : 12.09.22 19:28
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : evaluation_single_image.py
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

# learned_perceptual_image_patch_similarity = lpips.LPIPS(net='alex')
learned_perceptual_image_patch_similarity = lpips.LPIPS(net='vgg')

image_name = '01102.png'

gt_root = '/home/mhy/firenet-pdavis/output/compare_single_image/gt'
method1_root = '/home/mhy/firenet-pdavis/output/compare_single_image/m88'
method2_root = '/home/mhy/firenet-pdavis/output/compare_single_image/m88_new_pae_2'

gt_path = os.path.join(gt_root, image_name)
method1_path = os.path.join(method1_root, image_name)
method2_path = os.path.join(method2_root, image_name)

gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
method1 = cv2.imread(method1_path, cv2.IMREAD_GRAYSCALE)
method2 = cv2.imread(method2_path, cv2.IMREAD_GRAYSCALE)

gt_i, gt_a, gt_d = np.hsplit(gt / 255, 3)
method1_i, method1_a, method1_d = np.hsplit(method1 / 255, 3)
method2_i, method2_a, method2_d = np.hsplit(method2 / 255, 3)

method1_i_mse = mean_squared_error(gt_i, method1_i)
method2_i_mse = mean_squared_error(gt_i, method2_i)
method1_a_mse = mean_squared_error(gt_a, method1_a)
method2_a_mse = mean_squared_error(gt_a, method2_a)
method1_d_mse = mean_squared_error(gt_d, method1_d)
method2_d_mse = mean_squared_error(gt_d, method2_d)

method1_i_ssim = structural_similarity(gt_i, method1_i)
method2_i_ssim = structural_similarity(gt_i, method2_i)
method1_a_ssim = structural_similarity(gt_a, method1_a)
method2_a_ssim = structural_similarity(gt_a, method2_a)
method1_d_ssim = structural_similarity(gt_d, method1_d)
method2_d_ssim = structural_similarity(gt_d, method2_d)

method1_i_lpips = learned_perceptual_image_patch_similarity(
                torch.from_numpy(np.repeat(gt_i[None, None, :, :], 3, axis=1) * 2 - 1).float(),
                torch.from_numpy(np.repeat(method1_i[None, None, :, :], 3, axis=1) * 2 - 1).float(),
            ).data.squeeze().numpy().astype(float)
method2_i_lpips = learned_perceptual_image_patch_similarity(
                torch.from_numpy(np.repeat(gt_i[None, None, :, :], 3, axis=1) * 2 - 1).float(),
                torch.from_numpy(np.repeat(method2_i[None, None, :, :], 3, axis=1) * 2 - 1).float(),
            ).data.squeeze().numpy().astype(float)
method1_a_lpips = learned_perceptual_image_patch_similarity(
                torch.from_numpy(np.repeat(gt_a[None, None, :, :], 3, axis=1) * 2 - 1).float(),
                torch.from_numpy(np.repeat(method1_a[None, None, :, :], 3, axis=1) * 2 - 1).float(),
            ).data.squeeze().numpy().astype(float)
method2_a_lpips = learned_perceptual_image_patch_similarity(
                torch.from_numpy(np.repeat(gt_a[None, None, :, :], 3, axis=1) * 2 - 1).float(),
                torch.from_numpy(np.repeat(method2_a[None, None, :, :], 3, axis=1) * 2 - 1).float(),
            ).data.squeeze().numpy().astype(float)
method1_d_lpips = learned_perceptual_image_patch_similarity(
                torch.from_numpy(np.repeat(gt_d[None, None, :, :], 3, axis=1) * 2 - 1).float(),
                torch.from_numpy(np.repeat(method1_d[None, None, :, :], 3, axis=1) * 2 - 1).float(),
            ).data.squeeze().numpy().astype(float)
method2_d_lpips = learned_perceptual_image_patch_similarity(
                torch.from_numpy(np.repeat(gt_d[None, None, :, :], 3, axis=1) * 2 - 1).float(),
                torch.from_numpy(np.repeat(method2_d[None, None, :, :], 3, axis=1) * 2 - 1).float(),
            ).data.squeeze().numpy().astype(float)

print('-------------------------------------')
print('Intensity for {}'.format(image_name))
print('Method1: mse / ssim / lpips --- {:.4f} / {:.4f} / {:.4f}'.format(method1_i_mse, method1_i_ssim, method1_i_lpips))
print('Method2: mse / ssim / lpips --- {:.4f} / {:.4f} / {:.4f}'.format(method2_i_mse, method2_i_ssim, method2_i_lpips))

print('-------------------------------------')
print('AoLP for {}'.format(image_name))
print('Method1: mse / ssim / lpips --- {:.4f} / {:.4f} / {:.4f}'.format(method1_a_mse, method1_a_ssim, method1_a_lpips))
print('Method2: mse / ssim / lpips --- {:.4f} / {:.4f} / {:.4f}'.format(method2_a_mse, method2_a_ssim, method2_a_lpips))

print('-------------------------------------')
print('DoLP for {}'.format(image_name))
print('Method1: mse / ssim / lpips --- {:.4f} / {:.4f} / {:.4f}'.format(method1_d_mse, method1_d_ssim, method1_d_lpips))
print('Method2: mse / ssim / lpips --- {:.4f} / {:.4f} / {:.4f}'.format(method2_d_mse, method2_d_ssim, method2_d_lpips))
