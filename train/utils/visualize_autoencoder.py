"""
 @Time    : 05.09.22 09:49
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : visualize_autoencoder.py
 @Function:
 
"""
# import os
# import cv2
# from tqdm import tqdm
#
# # input_root = '/home/mhy/ECNet/checkpoints/ckpt/dolp_patch/images'
# # output_root = '/home/mhy/ECNet/checkpoints/ckpt/dolp_patch/images_color'
# input_root = '/home/mhy/ae/results/AutoEncoder_DoLP/DoLP'
# output_root = '/home/mhy/ae/results/AutoEncoder_DoLP/DoLP_color'
# if not os.path.exists(output_root):
#     os.makedirs(output_root)
#
# image_list = os.listdir(input_root)
# for image_name in tqdm(image_list):
#     image_path = os.path.join(input_root, image_name)
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     # output = cv2.applyColorMap(image, cv2.COLORMAP_HSV)
#     output = cv2.applyColorMap(image, cv2.COLORMAP_HOT)
#     output_path = os.path.join(output_root, image_name)
#     cv2.imwrite(output_path, output)
#
# print('Succeed!')

import os
import cv2
from tqdm import tqdm

# input_root = '/home/mhy/ae/results/AutoEncoder_AoLP/AoLP'
# output_root = '/home/mhy/ae/results/AutoEncoder_AoLP/AoLP_color'
input_root = '/home/mhy/ae/data/dolp_patch'
output_root = '/home/mhy/ae/data/dolp_patch_color'
if not os.path.exists(output_root):
    os.makedirs(output_root)

image_list = os.listdir(input_root)
image_list = sorted(image_list)[:100]
for image_name in tqdm(image_list):
    image_path = os.path.join(input_root, image_name)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # output = cv2.applyColorMap(image, cv2.COLORMAP_HSV)
    output = cv2.applyColorMap(image, cv2.COLORMAP_HOT)
    output_path = os.path.join(output_root, image_name)
    cv2.imwrite(output_path, output)

print('Succeed!')
