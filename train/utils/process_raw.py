"""
 @Time    : 18.05.22 12:38
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : process_raw.py
 @Function:
 
"""
import os
import cv2
from tqdm import tqdm
import numpy as np
from polarization_functions import *

raw_dir = '/home/mhy/data/movingcam/raw'
output_dir = raw_dir + '_extracted'


def getADoLP(dir, name, data):
    aolp_rgb, aolp_rgb_c, dolp_rgb, dolp_rgb_c, color = CalcADoLP(data)

    path_aolp_r = os.path.join(dir, 'aolp_r')
    path_aolp_g = os.path.join(dir, 'aolp_g')
    path_aolp_b = os.path.join(dir, 'aolp_b')
    check_mkdir(path_aolp_r)
    check_mkdir(path_aolp_g)
    check_mkdir(path_aolp_b)
    cv2.imwrite(os.path.join(path_aolp_r, name + '_aolp_r.tiff'), aolp_rgb[0])
    cv2.imwrite(os.path.join(path_aolp_g, name + '_aolp_g.tiff'), aolp_rgb[1])
    cv2.imwrite(os.path.join(path_aolp_b, name + '_aolp_b.tiff'), aolp_rgb[2])

    path_aolp_r_color = os.path.join(dir, 'aolp_r_color')
    path_aolp_g_color = os.path.join(dir, 'aolp_g_color')
    path_aolp_b_color = os.path.join(dir, 'aolp_b_color')
    check_mkdir(path_aolp_r_color)
    check_mkdir(path_aolp_g_color)
    check_mkdir(path_aolp_b_color)
    cv2.imwrite(os.path.join(path_aolp_r_color, name + '_aolp_r_color.tiff'), aolp_rgb_c[0])
    cv2.imwrite(os.path.join(path_aolp_g_color, name + '_aolp_g_color.tiff'), aolp_rgb_c[1])
    cv2.imwrite(os.path.join(path_aolp_b_color, name + '_aolp_b_color.tiff'), aolp_rgb_c[2])

    path_dolp_r = os.path.join(dir, 'dolp_r')
    path_dolp_g = os.path.join(dir, 'dolp_g')
    path_dolp_b = os.path.join(dir, 'dolp_b')
    check_mkdir(path_dolp_r)
    check_mkdir(path_dolp_g)
    check_mkdir(path_dolp_b)
    cv2.imwrite(os.path.join(path_dolp_r, name + '_dolp_r.tiff'), dolp_rgb[0])
    cv2.imwrite(os.path.join(path_dolp_g, name + '_dolp_g.tiff'), dolp_rgb[1])
    cv2.imwrite(os.path.join(path_dolp_b, name + '_dolp_b.tiff'), dolp_rgb[2])

    path_dolp_r_color = os.path.join(dir, 'dolp_r_color')
    path_dolp_g_color = os.path.join(dir, 'dolp_g_color')
    path_dolp_b_color = os.path.join(dir, 'dolp_b_color')
    check_mkdir(path_dolp_r_color)
    check_mkdir(path_dolp_g_color)
    check_mkdir(path_dolp_b_color)
    cv2.imwrite(os.path.join(path_dolp_r_color, name + '_dolp_r_color.tiff'), dolp_rgb_c[0])
    cv2.imwrite(os.path.join(path_dolp_g_color, name + '_dolp_g_color.tiff'), dolp_rgb_c[1])
    cv2.imwrite(os.path.join(path_dolp_b_color, name + '_dolp_b_color.tiff'), dolp_rgb_c[2])

    path_image = os.path.join(dir, 'image')
    check_mkdir(path_image)
    cv2.imwrite(os.path.join(path_image, name + '_rgb.tiff'), color)


if __name__ == '__main__':
    raw_list = os.listdir(raw_dir)
    for raw in tqdm(raw_list):
        raw_path = os.path.join(raw_dir, raw)
        raw_data = np.load(raw_path)
        for i in range(raw_data.shape[0]):
            name = raw.split('.')[0] + '_' + str(i)
            getADoLP(output_dir, name, raw_data[i, :, :])

    print('Succeed!')

