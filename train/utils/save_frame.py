"""
 @Time    : 20.04.22 18:24
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : save_frame.py
 @Function:
 
"""
import os
import sys
sys.path.append('..')
import cv2
import h5py
from tqdm import tqdm
from misc import check_mkdir

# dir = '/home/mhy/firenet-pdavis/data/train_p'
# file_name = 'subject07_group2_time1_p.h5'
dir = '/home/mhy/firenet-pdavis/data/test_p'
file_name = 'subject09_group2_time1_p.h5'
output_dir = os.path.join(dir, 'intensity_' + file_name[:-3])
check_mkdir(output_dir)

file_path = os.path.join(dir, file_name)

f_input = h5py.File(file_path, 'r')

intensity = f_input['/intensity']

print(intensity.shape)
print(intensity.dtype)

for i in tqdm(range(intensity.shape[0])):
    output_path = os.path.join(output_dir, '%06d.png'%i)
    cv2.imwrite(output_path, intensity[i, :, :])
