"""
 @Time    : 21.09.22 10:45
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : h5gt2s012iad.py
 @Function:
 
"""
import os
import sys
import h5py

sys.path.append('..')
from misc import check_mkdir
import cv2
from tqdm import tqdm

input_root = '/home/mhy/v2e/output/raw_demosaicing_polarization_5s'
output_root = '/home/mhy/firenet-pdavis/output/m_gt'
check_mkdir(output_root)

dir_list = os.listdir(input_root)

for name in tqdm(dir_list):
    input_path = os.path.join(input_root, name, name + '_s012_iad.h5')
    output_dir = os.path.join(output_root, name)
    check_mkdir(output_dir)

    f = h5py.File(input_path, 'r')

    for i in tqdm(range(f['/frame'].shape[0])):
        intensity = f['/intensity'][i, :, :]
        aolp = f['/aolp'][i, :, :]
        dolp = f['/dolp'][i, :, :]

        s0 = f['/s0'][i, :, :]
        s1 = f['/s1'][i, :, :]
        s2 = f['/s2'][i, :, :]

        output_s012 = cv2.hconcat([s0, s1, s2])
        output_iad = cv2.hconcat([intensity, aolp, dolp])
        output = cv2.vconcat([output_s012, output_iad])

        output_path = os.path.join(output_dir, '%05d.png' % i)
        # cv2.imwrite(output_path, output)
        cv2.imwrite(output_path, output_iad)

print('Succeed!')
