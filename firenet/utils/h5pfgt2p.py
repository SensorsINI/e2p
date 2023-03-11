"""
 @Time    : 23.05.22 18:04
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : h5pfgt2p.py
 @Function:
 
"""
import os
import sys
import h5py
sys.path.append('..')
from misc import check_mkdir
import numpy as np
import cv2
import math
from tqdm import tqdm

root_dir = '/home/mhy/v2e/output/raw_demosaicing_polarization'
list = [
    '00702',
    '00704',
    '00705',
    '01402',
    '01404',
    '01405',
]

for name in list:
    print(name)

    input_path = os.path.join(root_dir, name, name + '_pf.h5')

    output_dir = os.path.join(root_dir, name, name + '_pf_iadgt')
    check_mkdir(output_dir)

    f = h5py.File(input_path, 'r')

    for i in tqdm(range(f['/frame'].shape[0])):
        intensity = f['/intensity'][i, :, :]
        aolp = f['/aolp'][i, :, :]
        dolp = f['/dolp'][i, :, :]

        iad = cv2.hconcat([intensity, aolp, dolp])

        output_path = os.path.join(output_dir, '%05d.png'%i)
        cv2.imwrite(output_path, iad)

print('Succeed!')
