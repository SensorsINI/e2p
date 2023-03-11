"""
 @Time    : 04.04.22 09:50
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : h5gt2p.py
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

input_root = '/home/mhy/v2e/output/raw_demosaicing_polarization'
output_root = '/home/mhy/firenet-pdavis/output/m_gt'
check_mkdir(output_root)

# dir_list = os.listdir(input_root)
dir_list = [
        '00702',
        '00704',
        '00705',
        '01402',
        '01404',
        '01405'
]

for name in tqdm(dir_list):
        input_path = os.path.join(input_root, name, name + '.h5')
        output_dir = os.path.join(output_root, name)
        check_mkdir(output_dir)

        f = h5py.File(input_path, 'r')

        for i in tqdm(range(f['/frame'].shape[0])):
                polarization = f['/frame'][i, :, :]
                i90 = polarization[0::2, 0::2]
                i45 = polarization[0::2, 1::2]
                i135 = polarization[1::2, 0::2]
                i0 = polarization[1::2, 1::2]

                s0 = i0.astype(float) + i90.astype(float)
                s1 = i0.astype(float) - i90.astype(float)
                s2 = i45.astype(float) - i135.astype(float)

                intensity = (s0 / 2).astype(np.uint8)

                aolp = 0.5 * np.arctan2(s2, s1)
                aolp = aolp + 0.5 * math.pi
                aolp = (aolp * (255 / math.pi)).astype(np.uint8)

                dolp = np.divide(np.sqrt(np.square(s1) + np.square(s2)), s0, out=np.zeros_like(s0).astype(float), where=s0 != 0)
                dolp = dolp.clip(0.0, 1.0)
                dolp = (dolp * 255).astype(np.uint8)

                output = cv2.hconcat([intensity, aolp, dolp])

                output_path = os.path.join(output_dir, '%05d.png'%i)
                cv2.imwrite(output_path, output)

print('Succeed!')
