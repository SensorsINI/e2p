"""
 @Time    : 13.10.22 15:28
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : split_visual3.py
 @Function:
 
"""
import os
import numpy as np
import sys
sys.path.append('..')
from misc import check_mkdir
from tqdm import tqdm
import cv2

# for simulated data
# method1 = 'v_test_gt'
# method2 = 'firenet_direction_iad'
# method3 = 'v16_mix2'
#
# input_dir = '/home/mhy/firenet-pdavis/visual3_v16_mix2'
# output_dir = '/home/mhy/firenet-pdavis/visual3_v16_mix2_separate'
# check_mkdir(output_dir)

# for real data
method1 = 'test_gt'
method2 = 'firenet_direction_iad'
method3 = 'v16_mix2'

# input_dir = '/home/mhy/firenet-pdavis/output_real/visual3_real_v16_mix2'
# output_dir = '/home/mhy/firenet-pdavis/output_real/visual3_real_v16_mix2_separate'
# input_dir = '/home/mhy/firenet-pdavis/output_real/supp_real_v16_mix2'
# output_dir = '/home/mhy/firenet-pdavis/output_real/supp_real_v16_mix2_separate'
input_dir = '/home/mhy/firenet-pdavis/output_real/rebuttal_real_v16_mix2'
output_dir = '/home/mhy/firenet-pdavis/output_real/rebuttal_real_v16_mix2_separate'
# input_dir = '/home/mhy/firenet-pdavis/failure_case'
# output_dir = '/home/mhy/firenet-pdavis/failure_case_separate'
check_mkdir(output_dir)

list = sorted(os.listdir(input_dir))
for name in tqdm(list):
    input_path = os.path.join(input_dir, name)
    input = cv2.imread(input_path)

    intensity, aolp, dolp = np.hsplit(input, 3)
    intensity1, intensity2, intensity3 = np.vsplit(intensity, 3)
    aolp1, aolp2, aolp3 = np.vsplit(aolp, 3)
    dolp1, dolp2, dolp3 = np.vsplit(dolp, 3)

    intensity1_path = os.path.join(output_dir, name[:-4] + '_i_' + method1 + name[-4:])
    intensity2_path = os.path.join(output_dir, name[:-4] + '_i_' + method2 + name[-4:])
    intensity3_path = os.path.join(output_dir, name[:-4] + '_i_' + method3 + name[-4:])

    aolp1_path = os.path.join(output_dir, name[:-4] + '_a_' + method1 + name[-4:])
    aolp2_path = os.path.join(output_dir, name[:-4] + '_a_' + method2 + name[-4:])
    aolp3_path = os.path.join(output_dir, name[:-4] + '_a_' + method3 + name[-4:])

    dolp1_path = os.path.join(output_dir, name[:-4] + '_d_' + method1 + name[-4:])
    dolp2_path = os.path.join(output_dir, name[:-4] + '_d_' + method2 + name[-4:])
    dolp3_path = os.path.join(output_dir, name[:-4] + '_d_' + method3 + name[-4:])
    
    cv2.imwrite(intensity1_path, intensity1)
    cv2.imwrite(intensity2_path, intensity2)
    cv2.imwrite(intensity3_path, intensity3)
    cv2.imwrite(aolp1_path, aolp1)
    cv2.imwrite(aolp2_path, aolp2)
    cv2.imwrite(aolp3_path, aolp3)
    cv2.imwrite(dolp1_path, dolp1)
    cv2.imwrite(dolp2_path, dolp2)
    cv2.imwrite(dolp3_path, dolp3)

print('Succeed!')
