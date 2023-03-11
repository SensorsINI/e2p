"""
 @Time    : 29.08.22 20:18
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : split_iad.py
 @Function:
 
"""
import os
import cv2
import numpy as np

root = '/home/mhy/SinGAN/Input/Images'
iad_path = os.path.join(root, '00033.png')
intensity_path = os.path.join(root, 'intensity.png')
aolp_path = os.path.join(root, 'aolp.png')
dolp_path = os.path.join(root, 'dolp.png')

concat = cv2.imread(iad_path, cv2.IMREAD_GRAYSCALE)
intensity, aolp, dolp = np.hsplit(concat, 3)

cv2.imwrite(intensity_path, intensity)
cv2.imwrite(aolp_path, aolp)
cv2.imwrite(dolp_path, dolp)

print('Succeed!')
