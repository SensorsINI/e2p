"""
 @Time    : 06.07.22 18:07
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : edge_extraction.py
 @Function:
 
"""
import os
import cv2
import numpy as np
from tqdm import tqdm

input_dir = '/home/mhy/firenet-pdavis/output/m_gt/00702'
output_dir = '/home/mhy/firenet-pdavis/output/m_gt/00702_edge'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

image_list = os.listdir(input_dir)
for image_name in tqdm(image_list):
    image_path = os.path.join(input_dir, image_name)
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    intensity, aolp, dolp = np.hsplit(image, 3)

    intensity_edge = cv2.Laplacian(intensity, ddepth=cv2.CV_64F)
    aolp_edge = cv2.Laplacian(aolp, ddepth=cv2.CV_64F, ksize=3)
    dolp_edge = cv2.Laplacian(dolp, ddepth=cv2.CV_64F, ksize=3)

    intensity_edge = (intensity_edge).astype(np.uint8)
    aolp_edge = (aolp_edge).astype(np.uint8)
    dolp_edge = (dolp_edge).astype(np.uint8)

    edge = cv2.hconcat([intensity_edge, aolp_edge, dolp_edge])

    output_path = os.path.join(output_dir, image_name)
    cv2.imwrite(output_path, edge)

print('Succeed!')
