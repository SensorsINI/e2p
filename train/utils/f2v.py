"""
 @Time    : 2/16/22 00:10
 @Author  : Haiyang Mei
 @E-mail  : mhy666@mail.dlut.edu.cn
 
 @Project : firenet-pdavis
 @File    : f2v.py
 @Function:
 
"""
import os
import cv2
from tqdm import tqdm

root = '/home/iccd/firenet-pdavis/output/FireNet/demo_f42p'

dir_name = 'aolp'
rate = 10
size = (173, 130)

image_dir = os.path.join(root, dir_name)

image_list = [x for x in os.listdir(image_dir) if x.endswith('.png')]
image_list.sort()

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
video = cv2.VideoWriter(os.path.join(root, dir_name + '.mp4'), fourcc, rate, size)

for image in tqdm(image_list):
    video.write(cv2.imread(os.path.join(image_dir, image)))

cv2.destroyAllWindows()
video.release()

print('Succeed!')