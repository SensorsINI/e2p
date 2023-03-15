"""
 @Time    : 17.10.22 16:23
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : my_video2frame.py
 @Function:
 
"""
import os
import sys
sys.path.append('..')
from misc import check_mkdir
import cv2

dir = 'data/new/real'
video_list = sorted([x for x in os.listdir(dir) if x.endswith('.avi')])

for video_name in video_list:
    video_path = os.path.join(dir, video_name)
    output_dir = os.path.join(dir, video_name.split('.')[0])
    check_mkdir(output_dir)

    cap = cv2.VideoCapture(video_path)

    success, image = cap.read()
    count = 0
    while success:
        cv2.imwrite(os.path.join(output_dir, "%04d.png" % count), image)
        success, image = cap.read()
        count += 1

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(video_name)
    print(length)
    print(count)

print('Nice!')
