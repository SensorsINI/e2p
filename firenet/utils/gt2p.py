"""
 @Time    : 17.03.22 16:30
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : gt2p.py
 @Function:
 
"""
import os
import sys
sys.path.append('..')
from misc import check_mkdir
import numpy as np
import cv2
import math
from tqdm import tqdm

rate = 30
# size = (519, 130)
# size = (173, 130)
# size = (1224, 1024)

input_dir = '/home/mhy/firenet-pdavis/data/v2e/gt_ori'
output_dir = '/home/mhy/firenet-pdavis/data/v2e/gt'
check_mkdir(output_dir)

video_name = 'subject09_group2_time4.mp4'
output_frame_dir = os.path.join(output_dir, video_name[:-4])
check_mkdir(output_frame_dir)

capture = cv2.VideoCapture(os.path.join(input_dir, video_name))
# fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# fourcc = cv2.VideoWriter_fourcc(*'divx')
# fourcc = cv2.VideoWriter_fourcc('I', '4', '2', '0')
# fourcc = cv2.VideoWriter_fourcc('P', 'I', 'M', 'I')
# video = cv2.VideoWriter(os.path.join(output_dir, video_name), fourcc, rate, size, isColor=False)

if capture.isOpened():
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print('{} frames.'.format(total))
    # while True:
    for i in tqdm(range(total)):
        ret, img = capture.read()
        if not ret:
            break

        # test
        # output1 = img[:, :, 0] - img[:, :, 1]
        # output2 = img[:, :, 0] - img[:, :, 2]
        # print(output1.shape)
        # print(output2.shape)
        # print(np.max(output1))
        # print(np.min(output1))
        # print(np.max(output2))
        # print(np.min(output2))

        polarization = img[:, :, 0]
        i90 = polarization[0::2, 0::2]
        i45 = polarization[0::2, 1::2]
        i135 = polarization[1::2, 0::2]
        i0 = polarization[1::2, 1::2]

        s0 = i0.astype(float) + i90.astype(float)
        s1 = i0.astype(float) - i90.astype(float)
        s2 = i45.astype(float) - i135.astype(float)

        intensity = (s0 / 2).astype(np.uint8)

        aolp = 0.5 * np.arctan2(s2, s1)
        aolp[s2 < 0] += math.pi
        aolp = (aolp * (255 / math.pi)).astype(np.uint8)

        dolp = np.divide(np.sqrt(np.square(s1) + np.square(s2)), s0, out=np.zeros_like(s0).astype(float), where=s0 != 0)
        dolp = dolp.clip(0.0, 1.0)
        dolp = (dolp * 255).astype(np.uint8)

        # mask operation
        mask = np.where(dolp[:, :] >= 25.5, 255, 0).astype(np.uint8)
        aolp_masked = np.where(mask == 255, aolp, 0).astype(np.uint8)

        output = cv2.hconcat([intensity, aolp_masked, dolp])

        output_path = os.path.join(output_frame_dir, str(i) + '.png')
        cv2.imwrite(output_path, output)

        # print(intensity.shape)
        # print(aolp.shape)
        # print(dolp.shape)
        # print(output.shape)
        # # video.write(output)
        # # video.write(intensity[:, 0:172])
        # video.write(intensity)
        # print(output.shape)
else:
    print('Failed to open video')

cv2.destroyAllWindows()
# video.release()
