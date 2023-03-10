"""
 @Time    : 17.09.22 20:33
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : align3_real.py
 @Function:
 
"""
import os
import sys

sys.path.append('..')
from misc import check_mkdir
from tqdm import tqdm
import cv2

# name = 'Davis346B-2022-09-11T20-33-30+0200-00000000-0-Hallway'
# name = 'UIUC_183_121_196_134_RPM_1000'
name = 'UIUC_RPM_1000'

method1_path = os.path.join('/home/mhy/aedat2pvideo/output_pdavis_rpm0430/firenet_direction_iad/{}.avi'.format(name))
method2_path = os.path.join('/home/mhy/firenet-pdavis/output_pdavis/m88_new_pae_2_5ms_rpm0430/{}.avi').format(name)
method3_path = os.path.join('/home/mhy/firenet-pdavis/output_pdavis/v3_1/{}.avi').format(name)
compare_dir = '/home/mhy/firenet-pdavis/output_pdavis/compare_real_firenet_m88_v3_1'
check_mkdir(compare_dir)

print(name)

rate = 25
size = (519, 390)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter(
    os.path.join(compare_dir, name + '_firenet_m88_v3_1.avi'), fourcc,
    rate, size, isColor=True)

cap_method1 = cv2.VideoCapture(method1_path)
cap_method2 = cv2.VideoCapture(method2_path)
cap_method3 = cv2.VideoCapture(method3_path)

length_method1 = int(cap_method1.get(cv2.CAP_PROP_FRAME_COUNT))
length_method2 = int(cap_method2.get(cv2.CAP_PROP_FRAME_COUNT))
length_method3 = int(cap_method3.get(cv2.CAP_PROP_FRAME_COUNT))
if length_method1 != length_method2 or length_method1 != length_method3:
    # if length_method1 != length_method2 or length_method1 != length_method3:
    print('Inconsistency in frame numbers!')
    print(length_method1, length_method2, length_method3)
    exit(0)
else:
    print('Method has %d frames.' % length_method1)

for i in tqdm(range(length_method1)):
    cap_method1.set(cv2.CAP_PROP_POS_FRAMES, i)
    cap_method2.set(cv2.CAP_PROP_POS_FRAMES, i)
    cap_method3.set(cv2.CAP_PROP_POS_FRAMES, i)

    _, frame_method1 = cap_method1.read()
    _, frame_method2 = cap_method2.read()
    _, frame_method3 = cap_method3.read()

    frame = cv2.vconcat([frame_method1, frame_method2, frame_method3])

    video.write(frame)

cap_method1.release()
cap_method2.release()
cap_method3.release()

print('Done!')
