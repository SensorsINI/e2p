"""
 @Time    : 25.04.22 09:41
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : h5gt2iad.py
 @Function:
 
"""
import os
import h5py
import cv2
from tqdm import tqdm
from util import append_timestamp

def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

# input_root = '/home/mhy/firenet-pdavis/data/raw_demosaicing_polarization_5s_iad'
# output_root = '/home/mhy/firenet-pdavis/output/v_test_gt'
# check_mkdir(output_root)
input_root = 'train/data/E2PD/h5new/Davis346B-2023-03-16T15-42-43+0100-00000000-0-pdavis-polfilter-tobi-office-window3.h5'
output_root = 'output_test/test_gt'
check_mkdir(output_root)

# test txt
# test_list_txt = '/home/mhy/firenet-pdavis/data/movingcam/test_5s.txt'
test_list_txt = 'data/movingcam/test_real2.txt'
with open(test_list_txt, 'r') as f:
    test_list = [line.strip() for line in f]

# dir_list = os.listdir(input_root)
dir_list = test_list

for name in tqdm(dir_list):
    # name = name.split('/')[-1].split('_')[0]
    name = name.split('/')[-1].split('.')[0]
    # input_path = os.path.join(input_root, name, name + '_iad.h5')
    input_path = os.path.join(input_root, name + '.h5')
    output_dir = os.path.join(output_root, name)
    check_mkdir(output_dir)

    f = h5py.File(input_path, 'r')

    for i in tqdm(range(f['/frame'].shape[0])):
        intensity = f['/intensity'][i, :, :]
        aolp = f['/aolp'][i, :, :]
        dolp = f['/dolp'][i, :, :]

        output = cv2.hconcat([intensity, aolp, dolp])

        output_path = os.path.join(output_dir, '%05d.png' % i)
        cv2.imwrite(output_path, output)
        # cv2.imwrite(output_path, aolp)
        # cv2.imwrite(output_path, dolp)

        ts_fname = os.path.join(output_dir, 'timestamps.txt')
        append_timestamp(ts_fname, '%05d.png' % i, f['/frame_ts'][i])

print('Succeed!')
