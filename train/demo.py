"""
 @Time    : 20.07.22 16:59
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : demo.py
 @Function:
 
"""
import os

test_txt = '/home/mhy/aedat2pvideo/data/driving_polarization.txt'

# for my own method
method = 'm27_new'
date = '0716_184945'
ckpt_path = '/home/mhy/firenet-pdavis/ckpt/models/{}/{}/model_best.pth'.format(method, date)

list = []

with open(test_txt) as f:
    line = f.readline()
    if line[:-1].endswith('.h5'):
        list.append(line[:-1])
    while line:
        line = f.readline()
        if line[:-1].endswith('.h5'):
            list.append(line[:-1])

for name in list:
    call_with_args = 'python inference.py --checkpoint_path {} --height 260 --width 346 --device 0 --events_file_path {} --output_folder ./output_pdavis/{}/{} --legacy_norm --voxel_method t_seconds --t 5000 --sliding_window_t 0 '.format(ckpt_path, name, method, name.split('/')[-1].split('.')[0])

    print(call_with_args)

    os.system(call_with_args)

print('Succeed!')

