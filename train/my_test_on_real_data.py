"""
 @Time    : 15.09.22 16:54
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : my_test_on_real_data.py
 @Function:
 
"""
import os

# test_txt = '/home/mhy/aedat2pvideo/data/rpm0430_1000.txt'
# test_txt = '/home/mhy/aedat2pvideo/data/my_hallway.txt'
# test_txt = '/home/mhy/aedat2pvideo/data/my_desktop.txt'
# test_root = '/home/mhy/aedat2pvideo/aedat/1610pm'
test_root = '/home/mhy/aedat2pvideo/aedat_h5_fixed_duration_pre'

# method = 'm88_new_pae_2'
# date = '0910_193652'
method = 'v11_s'
date = '1011_224058'
ckpt_path = '/home/mhy/firenet-pdavis/ckpt/models/{}/{}/model_best.pth'.format(method, date)

# with open(test_txt, 'r') as f:
#     list = [line.strip() for line in f]

list = sorted([os.path.join(test_root, x) for x in os.listdir(test_root) if x.endswith('.h5')])
print(list)

for name in list:

    call_with_args = 'python inference.py --checkpoint_path {} --height 260 --width 346 --device 0 --events_file_path {} --output_folder ./output_pdavis/{}/{} --voxel_method t_seconds --t 10000 --sliding_window_t 0 --real_data --legacy_norm'.format(ckpt_path, name, method, name.split('/')[-1].split('.')[0])
    # call_with_args = 'python inference.py --checkpoint_path {} --height 260 --width 346 --device 0 --events_file_path {} --output_folder ./output_pdavis/{}/{} --voxel_method k_events --k 20000 --sliding_window_w 0 --real_data --legacy_norm'.format(ckpt_path, name, method, name.split('/')[-1].split('.')[0])
    # call_with_args = 'python inference.py --checkpoint_path {} --height 260 --width 346 --device 0 --events_file_path {} --output_folder ./output_pdavis/{}/{} --voxel_method t_seconds --t 5000 --sliding_window_t 0 --real_data'.format(ckpt_path, name, method, name.split('/')[-1].split('.')[0])
    # call_with_args = 'python inference.py --checkpoint_path {} --height 260 --width 346 --device 0 --events_file_path {} --output_folder ./output_pdavis/{}/{} --voxel_method t_seconds --t 5000 --sliding_window_t 0 --real_data'.format(ckpt_path, name, method, name.split('/')[-1].split('.')[0])

    print(call_with_args)

    os.system(call_with_args)

print('Succeed!')
