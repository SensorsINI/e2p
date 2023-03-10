"""
 @Time    : 21.10.22 16:06
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : my_test_firenet.py
 @Function:
 
"""
import os

# test_txt = '/home/mhy/firenet-pdavis/data/movingcam/test_calculate.txt'
# test_txt = '/home/mhy/firenet-pdavis/data/movingcam/test_5s.txt'
test_txt = '/home/mhy/firenet-pdavis/data/movingcam/test_real2.txt'

with open(test_txt, 'r') as f:
    list = [line.strip() for line in f]

ckpt_path = '/home/mhy/firenet-pdavis/ckpt/firenet_1000.pth.tar'

method_list = ['firenet_90', 'firenet_45', 'firenet_135', 'firenet_0']

for method in method_list:
    for name in list:
        # call_with_args = 'python inference_firenet.py --checkpoint_path {} --height 240 --width 320 --device 0 --events_file_path {} --output_folder /home/mhy/firenet-pdavis/output/{}/{} --firenet_legacy --direction {}'.format(ckpt_path, name, method, name.split('/')[-1].split('_')[0], method.split('_')[-1])
        # call_with_args = 'python inference_firenet.py --checkpoint_path {} --height 130 --width 173 --device 0 --events_file_path {} --output_folder /home/mhy/firenet-pdavis/output_real/{}/{} --firenet_legacy --direction {}'.format(ckpt_path, name, method, name.split('/')[-1].split('.')[0], method.split('_')[-1])
        call_with_args = 'python inference_firenet.py --checkpoint_path {} --height 260 --width 346 --device 0 --events_file_path {} --output_folder /home/mhy/firenet-pdavis/output_real_10ms/{}/{} --firenet_legacy --direction {} --voxel_method t_seconds --t 10000 --sliding_window_t 0'.format(ckpt_path, name, method, name.split('/')[-1].split('.')[0], method.split('_')[-1])

        # for calculation
        # call_with_args = 'python inference_firenet.py --checkpoint_path {} --height 240 --width 320 --device 0 --events_file_path {} --output_folder /home/mhy/firenet-pdavis/output_calculation/{}/{} --firenet_legacy --direction {} --calculate_mode'.format(ckpt_path, name, method, name.split('/')[-1].split('_')[0], method.split('_')[-1])

        # for test speed
        # call_with_args = 'python inference_firenet.py --checkpoint_path {} --height 240 --width 320 --device 0 --events_file_path {} --output_folder /home/mhy/firenet-pdavis/output_calculation/{}/{} --firenet_legacy --direction {}'.format(ckpt_path, name, method, name.split('/')[-1].split('_')[0], method.split('_')[-1])

        print(call_with_args)

        os.system(call_with_args)

call = 'python utils/direction2p.py'
os.system(call)

print('Succeed!')
