"""
 @Time    : 26.10.22 20:07
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : my_test_e2vid.py
 @Function:
 
"""
import os

# test_txt = '/home/mhy/firenet-pdavis/data/movingcam/test_calculate.txt'
test_txt = '/home/mhy/firenet-pdavis/data/movingcam/test_5s.txt'
# test_txt = '/home/mhy/firenet-pdavis/data/movingcam/test_real2.txt'

with open(test_txt, 'r') as f:
    list = [line.strip() for line in f]

ckpt_path = '/home/mhy/firenet-pdavis/ckpt/E2VID_lightweight.pth.tar'

method_list = ['e2vid_90', 'e2vid_45', 'e2vid_135', 'e2vid_0']

for method in method_list:
    for name in list:
        call_with_args = 'python inference_firenet.py --checkpoint_path {} --height 240 --width 320 --device 0 --events_file_path {} --output_folder /home/mhy/firenet-pdavis/output/{}/{} --e2vid --direction {}'.format(ckpt_path, name, method, name.split('/')[-1].split('_')[0], method.split('_')[-1])
        # call_with_args = 'python inference_firenet.py --checkpoint_path {} --height 130 --width 173 --device 0 --events_file_path {} --output_folder /home/mhy/firenet-pdavis/output_real/{}/{} --firenet_legacy --direction {}'.format(ckpt_path, name, method, name.split('/')[-1].split('.')[0], method.split('_')[-1])
        # call_with_args = 'python inference_firenet.py --checkpoint_path {} --height 130 --width 173 --device 0 --events_file_path {} --output_folder /home/mhy/firenet-pdavis/output_real_fixedt/{}/{} --firenet_legacy --voxel_method t_seconds --t 10000 --sliding_window_t 0'.format(ckpt_path, name, method, name.split('/')[-1].split('.')[0])

        # for calculation
        # call_with_args = 'python inference_firenet.py --checkpoint_path {} --height 240 --width 320 --device 0 --events_file_path {} --output_folder /home/mhy/firenet-pdavis/output_calculation/{}/{} --e2vid --direction {} --calculate_mode'.format(
            # ckpt_path, name, method, name.split('/')[-1].split('_')[0], method.split('_')[-1])

        # for test speed
        # call_with_args = 'python inference_firenet.py --checkpoint_path {} --height 240 --width 320 --device 0 --events_file_path {} --output_folder /home/mhy/firenet-pdavis/output_calculation/{}/{} --e2vid --direction {}'.format(
        #     ckpt_path, name, method, name.split('/')[-1].split('_')[0], method.split('_')[-1])

        print(call_with_args)

        os.system(call_with_args)

call = 'python utils/direction2p.py'
os.system(call)

print('Succeed!')
