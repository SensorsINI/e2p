"""
 @Time    : 21.10.22 16:06
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : my_test_firenet.py
 @Function:
 
"""
import os

root=os.getcwd()
test_txt = root+'/data/E2PD/test.txt'

with open(test_txt, 'r') as f:
    list = [line.strip() for line in f]

ckpt_path = './ckpt/firenet_1000.pth.tar'

method_list = ['firenet_90', 'firenet_45', 'firenet_135', 'firenet_0']

for method in method_list:
    for name in synthetic_list:
        call_with_args = 'python inference_firenet.py --checkpoint_path {} --height 480 --width 640 --device 0 --events_file_path {} --output_folder ./output_synthetic/{}/{} --firenet_legacy --direction {}'.format(ckpt_path, name, method, name.split('/')[-1].split('_')[0], method.split('_')[-1])

        print(call_with_args)

        os.system(call_with_args)

    for name in real_list:
        call_with_args = 'python inference_firenet.py --checkpoint_path {} --height 260 --width 346 --device 0 --events_file_path {} --output_folder ./output_real/{}/{} --firenet_legacy --direction {}'.format(
            ckpt_path, name, method, name.split('/')[-1].split('.')[0], method.split('_')[-1])

        print(call_with_args)

        os.system(call_with_args)

call_synthetic = 'python utils/direction2p_synthetic.py'
call_real = 'python utils/direction2p_real.py'
os.system(call_synthetic)
os.system(call_real)

print('Succeed!')