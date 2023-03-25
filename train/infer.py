"""
 @Time    : 25.03.23 16:16
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com

 @Project : e2p
 @File    : infer.py
 @Function:

"""
import os

root = os.getcwd()
test_txt = root + '/data/E2PD/test.txt'

method = 'e2p'

ckpt_path = '../{}.pth'.format(method)

with open(test_txt, 'r') as f:
    list = [line.strip() for line in f]

synthetic_list = list[:29]
real_list = list[29:]

for name in synthetic_list:
    call_with_args = 'python inference.py --checkpoint_path {} --height 480 --width 640 --device 0 --events_file_path {} --output_folder ./output_synthetic/{}/{}'.format(
        ckpt_path, name, method, name.split('/')[-1].split('_')[0])

    print(call_with_args)

    os.system(call_with_args)

for name in real_list:
    call_with_args = 'python inference.py --checkpoint_path {} --height 260 --width 346 --device 0 --events_file_path {} --output_folder ./output_real/{}/{}'.format(
        ckpt_path, name, method, name.split('/')[-1].split('.')[0])

    print(call_with_args)

    os.system(call_with_args)

print('Succeed!')
