"""
 @Time    : 23.05.22 16:16
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : my_test.py
 @Function:

"""
import os

root=os.getcwd()
# test_txt = '/home/mhy/firenet-pdavis/data/movingcam/test_calculate.txt'
# test_txt = '/home/mhy/firenet-pdavis/data/movingcam/test_5s.txt'
# test_txt = '/home/mhy/firenet-pdavis/data/movingcam/test_5s_attention.txt'
test_txt = root+'/data/E2PD/test.txt'

# for original firenet
# method = 'firenet_0'
# ckpt_path = '/home/mhy/firenet-pdavis/ckpt/firenet_1000.pth.tar'

# for original e2vid
# method = 'e2vid_0'
# ckpt_path = '/home/mhy/firenet-pdavis/ckpt/E2VID_lightweight.pth.tar'

# for my own method
# method = 'v16_mix2'
# date = '1021_001330'
# method = 'v16_so_1'
# date = '0127_172349'
method = 'e2p'
# date = '0127_172349'
# method = 'v16_b_c16_2'
# date = '1030_220415'
# method = 'v16_b_2'
# date = '1030_215908'
# method = 'v16_br'
# date = '1027_210855'
# method = 'v16_br_2'
# date = '1031_231603'
# method = 'v16_b_3'
# date = '1031_232205'
# method = 'v16_b_k4_3'
# date = '1103_184322'
# method = 'v16_b_c16_s'
# date = '1031_201238'
# method = 'v16_b_c16_i'
# date = '1104_192744'
# ckpt_path = root+'/ckpt/models/{}/{}/model_best.pth'.format(method, date)
ckpt_path = '../{}.pth'.format(method)
# ckpt_path = '/home/mhy/firenet-pdavis/ckpt/models/{}/{}/checkpoint-epoch30.pth'.format(method, date)

# for finetuned model
# method = 'v16'
# date = '1017_230949'
# ckpt_path = '/home/mhy/firenet-pdavis/ckpt/models/{}/{}/checkpoint-epoch60.pth'.format(method, date)

with open(test_txt, 'r') as f:
    list = [line.strip() for line in f]

synthetic_list = list[:29]
real_list = list[29:]

for name in list:

    # for directly test the original firenet model
    # call_with_args = 'python my_inference.py --checkpoint_path {} --height 480 --width 640 --device 0 --events_file_path {} --output_folder /home/mhy/firenet-pdavis/output/{}/{} --firenet_legacy'.format(ckpt_path, name, method, name.split('/')[-1].split('_')[0])
    # call_with_args = 'python my_inference.py --checkpoint_path {} --height 240 --width 320 --device 0 --events_file_path {} --output_folder /home/mhy/firenet-pdavis/output/{}/{} --firenet_legacy'.format(ckpt_path, name, method, name.split('/')[-1].split('_')[0])
    # call_with_args = 'python my_inference.py --checkpoint_path {} --height 130 --width 173 --device 0 --events_file_path {} --output_folder /home/mhy/firenet-pdavis/output/{}/{} --firenet_legacy --voxel_method t_seconds --t 10000 --sliding_window_t 0'.format(ckpt_path, name, method, name.split('/')[-1].split('.')[0])

    # for directly test the original e2vid model
    # call_with_args = 'python my_inference.py --checkpoint_path {} --height 240 --width 320 --device 0 --events_file_path {} --output_folder /home/mhy/firenet-pdavis/output/{}/{} --e2vid'.format(ckpt_path, name, method, name.split('/')[-1].split('_')[0])

    # for test my own model
    # five input channels
    ############# norm
    # call_with_args = 'python my_inference.py --checkpoint_path {} --height 480 --width 640 --device 0 --events_file_path {} --output_folder /home/mhy/firenet-pdavis/output/{}/{} --legacy_norm'.format(ckpt_path, name, method, name.split('/')[-1].split('_')[0])
    # call_with_args = 'python my_inference.py --checkpoint_path {} --height 260 --width 346 --device 0 --events_file_path {} --output_folder /home/mhy/firenet-pdavis/output_real/{}/{} --legacy_norm'.format(ckpt_path, name, method, name.split('/')[-1].split('.')[0])
    # call_with_args = 'python my_inference.py --checkpoint_path {} --height 480 --width 640 --device 0 --events_file_path {} --output_folder /home/mhy/firenet-pdavis/output/{}/{} --robust_norm'.format(ckpt_path, name, method, name.split('/')[-1].split('_')[0])
    ############ without norm
    # call_with_args = 'python my_inference.py --checkpoint_path {} --height 480 --width 640 --device 0 --events_file_path {} --output_folder /home/mhy/firenet-pdavis/output/{}/{}'.format(ckpt_path, name, method, name.split('/')[-1].split('_')[0])
    call_with_args = 'python my_inference.py --checkpoint_path {} --height 260 --width 346 --device 0 --events_file_path {} --output_folder ./output_real/{}/{}'.format(ckpt_path, name, method, name.split('/')[-1].split('.')[0])
    ########### without norm for attention visualization
    # call_with_args = 'python inference_v.py --checkpoint_path {} --height 480 --width 640 --device 0 --events_file_path {} --output_folder /home/mhy/firenet-pdavis/output_attention/{}/{}'.format(ckpt_path, name, method, name.split('/')[-1].split('_')[0])
    ########### without norm for ablation study
    # call_with_args = 'python my_inference.py --checkpoint_path {} --height 480 --width 640 --device 0 --events_file_path {} --output_folder /home/mhy/firenet-pdavis/output_ablation/{}/{}'.format(ckpt_path, name, method, name.split('/')[-1].split('_')[0])
    ########## without norm for ablation study predict s
    # call_with_args = 'python inference_s.py --checkpoint_path {} --height 480 --width 640 --device 0 --events_file_path {} --output_folder /home/mhy/firenet-pdavis/output_ablation/{}/{}'.format(ckpt_path, name, method, name.split('/')[-1].split('_')[0])
    ########## without norm for ablation study predict i
    # call_with_args = 'python inference_i.py --checkpoint_path {} --height 480 --width 640 --device 0 --events_file_path {} --output_folder /home/mhy/firenet-pdavis/output_ablation/{}/{}'.format(ckpt_path, name, method, name.split('/')[-1].split('_')[0])
    # without norm + calculation mode
    # call_with_args = 'python my_inference.py --checkpoint_path {} --height 480 --width 640 --device 0 --events_file_path {} --output_folder /home/mhy/firenet-pdavis/output_calculation/{}/{} --calculate_mode'.format(ckpt_path, name, method, name.split('/')[-1].split('_')[0])
    # without norm + test speed
    # call_with_args = 'python my_inference.py --checkpoint_path {} --height 480 --width 640 --device 0 --events_file_path {} --output_folder /home/mhy/firenet-pdavis/output_calculation/{}/{}'.format(ckpt_path, name, method, name.split('/')[-1].split('_')[0])
    # without norm + fixed time duration
    # call_with_args = 'python my_inference.py --checkpoint_path {} --height 260 --width 346 --device 0 --events_file_path {} --output_folder ./output_real_33ms/{}/{} --voxel_method t_seconds --t 33000 --sliding_window_t 0'.format(ckpt_path, name, method, name.split('/')[-1].split('.')[0])
    # call_with_args = 'python my_inference.py --checkpoint_path {} --height 260 --width 346 --device 0 --events_file_path {} --output_folder ./output_real_25ms/{}/{} --voxel_method t_seconds --t 25000 --sliding_window_t 0'.format(ckpt_path, name, method, name.split('/')[-1].split('.')[0])
    # call_with_args = 'python my_inference.py --checkpoint_path {} --height 260 --width 346 --device 0 --events_file_path {} --output_folder ./output_real_10ms/{}/{} --voxel_method t_seconds --t 10000 --sliding_window_t 0'.format(ckpt_path, name, method, name.split('/')[-1].split('.')[0])
    # call_with_args = 'python my_inference.py --checkpoint_path {} --height 260 --width 346 --device 0 --events_file_path {} --output_folder ./output_real_5ms/{}/{} --voxel_method t_seconds --t 5000 --sliding_window_t 0'.format(ckpt_path, name, method, name.split('/')[-1].split('.')[0])
    # ten input channels
    # call_with_args = 'python my_inference.py --checkpoint_path {} --height 480 --width 640 --device 0 --events_file_path {} --output_folder /home/mhy/firenet-pdavis/output/{}/{} --update'.format(ckpt_path, name, method, name.split('/')[-1].split('.')[0])
    # five input channels + davis346
    # call_with_args = 'python my_inference.py --checkpoint_path {} --height 260 --width 346 --device 0 --events_file_path {} --output_folder /home/mhy/firenet-pdavis/output/{}/{} --legacy_norm'.format(ckpt_path, name, method, name.split('/')[-1].split('_')[0])

    print(call_with_args)

    os.system(call_with_args)

print('Succeed!')
