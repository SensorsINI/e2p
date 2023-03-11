"""
 @Time    : 17.12.22 11:33
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : pdavis_demo
 @File    : consumer.py
 @Function:
 
"""

import argparse
import glob
import pickle
import cv2
import sys
import serial
import socket
from select import select

from thop import profile, clever_format

from globals_and_utils import *
from engineering_notation import EngNumber as eng # only from pip
import collections
from pathlib import Path
import random

# for network inference
import torch
from utils.util import torch2cv2, torch2numpy

log=get_logger()

# Only used in mac osx
try:
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
except Exception as e:
    print(e)

# from inference.py
import e2p as model_arch
from utils.henri_compatible import make_henri_compatible
from parse_config import ConfigParser

model_info = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def print_key_help():
    print('producer keys to use in cv2 image window:\n'
          'h or ?: print this help\n'
          'p: print timing info\n'
          'space: toggle or turn on while space down recording\n'
          'm: toggle between e2p and firenet models\n'
          'q or x: exit')


def legacy_compatibility(args, checkpoint):
    assert args.e2p and not (args.e2vid and args.firenet_legacy)
    if args.e2vid:
        args.legacy_norm = True
        final_activation = 'sigmoid'
    elif args.firenet_legacy:
        args.legacy_norm = True
        final_activation = ''
    else:
        return args, checkpoint
    # Make compatible with Henri saved models
    if not isinstance(checkpoint.get('config', None), ConfigParser) or args.e2vid or args.firenet_legacy:
        checkpoint = make_henri_compatible(checkpoint, final_activation)
    if args.firenet_legacy:
        checkpoint['config']['arch']['type'] = 'FireNet_legacy'
    return args, checkpoint


def load_model(checkpoint):
    config = checkpoint['config']
    log.info(f'configuration is {config["arch"]}')
    state_dict = checkpoint['state_dict']

    try:
        model_info['num_bins'] = config['arch']['args']['unet_kwargs']['num_bins']
        log.info(f'number of voxel bins (frames sent at each timestamp to model) is {model_info["num_bins"]}')
    except KeyError:
        model_info['num_bins'] = config['arch']['args']['num_bins']
    # logger = config.get_logger('test')

    # build model architecture
    model = config.init_obj('arch', model_arch)
    log.info(model)
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)
    log.info('Load my trained weights succeeded!')

    model = model.to(device)
    model.eval()
    # if args.color:
    #     model = ColorNet(model)
    for param in model.parameters():
        param.requires_grad = False

    return model

def compute_firenet_output(output):
    # original firenet
    # image = crop.crop(output['image'])
    image = torch2cv2(output['image'])
    # output raw
    # image = crop.crop(output['image'])
    image = torch2numpy(image)
    image = np.clip(image, 0, 1)
    #
    i90 = image[0::2, 0::2]
    i45 = image[0::2, 1::2]
    i135 = image[1::2, 0::2]
    i0 = image[1::2, 1::2]
    #
    s0 = i0.astype(float) + i90.astype(float)
    s1 = i0.astype(float) - i90.astype(float)
    s2 = i45.astype(float) - i135.astype(float)
    #

    # output stocks parameters
    # s0 = crop.crop(output['s0']) * 2
    # s1 = crop.crop(output['s1']) * 2 - 1
    # s2 = crop.crop(output['s2']) * 2 - 1
    #
    # s0 = torch2numpy(s0)
    # s1 = torch2numpy(s1)
    # s2 = torch2numpy(s2)
    #
    # intensity = s0 / 2
    # intensity = numpy2cv2(intensity)
    #
    aolp = 0.5 * np.arctan2(s2, s1)
    aolp = aolp + 0.5 * math.pi
    aolp = aolp / math.pi
    aolp = numpy2cv2(aolp)
    #
    # dolp = np.divide(np.sqrt(np.square(s1) + np.square(s2)), s0, out=np.zeros_like(s0).astype(float), where=s0 != 0)
    # dolp = numpy2cv2(dolp)

    # output polarization with crop
    # intensity = crop.crop(output['i'])
    # intensity = torch2cv2(intensity)
    # aolp = crop.crop(output['a'])
    # aolp = torch2cv2(aolp)
    # dolp = crop.crop(output['d'])
    # dolp = torch2cv2(dolp)

    # output polarization without crop
    # output['i'] = minmax_normalization(output['i'], output['i'].device)
    intensity = torch2cv2(output['i'])
    aolp = torch2cv2(output['a'])
    dolp = torch2cv2(output['d'])
    # intensity_f = torch2cv2(output['i_f'])
    # aolp_f = torch2cv2(output['a_f'])
    # dolp_f = torch2cv2(output['d_f'])

    # intensity = torch2cv2(output['i90'])
    # aolp = torch2cv2(output['i45'])
    # dolp = torch2cv2(output['i135'])

    # for dct output
    # intensity = crop.crop(output['i_f'])
    # intensity = torch2cv2(intensity)
    # aolp = crop.crop(output['a_f'])
    # aolp = torch2cv2(aolp)
    # dolp = crop.crop(output['d_f'])
    # dolp = torch2cv2(dolp)

    # new representation
    # aolp = output['image']
    # aolp = output['a']
    # aolp = torch2numpy(aolp)
    # aolp = np.clip(aolp, 0, 1)
    # aolp = aolp - 0.5
    # aolp = np.where(aolp < 0, aolp+1, aolp)
    # aolp = numpy2cv2(aolp)
    # intensity = aolp
    # dolp = aolp

    # dolp = crop.crop(output['image'])
    # dolp = crop.crop(output['d'])
    # dolp = torch2cv2(dolp)
    # intensity = dolp
    # aolp = dolp

    # output stocks parameters
    # s0 = crop.crop(output['s0'])
    # s0 = torch2numpy(s0)
    # s0 = np.clip(s0, 0, 1)
    # s1 = crop.crop(output['s1'])
    # s1 = torch2numpy(s1)
    # s1 = np.clip(s1, 0, 1)
    # s1 = s1 * 2 - 1
    # s2 = crop.crop(output['s2'])
    # s2 = torch2numpy(s2)
    # s2 = np.clip(s2, 0, 1)
    # s2 = s2 * 2 - 1
    #
    # intensity = numpy2cv2(s0)
    #
    # aolp = 0.5 * np.arctan2(s2, s1)
    # aolp = aolp + 0.5 * math.pi
    # aolp = aolp / math.pi
    # aolp = numpy2cv2(aolp)
    #
    # dolp = np.divide(np.sqrt(np.square(s1) + np.square(s2)), s0, out=np.zeros_like(s0).astype(float), where=s0 != 0)
    # dolp = numpy2cv2(dolp)

    # output four-direction intensity
    # i90 = crop.crop(output['i_90'])
    # i90 = torch2numpy(i90)
    # i90 = np.clip(i90, 0, 1)
    # i45 = crop.crop(output['i_45'])
    # i45 = torch2numpy(i45)
    # i45 = np.clip(i45, 0, 1)
    # i135 = crop.crop(output['i_135'])
    # i135 = torch2numpy(i135)
    # i135 = np.clip(i135, 0, 1)
    # i0 = crop.crop(output['i_0'])
    # i0 = torch2numpy(i0)
    # i0 = np.clip(i0, 0, 1)
    #
    # s0 = i0.astype(float) + i90.astype(float)
    # s1 = i0.astype(float) - i90.astype(float)
    # s2 = i45.astype(float) - i135.astype(float)
    #
    # intensity = s0 / 2
    # intensity = numpy2cv2(intensity)
    #
    # aolp = 0.5 * np.arctan2(s2, s1)
    # aolp = aolp + 0.5 * math.pi
    # aolp = aolp / math.pi
    # aolp = numpy2cv2(aolp)
    #
    # dolp = np.divide(np.sqrt(np.square(s1) + np.square(s2)), s0, out=np.zeros_like(s0).astype(float), where=s0 != 0)
    # dolp = numpy2cv2(dolp)
    return intensity,aolp,dolp

def compute_e2p_output(output):
    intensity = torch2cv2(output['i'])
    aolp = torch2cv2(output['a'])
    dolp = torch2cv2(output['d'])

    max = np.max(intensity)
    min = np.min(intensity)
    intensity = (intensity - min) / (max - min) * 255
    intensity = np.repeat(intensity[:, :, None], 3, axis=2).astype(np.uint8)
    aolp = cv2.applyColorMap(aolp, cv2.COLORMAP_HSV)
    dolp = cv2.applyColorMap(dolp, cv2.COLORMAP_HOT)
    return intensity,aolp,dolp

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='consumer: Consumes DVS frames to process', allow_abbrev=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--use_firenet', type=bool, default=USE_FIRENET, help='use (legacy) firenet instead of e2p')
    parser.add_argument('--checkpoint_path', type=str, default='e2p.pth', help='path to latest checkpoint')
    parser.add_argument('--output_folder', default="/tmp/output", type=str,
                        help='where to save outputs to')
    parser.add_argument('--height', type=int, default=260,
                        help='sensor resolution: height')
    parser.add_argument('--width', type=int, default=346,
                        help='sensor resolution: width')
    parser.add_argument('--device', default='0', type=str,
                        help='indices of GPUs to enable')
    parser.add_argument('--is_flow', action='store_true',
                        help='If true, save output to flow npy file')
    parser.add_argument('--update', action='store_true',
                        help='Set this if using updated models')
    parser.add_argument('--color', action='store_true', default=False,
                        help='Perform color reconstruction')
    parser.add_argument('--voxel_method', default='between_frames', type=str,
                        help='which method should be used to form the voxels',
                        choices=['between_frames', 'k_events', 't_seconds'])
    parser.add_argument('--k', type=int,
                        help='new voxels are formed every k events (required if voxel_method is k_events)')
    parser.add_argument('--sliding_window_w', type=int,
                        help='sliding_window size (required if voxel_method is k_events)')
    parser.add_argument('--t', type=float,
                        help='new voxels are formed every t seconds (required if voxel_method is t_seconds)')
    parser.add_argument('--sliding_window_t', type=float,
                        help='sliding_window size in seconds (required if voxel_method is t_seconds)')
    parser.add_argument('--loader_type', default='H5', type=str,
                        help='Which data format to load (HDF5 recommended)')
    parser.add_argument('--filter_hot_events', action='store_true',
                        help='If true, auto-detect and remove hot pixels')
    parser.add_argument('--legacy_norm', action='store_true', default=False,
                        help='Normalize nonzero entries in voxel to have mean=0, std=1 according to Rebecq20PAMI and Scheerlinck20WACV.'
                             'If --e2vid or --firenet_legacy are set, --legacy_norm will be set to True (default False).')
    parser.add_argument('--robust_norm', action='store_true', default=False,
                        help='Normalize voxel')
    parser.add_argument('--e2p', action='store_true', default=True,
                        help='set required parameters to run events to polarity e2p DNN')
    parser.add_argument('--e2vid', action='store_true', default=False,
                        help='set required parameters to run original e2vid as described in Rebecq20PAMI for polariziation reconstruction')
    parser.add_argument('--firenet_legacy', action='store_true', default=False,
                        help='set required parameters to run legacy firenet as described in Scheerlinck20WACV (not for retrained models using updated code)')
    parser.add_argument('--calculate_mode', action='store_true', default=False,
                        help='Calculate the parameters and FLOPs.')
    parser.add_argument('--real_data', action='store_true', default=False,
                        help='currently our own real data has no frame')
    parser.add_argument('--direction', default=None, type=str,
                        help='Specify which dataloader will be used for FireNet inference.')

    args = parser.parse_args()

    if args.device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    log.info('Loading checkpoint: {} ...'.format(args.checkpoint_path))

    log.info('opening UDP port {} to receive frames from producer'.format(PORT))
    server_socket: socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    address = ("", PORT)
    server_socket.bind(address)

    # initial network model
    log.info(f'checkpoint path is {args.checkpoint_path}')
    checkpoint = torch.load(args.checkpoint_path)
    args, checkpoint = legacy_compatibility(args, checkpoint)
    model = load_model(checkpoint)


    log.info(f'Using UDP buffer size {UDP_BUFFER_SIZE} to receive the {IMSIZE}x{IMSIZE} images')

    cv2_resized = dict()

    log.info('GPU is {}'.format('available' if args.device is not None else 'not available (check cuda setup)'))

    def show_frame(frame, name, resized_dict):
        """ Show the frame in named cv2 window and handle resizing
        :param frame: 2d array of float
        :param name: string name for window
        :param resized_dict: dict to hold this frame for resizing operations
        """
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.imshow(name, frame)
        if not (name in resized_dict):
            cv2.resizeWindow(name, 1800, 600)
            resized_dict[name] = True
            # wait minimally since interp takes time anyhow
            cv2.waitKey(1)

    last_frame_number=0
    voxel_five_float32 = np.zeros((NUM_BINS, IMSIZE, IMSIZE))
    c = 0
    print_key_help()
    while True:
        # todo: reset state after a long period
        model.reset_states_i()
        model.reset_states_a()
        model.reset_states_d()
        timestr = time.strftime("%Y%m%d-%H%M")
        with Timer('overall consumer loop', numpy_file=f'{DATA_FOLDER}/consumer-frame-rate-{timestr}.npy', show_hist=True):
            with Timer('receive UDP'):
                receive_data = server_socket.recv(UDP_BUFFER_SIZE)

            with Timer('unpickle and normalize/reshape'):
                (frame_number, timestamp, x, voxel) = pickle.loads(receive_data)
                if x == 0:
                    voxel_five_float32 = np.zeros((1, NUM_BINS, IMSIZE, IMSIZE))
                    c = 0
                dropped_frames=frame_number-last_frame_number-1
                if dropped_frames>0:
                    log.warning(f'Dropped {dropped_frames} frames from producer')
                last_frame_number=frame_number
                voxel_float32 = ((1. / 255) * np.array(voxel, dtype=np.float32)) * 2 - 1
                voxel_five_float32[:, x, :, :] = voxel_float32
                c += 1
            if c == NUM_BINS:
                with Timer('run CNN'):
                    input=torch.from_numpy(voxel_five_float32).float().to(device)
                    output = model(input)
                    if args.e2p:
                        intensity, aolp, dolp=compute_e2p_output(output)
                    elif args.firenet_legacy:
                        intensity, aolp, dolp=compute_firenet_output(output)


                with Timer('show output frame'):
                    image = cv2.hconcat([intensity, aolp, dolp])
                    show_frame(image, 'polarization', cv2_resized)

            # save time since frame sent from producer
            dt=time.time()-timestamp
            with Timer('producer->consumer inference delay', delay=dt, show_hist=True):
                pass

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q') or k==ord('x'):
                print('quitting....')
                break
            elif k == ord('h') or k == ord('?'):
                print_key_help()
            elif k == ord('p'):
                print_timing_info()

    # calculate the computational efficiency
    if args.calculate_mode:
        flops, params = profile(model, inputs=(input,))
        flops, params = clever_format([flops, params], "%.3f")
        print(model)
        print('[Statistics Information]\nFLOPs: {}\nParams: {}'.format(flops, params))
        exit(0)
