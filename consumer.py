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
from globals_and_utils import *
from engineering_notation import EngNumber as eng # only from pip
import collections
from pathlib import Path
import random

from e2p import V16 as Model

# for network inference
import torch
from inference import legacy_compatibility, load_model
from utils.util import torch2cv2

log=my_logger(__name__)

# Only used in mac osx
try:
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
except Exception as e:
    print(e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='consumer: Consumes DVS frames to process', allow_abbrev=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--checkpoint_path', required=True, type=str, help='path to latest checkpoint (default: None)')
    parser.add_argument('--events_file_path', required=True, type=str,
                        help='path to events (HDF5)')
    parser.add_argument('--output_folder', default="/tmp/output", type=str,
                        help='where to save outputs to')
    parser.add_argument('--height', required=True, type=int, default=260,
                        help='sensor resolution: height')
    parser.add_argument('--width', required=True, type=int, default=346,
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
    parser.add_argument('--e2vid', action='store_true', default=False,
                        help='set required parameters to run original e2vid as described in Rebecq20PAMI')
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
    print('Loading checkpoint: {} ...'.format(args.checkpoint_path))

    log.info('opening UDP port {} to receive frames from producer'.format(PORT))
    server_socket: socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    address = ("", PORT)
    server_socket.bind(address)

    # initial network model
    checkpoint = torch.load(args.checkpoint_path)
    args, checkpoint = legacy_compatibility(args, checkpoint)
    model = load_model(checkpoint)

    log.info(f'Using UDP buffer size {UDP_BUFFER_SIZE} to receive the {IMSIZE}x{IMSIZE} images')

    saved_non_jokers = collections.deque(maxlen=NUM_NON_JOKER_IMAGES_TO_SAVE_PER_JOKER)  # lists of images to save
    Path(JOKERS_FOLDER).mkdir(parents=True, exist_ok=True)
    Path(NONJOKERS_FOLDER).mkdir(parents=True, exist_ok=True)

    def next_path_index(path):
        l = glob.glob(path + '/[0-9]*.png')
        if len(l) == 0:
            return 0
        else:
            l2 = sorted(l)
            last = l2[-1]
            last2 = last.split('/')[-1]
            last3 = last2.split('.')[0]
            next = int(last3) + 1  # strip .png
            return next

    next_joker_index = next_path_index(JOKERS_FOLDER)
    next_non_joker_index = next_path_index(NONJOKERS_FOLDER)
    cv2_resized = dict()

    def print_num_saved_images():
        log.info(f'saved {next_non_joker_index} nonjokers to {NONJOKERS_FOLDER} and {next_joker_index} jokers to {JOKERS_FOLDER}')

    atexit.register(print_num_saved_images)

    log.info('GPU is {}'.format('available' if args.device is not None else 'not available (check cuda setup)'))

    def show_frame(frame, name, resized_dict):
        """ Show the frame in named cv2 window and handle resizing
        :param frame: 2d array of float
        :param name: string name for window
        """
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.imshow(name, frame)
        if not (name in resized_dict):
            cv2.resizeWindow(name, 300, 300)
            resized_dict[name] = True
            # wait minimally since interp takes time anyhow
            cv2.waitKey(1)

    last_frame_number=0
    while True:
        timestr = time.strftime("%Y%m%d-%H%M")
        with Timer('overall consumer loop', numpy_file=f'{DATA_FOLDER}/consumer-frame-rate-{timestr}.npy', show_hist=True):
            with Timer('receive UDP'):
                receive_data = server_socket.recv(UDP_BUFFER_SIZE)

            with Timer('unpickle and normalize/reshape'):
                (frame_number, timestamp, img255) = pickle.loads(receive_data)
                dropped_frames=frame_number-last_frame_number-1
                if dropped_frames>0:
                    log.warning(f'Dropped {dropped_frames} frames from producer')
                last_frame_number=frame_number
                img_01_float32 = (1. / 255) * np.array(img255, dtype=np.float32)
            with Timer('run CNN'):
                # pred = model.predict(img[None, :])
                output = model(img_01_float32)
                intensity = torch2cv2(output['i'])
                aolp = torch2cv2(output['a'])
                dolp = torch2cv2(output['d'])
                image = cv2.hconcat([intensity, aolp, dolp])
                show_frame(image, 'polarization', cv2_resized)

            # save time since frame sent from producer
            dt=time.time()-timestamp
            with Timer('producer->consumer inference delay', delay=dt, show_hist=True):
                pass

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
