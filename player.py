"""
 @Time    : 03.04.22 16:46
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : player.py
 @Function:
 
"""
import time
import datetime
import argparse
import torch
import numpy as np
from easygui import fileopenbox
from numpy import mean
from os.path import join
import os
import sys
import cv2
from tqdm import tqdm
from thop import profile
# from thop import clever_format
from engineering_notation import EngNumber as eng # only from pip

from utils.prefs import MyPreferences
prefs=MyPreferences()
from utils.get_logger import get_logger
log=get_logger(__name__)

# import math
from pathlib import Path

from globals_and_utils import DOLP_AOLP_MASK_LEVEL, mycv2_put_text
# for run original model
# from model_ori.model import *
# from model_ori import model as model_arch

# for train and test new model
# from model.model import *
# from model import model as model_arch
# from model import model_mhy as model_arch
# from model import model_original as model_arch
from train.model import model_v as model_arch
from train.utils.render_e2p_output import render_e2p_output
# from model import model_ed as model_arch
# from model.model import ColorNet

from train.utils.util import ensure_dir, flow2bgr_np
from train.data_loader.data_loaders import InferenceDataLoader
from train.utils.util import CropParameters, get_height_width, torch2cv2, \
    append_timestamp, setup_output_folder, torch2numpy, numpy2cv2
from utils.timers import CudaTimer
from train.utils.henri_compatible import make_henri_compatible

from train.parse_config import ConfigParser

model_info = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(args):
    checkpoint = torch.load(args.checkpoint_path)
    config = checkpoint['config']
    log.info(f"config['arch']={config['arch']}")

    try:
        model_info['num_bins'] = config['arch']['args']['unet_kwargs']['num_bins']
    except KeyError:
        model_info['num_bins'] = config['arch']['args']['num_bins']
    log.info(f"model_info['num_bins']={model_info['num_bins']}")
    # logger = config.get_logger('test')

    # build model architecture
    model = config.init_obj('arch', model_arch)
    log.info(f"model={model}")
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    log.info('Load my trained weights succeeded!')

    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return model


def minmax_normalization(image, device):
    mini = np.percentile(torch.flatten(image).cpu().detach().numpy(), 1)
    maxi = np.percentile(torch.flatten(image).cpu().detach().numpy(), 99)
    image_morm = (image - mini) / (maxi - mini + 1e-5)
    image_morm = torch.clamp(image_morm, 0, 1)

    return image_morm.to(device)


def main(args):
    sys.path.append('train')  # needed to get model to load using torch.load with train.parse_config ConfigParser.. don't understand why
    if args.events_file_path is None:
        events_file_path = get_events_file_path()
    else:
        events_file_path=Path(args.events_file_path)
    if events_file_path is None:
        print('no file specified, quitting')
        quit(0)
    log.info(f'playing file "{events_file_path}"')

    data_loader, dataset = open_dataset(args, events_file_path)
    n_samples = len(dataset)

    height, width = get_height_width(data_loader)
    model_info['input_shape'] = height, width
    # crop = CropParameters(width, height, model.num_encoders)
    crop = CropParameters(width, height, 1)

    if args.device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    log.info('Loading checkpoint: {} ...'.format(args.checkpoint_path))

    model = load_model(args)

    reset_e2p_network(model)
    frame_number = -1
    time_list = []
    paused=False # to pause playback
    forwards=True # what direction to go
    if not args.quiet:  # show video
        cv2.namedWindow('pdavis',cv2.WINDOW_NORMAL)
    recording_activated=False
    frame_interval_ms=100
    # https://stackoverflow.com/questions/53570732/get-single-random-example-from-pytorch-dataloader/61389393#61389393
    # for item in tqdm(data_loader):
    with tqdm(total=n_samples) as pbar:
        while True:
            k = cv2.waitKey(frame_interval_ms) & 0xFF
            # https://stackoverflow.com/questions/75030061/python-opencv-waitkeyex-stops-picking-up-arrow-keys-after-mouse-click-or-tab
            if k == 27 or k == ord('x'):  # ESC or 'x' exits
                print('quitting...')
                cv2.destroyAllWindows()
                break
            elif k==ord(' '):
                paused=not paused
                print(f'paused={paused}')
            elif k==ord('b'):
                forwards=not forwards
                print(f'forwards={forwards}')
            elif k==ord('r'):
                print('rewound')
                frame_number=-1
                forwards=True
                reset_e2p_network(model)
                pbar.reset(-1)
                continue
            elif k==ord('['):
                frame_number-=20
                print(f'jogged backwards to {frame_number}')
            elif k==ord(']'):
                frame_number+=20
                print(f'jogged forwards to {frame_number}')
            elif k == ord('f'):
                frame_interval_ms = int(decrease(frame_interval_ms, 4))
                print(f'shorter frame intervals is now {frame_interval_ms:.2f}ms')
            elif k == ord('s'):
                frame_interval_ms = int(increase(frame_interval_ms, 1000))
                print(f'longer frame intervals is now {frame_interval_ms:.2f}ms')
            elif k==ord('o'):
                events_file_path=get_events_file_path()
                if events_file_path is None:
                    continue
                data_loader, dataset = open_dataset(args, events_file_path)
                n_samples = len(dataset)
                frame_number=-1
                pbar.reset(n_samples)
                continue
            elif k == ord('l'):
                if not recording_activated:
                    recording_activated = True
                    output_folder = os.path.join(args.output_folder, events_file_path.stem)
                    ts_fname = setup_output_folder(output_folder)  # create folder and timestamps file in it
                    log.info(f'started logging recording PNG frames to folder {output_folder}')
                else:
                    recording_activated = False
                    log.info(f'stopped logging recording PNG frames to folder {output_folder}')
            elif k==ord('e'):
                reset_e2p_network(model)
            elif k == ord('-'):
                args.dolp_aolp_mask_level *= .9
                print(f'decrased AoLP DoLP mask level to {args.dolp_aolp_mask_level}')
            elif k == ord('='):  # change mask level for displaying AoLP
                args.dolp_aolp_mask_level /= .9
                print(f'increased AoLP DoLP mask level to {args.dolp_aolp_mask_level}')
            elif k==ord('m'):
                lastmodel = prefs.get('last_model_selected', 'models/*.pth')
                f = fileopenbox(msg='select model checkpoint', title='Model checkpoint', default=lastmodel,
                                filetypes=['*.pth'])
                if f is not None:
                    prefs.put('last_model_selected', f)
                    args.checkpoint_path = f
                    model=load_model(args)
                    print(f'changed model to {args.checkpoint_path}')
            elif k==ord('h') or k==ord('?'):
                print('ESC or x: exit\n'
                      'space: toggle pause\n'
                      'r: rewind\n'
                      'b: toggle direction backwards/forwards\n'
                      's or f: slower or faster playback\n'
                      '[ or ]: jog backwards or forwards\n'
                      'o: open a new h5 to play back\n'
                      'm: load a new E2P network model\n'
                      'l: toggle logging (recording) frames to disk'
                      f'- or =: decrease or increase the AoLP DoLP mask level which is currently {args.dolp_aolp_mask_level}'
                      'e: rEset E2P hidden states\n'
                      '? or h: print this help'
                      )
            elif k!=255:
                print(f'unknown key {k}')
            if paused:
                continue

            frame_number += 1 if forwards else -1
            if frame_number>=n_samples:
                print('rewound')
                frame_number=-1
                if recording_activated:
                    log.info(f'stopped logging recording frames to {output_folder}')
                    recording_activated=False
                continue
            if frame_number<0:
                print('done going backwards, changed to forwards')
                forwards=True
                frame_number=-1
                if recording_activated:
                    log.info(f'stopped logging recording frames to {output_folder}')
                    recording_activated=False
                continue

            pbar.update(1 if forwards else -1)
            item=dataset[frame_number] # get a particular frame, is dict with intensity, aolp, dolp, events (voxel grid), etc
            # even though we directly access the dataset item by index, it still calls the get_item that calls the code in data_loaders.py to build the voxel grid from raw events
            # item['events'] is actually the complete 5-frame voxel grid
            # see my_merge_h5.py for how the original data is created for training E2P
            voxel = torch.unsqueeze(item['events'], dim=0).to(device)# we need to add a singleton dimension at front to make it look like batch 1 sample
            # voxel = item['events'].to(device)
            # important # tobi doesn't think we need to do this for E2P... maybe holdover from firenet?
            # if args.real_data:
            #     voxel = torch.flip(voxel, [2])
            #     voxel = torch.flip(voxel, [2, 3])
            # with CudaTimer('Inference'):
            #     output = model(voxel)
            start_each = time.time()
            output = model(voxel)
            time_each = time.time() - start_each
            time_list.append(time_each)

            # calculate the computational efficiency
            if args.measure_cost:
                args.measure_cost=False # only first one
                flops, params = profile(model, inputs=(voxel,))
                # print(model)
                log.info(f'\n[Network cost information]\nFLOPs/sample: {eng(flops)}\nParams: {eng(params)}')

            # code moved to bottom of file ****************************************************
            # intensity = torch2cv2(output['i']) # these 3 outputs are scaled 0-1, torch2cv2 rescales to 0-255 uint8 2d images
            # aolp = torch2cv2(output['a'])
            # dolp = torch2cv2(output['d'])
            intensity,aolp,dolp=render_e2p_output(output, args.dolp_aolp_mask_level, 1.0)
            iad = cv2.hconcat([intensity, aolp, dolp])
            # iad_f = cv2.hconcat([intensity_f, aolp_f, dolp_f])
            gt={}
            gt['i']=(torch.squeeze(item['intensity']))
            gt['a']=(torch.squeeze(item['aolp']))
            gt['d']=(torch.squeeze(item['dolp']))
            # intensity_gt=(torch.squeeze(item['intensity']).numpy() * 255).astype(np.uint8)
            # aolp_gt=(torch.squeeze(item['aolp']).numpy() * 255).astype(np.uint8)
            # dolp_gt=(torch.squeeze(item['dolp']).numpy() * 255).astype(np.uint8)
            intensity_gt,aolp_gt,dolp_gt=render_e2p_output(gt, args.dolp_aolp_mask_level, 1.0)
            iad_gt = cv2.hconcat([intensity_gt,aolp_gt,dolp_gt])
            mycv2_put_text(iad_gt, f'GT fr:{frame_number:,}/{n_samples:,}',fontScale=1.5,org=(10,20))
            mycv2_put_text(iad, 'E2P',fontScale=1.5,org=(10,20))

            iad_both = cv2.vconcat([iad_gt, iad])
            if recording_activated:
                iad_name = '{:05d}.png'.format(frame_number)
                cv2.imwrite(join(output_folder, iad_name), iad_both)
                print('.', end='')
                append_timestamp(ts_fname, iad_name, item['timestamp'].item())
                if frame_number % 80 == 0:
                    print('')
            if not args.quiet: # show video
                cv2.namedWindow('pdavis')
                cv2.imshow('pdavis',iad_both)



    log.info(f"\n{args.checkpoint_path}'s average inference time Is : {eng(mean(time_list) * 1000)} ms")


def get_events_file_path():
    last_h5_folder = prefs.get('last_h5_folder', '')
    events_file_path = fileopenbox(msg='Select h5 dataset file', title='H5 chooser', filetypes=['*.h5'],
                                   default=last_h5_folder)
    if events_file_path is None:
        print('no file selected')
        return None
    events_file_path = Path(events_file_path)
    prefs.put('last_h5_folder', str(events_file_path))
    return events_file_path


def open_dataset(args, events_file_path):
    dataset_kwargs = {'transforms': {},
                      'max_length': None,
                      'sensor_resolution': (args.height, args.width),
                      'num_bins': 5,
                      'filter_hot_events': args.filter_hot_events,
                      'voxel_method': {'method': args.voxel_method,
                                       'k': args.k,
                                       't': args.t,
                                       'sliding_window_w': args.sliding_window_w,
                                       'sliding_window_t': args.sliding_window_t}
                      }
    if args.update:
        print("Updated style model")
        dataset_kwargs['combined_voxel_channels'] = False
    if args.legacy_norm:
        print('Using legacy voxel normalization')
        dataset_kwargs['transforms'] = {'LegacyNorm': {}}
    if args.robust_norm:
        print('Using Robust voxel normalization')
        dataset_kwargs['transforms'] = {'RobustNorm': {}}
    data_loader = InferenceDataLoader(events_file_path, dataset_kwargs=dataset_kwargs, ltype=args.loader_type,
                                      real_data=args.real_data, direction=args.direction)
    dataset = data_loader.dataset
    return data_loader, dataset


def reset_e2p_network(model):
    print('reset E2P hidden states')
    model.reset_states_i()
    model.reset_states_a()
    model.reset_states_d()

SPEED_UP_FACTOR=2
def increase(val,limit):
    return val*SPEED_UP_FACTOR if val*SPEED_UP_FACTOR<=limit else limit
def decrease(val,limit):
    return val/SPEED_UP_FACTOR if val/SPEED_UP_FACTOR>=limit else limit

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('--checkpoint_path', type=str, default='models/e2p-0317_215454-e2p-paper_plus_tobi_office-from-scratch.pth',
                        help='path to latest checkpoint (default: None)')
    # parser.add_argument('--events_file_path', type=str, default='/mnt/c/Users/tobid/Downloads/Davis346B-2023-03-16T15-42-43+0100-00000000-0-pdavis-polfilter-tobi-office-window2.h5',
    parser.add_argument('--events_file_path', type=str, default=None,
                        help='path to events (HDF5)')
    parser.add_argument('--output_folder', default="/tmp/output", type=str,
                        help='where to save outputs to')
    parser.add_argument('--dolp_aolp_mask_level', type=float, default=DOLP_AOLP_MASK_LEVEL, help='level of DoLP below which to mask the AoLP value since it is likely not meaningful')
    parser.add_argument('--height',  type=int, default=260,
                        help='sensor resolution: height')
    parser.add_argument('--width',  type=int, default=346,
                        help='sensor resolution: width')
    parser.add_argument('--device', default='0', type=str,
                        help='indices of GPUs to enable')
    parser.add_argument('--update', action='store_true',
                        help='Set this if using updated models')
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
    parser.add_argument('--measure_cost', action='store_true', default=False,
                        help='Calculate the parameters and FLOPs.')
    parser.add_argument('--real_data', action='store_true', default=False,
                        help='currently our own real data has no frame')
    parser.add_argument('--direction', default=None, type=str,
                        help='Specify which dataloader will be used for FireNet inference.')
    parser.add_argument('--quiet',action='store_true',help='quiet mode - do not show video during reconstruction')

    args = parser.parse_args()

    main(args)

    # model.firenet_i90.reset_states()
    # model.firenet_i45.reset_states()
    # model.firenet_i135.reset_states()
    # model.firenet_i0.reset_states()
    # model.reset_states_intensity()
    # model.reset_states_aolp()
    # model.reset_states_dolp()
    # m13
    # model.reset_states_i()
    # v
    # model.reset_states_s0()
    # model.reset_states_s1()
    # model.reset_states_s2()
    # m31
    # model.reset_states_i2()
    # model.reset_states_i4()
    # model.reset_states_i8()
    # model.reset_states_a2()
    # model.reset_states_a4()
    # model.reset_states_a8()
    # model.reset_states_d2()
    # model.reset_states_d4()
    # model.reset_states_d8()


# original firenet
# image = crop.crop(output['image'])
# image = torch2cv2(image)
# output raw
# image = crop.crop(output['image'])
# image = torch2numpy(image)
# image = np.clip(image, 0, 1)
#
# i90 = image[0::2, 0::2]
# i45 = image[0::2, 1::2]
# i135 = image[1::2, 0::2]
# i0 = image[1::2, 1::2]
#
# s0 = i0.astype(float) + i90.astype(float)
# s1 = i0.astype(float) - i90.astype(float)
# s2 = i45.astype(float) - i135.astype(float)
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
# aolp = 0.5 * np.arctan2(s2, s1)
# aolp = aolp + 0.5 * math.pi
# aolp = aolp / math.pi
# aolp = numpy2cv2(aolp)
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
# output raw
# fname = '{:05d}.png'.format(i + 1)
# cv2.imwrite(join(output_folder, fname), image)
# output polarization iad
