"""
 @Time    : 14.11.22 19:54
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : inference_v.py
 @Function:
 
"""
import time
import datetime
import argparse
import torch
from torchvision.utils import make_grid, save_image
import numpy as np
from numpy import mean
from os.path import join
import os
import cv2
from tqdm import tqdm
from thop import profile
from thop import clever_format
import math

# for run original model
# from model_ori.model import *
# from model_ori import model as model_arch

# for train and test new model
from model.model import *
# from model import model as model_arch
# from model import model_mhy as model_arch
# from model import model_original as model_arch
from model import model_v as model_arch
# from model import model_ed as model_arch
# from model.model import ColorNet

from utils.util import ensure_dir, flow2bgr_np
from data_loader.data_loaders import InferenceDataLoader
from utils.util import CropParameters, get_height_width, torch2cv2, \
    append_timestamp, setup_output_folder, torch2numpy, numpy2cv2
from utils.timers import CudaTimer
from utils.henri_compatible import make_henri_compatible

from parse_config import ConfigParser

model_info = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def legacy_compatibility(args, checkpoint):
    assert not (args.e2vid and args.firenet_legacy)
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
    print(config['arch'])
    state_dict = checkpoint['state_dict']

    try:
        model_info['num_bins'] = config['arch']['args']['unet_kwargs']['num_bins']
        print(model_info['num_bins'])
    except KeyError:
        model_info['num_bins'] = config['arch']['args']['num_bins']
    logger = config.get_logger('test')

    # build model architecture
    model = config.init_obj('arch', model_arch)
    logger.info(model)
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)
    print('Load my trained weights succeed!')

    model = model.to(device)
    model.eval()
    # if args.color:
    #     model = ColorNet(model)
    for param in model.parameters():
        param.requires_grad = False

    return model


def minmax_normalization(image, device):
    mini = np.percentile(torch.flatten(image).cpu().detach().numpy(), 1)
    maxi = np.percentile(torch.flatten(image).cpu().detach().numpy(), 99)
    image_morm = (image - mini) / (maxi - mini + 1e-5)
    image_morm = torch.clamp(image_morm, 0, 1)

    return image_morm.to(device)


def main(args, model):
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

    data_loader = InferenceDataLoader(args.events_file_path, dataset_kwargs=dataset_kwargs, ltype=args.loader_type,
                                      real_data=args.real_data, direction=args.direction)

    height, width = get_height_width(data_loader)

    model_info['input_shape'] = height, width
    # crop = CropParameters(width, height, model.num_encoders)
    crop = CropParameters(width, height, 1)

    ts_fname = setup_output_folder(args.output_folder)

    output_folder = args.output_folder

    # original
    # model.reset_states()
    # four-direction
    # model.reset_states_i90()
    # model.reset_states_i45()
    # model.reset_states_i135()
    # model.reset_states_i0()
    model.reset_states_i()
    model.reset_states_a()
    model.reset_states_d()
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
    i = 0
    time_list = []
    for item in tqdm(data_loader):
        voxel = item['events'].to(device)
        if not args.color:
            voxel = crop.pad(voxel)
        # important
        # if args.real_data:
        # voxel = torch.flip(voxel, [2])
        # voxel = torch.flip(voxel, [2, 3])
        # with CudaTimer('Inference'):
        #     output = model(voxel)
        start_each = time.time()
        output = model(voxel)
        time_each = time.time() - start_each
        time_list.append(time_each)
        # save sample images, or do something with output here

        # calculate the computational efficiency
        if args.calculate_mode:
            flops, params = profile(model, inputs=(voxel,))
            flops, params = clever_format([flops, params], "%.3f")
            print(model)
            print('[Statistics Information]\nFLOPs: {}\nParams: {}'.format(flops, params))
            exit(0)

        if args.is_flow:
            flow_t = torch.squeeze(crop.crop(output['flow']))
            # Convert displacement to flow
            if item['dt'] == 0:
                flow = flow_t.cpu().numpy()
            else:
                flow = flow_t.cpu().numpy() / item['dt'].numpy()
            ts = item['timestamp'].cpu().numpy()
            flow_dict = flow
            fname = 'flow_{:010d}.npy'.format(i)
            np.save(os.path.join(args.output_folder, fname), flow_dict)
            with open(os.path.join(args.output_folder, fname), "a") as myfile:
                myfile.write("\n")
                myfile.write("timestamp: {:.10f}".format(ts[0]))
            flow_img = flow2bgr_np(flow[0, :, :], flow[1, :, :])
            fname = 'flow_{:010d}.png'.format(i)
            cv2.imwrite(os.path.join(args.output_folder, fname), flow_img)
        else:
            if args.color:
                image = output['image']
            else:
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
                wi = output['wi'][0, :, :, :]
                intensity = torch2cv2(wi)

                wa = output['wa'][0, :, :, :]
                aolp = torch2cv2(wa)

                wd = output['wd'][0, :, :, :]
                dolp = torch2cv2(wd)

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
            # cv2.imwrite(join(args.output_folder, fname), image)
            # output polarization iad
            iad_name = '{:05d}.png'.format(i)
            i_name = '{:05d}_i.png'.format(i)
            a_name = '{:05d}_a.png'.format(i)
            d_name = '{:05d}_d.png'.format(i)
            # iad = cv2.hconcat([intensity, aolp, dolp])
            # iad_f = cv2.hconcat([intensity_f, aolp_f, dolp_f])
            # iad_gt = cv2.hconcat([(torch.squeeze(item['intensity']).numpy() * 255).astype(np.uint8),
            #                       (torch.squeeze(item['aolp']).numpy() * 255).astype(np.uint8),
            #                       (torch.squeeze(item['dolp']).numpy() * 255).astype(np.uint8)])
            # iad = cv2.vconcat([iad_gt, iad_f, iad])
            # iad = cv2.vconcat([iad_gt, iad])
            for j in range(32):
                # cv2.imwrite(join(output_folder, i_name[:-4] + '_{:02d}.png'.format(j)), cv2.applyColorMap(intensity[j, :, :], cv2.COLORMAP_SUMMER))
                # cv2.imwrite(join(output_folder, a_name[:-4] + '_{:02d}.png'.format(j)), cv2.applyColorMap(aolp[j, :, :], cv2.COLORMAP_SUMMER))
                # cv2.imwrite(join(output_folder, d_name[:-4] + '_{:02d}.png'.format(j)), cv2.applyColorMap(dolp[j, :, :], cv2.COLORMAP_SUMMER))
                cv2.imwrite(join(output_folder, i_name[:-4] + '_{:02d}.png'.format(j)), cv2.applyColorMap(intensity[j, :, :], cv2.COLORMAP_AUTUMN))
                cv2.imwrite(join(output_folder, a_name[:-4] + '_{:02d}.png'.format(j)), cv2.applyColorMap(aolp[j, :, :], cv2.COLORMAP_AUTUMN))
                cv2.imwrite(join(output_folder, d_name[:-4] + '_{:02d}.png'.format(j)), cv2.applyColorMap(dolp[j, :, :], cv2.COLORMAP_AUTUMN))
            # cv2.imwrite(join(output_folder, iad_name), np.flipud(iad))
            # output four-direction intensity
            # direction_name = '{:05d}.png'.format(i + 1)
            # direction = cv2.hconcat([i90, i45, i135, i0])
            # cv2.imwrite(join(output_folder, direction_name), direction)
        # output raw
        # append_timestamp(ts_fname, fname, item['timestamp'].item())
        # output polarization iad
        append_timestamp(ts_fname, iad_name, item['timestamp'].item())
        # output four-direction intensity
        # append_timestamp(ts_fname, direction_name, item['timestamp'].item())

        i += 1

    print("{}'s average Time Is : {:.3f} ms".format(args.checkpoint_path, mean(time_list) * 1000))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('--checkpoint_path', required=True, type=str,
                        help='path to latest checkpoint (default: None)')
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
    checkpoint = torch.load(args.checkpoint_path)
    args, checkpoint = legacy_compatibility(args, checkpoint)
    model = load_model(checkpoint)
    main(args, model)

