"""
 @Time    : 28.3.23
 @Author  : Tobi Delbruck, Haiyang Mei, Zoowen Wang
 @E-mail  : tobi@ini.uzh.ch
 
 @Project : pdavis_demo
 @File    : consumer.py
 @Function:
 
"""

import argparse
import pickle
from queue import Empty

import socket

from thop import profile, clever_format

from globals_and_utils import *
# from engineering_notation import EngNumber as eng  # only from pip
from pathlib import Path

import torch

from train.events_contrast_maximization.utils.event_utils import events_to_voxel_torch
from train.utils.render_e2p_output import render_e2p_output
from train.utils.util import torch2numpy, numpy2cv2
from easygui import fileopenbox
from utils.prefs import MyPreferences
prefs=MyPreferences()

log = get_logger(__name__)

# Only used in mac osx
try:
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
except Exception as e:
    print(e)

# from player.py
import e2p as model_arch
def consumer(queue:Queue):
    """
    consume frames to predict polarization
    :param queue: if started with a queue, uses that for input of voxel volume
    """
    args=get_args()

    states = None  # firenet states, to feed back into firenet
    model = None

    playback_file=None # used to play back recording from producer
    playback_events=None # the events we will play back, totally loaded into RAM from numpy recording
    playback_current_time=0.0
    playback_timestamps=None # the timestamps alone, for searching current frame
    playback_frame_duration=FRAME_DURATION_US*1e-6

    recording_folder_base = args.recording_folder
    recording_folder_current=None
    recording_frame_number = 0
    server_socket=None

    aolp_colorwheel = cv2.imread('media/aolp_dolp_colorscales.png')
    cv2.namedWindow('aolp_colorwheel')
    cv2.imshow('aolp_colorwheel', aolp_colorwheel)
    cv2.waitKey(100)

    brightness=prefs.get('brightness',1.0)

    if queue is None:
        log.info('opening UDP port {} to receive frames from producer'.format(PORT))
        server_socket: socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        address = ("", PORT)
        server_socket.bind(address)
        args.sensor_resolution=(IMSIZE,IMSIZE) #using UDP, so need to limit UDP packet size
        log.info(f'Using UDP buffer size {UDP_BUFFER_SIZE} to receive the {args.sensor_resolution} images')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cpu':
        log.warning(f'CUDA GPU is not available, running on CPU which will be very slow')
        time.sleep(1)
    if args.device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    log.info('Loading checkpoint: {} ...'.format(args.checkpoint_path))
    model,checkpoint_path = load_selected_model(args, device)
    log.info('GPU is {}'.format('available' if args.device is not None else 'not available (check cuda setup)'))

    cv2_resized = dict()

    def show_frame(frame, name, resized_dict):
        """ Show the frame in named cv2 window and handle resizing
        :param frame: 2D array of RGB values uint8 [y,x,RGB]
        :param name: string name for window
        :param resized_dict: dict to hold this frame for resizing operations
        """
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.imshow(name, frame)
        if not (name in resized_dict):
            cv2.resizeWindow(name, 800, 600)
            resized_dict[name] = True
            # wait minimally since interp takes time anyhow
            cv2.waitKey(1)

    recording_activated=False
    last_frame_number = 0
    voxel_five_float32 = None
    c = 0
    print_key_help(args)
    frames_without_drop = 0
    reset_e2p_state(args,model)

    while True: # main animation loop start
        with Timer('overall consumer loop', show_hist=True, savefig=True):
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q') or k == ord('x'):
                if recording_folder_current is not None:
                    log.info(f'*** recordings of {recording_frame_number - 1} frames saved in {recording_folder_current}')
                print('quitting....')
                break
            elif k == ord('h') or k == ord('?'):
                print_key_help(args)
            elif k == ord('p'):
                print_timing_info()
            elif k == ord('-'):
                args.dolp_aolp_mask_level*= .9
                print(f'decrased AoLP DoLP mask level to {args.dolp_aolp_mask_level}')
            elif k == ord('='): # change mask level for displaying AoLP
                args.dolp_aolp_mask_level/= .9
                print(f'increased AoLP DoLP mask level to {args.dolp_aolp_mask_level}')
            elif k==ord('e'): # reset network state
                log.info('resetting E2P state')
                reset_e2p_state(args, model)
            # elif k == ord('m'): # TODO currently broken
            #     args.use_firenet = not args.use_firenet
            #     model,checkpoint_path = load_selected_model(args, device)
            #     print(f' changed mode to args.use_firenet={args.use_firenet}')
            elif k == ord('b'):
                brightness *= 1.1
                prefs.put('brightness', brightness)
                print(f'increased brightness of intensity reconstruction to {brightness:.2f}')
            elif k == ord('d'):
                brightness /= 1.1
                prefs.put('brightness', brightness)
                print(f'decreased brightness of intensity reconstruction to {brightness:.2f}')
            elif k == ord('r'):
                if not recording_activated:
                    recording_activated = True
                    recording_folder_current = os.path.join(recording_folder_base, get_timestr())
                    Path(recording_folder_current).mkdir(parents=True, exist_ok=True)
                    log.info(f'started recording PNG frames to folder {recording_folder_current}')
                else:
                    recording_activated = False
                    log.info(f'stopped recording PNG frames to folder {recording_folder_current}')
            elif k==ord('o'):
                if playback_file:
                    log.info(f'closing playback file {playback_file}')
                    playback_file=None
                    playback_events=None
                    playback_timestamps=None
                else:
                    last_playback_file = prefs.get('last_playback_file', RECORDING_FOLDER)
                    playback_file = fileopenbox(msg='select numpy recording', title='Select recording', default=last_playback_file,
                                    filetypes=['*.npy'])
                    if playback_file is not None:
                        prefs.put('last_playback_file', playback_file)
                        try:
                            playback_events=np.load(playback_file)
                            playback_timestamps=playback_events[:,0]*1e-6 # make time in seconds
                            playback_current_time=playback_timestamps[0]
                            reset_e2p_state(args, model)
                            log.info(f'playing {playback_events.shape} events with duration {(playback_timestamps[-1]-playback_timestamps[0]):.3f}s from {playback_file}')
                        except Exception as e:
                            log.error(f'could not load events from {playback_file}: got {str(e)}')
                            playback_file=None
                    else:
                        log.warning('no file selected')
            elif k==ord('s'):
                playback_frame_duration/=np.sqrt(2)
                print(f'frame duration decreased to {playback_frame_duration*1e3:.2f}ms frames')
            elif k==ord('f'):
                playback_frame_duration*=np.sqrt(2)
                print(f'frame duration increased to {playback_frame_duration*1e3:.2f}ms frames')


            elif k != 255:
                print_key_help(args)
                print(f'unknown key {k}')

            with Timer('receive voxels'):
                if playback_file:
                    playback_next_time=playback_current_time+playback_frame_duration

                    idx_start=np.searchsorted(playback_timestamps,playback_current_time)
                    # idx_end=np.searchsorted(playback_timestamps,playback_current_time+FRAME_DURATION_US*1e-6)
                    idx_end=np.searchsorted(playback_timestamps,playback_current_time+playback_frame_duration)
                    events=playback_events[idx_start:idx_end,:]
                    playback_current_time=playback_next_time
                    if playback_current_time>playback_timestamps[-1]:
                        playback_current_time=playback_timestamps[0]
                        reset_e2p_state(args, model)
                        log.info('rewound to start, reset network')
                        continue
                    if len(events)==0:
                        log.info('no events')
                        continue
                    xs = torch.from_numpy(events[:, 1].astype(np.float32))  # event x addreses
                    ys = torch.from_numpy(events[:, 2].astype(np.float32))  # event y addresses
                    ts = torch.from_numpy(
                        (events[:, 0] - events[0, 0]).astype(np.float32))  # event relative timesamps in us
                    ps = torch.from_numpy((events[:, 3] * 2 - 1).astype(
                        np.float32))  # change polarity from 0,1 to -1,+1 so that polarities are signed
                    voxel = events_to_voxel_torch(xs, ys, ts, ps, NUM_BINS, sensor_size=SENSOR_RESOLUTION,temporal_bilinear=True)
                    voxel = voxel[:, 0:args.sensor_resolution[0], 0:args.sensor_resolution[1]]  # crop out UL corner from entire voxel frame to limit to max possible UDP packet size
                    voxel_five_float32 = np.expand_dims(voxel.numpy(), 0)  # need to expand to 4d for input to DNN

                elif queue:
                    # log.debug('receiving entire voxel volume on pipe')
                    try:
                        (voxel_five_float32, frame_number, time_last_frame_sent) = queue.get(block=True, timeout=1)
                        dropped_frames = frame_number - last_frame_number - 1
                        if dropped_frames > 0:
                            log.warning(
                                f'Dropped {dropped_frames} producer frames after {frames_without_drop} good frames')
                            frames_without_drop = 0
                        else:
                            frames_without_drop += 1
                        last_frame_number = frame_number
                        time_now = time.time()
                        delta_time = time_now - time_last_frame_sent
                        # print(delta_time)
                        if delta_time > .3:
                            log.warning(
                                f'time since this frame was sent ({eng(delta_time)}s) is longer than .3s; consider increasing producer FRAME_DURATION_US which is currently {FRAME_DURATION_US}us')
                    except Empty:
                        image=np.zeros(args.sensor_resolution,dtype=np.uint8)
                        show_frame(image, 'polarization', cv2_resized)
                        continue
                    # log.debug(f'received entire voxel volume on pipe with shape={voxel_five_float32.shape}')
                    c=NUM_BINS
                else:
                    receive_data = server_socket.recv(UDP_BUFFER_SIZE)


                    (frame_number, time_last_frame_sent, bin, frame_255, frame_min, frame_max) = pickle.loads(receive_data)
                    if bin == 0:
                        voxel_five_float32 = np.zeros((1, NUM_BINS, args.sensor_resolution[0], args.sensor_resolution[1]),dtype=np.float32)
                        c = 0
                    dropped_frames = frame_number - last_frame_number - 1
                    if dropped_frames > 0:
                        log.warning(f'Dropped {dropped_frames} producer frames after {frames_without_drop} good frames')
                        frames_without_drop = 0
                    else:
                        frames_without_drop += 1
                    last_frame_number = frame_number
                    # voxel_float32 = ((1. / 255) * np.array(voxel, dtype=np.float32)) * 2 - 1 # map 0-255 range to -1,+1 range
                    # voxel_five_float32[:, bin, :, :] = voxel.astype(np.float32)
                    # for UDP, convert voxel frame from uint8 as well as possible, since producer mapped to uint8 with inverse of following
                    voxel_five_float32[:, bin, :, :] = ((frame_255.astype(np.float32)) / 255) * (frame_max - frame_min) + frame_min
                    c += 1
            if c == NUM_BINS:
                with Timer('run CNN', show_hist=True, savefig=True):
                    input = torch.from_numpy(voxel_five_float32).to(device)
                    if not args.use_firenet:  # e2p, just use voxel grid from producer
                        output = model(input)
                        intensity, aolp, dolp = render_e2p_output(output, args.dolp_aolp_mask_level, brightness) # output are RGB images with gray, HSV, and HOT coding
                        if frames_without_drop>0 and args.reset_period>0 and frames_without_drop%args.reset_period==0:
                            reset_e2p_state(args,model)
                    else:  # firenet, we need to extract the 4 angle channels and run firenet on each one

                        intensity, aolp, dolp, aolp_mask = compute_firenet_output(model,input, states)

                with Timer('show output frame'):
                    if queue: # make DVS frame to show from float of last voxel frame
                        frame_255=(255*((voxel_five_float32[0,-1,:,:].squeeze())+1)/2).astype(np.uint8)
                        pass

                    dvs = cv2.cvtColor(frame_255[::2,::2],cv2.COLOR_GRAY2RGB)
                    # dvs = np.repeat(frame_255[::2,::2], 3, axis=1).astype(np.uint8)  # need to dup channels and reduce res to display together



                    if playback_file:
                        mycv2_put_text(dvs, f'DVS {playback_frame_duration*1e3:.1f}ms@{playback_current_time:.2f}s')
                    else:
                        mycv2_put_text(dvs, 'DVS')
                    mycv2_put_text(intensity, 'intensity')
                    mycv2_put_text(aolp, 'AoLP')
                    mycv2_put_text(dolp, 'DoLP')

                    image = cv2.vconcat([cv2.hconcat([dvs,intensity]),cv2.hconcat([aolp, dolp])])
                    show_frame(image, 'polarization', cv2_resized)
                    if recording_activated:
                        recording_frame_number = write_next_image(recording_folder_current, recording_frame_number, image)
                        print('.', end='')
                        if recording_frame_number % 80 == 0:
                            print('')

    cv2.destroyAllWindows()

    # calculate the computational efficiency
    if args.calculate_model_flops_size:
        flops, params = profile(model, inputs=(input,))
        flops, params = clever_format([flops, params], "%.3f")
        print(model)
        print('[Model Ops/Size Information]\nFLOPs: {}\nParams: {}'.format(flops, params))

    print_timing_info()
    print(f'****** done running model {checkpoint_path}')
from train.utils.henri_compatible import make_henri_compatible
from train.parse_config import ConfigParser

from multiprocessing import Queue



sys.path.append('train') # needed to get model to load using torch.load with train.parse_config ConfigParser.. don't understand why
################### end of main

def reset_e2p_state(args, model):
    """ Reset the internal hidden states of E2P."""
    if args.e2p:
        log.debug('resetting E2P state')
        model.reset_states_i()
        model.reset_states_a()
        model.reset_states_d()


def get_timestr():
    """ Get a timestamp str for recording folder names."""
    timestr = time.strftime("%Y%m%d-%H%M%S")
    return timestr

def print_key_help(args):
    """Print keyboard help."""
    print(f'**********************\nproducer keys to use in cv2 image window:\n'
          f'- or =: decrease or increase the AoLP DoLP mask level which is currently {args.dolp_aolp_mask_level}'
          'h or ?: print this help\n'
          'p: print timing info\n'
          'o: open recording numpy file to play back\n'
          's or f: slow down (briefer frames) or speed up (longer frames) playback'
          'b or d: brighter or darker frames\n'
          'r: toggle recording on/off; see console output for timestamped output folder\n'
          f'e: reset E2P hidden state; it is currently reset every {args.reset_period} frames by --reset_period argument\n'
          'space: toggle pause\n'
          # 'm: toggle between e2p and firenet models\n'
          'q or x: exit\n*****************************')


def legacy_compatibility(args, checkpoint):
    """ Modify checkpoint so that all model types have same signature."""
    assert args.e2p and not (args.e2vid and args.firenet_legacy) # TODO clean up options
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


def load_e2p_model(checkpoint, device):
    """ Load the E2P model checkpoint and put on particular device (cuda or cpu)"""
    config = checkpoint['config']
    log.info(f'configuration is {config["arch"]}')
    state_dict = checkpoint['state_dict']
    model_info = {}

    try:
        model_info['num_bins'] = config['arch']['args']['unet_kwargs']['num_bins']
        log.info(f'number of voxel bins (frames sent at each timestamp to model) is {model_info["num_bins"]}')
    except KeyError:
        model_info['num_bins'] = config['arch']['args']['num_bins']
    # logger = config.get_logger('test')

    # build model architecture
    model = config.init_obj('arch', model_arch)
    # log.info(model)
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    log.debug('Loading state dictionary ....')
    model.load_state_dict(state_dict)
    log.debug('Loading my trained weights succeeded!')

    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return model


def compute_firenet_output(model,input,states):
    output, states = model(input,
                           states)  # TODO not correct, should build voxel grid for each polarization channel and run on each independently

    # original firenet
    image = torch2numpy(output)
    image = np.clip(image, 0, 1)
    i0 = image[1::2, 1::2]
    i45 = image[0::2, 1::2]
    i90 = image[0::2, 0::2]
    i135 = image[1::2, 0::2]
    s0 = i0.astype(float) + i90.astype(float)
    s1 = i0.astype(float) - i90.astype(float)
    s2 = i45.astype(float) - i135.astype(float)

    intensity = s0 / 2
    aolp = 0.5 * np.arctan2(s2, s1)
    aolp = aolp + 0.5 * math.pi
    aolp = aolp / math.pi
    dolp = np.divide(np.sqrt(np.square(s1) + np.square(s2)), s0, out=np.zeros_like(s0).astype(float), where=s0 != 0)
    intensity = numpy2cv2(intensity)
    aolp = numpy2cv2(aolp)
    dolp = numpy2cv2(dolp)

    max = np.max(intensity)
    min = np.min(intensity)
    intensity = (intensity - min) / (max - min) * 255
    intensity = np.repeat(intensity[:, :, None], 3, axis=2).astype(np.uint8)
    aolp = cv2.applyColorMap(aolp, cv2.COLORMAP_HSV)
    dolp = cv2.applyColorMap(dolp, cv2.COLORMAP_HOT)

    return intensity, aolp, dolp


def norm_max_min(v):
    max = np.max(v)
    min = np.min(v)
    v = (v - min) / (max - min + 1e-5) * 255
    return v




def load_selected_model(args, device):
    """ Loads desired checkpoint E2P model, either from file browser or args. Put it on particular device (cuda or cpu)."""
    if args.browse_checkpoint:
        lastmodel=prefs.get('last_model_selected','models/*.pth')
        f = fileopenbox(msg='select model checkpoint', title='Model checkpoint', default=lastmodel,
                        filetypes=['*.pth'])
        if f is not None:
            prefs.put('last_model_selected',f)
            args.checkpoint_path = f
        else:
            log.warning('no model selected, exiting')
            quit(0)

    # initial network model
    if not args.use_firenet:
        checkpoint_path = E2P_MODEL if args.checkpoint_path is None else args.checkpoint_path
        p = Path(checkpoint_path)
        if not p.is_file():
            raise FileNotFoundError(f'model --checkpoint_path={args.checkpoint_path} does not exist. Maybe you used single quote in args? Use double quote.')
        log.info(f'loading checkpoint model path from {checkpoint_path} with torch.load()')
        from train.parse_config import ConfigParser
        checkpoint = torch.load(checkpoint_path)
        # args, checkpoint = legacy_compatibility(args, checkpoint)
        model = load_e2p_model(checkpoint,device)
    else:  # load firenet
        from rpg_e2vid.utils.loading_utils import load_model as load_firenet_model
        checkpoint_path = FIRENET_MODEL if args.checkpoint_path is None else args.checkpoint_path
        model = load_firenet_model(checkpoint_path)
        model = model.to(device)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    return model,checkpoint_path

def get_args():
    parser = argparse.ArgumentParser(description='consumer: Consumes DVS frames to process', allow_abbrev=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--sensor_resolution", type=tuple, default=SENSOR_RESOLUTION, help="sensor resolution as tuple (height, width)")
    parser.add_argument("--recording_folder", type=str, default=RECORDING_FOLDER, help=f"record DVS frames into folder {RECORDING_FOLDER}")
    parser.add_argument('--dolp_aolp_mask_level', type=float, default=DOLP_AOLP_MASK_LEVEL, help='level of DoLP below which to mask the AoLP value since it is likely not meaningful')
    parser.add_argument('--reset_period', type=int, default=E2P_RESET_PERIOD, help='The E2P state is reset every this many frames')
    parser.add_argument('--e2p', action='store_true', default=True, help='set required parameters to run events to polarity e2p DNN')
    parser.add_argument('--use_firenet', action='store_true', default=USE_FIRENET, help='use (legacy) firenet instead of e2p')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='path to latest checkpoint, if not specified, uses E2P_MODEL global')
    parser.add_argument('--browse_checkpoint', action='store_true', default=False, help='open file dialog to select model checkpoint')
    # parser.add_argument('--output_folder', default="/tmp/output", type=str, help='where to save outputs to')
    parser.add_argument('--device', default='0', type=str, help='indices of GPUs to enable')
    # parser.add_argument('--legacy_norm', action='store_true', default=False, help='Normalize nonzero entries in voxel to have mean=0, std=1 according to Rebecq20PAMI and Scheerlinck20WACV.If --e2vid or --firenet_legacy are set, --legacy_norm will be set to True (default False).')
    # parser.add_argument('--robust_norm', action='store_true', default=False, help='Normalize voxel')
    parser.add_argument('--calculate_model_flops_size', action='store_true', default=False, help='Calculate the model num parameters and FLOPs per frame.')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    consumer(None)
