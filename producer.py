"""
 @Time    : 28.3.23
 @Author  : Tobi Delbruck, Haiyang Mei, Zoowen Wang
 @E-mail  : tobi@ini.uzh.ch

 @Project : pdavis_demo
 @File    : consumer.py
 @Function:

"""

import atexit
import pickle
from pathlib import Path

import cv2
import sys
import math
import time
import numpy.ma as ma
import socket
import numpy as np
# from tqdm import tqdm
import select
import multiprocessing.connection as mpc
from multiprocessing import  Pipe,Queue

from tqdm import tqdm

from globals_and_utils import *
from engineering_notation import EngNumber as eng # only from pip
import argparse
# import psutil
import torch

from pyaer.davis import DAVIS
from pyaer import libcaer

from utils.get_logger import get_logger
import desktop # tobi's patch to support xdg-open https://github.com/eight04/pyDesktop3/issues/5 never pulled, using local copy of desktop package developed for v2e project

log=get_logger(__name__)

# from events_contrast_maximization.utils.event_utils import events_to_voxel_torch  # WARNING: this function is not the same one used for e2p training

from train.events_contrast_maximization.utils.event_utils import events_to_voxel_torch  # This one is the same as used for e2p training


def producer(queue:Queue):
    """ produce frames for consumer
    :param queue: if started with a queue, uses that for sending voxel volume
    """
    args=get_args()
    if queue is None:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_address = ('', PORT)
        args.sensor_resolution=(IMSIZE,IMSIZE) # if we use UDP, we need to limit UDP packet size
    else:
        args.sensor_resolution=SENSOR_RESOLUTION # we can send the whole voxel volume over Pipe

    # arg parser args
    recording_folder_base = args.recording_folder
    recording_folder_current=None
    flex_time_mode=args.flex_time_mode
    frame_duration_ms=args.frame_duration_ms
    frame_count_k_events=args.frame_count_k_events
    save_numpy=args.save_numpy
    num_bins=args.num_bins
    sensor_resolution=args.sensor_resolution

    recording_frame_number = 0
    warning_counter=0
    paused=False

    # open davis camera, set biases
    log.info('opening PDAVIS camera')
    device = DAVIS(noise_filter=True)
    def cleanup():
        log.info('closing {}'.format(device))
        device.shutdown()
        cv2.destroyAllWindows()
        if recording_folder_base is not None and recording_frame_number>0:
            log.info(f'*** recordings of {recording_frame_number - 1} frames saved in {recording_folder_base}')
            desktop.open(recording_folder_base)
    atexit.register(cleanup)

    print("DVS USB ID:", device.device_id)
    if device.device_is_master:
        print("DVS is master.")
    else:
        print("DVS is slave.")
    print("DVS Serial Number:", device.device_serial_number)
    print("DVS String:", device.device_string)
    print("DVS USB bus Number:", device.device_usb_bus_number)
    print("DVS USB device address:", device.device_usb_device_address)
    print("DVS size X:", device.dvs_size_X)
    print("DVS size Y:", device.dvs_size_Y)
    dvs_resolution=(device.dvs_size_Y,device.dvs_size_X)
    print("Logic Version--checkpoint_path=models/checkpoint-epoch106.pth:", device.logic_version)
    print("Background Activity Filter:",
          device.dvs_has_background_activity_filter)
    print("Color Filter", device.aps_color_filter, type(device.aps_color_filter))
    print(device.aps_color_filter == 1)
    # device.start_data_stream()
    assert (device.send_default_config())
    # following buffer size/number commands fail on WSL2, nothing comes from DAVIS
    # attempt to set up USB host buffers for acquisition thread to minimize latency
    # assert (device.set_config(
    #     libcaer.CAER_HOST_CONFIG_USB,
    #     libcaer.CAER_HOST_CONFIG_USB_BUFFER_NUMBER,
    #     8))
    # assert (device.set_config(
    #     libcaer.CAER_HOST_CONFIG_USB,
    #     libcaer.CAER_HOST_CONFIG_USB_BUFFER_SIZE,
    #     64000))
    # assert (device.set_config(
    #     libcaer.CAER_HOST_CONFIG_PACKETS,
    #     libcaer.CAER_HOST_CONFIG_PACKETS_MAX_CONTAINER_INTERVAL,
    #     10000)) # set max interval to this value in us. Set to not produce too many packets/sec here, not sure about reasoning
    assert (device.data_start())
    assert (device.set_data_exchange_blocking())

    # setting bias after data stream started
    log.info(f'setting biases from {BIASES_CONFIG_FILE}')
    device.set_bias_from_json(BIASES_CONFIG_FILE)
    biases_config_file_path=Path(BIASES_CONFIG_FILE)
    biases_mtime=biases_config_file_path.stat().st_mtime  # get modification time of config


    # if recording_folder is not None:
    #     log.info(f'will record to {recording_folder}')
    #     Path(recording_folder).mkdir(parents=True, exist_ok=True)

    cv2_resized = False
    last_cv2_frame_time = time.time()
    # frame=None
    frame_number=0
    time_last_frame_sent=time.time()
    frames_dropped_counter=0
    recording_activated=False
    saved_events=[]

    #https://stackoverflow.com/questions/45318798/how-to-detect-multiprocessing-pipe-is-full
    def pipe_full(conn):
        r, w, x = select.select([], [conn], [], 0.0)
        return 0 == len(w)


    print_key_help()
    try:
        printed_udp_size=False
        while True:

            # check if biases config file changed, if so apply it
            new_biases_mtime = biases_config_file_path.stat().st_mtime  # get modification time of config
            if new_biases_mtime>biases_mtime:
                log.info(f'bias config file change detected, reloading {BIASES_CONFIG_FILE}')
                try:
                    device.set_bias_from_json(BIASES_CONFIG_FILE)
                except Exception as e:
                    log.warning(f'formatting error in biases file {BIASES_CONFIG_FILE}: {str(e)}')
                biases_mtime=new_biases_mtime

            with Timer('overall producer frame rate') as timer_overall:
                with Timer('accumulating DVS events'):
                    events = None
                    # while events is None or duration of collected events is less than desired keep accumulating events
                    while is_frame_not_complete(events, flex_time_mode, frame_duration_ms, frame_count_k_events): # 0 is index of timestamp
                        pol_events, num_pol_event,_, _, _, _, _, _ = device.get_event()
                        if num_pol_event>0:
                            if events is None:
                                events=pol_events # events[:,x] where for x 0 is time, 1 and 2 are x and y, 3 is polarity (OFF,ON)=(0,1) values, 4 is filtered status when noise filtering enabled
                            else:
                                events = np.vstack([events, pol_events]) # otherwise tack new events to end
                        time.sleep(0.001) # yield

                # log.debug('got {} events (total so far {}/{} events)'
                #          .format(num_pol_event, 0 if events is None else len(events), EVENT_COUNT))
                dtMs = (time.time() - time_last_frame_sent)*1e3
                if recording_folder_base is None and dtMs<MIN_PRODUCER_FRAME_INTERVAL_MS:
                    if warning_counter<WARNING_INTERVAL or warning_counter%WARNING_INTERVAL==0:
                        warning_counter+=1
                        log.debug(f'frame )#{frames_dropped_counter} after only {dtMs:.3f}ms, discarding to avoid flooding consumer and instead to collect newer frame')
                    frames_dropped_counter+=1
                    continue # just collect another frame since it will be more timely
                if frames_dropped_counter>0:
                    if warning_counter<WARNING_INTERVAL or warning_counter%WARNING_INTERVAL==0:
                        warning_counter+=1
                        log.warning(f'after dropping {frames_dropped_counter} frames, got one after {dtMs:.1f}ms')
                if save_numpy:
                    if saved_events is None:
                        saved_events = pol_events
                    else:
                        saved_events.append(events)
                        print('.',end='')
                        # saved_events = np.vstack([saved_events, events])
                        # if psutil.virtual_memory().available < 1e9:
                        #     log.warning('available RAM too low, turning off numpy data saving')

                frames_dropped_counter=0

                # voxelization for network inference
                with Timer('computing voxels from events'):
                    # events[:,x] where for x, 0 is time, 1 and 2 are x and y, 3rd is polarity ON/OFF
                    xs = torch.from_numpy(events[:, 1].astype(np.float32)) # event x addreses
                    ys = torch.from_numpy(events[:, 2].astype(np.float32)) # event y addresses
                    ts = torch.from_numpy((events[:, 0] - events[0, 0]).astype(np.float32)) # event relative timesamps in us
                    ps = torch.from_numpy((events[:, 3] * 2 - 1).astype(np.float32)) # change polarity from 0,1 to -1,+1 so that polarities are signed
                    voxel = events_to_voxel_torch(xs, ys, ts, ps, num_bins, sensor_size=dvs_resolution, temporal_bilinear=True) # TODO temporal_bilinear=False broken
                    # The DNN is trained with 112x112 but can test on 346x260. We crop it to 224x224 to enable UDP transfer otherwise it will be too large.
                    voxel = voxel[:, 0:args.sensor_resolution[0], 0:args.sensor_resolution[1]] # crop out UL corner from entire voxel frame to limit to max possible UDP packet size
                    pass

                with Timer('sending voxels to consumer'):
                    if queue:
                        voxel_4d=np.expand_dims(voxel.numpy(),0) # need to expand to 4d for input to DNN
                        # log.debug(f'sending entire voxel volume on pipe with shape={voxel_4d.shape}')
                        if not queue.empty() and warning_counter < WARNING_INTERVAL or warning_counter % WARNING_INTERVAL == 0:
                            log.warning('queue is full, cannot send voxel volume')
                            warning_counter+=1
                        else:
                            frame_number+=1
                            time_last_frame_sent=time.time()
                            # following blocks until frame can be put on Pipe... it means if the source frame rate is too high, the consumer
                            # will not drop frames but just lag very far behind
                            queue.put((voxel_4d, frame_number, time_last_frame_sent))
                        # log.debug('sent entire voxel volume on pipe')
                    else:
                        # data = pickle.dumps((frame_number, time_last_frame_sent, voxel[0, :, :])) # send frame_number to allow determining dropped frames in consumer
                        # data = pickle.dumps((frame_number, time_last_frame_sent, frame)) # send frame_number to allow determining dropped frames in consumer
                        frame_float=voxel.numpy()
                        frame_min=np.min(frame_float, axis=(1,2))[:, np.newaxis, np.newaxis] # get the min value per channel/bin, shape should be (bin,1,1)
                        frame_max=np.max(frame_float, axis=(1,2))[:, np.newaxis, np.newaxis]
                        frame_255=(((frame_float-frame_min)/(frame_max-frame_min))*255).astype(np.uint8) # do per channel normalization to [0,1], then scale to [0,255]

                        for bin in range(NUM_BINS): # send bin by bin (really frame by frame) to consumer, each one is 224x224 bytes which is about 50kB, OK for UDP
                            frame_number+=1
                            time_last_frame_sent=time.time()
                            data = pickle.dumps((frame_number, time_last_frame_sent, bin, frame_255[bin],frame_min[bin],frame_max[bin]))
                            if not printed_udp_size:
                                if len(data)>64000:
                                    raise ValueError(f'UDP packet with length {len(data)} is too large')
                                else:
                                    printed_udp_size=True
                                    log.info(f'UDP packet size for first frame is {len(data)} bytes')
                            client_socket.sendto(data, udp_address)


                if SHOW_DVS_OUTPUT:
                    t=time.time()
                    if t-last_cv2_frame_time>1./MAX_SHOWN_DVS_FRAME_RATE_HZ:
                        last_cv2_frame_time=t
                        with Timer('show voxel image'):
                            # min = np.min(frame)
                            # img = ((frame - min) / (np.max(frame) - min))
                            cv2.namedWindow('DVS', cv2.WINDOW_NORMAL)
                            if queue: # we need to render from last frame of voxel volume here since we just sent the whole thing over pipe as float
                                frame_float = voxel.numpy()[-1]
                                frame_min = np.min(frame_float)
                                frame_max = np.max(frame_float)
                                frame_255 = (((frame_float - frame_min) / (frame_max - frame_min)) * 255).astype(
                                    np.uint8)
                                cv2.imshow('DVS', frame_255)
                            else:
                                cv2.imshow('DVS', frame_255[-1]) # just show last frame
                            if not cv2_resized:
                                cv2.resizeWindow('DVS', args.sensor_resolution[1]*2, args.sensor_resolution[0]*2)
                                cv2_resized = True

                            # process key commands
                            k = cv2.waitKey(1) & 0xFF
                            if k == ord('q') or k == ord('x'):
                                if recording_folder_base is not None:
                                    log.info(f'*** recordings of {recording_frame_number - 1} frames saved in {recording_folder_base}')
                                print_timing_info()
                                break
                            elif k==ord('t'):
                                flex_time_mode=not flex_time_mode
                                print(f'toggled flex time to {flex_time_mode}')
                                if flex_time_mode:
                                    print(f'frames are {frame_count_k_events}k events')
                                else:
                                    print(f'frames are {frame_duration_ms}ms long')

                            elif k==ord('f'):
                                if flex_time_mode:
                                    frame_count_k_events=decrease(frame_count_k_events, 4)
                                    print(f'shorter frames now are {frame_count_k_events}k events')
                                else:
                                    frame_duration_ms=decrease(frame_duration_ms,4)
                                    print(f'shorter frames now are {frame_duration_ms}ms long')
                            elif k==ord('s'):
                                if flex_time_mode:
                                    frame_count_k_events = increase(frame_count_k_events, 200)
                                    print(f'longer frames now are {frame_count_k_events}k events')
                                else:
                                    frame_duration_ms = increase(frame_duration_ms,200)
                                    print(f'longer frames now are {frame_duration_ms}ms long')
                            elif k == ord('p'):
                                print_timing_info()
                            elif k == ord('r'):
                                if not recording_activated:
                                    recording_activated=True
                                    recording_folder_current=os.path.join(recording_folder_base,get_timestr())
                                    Path(recording_folder_current).mkdir(parents=True, exist_ok=True)
                                    log.info(f'started recording PNG frames to folder {recording_folder_current}')
                                else:
                                    recording_activated=False
                                    log.info(f'stopped recording PNG frames to folder {recording_folder_current}')
                            elif k==ord('l'): # numpy file of events saved at the end
                                save_numpy= not save_numpy
                                if save_numpy:
                                    log.info(f'started saving events to RAM')
                                else:
                                    log.info(f'writing saved events to numpy file...')
                                    try:
                                        save_events_to_numpy(recording_folder_base, saved_events)
                                    except Exception as e:
                                        log.error(f'could not save events: {e}')

                            elif k==ord(' '):
                                paused=not paused
                                print(f'paused={paused}')
                            elif k==ord('h') or k==ord('?'):
                                print_key_help()
                            elif k!=255:
                                log.warning(f'unknown keystroke {k}')
                    if recording_activated:
                        recording_frame_number = write_next_image(recording_folder_current, recording_frame_number,
                                                                  frame_255[-1])
                        print('.', end='')
                        if recording_frame_number % 80 == 0:
                            print('')

    except KeyboardInterrupt as e:
        log.info(f'got KeyboardInterrupt {e}')
        cleanup()


def save_events_to_numpy(recording_folder_base, saved_events):
    if saved_events is not None and recording_folder_base is not None and len(saved_events) > 0:
        nevents = 0
        for a in saved_events:
            nevents += len(a)
        o = np.empty((nevents, 5), dtype=np.float32)
        idx = 0
        for a in tqdm(saved_events, desc='converting events to numpy'):
            o[idx:idx + a.shape[0]] = a
            idx += a.shape[0]
        data_path = os.path.join(recording_folder_base, f'events-{get_timestr()}.npy')
        log.info(f'saving {eng(nevents)} events to {data_path}')
        np.save(data_path, o)
        desktop.open(recording_folder_base) # if skype opens on gnome, see https://www.reddit.com/r/linuxquestions/comments/gxsqt3/skype_somehow_inserted_itself_into_xdgopen_and_i/


def get_timestr():
    timestr = time.strftime("%Y%m%d-%H%M%S")
    return timestr


def increase(val,limit):
    return val*2 if val*2<=limit else limit
def decrease(val,limit):
    return val/2 if val/2>=limit else limit

def is_frame_not_complete(events, flex_time_mode, frame_duration_ms, frame_count_k_events):
    if events is None:
        return True

    if not flex_time_mode:
        dtFrameUs = (events[-1, 0] - events[0, 0])
        return dtFrameUs < frame_duration_ms * 1000
    else:
        eventCount=events.shape[0]
        return eventCount<frame_count_k_events*1000

def print_key_help():
    print('producer keys to use in cv2 image window:\n'
          'h or ?: print this help\n'
          'p: print timing info\n'
          't: toggle flex time (constant-duration / constant-count frames)\n'
          'f or s: faster or slower frames (less duration or count vs more)'
          'r: toggle recording PNG frames\n'
          'l: toggle saving events to write numpy file at the end\n'
          'q or x or ESC: exit')

def get_args():
    parser = argparse.ArgumentParser(description='producer: Generates DVS frames for pdavis_demo to process in consumer', allow_abbrev=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--recording_folder", type=str, default=RECORDING_FOLDER, help=f"record DVS frames into folder {RECORDING_FOLDER}")
    parser.add_argument("--flex_time_mode", type=bool, default=FLEX_TIME_MODE, help="True to use frame_count_k_events input, False to use frame_duration_ms")
    parser.add_argument("--frame_duration_ms", type=int, default=FRAME_DURATION_US/1000, help="duration of frame exposure per total voxel volume")
    parser.add_argument("--frame_count_k_events", type=int, default=FRAME_COUNT_EVENTS/1000, help="duration of frame exposure per total voxel volume")
    parser.add_argument("--num_bins", type=int, default=NUM_BINS, help="number of bins for event voxel")
    parser.add_argument("--sensor_resolution", type=tuple, default=SENSOR_RESOLUTION, help="sensor resolution as tuple (height, width)")
    parser.add_argument("--save_numpy", action='store_true', default=False, help="saves raw AE data to RAM and writes as numpy at the end (will gobble RAM like crazy)")
    parser.add_argument('--e2p', action='store_true', default=True, help='set required parameters to run events to polarity e2p DNN')
    parser.add_argument('--e2vid', action='store_true', default=False, help='set required parameters to run original e2vid as described in Rebecq20PAMI for polariziation reconstruction')
    parser.add_argument('--firenet_legacy', action='store_true', default=False, help='set required parameters to run legacy firenet as described in Scheerlinck20WACV (not for retrained models using updated code)')
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    producer(None)
