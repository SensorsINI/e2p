"""
 @Time    : 17.12.22 11:33
 @Author  : Haiyang Mei, Tobi Delbruck
 @E-mail  : haiyang.mei@outlook.com, tobi@ini.uzh.ch
 
 @Project : pdavis_demo
 @File    : producer.py
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
from tqdm import tqdm

from globals_and_utils import *
from engineering_notation import EngNumber as eng # only from pip
import argparse
import psutil

from pyaer.davis import DAVIS
from pyaer import libcaer

from utils.get_logger import get_logger
log=get_logger(__name__)

import torch
from events_contrast_maximization.utils.event_utils import events_to_voxel_torch


def producer(args):
    """ produce frames for consumer
    :param record: record frames to a folder name record
    """
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_address = ('', PORT)
    recording_folder = None
    recording_frame_number = 0

    # arg parser args
    record = args.record
    spacebar_records = args.spacebar_records
    space_toggles_recording = args.space_toggles_recording
    flex_time_mode=args.flex_time_mode
    frame_duration_ms=args.frame_duration_ms
    frame_count_k_events=args.frame_count_k_events
    save_numpy_images=args.numpy
    num_bins=args.num_bins
    sensor_resolution=args.sensor_resolution

    warning_counter=0

    # open davis camera, set biases
    device = DAVIS(noise_filter=True)
    def cleanup():
        log.info('closing {}'.format(device))
        device.shutdown()
        cv2.destroyAllWindows()
        if recording_folder is not None:
            log.info(f'*** recordings of {recording_frame_number - 1} frames saved in {recording_folder}')
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
    print("Logic Version--checkpoint_path=models/checkpoint-epoch106.pth:", device.logic_version)
    print("Background Activity Filter:",
          device.dvs_has_background_activity_filter)
    print("Color Filter", device.aps_color_filter, type(device.aps_color_filter))
    print(device.aps_color_filter == 1)
    # device.start_data_stream()
    assert (device.send_default_config())
    # attempt to set up USB host buffers for acquisition thread to minimize latency
    # assert (device.set_config(
    #     libcaer.CAER_HOST_CONFIG_USB,
    #     libcaer.CAER_HOST_CONFIG_USB_BUFFER_NUMBER,
    #     8))
    # assert (device.set_config(
    #     libcaer.CAER_HOST_CONFIG_USB,
    #     libcaer.CAER_HOST_CONFIG_USB_BUFFER_SIZE,
    #     64000))
    assert (device.data_start())
    # assert (device.set_config(
    #     libcaer.CAER_HOST_CONFIG_PACKETS,
    #     libcaer.CAER_HOST_CONFIG_PACKETS_MAX_CONTAINER_INTERVAL,
    #     10000)) # set max interval to this value in us. Set to not produce too many packets/sec here, not sure about reasoning
    assert (device.set_data_exchange_blocking())

    # setting bias after data stream started
    log.info(f'setting biases from {BIASES_CONFIG_FILE}')
    device.set_bias_from_json(BIASES_CONFIG_FILE)
    biases_config_file_path=Path(BIASES_CONFIG_FILE)
    biases_mtime=biases_config_file_path.stat().st_mtime  # get modification time of config



    if record is not None and space_toggles_recording and spacebar_records:
        log.error('set either --spacebar_records or --space_toggles_recording')
        quit(1)
    if record is not None:
        log.info(f'recording to {record} with spacebar_records={spacebar_records} space_toggles_recording={space_toggles_recording} and args {str(args)}')
        recording_folder = os.path.join(DATA_FOLDER, 'recordings', record)
        log.info(f'recording frames to {recording_folder}')
        Path(recording_folder).mkdir(parents=True, exist_ok=True)


    xfac = float(IMSIZE) / device.dvs_size_X
    yfac = float(IMSIZE) / device.dvs_size_Y
    histrange = [(0, v) for v in (IMSIZE, IMSIZE)]  # allocate DVS frame histogram to desired output size
    npix = IMSIZE * IMSIZE
    cv2_resized = False
    last_cv2_frame_time = time.time()
    # frame=None
    frame_number=0
    time_last_frame_sent=time.time()
    frames_dropped_counter=0
    recording_activated=False
    save_next_frame=(not space_toggles_recording and not spacebar_records) # if we don't supply the option, it will be False and we want to then save all frames
    saved_events=[]

    print_key_help()
    try:
        timestr = time.strftime("%Y%m%d-%H%M")
        numpy_frame_rate_data_file_path = f'{DATA_FOLDER}/producer-frame-rate-{timestr}.npy'
        while True:

            # check if biases config file changed, if so apply it
            new_biases_mtime = biases_config_file_path.stat().st_mtime  # get modification time of config
            if new_biases_mtime>biases_mtime:
                log.info(f'bias config file change detected, reloading {BIASES_CONFIG_FILE}')
                device.set_bias_from_json(BIASES_CONFIG_FILE)
                biases_mtime=new_biases_mtime

            with Timer('overall producer frame rate', numpy_file=numpy_frame_rate_data_file_path , show_hist=True, savefig=True) as timer_overall:
                with Timer('accumulating DVS events'):
                    events = None
                    # while events is None or duration of collected events is less than desired keep accumulating events
                    while is_frame_not_complete(events, flex_time_mode, frame_duration_ms, frame_count_k_events): # 0 is index of timestamp
                        pol_events, num_pol_event,_, _, _, _, _, _ = device.get_event()
                        if num_pol_event>0:
                            if events is None:
                                events=pol_events
                            else:
                                events = np.vstack([events, pol_events]) # otherwise tack new events to end
                        time.sleep(0.001) # yield

                # log.debug('got {} events (total so far {}/{} events)'
                #          .format(num_pol_event, 0 if events is None else len(events), EVENT_COUNT))
                dtMs = (time.time() - time_last_frame_sent)*1e3
                if recording_folder is None and dtMs<MIN_PRODUCER_FRAME_INTERVAL_MS:
                    if warning_counter<WARNING_INTERVAL or warning_counter%WARNING_INTERVAL==0:
                        warning_counter+=1
                        log.debug(f'frame #{frames_dropped_counter} after only {dtMs:.3f}ms, discarding to avoid flooding consumer and instead to collect newer frame')
                    frames_dropped_counter+=1
                    continue # just collect another frame since it will be more timely
                if frames_dropped_counter>0:
                    if warning_counter<WARNING_INTERVAL or warning_counter%WARNING_INTERVAL==0:
                        warning_counter+=1
                        log.warning(f'after dropping {frames_dropped_counter} frames, got one after {dtMs:.1f}ms')
                if save_numpy_images:
                    if saved_events is None:
                        saved_events = pol_events
                    else:
                        saved_events.append(events)
                        # saved_events = np.vstack([saved_events, events])
                        # if psutil.virtual_memory().available < 1e9:
                        #     log.warning('available RAM too low, turning off numpy data saving')

                frames_dropped_counter=0

                # naive integration
                # with Timer('accum/norm frame from events'):
                #     # if frame is None: # debug timing
                #         # take DVS coordinates and scale x and y to output frame dimensions using flooring math
                #         # xs=np.floor(events[:,1]*xfac)
                #         # ys=np.floor(events[:,2]*yfac)
                #         xs=events[:,1]
                #         ys=events[:,2]
                #         ts=events[:,0]
                #         frame, _, _ = np.histogram2d(ys, xs, bins=(IMSIZE, IMSIZE), range=histrange)
                #         # frame, _, _ = np.histogram2d(ys, xs, bins=(257, 255), range=histrange)
                #         # fmax_count=np.max(frame) # todo check if fmax is frequenty exceeded, increase contrast
                #         frame[frame > args.clip_count]=args.clip_count
                #         frame= ((255. / args.clip_count) * frame).astype('uint8') # max pixel will have value 255

                # voxelization for network inference
                with Timer('computing voxels from events'):
                    xs = torch.from_numpy(events[:, 1].astype(np.float32))
                    ys = torch.from_numpy(events[:, 2].astype(np.float32))
                    ts = torch.from_numpy((events[:, 0] - events[0, 1]).astype(np.float32))
                    ps = torch.from_numpy((events[:, 3] * 2 - 1).astype(np.float32))
                    voxel = events_to_voxel_torch(xs, ys, ts, ps, num_bins, sensor_size=sensor_resolution, temporal_bilinear=True) # TODO temporal_bilinear=False broken
                    voxel = (((voxel + 1) / 2) * 255)
                    voxel = voxel[:, 0:IMSIZE, 0:IMSIZE] # crop out UL corner from entire voxel frame to limit to max possible UDP packet size
                    voxel_224 = voxel.numpy().astype('uint8')
                    # The DNN is trained with 112x112 but can test on 346x260. We crop it to 224x224 to enable UDP transfer otherwise it will be too large.

                with Timer('sending voxels to consumer'):
                    time_last_frame_sent=time.time()
                    # data = pickle.dumps((frame_number, time_last_frame_sent, voxel[0, :, :])) # send frame_number to allow determining dropped frames in consumer
                    # data = pickle.dumps((frame_number, time_last_frame_sent, frame)) # send frame_number to allow determining dropped frames in consumer
                    for x in range(voxel_224.shape[0]): # send bin by bin (really frame by frame) to consumer, each one is 224x224 bytes which is about 50kB, OK for UDP
                        data = pickle.dumps((frame_number, time_last_frame_sent, x, voxel_224[x, :, :]))
                        if len(data)>64000:
                            log.error(f'UDP packet {data} is too large')
                        frame_number+=1
                        client_socket.sendto(data, udp_address)

                if recording_folder is not None and (save_next_frame or recording_activated):
                    recording_frame_number=write_next_image(recording_folder, recording_frame_number, voxel_224)
                    print('.', end='')
                    if recording_frame_number%80==0:
                        print('')

                if SHOW_DVS_OUTPUT:
                    t=time.time()
                    if t-last_cv2_frame_time>1./MAX_SHOWN_DVS_FRAME_RATE_HZ:
                        last_cv2_frame_time=t
                        with Timer('show voxel image'):
                            # min = np.min(frame)
                            # img = ((frame - min) / (np.max(frame) - min))
                            cv2.namedWindow('DVS', cv2.WINDOW_NORMAL)
                            voxel_224_01 = voxel_224 / 255.0
                            # pad = np.zeros([IMSIZE, 20])
                            # voxel_five = cv2.hconcat([voxel_224_01[0], pad, voxel_224_01[1], pad, voxel_224_01[2], pad,
                            #                           voxel_224_01[3], pad, voxel_224_01[4]])
                            cv2.imshow('DVS', voxel_224_01[-1]) # just show last frame
                            if not cv2_resized:
                                cv2.resizeWindow('DVS', IMSIZE*2, IMSIZE*2)
                                cv2_resized = True
                            k = cv2.waitKey(1) & 0xFF
                            if k == ord('q') or k == ord('x'):
                                if recording_folder is not None:
                                    log.info(f'*** recordings of {recording_frame_number - 1} frames saved in {recording_folder}')
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
                            elif k == ord(' ') and (spacebar_records or space_toggles_recording):
                                if spacebar_records:
                                    save_next_frame=True
                                else:
                                    recording_activated=not recording_activated
                                    if recording_activated:
                                        recordprint('recording activated - use space to stop recording\n')
                                    else:
                                        print('recording paused - use space to start recording\n')
                                    save_next_frame=recording_activated
                            elif k==ord('h') or k==ord('?'):
                                print_key_help()
                            else:
                                save_next_frame=(recording_activated or (not spacebar_records and not space_toggles_recording))
        if saved_events is not None and recording_folder is not None and len(saved_events)>0:
            nevents=0
            for a in saved_events:
                nevents+=len(a)
            o=np.empty((nevents,5),dtype=np.float32)
            idx=0
            for a in tqdm(saved_events,desc='converting events to numpy'):
                o[idx:idx+a.shape[0]]=a
                idx+=a.shape[0]
            data_path=os.path.join(recording_folder,f'events-{timestr}.npy')
            log.info(f'saving {eng(nevents)} events to {data_path}')
            np.save(data_path,o)
    except KeyboardInterrupt as e:
        log.info(f'got KeyboardInterrupt {e}')
        cleanup()

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
          'space: toggle or turn on while space down recording\n'
          'q or x or ESC: exit')

def get_args():
    parser = argparse.ArgumentParser(
        description='producer: Generates DVS frames for pdavis_demo to process in consumer', allow_abbrev=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--record", type=str, default=None,
        help=f"record DVS frames into folder {os.path.join(DATA_FOLDER, 'recordings','<name>')}")
    parser.add_argument(
        "--flex_time_mode", type=bool, default=FLEX_TIME_MODE,
        help="True to use frame_count_k_events input, False to use frame_duration_ms")
    parser.add_argument(
        "--frame_duration_ms", type=int, default=FRAME_DURATION_US/1000,
        help="duration of frame exposure per total voxel volume")
    parser.add_argument(
        "--frame_count_k_events", type=int, default=FRAME_COUNT_EVENTS/1000,
        help="duration of frame exposure per total voxel volume")
    parser.add_argument(
        "--num_bins", type=int, default=NUM_BINS,
        help="number of bins for event voxel")
    parser.add_argument(
        "--sensor_resolution", type=tuple, default=SENSOR_RESOLUTION,
        help="sensor resolution")
    parser.add_argument(
        "--spacebar_records", action='store_true',
        help="only record when spacebar pressed down")
    parser.add_argument(
        "--space_toggles_recording", action='store_true', default=True,
        help="space toggles recording on/off")
    parser.add_argument(
        "--numpy", action='store_true', default=True,
        help="saves raw AE data to RAM and writes as numpy at the end (will gobble RAM like crazy)")
    parser.add_argument('--e2p', action='store_true', default=True,
                        help='set required parameters to run events to polarity e2p DNN')
    parser.add_argument('--e2vid', action='store_true', default=False,
                        help='set required parameters to run original e2vid as described in Rebecq20PAMI for polariziation reconstruction')
    parser.add_argument('--firenet_legacy', action='store_true', default=False,
                        help='set required parameters to run legacy firenet as described in Scheerlinck20WACV (not for retrained models using updated code)')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args=get_args()
    producer(args)
