"""
 @Time    : 12.12.22 10:21
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : consumer.py
 @Function:
 
"""
"""
consumer of DVS frames for classification of joker/nonjoker by consumer processs
Authors: Tobi Delbruck, Shasha Guo, Yuhaung Hu, Min Liu, Oct 2020
"""
import argparse
import copy
import glob
import pickle
import cv2
import sys
import tensorflow as tf
# from keras.models import load_model
import serial
import socket
from select import select
from globals_and_utils import *
from engineering_notation import EngNumber  as eng # only from pip
import collections
from pathlib import Path
import random

from train import load_latest_model, classify_joker_img

log=my_logger(__name__)

# Only used in mac osx
try:
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
except Exception as e:
    print(e)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='consumer: Consumes DVS frames for trixy to process', allow_abbrev=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--serial_port", type=str, default=SERIAL_PORT,
        help="serial port, e.g. /dev/ttyUSB0")

    args = parser.parse_args()

    log.info('opening UDP port {} to receive frames from producer'.format(PORT))
    server_socket: socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    address = ("", PORT)
    server_socket.bind(address)
    model,interpreter, input_details, output_details,model_folder=load_latest_model()

    serial_port = args.serial_port
    log.info('opening serial port {} to send commands to finger'.format(serial_port))
    arduino_serial_port = serial.Serial(serial_port, 115200, timeout=5)

    log.info(f'Using UDP buffer size {UDP_BUFFER_SIZE} to recieve the {IMSIZE}x{IMSIZE} images')

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
    finger_out_time = 0
    STATE_IDLE = 0
    STATE_FINGER_OUT = 1
    state = STATE_IDLE

    def print_num_saved_images():
        log.info(f'saved {next_non_joker_index} nonjokers to {NONJOKERS_FOLDER} and {next_joker_index} jokers to {JOKERS_FOLDER}')

    atexit.register(print_num_saved_images)

    log.info('GPU is {}'.format('available' if len(tf.config.list_physical_devices('GPU')) > 0 else 'not available (check tensorflow/cuda setup)'))


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
    # receive_data=bytearray(UDP_BUFFER_SIZE)
    while True:
        timestr = time.strftime("%Y%m%d-%H%M")
        with Timer('overall consumer loop', numpy_file=f'{DATA_FOLDER}/consumer-frame-rate-{timestr}.npy', show_hist=True):
            with Timer('recieve UDP'):
                # num_bytes_recieved=0
                # receive_data=None
                # tries=0
                # while True: # read datagrams unti there are no more, so that we always get very latest one in our receive buffer
                #     inputready, _, _ = select([server_socket], [], [], .1)
                #     num_ready=len(inputready)
                #     if (r
                #     eceive_data is not None)  and (num_ready==0 or tries>2):
                #         # Has danger that as we recieve a datagram, another arrives, getting us stuck here.
                #         # Hence we break from loop only if  we have data AND (there is no more OR we already tried 3 times to empty the socket)
                #         break
                #     if num_ready>0:
                        receive_data = server_socket.recv(UDP_BUFFER_SIZE)
                    # tries+=1

            with Timer('unpickle and normalize/reshape'):
                (frame_number, timestamp, img255) = pickle.loads(receive_data)
                dropped_frames=frame_number-last_frame_number-1
                if dropped_frames>0:
                    log.warning(f'Dropped {dropped_frames} frames from producer')
                last_frame_number=frame_number
                img_01_float32 = (1. / 255) * np.array(img255, dtype=np.float32)
            with Timer('run CNN'):
                # pred = model.predict(img[None, :])
                is_joker, joker_prob, pred=classify_joker_img(img_01_float32, model, interpreter, input_details, output_details)


            if is_joker: # joker
                arduino_serial_port.write(b'1')
                finger_out_time=time.time()
                state=STATE_FINGER_OUT
                log.info('sent finger OUT')
            elif state==STATE_FINGER_OUT and time.time()-finger_out_time>FINGER_OUT_TIME_S:
                arduino_serial_port.write(b'0')
                state=STATE_IDLE
                log.info('sent finger IN')
            else:
                pass

            # save time since frame sent from producer
            dt=time.time()-timestamp
            with Timer('producer->consumer inference delay',delay=dt, show_hist=True):
                pass

            save_img_255= (img255.squeeze()).astype('uint8')
            if is_joker: # joker
                # find next name that is not taken yet
                next_joker_index= write_next_image(JOKERS_FOLDER, next_joker_index, save_img_255)
                show_frame(1-img_01_float32, 'joker', cv2_resized)
                non_joker_window_number=0
                for saved_img in saved_non_jokers:
                    next_non_joker_index= write_next_image(NONJOKERS_FOLDER, next_non_joker_index, saved_img)
                    show_frame(saved_img, f'nonjoker{non_joker_window_number}', cv2_resized)
                    non_joker_window_number+=1
                saved_non_jokers.clear()
            else:
                if random.random()<.03: # append random previous images to not just get previous almost jokers
                    saved_non_jokers.append(copy.deepcopy(save_img_255)) # we need to copy the frame otherwise the reference is overwritten by next frame and we just get the same frame over and over
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

