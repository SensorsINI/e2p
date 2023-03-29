"""
 @Time    : 17.12.22 21:26
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : pdavis_demo
 @File    : globals_and_utils.py
 @Function:
 
"""
""" Shared stuff between producer and consumer
 Author: Tobi Delbruck
 """
import logging
import math
import os
import sys
import time
from pathlib import Path
from subprocess import TimeoutExpired
import os
from typing import Optional
from utils.get_logger import get_logger
log=get_logger(__name__)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' # all TF messages

# import tensorflow as tf

import cv2
import numpy as np
import atexit
from engineering_notation import EngNumber  as eng  # only from pip
from matplotlib import pyplot as plt
import numpy as np
# https://stackoverflow.com/questions/35851281/python-finding-the-users-downloads-folder
import os
if os.name == 'nt':
    import ctypes
    from ctypes import windll, wintypes
    from uuid import UUID

    # ctypes GUID copied from MSDN sample code
    class GUID(ctypes.Structure):
        _fields_ = [
            ("Data1", wintypes.DWORD),
            ("Data2", wintypes.WORD),
            ("Data3", wintypes.WORD),
            ("Data4", wintypes.BYTE * 8)
        ]

        def __init__(self, uuidstr):
            uuid = UUID(uuidstr)
            ctypes.Structure.__init__(self)
            self.Data1, self.Data2, self.Data3, \
            self.Data4[0], self.Data4[1], rest = uuid.fields
            for i in range(2, 8):
                self.Data4[i] = rest>>(8-i-1)*8 & 0xff

    SHGetKnownFolderPath = windll.shell32.SHGetKnownFolderPath
    SHGetKnownFolderPath.argtypes = [
        ctypes.POINTER(GUID), wintypes.DWORD,
        wintypes.HANDLE, ctypes.POINTER(ctypes.c_wchar_p)
    ]

    def _get_known_folder_path(uuidstr):
        pathptr = ctypes.c_wchar_p()
        guid = GUID(uuidstr)
        if SHGetKnownFolderPath(ctypes.byref(guid), 0, 0, ctypes.byref(pathptr)):
            raise ctypes.WinError()
        return pathptr.value

    FOLDERID_Download = '{374DE290-123F-4565-9164-39C4925E467B}'

    def get_download_folder():
        return _get_known_folder_path(FOLDERID_Download)
else:
    def get_download_folder():
        home = os.path.expanduser("~")
        return os.path.join(home, "Downloads")

LOGGING_LEVEL = logging.INFO
WARNING_INTERVAL=30 # show first this many warnings and then only every this many

# EVENT_COUNT_PER_FRAME = 2300  # events per frame
BIASES_CONFIG_FILE='configs/davis346_sensitive_slow.json' # "./configs/davis346_config.json"
# EVENT_DURATION = 100000  # events per voxel frame
FLEX_TIME_MODE=False # True -> constant-count voxel volume, False -> constant-duration
FRAME_COUNT_EVENTS=50000 # default for constant-count volume entire voxel volume fed to DNN
FRAME_DURATION_US = 120000  # default for constant-duration entire voxel volume fed to DNN. Must be longer than consumer loop interval to avoid dropping frames.
NUM_BINS = 5 # number of bins for event voxel (frames), must be 5
SENSOR_RESOLUTION = (260, 346) # sensor resolution in pixels, vertical, horizontal
IMSIZE = 224  # CNN input image size, must be small enough that single frame of bytes is less than 64kB for UDP
DOLP_AOLP_MASK_LEVEL=.35 # level of DoLP below which to mask the AoLP value since it is likely not meaningful
EVENT_COUNT_CLIP_VALUE = 3  # full count value for collecting histograms of DVS events
SHOW_DVS_OUTPUT = True # producer shows the accumulated DVS frames as aid for focus and alignment
MIN_PRODUCER_FRAME_INTERVAL_MS=3.0 # inference takes about 15ms for PDAVIS reconstruction and normalization takes 3ms
        # limit rate that we send frames to about what the GPU can manage for inference time
        # after we collect sufficient events, we don't bother to normalize and send them unless this time has
        # passed since last frame was sent. That way, we make sure not to flood the consumer
MAX_SHOWN_DVS_FRAME_RATE_HZ=30 # limits cv2 rendering of DVS frames to reduce loop latency for the producer
ROOT_DATA_FOLDER= os.path.join(get_download_folder(), 'pdavis_demo_dataset') # does not properly find the Downloads folder under Windows if not on same disk as Windows
DATA_FOLDER = os.path.join(ROOT_DATA_FOLDER, 'data') #/home/tobi/Downloads/pdavis_demo_dataset/data' #'data'  # new samples stored here
RECORDING_FOLDER = os.path.join(DATA_FOLDER, 'recordings') #/home/tobi/Downloads/pdavis_demo_dataset/data' #'data'  # new samples stored here

PORT = 12000  # UDP port used to send frames from producer to consumer
UDP_BUFFER_SIZE = int(math.pow(2, math.ceil(math.log(IMSIZE * IMSIZE + 1000) / math.log(2))))


USE_FIRENET=False # True to use firenet DNN from Cedric, False to use e2p DNN
E2P_MODEL= 'models/e2p-0317_215454-e2p-paper_plus_tobi_office-from-scratch.pth' # 'models/e2ptobi-data.pth' #'./e2p-cvpr2023.pth'
FIRENET_MODEL='./firenet/ckpt/firenet_1000.pth.tar'

import signal
def alarm_handler(signum, frame):
    raise TimeoutError
def input_with_timeout(prompt, timeout=30):
    """ get input with timeout

    :param prompt: the prompt to print
    :param timeout: timeout in seconds, or None to disable

    :returns: the input
    :raises: TimeoutError if times out
    """
    # set signal handler
    if timeout is not None:
        signal.signal(signal.SIGALRM, alarm_handler)
        signal.alarm(timeout) # produce SIGALRM in `timeout` seconds
    try:
        time.sleep(.5) # get input to be printed after logging
        return input(prompt)
    except TimeoutError as to:
        raise to
    finally:
        if timeout is not None:
            signal.alarm(0) # cancel alarm

def mycv2_put_text(img,text:str, org=(0,10), font=cv2.FONT_HERSHEY_PLAIN, fontScale=.9, color=(255,255,255),thickness=2,):
    """" Draws text on cv2 image using some defaults
    :param img: the image to draw on; it will later be rendered with imshow
    :param text: the line of text to render
    """
    cv2.putText(img, text, org, font, fontScale, color, thickness)

def yes_or_no(question, default='y', timeout=None):
    """ Get y/n answer with default choice and optional timeout

    :param question: prompt
    :param default: the default choice, i.e. 'y' or 'n'
    :param timeout: the timeout in seconds, default is None

    :returns: True or False
    """
    if default is not None and (default!='y' and default!='n'):
        log.error(f'bad option for default: {default}')
        quit(1)
    y='Y' if default=='y' else 'y'
    n='N' if default=='n' else 'n'
    while "the answer is invalid":
        try:
            to_str='' if timeout is None or os.name=='nt' else f'(Timeout {default} in {timeout}s)'
            if os.name=='nt':
                log.warning('cannot use timeout signal on windows')
                time.sleep(.1) # make the warning come out first
                reply=str(input(f'{question} {to_str} ({y}/{n}): ')).lower().strip()
            else:
                reply = str(input_with_timeout(f'{question} {to_str} ({y}/{n}): ',timeout=timeout)).lower().strip()
        except TimeoutError:
            log.warning(f'timeout expired, returning default={default} answer')
            reply=''
        if len(reply)==0 or reply=='':
            return True if default=='y' else False
        elif reply[0].lower() == 'y':
            return True
        if reply[0].lower() == 'n':
            return False




timers = {}
times = {}
class Timer:
    def __init__(self, timer_name='', delay=None, show_hist=False, numpy_file=None, savefig=False):
        """ Make a Timer() in a _with_ statement for a block of code.
        The timer is started when the block is entered and stopped when exited.
        The Timer _must_ be used in a with statement.

        :param timer_name: the str by which this timer is repeatedly called and which it is named when summary is printed on exit
        :param delay: set this to a value to simply accumulate this externally determined interval
        :param show_hist: whether to plot a histogram with pyplot
        :param savefig: whether to save PDF of histogram; it is saved to timer_plots/<timer_name>_timing_histogram.pdf
        :param numpy_file: optional numpy file path
        """
        self.timer_name = timer_name
        self.show_hist = show_hist
        self.savefig = savefig
        self.numpy_file = numpy_file
        self.delay=delay

        if self.timer_name not in timers.keys():
            timers[self.timer_name] = self
        if self.timer_name not in times.keys():
            times[self.timer_name]=[]

    def __enter__(self):
        if self.delay is None:
            self.start = time.time()
        return self

    def __exit__(self, *args):
        if self.delay is None:
            self.end = time.time()
            self.interval = self.end - self.start  # measured in seconds
        else:
            self.interval=self.delay
        times[self.timer_name].append(self.interval)

    def print_timing_info(self, logger=None):
        """ Prints the timing information accumulated for this Timer

        :param logger: write to the supplied logger, otherwise use the built-in logger
        """
        if len(times)==0:
            log.error(f'Timer {self.timer_name} has no statistics; was it used without a "with" statement?')
            return
        a = np.array(times[self.timer_name])
        timing_mean = np.mean(a) # todo use built in print method for timer
        timing_std = np.std(a)
        timing_median = np.median(a)
        timing_min = np.min(a)
        timing_max = np.max(a)
        s='{} n={}: {}s +/- {}s (median {}s, min {}s max {}s)'.format(self.timer_name, len(a),
                                                                      eng(timing_mean), eng(timing_std),
                                                                      eng(timing_median), eng(timing_min),
                                                                      eng(timing_max))

        if logger is not None:
            logger.info(s)
        else:
            log.info(s)

def print_timing_info():
    print('*********Timing statistics ******')
    for timer,v in times.items():  # k is the name, v is the list of times
        a = np.array(v)
        timing_mean = np.mean(a)
        timing_std = np.std(a)
        timing_median = np.median(a)
        timing_min = np.min(a)
        timing_max = np.max(a)
        print('\tTimer {}: n={}: {}s +/- {}s (median {}s, min {}s max {}s)'.format(timer.ljust(40), len(a),
                                                                          eng(timing_mean), eng(timing_std),
                                                                          eng(timing_median), eng(timing_min),
                                                                          eng(timing_max)))
        if timers[timer].numpy_file is not None:
            try:
                log.info(f'saving timing data for {timer} in numpy file {timers[timer].numpy_file}')
                log.info('there are {} times'.format(len(a)))
                np.save(timers[timer].numpy_file, a)
            except Exception as e:
                log.error(f'could not save numpy file {timers[timer].numpy_file}; caught {e}')

        if timers[timer].show_hist or timers[timer].savefig:

            def plot_loghist(x, bins):
                hist, bins = np.histogram(x, bins=bins) # histogram x linearly
                if len(bins)<2 or bins[0]<=0:
                    log.error(f'cannot plot histogram since bins={bins}')
                    return
                logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins)) # use resulting bin ends to get log bins
                plt.hist(x, bins=logbins) # now again histogram x, but with the log-spaced bins, and plot this histogram
                plt.xscale('log')

            dt = np.clip(a,1e-6, None)
            # logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
            try:
                plot_loghist(dt,bins=100)
                plt.xlabel('interval[ms]')
                plt.ylabel('frequency')
                plt.title(timer)
                if timers[timer].savefig:
                    # python program to check if a path exists
                    # if path doesn’t exist we create a new path
                    from pathlib import Path
                    # creating a new directory called pythondirectory
                    Path("timer_plots").mkdir(exist_ok=True)
                    fn=os.path.join('timer_plots', timers[timer].timer_name+'_timer_hist.pdf')
                    plt.savefig(fn)
                    log.info(f'saved timing histogram to {fn}')
                if timers[timer].show_hist:
                    plt.show()
            except Exception as e:
                log.error(f'could not plot histogram: got {e}')

def write_next_image(dir:str, idx:int, img):
    """ Saves data sample image

    :param dir: the folder
    :param idx: the current index number
    :param img: the image to save, which should be monochrome uint8 and which is saved as default png format
    :returns: the next index
    """
    while True:
        n=f'{dir}/{idx:04d}.png'
        if not os.path.isfile(n):
            break
        idx+=1
    try:
        cv2.imwrite(n, img)
    except Exception as e:
        log.error(f'error saving {n}: caught {e}')
    return idx

# this will print all the timer values upon termination of any program that imported this file
# atexit.register(print_timing_info)

