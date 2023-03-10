"""
 @Time    : 29.03.22 15:12
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : dataset.py
 @Function:
 
"""
import cv2
from torch.utils.data import Dataset
import numpy as np
import random
import torch
import h5py
import os
# local modules
from utils.data_augmentation import *
from utils.data import data_sources
from events_contrast_maximization.utils.event_utils import events_to_voxel_torch, \
    events_to_neg_pos_voxel_torch, binary_search_torch_tensor, events_to_image_torch, \
    binary_search_h5_dset, get_hot_event_mask, save_image
from utils.util import read_json, write_json


class BaseVoxelDataset(Dataset):
    """
    Dataloader for voxel grids given file containing events.
    Also loads time-synchronized frames and optic flow if available.
    Voxel grids are formed on-the-fly.
    For each index, returns a dict containing:
        * frame is a H x W tensor containing the first frame whose
          timestamp >= event tensor
        * events is a C x H x W tensor containing the voxel grid
        * flow is a 2 x H x W tensor containing the flow (displacement) from
          the current frame to the last frame
        * dt is the time spanned by 'events'
        * data_source_idx is the index of the data source (simulated, IJRR, MVSEC etc)
    Subclasses must implement:
        - get_frame(index) method which retrieves the frame at index i
        - get_flow(index) method which retrieves the optic flow at index i
        - get_events(idx0, idx1) method which gets the events between idx0 and idx1
            (in format xs, ys, ts, ps, where each is a np array
            of x, y positions, timestamps and polarities respectively)
        - load_data() initialize the data loading method and ensure the following
            members are filled:
            sensor_resolution - the sensor resolution
            has_flow - if this dataset has optic flow
            t0 - timestamp of first event
            tk - timestamp of last event
            num_events - the total number of events
            frame_ts - list of the timestamps of the frames
            num_frames - the number of frames
        - find_ts_index(timestamp) given a timestamp, find the index of
            the corresponding event

    Parameters:
        data_path Path to the file containing the event/image data
        transforms Dict containing the desired augmentations
        sensor_resolution The size of the image sensor from which the events originate
        num_bins The number of bins desired in the voxel grid
        voxel_method Which method should be used to form the voxels.
            Currently supports:
            * "k_events" (new voxels are formed every k events)
            * "t_seconds" (new voxels are formed every t seconds)
            * "between_frames" (all events between frames are taken, requires frames to exist)
            A sliding window width must be given for k_events and t_seconds,
            which determines overlap (no overlap if set to 0). Eg:
            method={'method':'k_events', 'k':10000, 'sliding_window_w':100}
            method={'method':'t_events', 't':0.5, 'sliding_window_t':0.1}
            method={'method':'between_frames'}
            Default is 'between_frames'.
    """

    def get_frame(self, index):
        """
        Get frame at index
        """
        raise NotImplementedError

    def get_flow(self, index):
        """
        Get optic flow at index
        """
        raise NotImplementedError

    def get_events(self, idx0, idx1):
        """
        Get events between idx0, idx1
        """
        raise NotImplementedError

    def load_data(self, data_path):
        """
        Perform initialization tasks and ensure essential members are populated.
        Required members are:
            members are filled:
            self.sensor_resolution - the sensor resolution
            self.has_flow - if this dataset has optic flow
            self.t0 - timestamp of first event
            self.tk - timestamp of last event
            self.num_events - the total number of events
            self.frame_ts - list of the timestamps of the frames
            self.num_frames - the number of frames
        """
        raise NotImplementedError

    def find_ts_index(self, timestamp):
        """
        Given a timestamp, find the event index
        """
        raise NotImplementedError


    def __init__(self, data_path, transforms={}, sensor_resolution=None, num_bins=5,
                 voxel_method=None, max_length=None, combined_voxel_channels=True,
                 filter_hot_events=False):
        """
        self.transform applies to event voxels, frames and flow.
        self.vox_transform applies to event voxels only.
        """

        self.num_bins = num_bins
        self.data_path = data_path
        self.combined_voxel_channels = combined_voxel_channels
        self.sensor_resolution = tuple(sensor_resolution)
        # print(sensor_resolution)
        # print(self.sensor_resolution)
        # self.sensor_resolution = (256, 306)
        self.data_source_idx = -1
        self.has_flow = False
        self.channels = self.num_bins if combined_voxel_channels else self.num_bins*2

        # self.sensor_resolution, self.t0, self.tk, self.num_events, self.frame_ts, self.num_frames = \
        #     None, None, None, None, None, None

        self.t0, self.tk, self.num_events, self.frame_ts, self.num_frames = None, None, None, None, None

        self.load_data(data_path)

        if self.sensor_resolution is None or self.has_flow is None or self.t0 is None \
                or self.tk is None or self.num_events is None or self.frame_ts is None \
                or self.num_frames is None:
            raise Exception("Dataloader failed to initialize all required members ({})".format(self.data_path))

        self.num_pixels = self.sensor_resolution[0] * self.sensor_resolution[1]
        self.duration = self.tk - self.t0

        if filter_hot_events:
            secs_for_hot_mask = 0.2
            hot_pix_percent = 0.01
            hot_num = min(self.find_ts_index(secs_for_hot_mask+self.t0), self.num_events)
            xs, ys, ts, ps = self.get_events(0, hot_num)
            self.hot_events_mask = get_hot_event_mask(xs.astype(np.int), ys.astype(np.int), ps, self.sensor_resolution, num_hot=int(self.num_pixels*hot_pix_percent))
            self.hot_events_mask = np.stack([self.hot_events_mask]*self.channels, axis=2).transpose(2, 0, 1)
        else:
            self.hot_events_mask = np.ones([self.channels, *self.sensor_resolution])
        self.hot_events_mask = torch.from_numpy(self.hot_events_mask).float()

        if voxel_method is None:
            voxel_method = {'method': 'between_frames'}
        self.set_voxel_method(voxel_method)

        if 'LegacyNorm' in transforms.keys() and 'RobustNorm' in transforms.keys():
            raise Exception('Cannot specify both LegacyNorm and RobustNorm')

        # print(transforms.keys())
        self.normalize_voxels = False
        for norm in ['RobustNorm', 'LegacyNorm']:
            if norm in transforms.keys():
                vox_transforms_list = [eval(t)(**kwargs) for t, kwargs in transforms.items()]
                del (transforms[norm])
                self.normalize_voxels = True
                self.vox_transform = Compose(vox_transforms_list)
                break

        transforms_list = [eval(t)(**kwargs) for t, kwargs in transforms.items()]

        if len(transforms_list) == 0:
            self.transform = None
        elif len(transforms_list) == 1:
            self.transform = transforms_list[0]
        else:
            self.transform = Compose(transforms_list)
        if not self.normalize_voxels:
            self.vox_transform = self.transform

        if max_length is not None:
            self.length = min(self.length, max_length + 1)

    def __getitem__(self, index, seed=None):
        """
        Get data at index.
            :param index: index of data
            :param seed: random seed for data augmentation
        """
        assert 0 <= index < self.__len__(), "index {} out of bounds (0 <= x < {})".format(index, self.__len__())
        seed = random.randint(0, 2 ** 32) if seed is None else seed

        idx0, idx1 = self.get_event_indices(index)
        xs, ys, ts, ps = self.get_events(idx0, idx1)
        try:
            ts_0, ts_k = ts[0], ts[-1]
        except:
            ts_0, ts_k = 0, 0
        if len(xs) < 3:
            # print('xs is less than 3')
            voxel = self.get_empty_voxel_grid(self.combined_voxel_channels)
        else:
            xs = torch.from_numpy(xs.astype(np.float32))
            ys = torch.from_numpy(ys.astype(np.float32))
            ts = torch.from_numpy((ts-ts_0).astype(np.float32))
            ps = torch.from_numpy(ps.astype(np.float32))

            # added by Haiyang Mei, to prevent Inf or Nan in the input voxel
            if ts_k - ts_0 == 0:
                voxel = self.get_empty_voxel_grid(self.combined_voxel_channels)
                # print('dt is zero')
            else:
                voxel = self.get_voxel_grid(xs, ys, ts, ps, combined_voxel_channels=self.combined_voxel_channels)

        voxel = self.transform_voxel(voxel, seed).float()
        dt = ts_k - ts_0
        if dt == 0:
            dt = np.array(0.0)

        #print("Get voxel: event_t0={}, event_tk={}, image_ts={}".format(ts_0, ts_k, self.frame_ts[index]))
        if self.voxel_method['method'] == 'between_frames':
            # print('between frames')
            frame = self.get_frame(index)
            frame = self.transform_frame(frame, seed)

            if self.has_flow:
                flow = self.get_flow(index)
                # convert to displacement (pix)
                flow = flow / 5000 * dt
                flow = self.transform_flow(flow, seed)
            else:
                # print('no flow')
                flow = torch.zeros((2, frame.shape[-2], frame.shape[-1]), dtype=frame.dtype, device=frame.device)

            timestamp = torch.tensor(self.frame_ts[index], dtype=torch.float64)
            item = {'frame': frame,
                    'flow': flow,
                    'events': voxel,
                    'timestamp': timestamp,
                    'data_source_idx': self.data_source_idx,
                    'dt': torch.tensor(dt, dtype=torch.float64)}
        else:
            print("Not between")
            item = {'events': voxel,
                    'timestamp': torch.tensor(ts_k, dtype=torch.float64),
                    'data_source_idx': self.data_source_idx,
                    'dt': torch.tensor(dt, dtype=torch.float64)}
        return item

    def compute_frame_indices(self):
        """
        For each frame, find the start and end indices of the
        time synchronized events
        """
        frame_indices = []
        start_idx = 0
        for ts in self.frame_ts:
            end_index = self.find_ts_index(ts)
            frame_indices.append([start_idx, end_index])
            start_idx = end_index
        return frame_indices

    def compute_timeblock_indices(self):
        """
        For each block of time (using t_events), find the start and
        end indices of the corresponding events
        """
        timeblock_indices = []
        start_idx = 0
        for i in range(self.__len__()):
            start_time = ((self.voxel_method['t'] - self.voxel_method['sliding_window_t']) * i) + self.t0
            end_time = start_time + self.voxel_method['t']
            end_idx = self.find_ts_index(end_time)
            timeblock_indices.append([start_idx, end_idx])
            start_idx = end_idx
        return timeblock_indices

    def compute_k_indices(self):
        """
        For each block of k events, find the start and
        end indices of the corresponding events
        """
        k_indices = []
        start_idx = 0
        for i in range(self.__len__()):
            idx0 = (self.voxel_method['k'] - self.voxel_method['sliding_window_w']) * i
            idx1 = idx0 + self.voxel_method['k']
            k_indices.append([idx0, idx1])
        return k_indices

    def set_voxel_method(self, voxel_method):
        """
        Given the desired method of computing voxels,
        compute the event_indices lookup table and dataset length
        """
        self.voxel_method = voxel_method
        if self.voxel_method['method'] == 'k_events':
            self.length = max(int(self.num_events / (voxel_method['k'] - voxel_method['sliding_window_w'])), 0)
            self.event_indices = self.compute_k_indices()
        elif self.voxel_method['method'] == 't_seconds':
            self.length = max(int(self.duration / (voxel_method['t'] - voxel_method['sliding_window_t'])), 0)
            self.event_indices = self.compute_timeblock_indices()
        elif self.voxel_method['method'] == 'between_frames':
            self.length = self.num_frames - 1
            self.event_indices = self.compute_frame_indices()
        else:
            raise Exception("Invalid voxel forming method chosen ({})".format(self.voxel_method))
        if self.length == 0:
            raise Exception("Current voxel generation parameters lead to sequence length of zero")

    def __len__(self):
        return self.length

    def get_event_indices(self, index):
        """
        Get start and end indices of events at index
        """
        idx0, idx1 = self.event_indices[index]
        if not (idx0 >= 0 and idx1 <= self.num_events):
            raise Exception("WARNING: Event indices {},{} out of bounds 0,{}".format(idx0, idx1, self.num_events))
        return idx0, idx1

    def get_empty_voxel_grid(self, combined_voxel_channels=True):
        """Return an empty voxel grid filled with zeros"""
        if combined_voxel_channels:
            size = (self.num_bins, *self.sensor_resolution)
        else:
            size = (2*self.num_bins, *self.sensor_resolution)
        return torch.zeros(size, dtype=torch.float32)

    def get_voxel_grid(self, xs, ys, ts, ps, combined_voxel_channels=True):
        """
        Given events, return voxel grid
        :param xs: tensor containg x coords of events
        :param ys: tensor containg y coords of events
        :param ts: tensor containg t coords of events
        :param ps: tensor containg p coords of events
        :param combined_voxel_channels: if True, create voxel grid merging positive and
            negative events (resulting in NUM_BINS x H x W tensor). Otherwise, create
            voxel grid for positive and negative events separately
            (resulting in 2*NUM_BINS x H x W tensor)
        """
        if combined_voxel_channels:
            # generate voxel grid which has size self.num_bins x H x W
            voxel_grid = events_to_voxel_torch(xs, ys, ts, ps, self.num_bins, sensor_size=self.sensor_resolution)
        else:
            # generate voxel grid which has size 2*self.num_bins x H x W
            voxel_grid = events_to_neg_pos_voxel_torch(xs, ys, ts, ps, self.num_bins,
                                                       sensor_size=self.sensor_resolution)
            voxel_grid = torch.cat([voxel_grid[0], voxel_grid[1]], 0)

        voxel_grid = voxel_grid*self.hot_events_mask

        return voxel_grid

    def transform_frame(self, frame, seed):
        """
        Augment frame and turn into tensor
        """
        frame = torch.from_numpy(frame).float().unsqueeze(0) / 255
        if self.transform:
            random.seed(seed)
            frame = self.transform(frame)
        return frame

    def transform_voxel(self, voxel, seed):
        """
        Augment voxel and turn into tensor
        """
        if self.vox_transform:
            random.seed(seed)
            voxel = self.vox_transform(voxel)
        return voxel

    def transform_flow(self, flow, seed):
        """
        Augment flow and turn into tensor
        """
        flow = torch.from_numpy(flow)  # should end up [2 x H x W]
        if self.transform:
            random.seed(seed)
            flow = self.transform(flow, is_flow=True)
        return flow


class BaseVoxelDataset_out_of_init(Dataset):
    """
    Dataloader for voxel grids given file containing events.
    Also loads time-synchronized frames and optic flow if available.
    Voxel grids are formed on-the-fly.
    For each index, returns a dict containing:
        * frame is a H x W tensor containing the first frame whose
          timestamp >= event tensor
        * events is a C x H x W tensor containing the voxel grid
        * flow is a 2 x H x W tensor containing the flow (displacement) from
          the current frame to the last frame
        * dt is the time spanned by 'events'
        * data_source_idx is the index of the data source (simulated, IJRR, MVSEC etc)
    Subclasses must implement:
        - get_frame(index) method which retrieves the frame at index i
        - get_flow(index) method which retrieves the optic flow at index i
        - get_events(idx0, idx1) method which gets the events between idx0 and idx1
            (in format xs, ys, ts, ps, where each is a np array
            of x, y positions, timestamps and polarities respectively)
        - load_data() initialize the data loading method and ensure the following
            members are filled:
            sensor_resolution - the sensor resolution
            has_flow - if this dataset has optic flow
            t0 - timestamp of first event
            tk - timestamp of last event
            num_events - the total number of events
            frame_ts - list of the timestamps of the frames
            num_frames - the number of frames
        - find_ts_index(timestamp) given a timestamp, find the index of
            the corresponding event

    Parameters:
        data_path Path to the file containing the event/image data
        transforms Dict containing the desired augmentations
        sensor_resolution The size of the image sensor from which the events originate
        num_bins The number of bins desired in the voxel grid
        voxel_method Which method should be used to form the voxels.
            Currently supports:
            * "k_events" (new voxels are formed every k events)
            * "t_seconds" (new voxels are formed every t seconds)
            * "between_frames" (all events between frames are taken, requires frames to exist)
            A sliding window width must be given for k_events and t_seconds,
            which determines overlap (no overlap if set to 0). Eg:
            method={'method':'k_events', 'k':10000, 'sliding_window_w':100}
            method={'method':'t_events', 't':0.5, 'sliding_window_t':0.1}
            method={'method':'between_frames'}
            Default is 'between_frames'.
    """

    def get_frame(self, index):
        """
        Get frame at index
        """
        raise NotImplementedError

    def get_flow(self, index):
        """
        Get optic flow at index
        """
        raise NotImplementedError

    def get_events(self, idx0, idx1):
        """
        Get events between idx0, idx1
        """
        raise NotImplementedError

    def load_data(self, data_path):
        """
        Perform initialization tasks and ensure essential members are populated.
        Required members are:
            members are filled:
            self.sensor_resolution - the sensor resolution
            self.has_flow - if this dataset has optic flow
            self.t0 - timestamp of first event
            self.tk - timestamp of last event
            self.num_events - the total number of events
            self.frame_ts - list of the timestamps of the frames
            self.num_frames - the number of frames
        """
        raise NotImplementedError

    def find_ts_index(self, timestamp):
        """
        Given a timestamp, find the event index
        """
        raise NotImplementedError


    def __init__(self, data_path, transforms={}, sensor_resolution=None, num_bins=5,
                 voxel_method=None, max_length=None, combined_voxel_channels=True,
                 filter_hot_events=False):
        """
        self.transform applies to event voxels, frames and flow.
        self.vox_transform applies to event voxels only.
        """

        self.num_bins = num_bins
        self.data_path = data_path
        self.combined_voxel_channels = combined_voxel_channels
        self.sensor_resolution = tuple(sensor_resolution)
        # print(sensor_resolution)
        # print(self.sensor_resolution)
        # self.sensor_resolution = (256, 306)
        self.data_source_idx = -1
        self.has_flow = False
        self.channels = self.num_bins if combined_voxel_channels else self.num_bins*2

        # self.sensor_resolution, self.t0, self.tk, self.num_events, self.frame_ts, self.num_frames = \
        #     None, None, None, None, None, None

        self.t0, self.tk, self.num_events, self.frame_ts, self.num_frames = None, None, None, None, None

        # self.load_data(data_path)

        # if self.sensor_resolution is None or self.has_flow is None or self.t0 is None \
        #         or self.tk is None or self.num_events is None or self.frame_ts is None \
        #         or self.num_frames is None:
        #     raise Exception("Dataloader failed to initialize all required members ({})".format(self.data_path))

        self.num_pixels = self.sensor_resolution[0] * self.sensor_resolution[1]
        self.duration = 0

        if filter_hot_events:
            secs_for_hot_mask = 0.2
            hot_pix_percent = 0.01
            hot_num = min(self.find_ts_index(secs_for_hot_mask+self.t0), self.num_events)
            xs, ys, ts, ps = self.get_events(0, hot_num)
            self.hot_events_mask = get_hot_event_mask(xs.astype(np.int), ys.astype(np.int), ps, self.sensor_resolution, num_hot=int(self.num_pixels*hot_pix_percent))
            self.hot_events_mask = np.stack([self.hot_events_mask]*self.channels, axis=2).transpose(2, 0, 1)
        else:
            self.hot_events_mask = np.ones([self.channels, *self.sensor_resolution])
        self.hot_events_mask = torch.from_numpy(self.hot_events_mask).float()

        if voxel_method is None:
            voxel_method = {'method': 'between_frames'}
        self.set_voxel_method(voxel_method)

        if 'LegacyNorm' in transforms.keys() and 'RobustNorm' in transforms.keys():
            raise Exception('Cannot specify both LegacyNorm and RobustNorm')

        # print(transforms.keys())
        self.normalize_voxels = False
        for norm in ['RobustNorm', 'LegacyNorm']:
            if norm in transforms.keys():
                vox_transforms_list = [eval(t)(**kwargs) for t, kwargs in transforms.items()]
                del (transforms[norm])
                self.normalize_voxels = True
                self.vox_transform = Compose(vox_transforms_list)
                break

        transforms_list = [eval(t)(**kwargs) for t, kwargs in transforms.items()]

        if len(transforms_list) == 0:
            self.transform = None
        elif len(transforms_list) == 1:
            self.transform = transforms_list[0]
        else:
            self.transform = Compose(transforms_list)
        if not self.normalize_voxels:
            self.vox_transform = self.transform

        if max_length is not None:
            self.length = min(self.length, max_length + 1)

    def __getitem__(self, index, seed=None):
        """
        Get data at index.
            :param index: index of data
            :param seed: random seed for data augmentation
        """
        if not hasattr(self, 'h5_file'):
            self.load_data(self.data_path)

        assert 0 <= index < self.__len__(), "index {} out of bounds (0 <= x < {})".format(index, self.__len__())
        seed = random.randint(0, 2 ** 32) if seed is None else seed

        idx0, idx1 = self.get_event_indices(index)
        xs, ys, ts, ps = self.get_events(idx0, idx1)
        try:
            ts_0, ts_k = ts[0], ts[-1]
        except:
            ts_0, ts_k = 0, 0
        if len(xs) < 3:
            # print('xs is less than 3')
            voxel = self.get_empty_voxel_grid(self.combined_voxel_channels)
        else:
            xs = torch.from_numpy(xs.astype(np.float32))
            ys = torch.from_numpy(ys.astype(np.float32))
            ts = torch.from_numpy((ts-ts_0).astype(np.float32))
            ps = torch.from_numpy(ps.astype(np.float32))

            # added by Haiyang Mei, to prevent Inf or Nan in the input voxel
            if ts_k - ts_0 == 0:
                voxel = self.get_empty_voxel_grid(self.combined_voxel_channels)
                # print('dt is zero')
            else:
                voxel = self.get_voxel_grid(xs, ys, ts, ps, combined_voxel_channels=self.combined_voxel_channels)

        voxel = self.transform_voxel(voxel, seed).float()
        dt = ts_k - ts_0
        if dt == 0:
            dt = np.array(0.0)

        #print("Get voxel: event_t0={}, event_tk={}, image_ts={}".format(ts_0, ts_k, self.frame_ts[index]))
        if self.voxel_method['method'] == 'between_frames':
            # print('between frames')
            frame = self.get_frame(index)
            frame = self.transform_frame(frame, seed)

            if self.has_flow:
                flow = self.get_flow(index)
                # convert to displacement (pix)
                flow = flow / 5000 * dt
                flow = self.transform_flow(flow, seed)
            else:
                # print('no flow')
                flow = torch.zeros((2, frame.shape[-2], frame.shape[-1]), dtype=frame.dtype, device=frame.device)

            timestamp = torch.tensor(self.frame_ts[index], dtype=torch.float64)
            item = {'frame': frame,
                    'flow': flow,
                    'events': voxel,
                    'timestamp': timestamp,
                    'data_source_idx': self.data_source_idx,
                    'dt': torch.tensor(dt, dtype=torch.float64)}
        else:
            print("Not between")
            item = {'events': voxel,
                    'timestamp': torch.tensor(ts_k, dtype=torch.float64),
                    'data_source_idx': self.data_source_idx,
                    'dt': torch.tensor(dt, dtype=torch.float64)}
        return item

    def compute_frame_indices(self):
        """
        For each frame, find the start and end indices of the
        time synchronized events
        """
        frame_indices = []
        start_idx = 0
        for ts in self.frame_ts:
            end_index = self.find_ts_index(ts)
            frame_indices.append([start_idx, end_index])
            start_idx = end_index
        return frame_indices

    def compute_timeblock_indices(self):
        """
        For each block of time (using t_events), find the start and
        end indices of the corresponding events
        """
        timeblock_indices = []
        start_idx = 0
        for i in range(self.__len__()):
            start_time = ((self.voxel_method['t'] - self.voxel_method['sliding_window_t']) * i) + self.t0
            end_time = start_time + self.voxel_method['t']
            end_idx = self.find_ts_index(end_time)
            timeblock_indices.append([start_idx, end_idx])
            start_idx = end_idx
        return timeblock_indices

    def compute_k_indices(self):
        """
        For each block of k events, find the start and
        end indices of the corresponding events
        """
        k_indices = []
        start_idx = 0
        for i in range(self.__len__()):
            idx0 = (self.voxel_method['k'] - self.voxel_method['sliding_window_w']) * i
            idx1 = idx0 + self.voxel_method['k']
            k_indices.append([idx0, idx1])
        return k_indices

    def set_voxel_method(self, voxel_method):
        """
        Given the desired method of computing voxels,
        compute the event_indices lookup table and dataset length
        """
        self.voxel_method = voxel_method
        if self.voxel_method['method'] == 'k_events':
            self.length = max(int(self.num_events / (voxel_method['k'] - voxel_method['sliding_window_w'])), 0)
            self.event_indices = self.compute_k_indices()
        elif self.voxel_method['method'] == 't_seconds':
            self.length = max(int(self.duration / (voxel_method['t'] - voxel_method['sliding_window_t'])), 0)
            self.event_indices = self.compute_timeblock_indices()
        elif self.voxel_method['method'] == 'between_frames':
            self.length = self.num_frames - 1
            self.event_indices = self.compute_frame_indices()
        else:
            raise Exception("Invalid voxel forming method chosen ({})".format(self.voxel_method))
        if self.length == 0:
            raise Exception("Current voxel generation parameters lead to sequence length of zero")

    def __len__(self):
        return self.length

    def get_event_indices(self, index):
        """
        Get start and end indices of events at index
        """
        idx0, idx1 = self.event_indices[index]
        if not (idx0 >= 0 and idx1 <= self.num_events):
            raise Exception("WARNING: Event indices {},{} out of bounds 0,{}".format(idx0, idx1, self.num_events))
        return idx0, idx1

    def get_empty_voxel_grid(self, combined_voxel_channels=True):
        """Return an empty voxel grid filled with zeros"""
        if combined_voxel_channels:
            size = (self.num_bins, *self.sensor_resolution)
        else:
            size = (2*self.num_bins, *self.sensor_resolution)
        return torch.zeros(size, dtype=torch.float32)

    def get_voxel_grid(self, xs, ys, ts, ps, combined_voxel_channels=True):
        """
        Given events, return voxel grid
        :param xs: tensor containg x coords of events
        :param ys: tensor containg y coords of events
        :param ts: tensor containg t coords of events
        :param ps: tensor containg p coords of events
        :param combined_voxel_channels: if True, create voxel grid merging positive and
            negative events (resulting in NUM_BINS x H x W tensor). Otherwise, create
            voxel grid for positive and negative events separately
            (resulting in 2*NUM_BINS x H x W tensor)
        """
        if combined_voxel_channels:
            # generate voxel grid which has size self.num_bins x H x W
            voxel_grid = events_to_voxel_torch(xs, ys, ts, ps, self.num_bins, sensor_size=self.sensor_resolution)
        else:
            # generate voxel grid which has size 2*self.num_bins x H x W
            voxel_grid = events_to_neg_pos_voxel_torch(xs, ys, ts, ps, self.num_bins,
                                                       sensor_size=self.sensor_resolution)
            voxel_grid = torch.cat([voxel_grid[0], voxel_grid[1]], 0)

        voxel_grid = voxel_grid*self.hot_events_mask

        return voxel_grid

    def transform_frame(self, frame, seed):
        """
        Augment frame and turn into tensor
        """
        frame = torch.from_numpy(frame).float().unsqueeze(0) / 255
        if self.transform:
            random.seed(seed)
            frame = self.transform(frame)
        return frame

    def transform_voxel(self, voxel, seed):
        """
        Augment voxel and turn into tensor
        """
        if self.vox_transform:
            random.seed(seed)
            voxel = self.vox_transform(voxel)
        return voxel

    def transform_flow(self, flow, seed):
        """
        Augment flow and turn into tensor
        """
        flow = torch.from_numpy(flow)  # should end up [2 x H x W]
        if self.transform:
            random.seed(seed)
            flow = self.transform(flow, is_flow=True)
        return flow


class BaseVoxelDataset_p(Dataset):
    """
    Dataloader for voxel grids given file containing events.
    Also loads time-synchronized intensities, aolp, and dolp if available.
    Voxel grids are formed on-the-fly.
    For each index, returns a dict containing:
        * frame is a H x W tensor containing the first frame whose
          timestamp >= event tensor
        * events is a C x H x W tensor containing the voxel grid
        * flow is a 2 x H x W tensor containing the flow (displacement) from
          the current frame to the last frame
        * dt is the time spanned by 'events'
        * data_source_idx is the index of the data source (simulated, IJRR, MVSEC etc)
    Subclasses must implement:
        - get_frame(index) method which retrieves the frame at index i
        - get_flow(index) method which retrieves the optic flow at index i
        - get_events(idx0, idx1) method which gets the events between idx0 and idx1
            (in format xs, ys, ts, ps, where each is a np array
            of x, y positions, timestamps and polarities respectively)
        - load_data() initialize the data loading method and ensure the following
            members are filled:
            sensor_resolution - the sensor resolution
            has_flow - if this dataset has optic flow
            t0 - timestamp of first event
            tk - timestamp of last event
            num_events - the total number of events
            frame_ts - list of the timestamps of the frames
            num_frames - the number of frames
        - find_ts_index(timestamp) given a timestamp, find the index of
            the corresponding event

    Parameters:
        data_path Path to the file containing the event/image data
        transforms Dict containing the desired augmentations
        sensor_resolution The size of the image sensor from which the events originate
        num_bins The number of bins desired in the voxel grid
        voxel_method Which method should be used to form the voxels.
            Currently supports:
            * "k_events" (new voxels are formed every k events)
            * "t_seconds" (new voxels are formed every t seconds)
            * "between_frames" (all events between frames are taken, requires frames to exist)
            A sliding window width must be given for k_events and t_seconds,
            which determines overlap (no overlap if set to 0). Eg:
            method={'method':'k_events', 'k':10000, 'sliding_window_w':100}
            method={'method':'t_events', 't':0.5, 'sliding_window_t':0.1}
            method={'method':'between_frames'}
            Default is 'between_frames'.
    """

    def get_intensity(self, index):
        """
        Get intensity at index
        """
        raise NotImplementedError

    def get_aolp(self, index):
        """
        Get aolp at index
        """
        raise NotImplementedError

    def get_dolp(self, index):
        """
        Get dolp at index
        """
        raise NotImplementedError

    def get_flow(self, index):
        """
        Get optic flow at index
        """
        raise NotImplementedError

    def get_events(self, idx0, idx1):
        """
        Get events between idx0, idx1
        """
        raise NotImplementedError

    def load_data(self, data_path):
        """
        Perform initialization tasks and ensure essential members are populated.
        Required members are:
            members are filled:
            self.sensor_resolution - the sensor resolution
            self.has_flow - if this dataset has optic flow
            self.t0 - timestamp of first event
            self.tk - timestamp of last event
            self.num_events - the total number of events
            self.frame_ts - list of the timestamps of the frames
            self.num_frames - the number of frames
        """
        raise NotImplementedError

    def find_ts_index(self, timestamp):
        """
        Given a timestamp, find the event index
        """
        raise NotImplementedError


    def __init__(self, data_path, transforms={}, sensor_resolution=None, num_bins=5,
                 voxel_method=None, max_length=None, combined_voxel_channels=True,
                 filter_hot_events=False):
        """
        self.transform applies to event voxels, frames and flow.
        self.vox_transform applies to event voxels only.
        """

        self.num_bins = num_bins
        self.data_path = data_path
        self.combined_voxel_channels = combined_voxel_channels
        self.sensor_resolution = sensor_resolution
        self.data_source_idx = -1
        self.has_intensity = False
        self.has_aolp = False
        self.has_dolp = False
        self.has_flow = False
        self.channels = self.num_bins if combined_voxel_channels else self.num_bins*2

        self.t0, self.tk, self.num_events, self.frame_ts, self.num_frames = None, None, None, None, None

        self.load_data(data_path)

        # if self.sensor_resolution is None or self.t0 is None or self.tk is None or self.num_events is None or self.frame_ts is None or self.num_frames is None:
        if self.sensor_resolution is None or self.t0 is None or self.tk is None or self.num_events is None:
            raise Exception("Dataloader failed to initialize all required members ({})".format(self.data_path))

        self.num_pixels = self.sensor_resolution[0] * self.sensor_resolution[1]
        self.duration = self.tk - self.t0

        if filter_hot_events:
            secs_for_hot_mask = 0.2
            hot_pix_percent = 0.01
            hot_num = min(self.find_ts_index(secs_for_hot_mask+self.t0), self.num_events)
            xs, ys, ts, ps = self.get_events(0, hot_num)
            self.hot_events_mask = get_hot_event_mask(xs.astype(np.int), ys.astype(np.int), ps, self.sensor_resolution, num_hot=int(self.num_pixels*hot_pix_percent))
            self.hot_events_mask = np.stack([self.hot_events_mask]*self.channels, axis=2).transpose(2, 0, 1)
        else:
            self.hot_events_mask = np.ones([self.channels, *self.sensor_resolution])
        self.hot_events_mask = torch.from_numpy(self.hot_events_mask).float()

        if voxel_method is None:
            voxel_method = {'method': 'between_frames'}
        self.set_voxel_method(voxel_method)

        if 'LegacyNorm' in transforms.keys() and 'RobustNorm' in transforms.keys():
            raise Exception('Cannot specify both LegacyNorm and RobustNorm')

        self.normalize_voxels = False
        for norm in ['RobustNorm', 'LegacyNorm']:
            if norm in transforms.keys():
                vox_transforms_list = [eval(t)(**kwargs) for t, kwargs in transforms.items()]
                del (transforms[norm])
                self.normalize_voxels = True
                self.vox_transform = Compose(vox_transforms_list)
                break

        transforms_list = [eval(t)(**kwargs) for t, kwargs in transforms.items()]

        if len(transforms_list) == 0:
            self.transform = None
        elif len(transforms_list) == 1:
            self.transform = transforms_list[0]
        else:
            self.transform = Compose(transforms_list)
        if not self.normalize_voxels:
            self.vox_transform = self.transform

        # print(self.transform)
        # print(self.vox_transform)
        # exit(0)

        if max_length is not None:
            self.length = min(self.length, max_length + 1)

    def __getitem__(self, index, seed=None):
        """
        Get data at index.
            :param index: index of data
            :param seed: random seed for data augmentation
        """
        assert 0 <= index < self.__len__(), "index {} out of bounds (0 <= x < {})".format(index, self.__len__())
        seed = random.randint(0, 2 ** 32) if seed is None else seed

        idx0, idx1 = self.get_event_indices(index)
        xs, ys, ts, ps = self.get_events(idx0, idx1)
        try:
            ts_0, ts_k = ts[0], ts[-1]
        except:
            ts_0, ts_k = 0, 0
        if len(xs) < 3:
            # print('xs is less than 3')
            voxel = self.get_empty_voxel_grid(self.combined_voxel_channels)
        else:
            xs = torch.from_numpy(xs.astype(np.float32))
            ys = torch.from_numpy(ys.astype(np.float32))
            ts = torch.from_numpy((ts-ts_0).astype(np.float32))
            ps = torch.from_numpy(ps.astype(np.float32))

            # added by Haiyang Mei, to prevent Inf or Nan in the input voxel
            if ts_k - ts_0 == 0:
                voxel = self.get_empty_voxel_grid(self.combined_voxel_channels)
                # print('dt is zero')
            else:
                voxel = self.get_voxel_grid(xs, ys, ts, ps, combined_voxel_channels=self.combined_voxel_channels)

        # events voxel
        voxel = self.transform_voxel(voxel, seed).float()
        dt = ts_k - ts_0
        if dt == 0:
            dt = np.array(0.0)
        # print('dt is:', dt)

        # print("Get voxel: event_t0={}, event_tk={}, image_ts={}".format(ts_0, ts_k, self.frame_ts[index]))
        if self.voxel_method['method'] == 'between_frames':
            # intensity
            if self.has_intensity:
                intensity = self.get_intensity(index)
                intensity = self.transform_intensity(intensity, seed)
            else:
                print('no intensity')
                intensity = torch.zeros((1, int(self.sensor_resolution[0] / 2), int(self.sensor_resolution[0] / 2)))

            # aolp
            if self.has_aolp:
                aolp = self.get_aolp(index)
                aolp = self.transform_aolp(aolp, seed)
            else:
                print('no aolp')
                aolp = torch.zeros((1, int(self.sensor_resolution[0] / 2), int(self.sensor_resolution[0] / 2)))

            # dolp
            if self.has_dolp:
                dolp = self.get_dolp(index)
                dolp = self.transform_dolp(dolp, seed)
            else:
                print('no dolp')
                dolp = torch.zeros((1, int(self.sensor_resolution[0] / 2), int(self.sensor_resolution[0] / 2)))

            # flow
            if self.has_flow:
                flow = self.get_flow(index)
                # convert to displacement (pix)
                # print(dt)
                # print(flow)
                # exit(0)
                # flow = flow / 33333 * dt
                flow = flow / 5000 * dt
                # print(np.max(flow))
                # print(np.min(flow))
                flow = self.transform_flow(flow, seed)
            else:
                flow = torch.zeros((2, intensity.shape[-2], intensity.shape[-1]), dtype=intensity.dtype, device=intensity.device)

            timestamp = torch.tensor(self.frame_ts[index], dtype=torch.float64)
            item = {'intensity': intensity,
                    'aolp': aolp,
                    'dolp': dolp,
                    'flow': flow,
                    'events': voxel,
                    'timestamp': timestamp,
                    'data_source_idx': self.data_source_idx,
                    'dt': torch.tensor(dt, dtype=torch.float64)}
        else:
            # print("Not between")
            item = {'events': voxel,
                    'timestamp': torch.tensor(ts_k, dtype=torch.float64),
                    'data_source_idx': self.data_source_idx,
                    'dt': torch.tensor(dt, dtype=torch.float64)}
        return item

    def compute_frame_indices(self):
        """
        For each frame, find the start and end indices of the
        time synchronized events
        """
        frame_indices = []
        start_idx = 0
        for ts in self.frame_ts:
            end_index = self.find_ts_index(ts)
            frame_indices.append([start_idx, end_index])
            start_idx = end_index
        return frame_indices

    def compute_timeblock_indices(self):
        """
        For each block of time (using t_events), find the start and
        end indices of the corresponding events
        """
        timeblock_indices = []
        start_idx = 0
        for i in range(self.__len__()):
            start_time = ((self.voxel_method['t'] - self.voxel_method['sliding_window_t']) * i) + self.t0
            end_time = start_time + self.voxel_method['t']
            end_idx = self.find_ts_index(end_time)
            timeblock_indices.append([start_idx, end_idx])
            start_idx = end_idx
        return timeblock_indices

    def compute_k_indices(self):
        """
        For each block of k events, find the start and
        end indices of the corresponding events
        """
        k_indices = []
        start_idx = 0
        for i in range(self.__len__()):
            idx0 = (self.voxel_method['k'] - self.voxel_method['sliding_window_w']) * i
            idx1 = idx0 + self.voxel_method['k']
            k_indices.append([idx0, idx1])
        return k_indices

    def set_voxel_method(self, voxel_method):
        """
        Given the desired method of computing voxels,
        compute the event_indices lookup table and dataset length
        """
        self.voxel_method = voxel_method
        if self.voxel_method['method'] == 'k_events':
            self.length = max(int(self.num_events / (voxel_method['k'] - voxel_method['sliding_window_w'])), 0)
            self.event_indices = self.compute_k_indices()
        elif self.voxel_method['method'] == 't_seconds':
            # print(voxel_method['t'])
            # print(voxel_method['sliding_window_t'])
            self.length = max(int(self.duration / (voxel_method['t'] - voxel_method['sliding_window_t'])), 0)
            print(self.length)
            self.event_indices = self.compute_timeblock_indices()
            # print('ok')
        elif self.voxel_method['method'] == 'between_frames':
            # self.length = self.num_frames - 1
            # length should be same with self.num_frames
            self.length = self.num_frames
            self.event_indices = self.compute_frame_indices()
        else:
            raise Exception("Invalid voxel forming method chosen ({})".format(self.voxel_method))
        if self.length == 0:
            raise Exception("Current voxel generation parameters lead to sequence length of zero")

    def __len__(self):
        return self.length

    def get_event_indices(self, index):
        """
        Get start and end indices of events at index
        """
        idx0, idx1 = self.event_indices[index]
        if not (idx0 >= 0 and idx1 <= self.num_events):
            raise Exception("WARNING: Event indices {},{} out of bounds 0,{}".format(idx0, idx1, self.num_events))
        return idx0, idx1

    def get_empty_voxel_grid(self, combined_voxel_channels=True):
        """Return an empty voxel grid filled with zeros"""
        if combined_voxel_channels:
            size = (self.num_bins, *self.sensor_resolution)
        else:
            size = (2*self.num_bins, *self.sensor_resolution)
        return torch.zeros(size, dtype=torch.float32)

    def get_voxel_grid(self, xs, ys, ts, ps, combined_voxel_channels=True):
        """
        Given events, return voxel grid
        :param xs: tensor containg x coords of events
        :param ys: tensor containg y coords of events
        :param ts: tensor containg t coords of events
        :param ps: tensor containg p coords of events
        :param combined_voxel_channels: if True, create voxel grid merging positive and
            negative events (resulting in NUM_BINS x H x W tensor). Otherwise, create
            voxel grid for positive and negative events separately
            (resulting in 2*NUM_BINS x H x W tensor)
        """
        if combined_voxel_channels:
            # generate voxel grid which has size self.num_bins x H x W
            voxel_grid = events_to_voxel_torch(xs, ys, ts, ps, self.num_bins, sensor_size=self.sensor_resolution)
        else:
            # generate voxel grid which has size 2*self.num_bins x H x W
            voxel_grid = events_to_neg_pos_voxel_torch(xs, ys, ts, ps, self.num_bins,
                                                       sensor_size=self.sensor_resolution)
            voxel_grid = torch.cat([voxel_grid[0], voxel_grid[1]], 0)

        voxel_grid = voxel_grid * self.hot_events_mask

        return voxel_grid

    def transform_voxel(self, voxel, seed):
        """
        Augment voxel and turn into tensor
        """
        if self.vox_transform:
            random.seed(seed)
            voxel = self.vox_transform(voxel)
        return voxel

    def transform_intensity(self, intensity, seed):
        """
        Augment intensity and turn into tensor
        """
        intensity = torch.from_numpy(intensity).float().unsqueeze(0) / 255
        if self.transform:
            random.seed(seed)
            intensity = self.transform(intensity, is_polarization=True)
        return intensity

    def transform_aolp(self, aolp, seed):
        """
        Augment aolp and turn into tensor
        """
        aolp = torch.from_numpy(aolp).float().unsqueeze(0) / 255
        if self.transform:
            random.seed(seed)
            aolp = self.transform(aolp, is_polarization=True, is_aolp=True)
        return aolp

    def transform_dolp(self, dolp, seed):
        """
        Augment dolp and turn into tensor
        """
        dolp = torch.from_numpy(dolp).float().unsqueeze(0) / 255
        if self.transform:
            random.seed(seed)
            dolp = self.transform(dolp, is_polarization=True)
        return dolp

    def transform_flow(self, flow, seed):
        """
        Augment flow and turn into tensor
        """
        flow = torch.from_numpy(flow)  # should end up [2 x H x W]
        if self.transform:
            random.seed(seed)
            flow = self.transform(flow, is_polarization=True, is_flow=True)
        return flow


class BaseVoxelDataset_rp(Dataset):
    """
    Dataloader for voxel grids given file containing events.
    Also loads time-synchronized intensities, aolp, and dolp if available.
    Voxel grids are formed on-the-fly.
    For each index, returns a dict containing:
        * frame is a H x W tensor containing the first frame whose
          timestamp >= event tensor
        * events is a C x H x W tensor containing the voxel grid
        * flow is a 2 x H x W tensor containing the flow (displacement) from
          the current frame to the last frame
        * dt is the time spanned by 'events'
        * data_source_idx is the index of the data source (simulated, IJRR, MVSEC etc)
    Subclasses must implement:
        - get_frame(index) method which retrieves the frame at index i
        - get_flow(index) method which retrieves the optic flow at index i
        - get_events(idx0, idx1) method which gets the events between idx0 and idx1
            (in format xs, ys, ts, ps, where each is a np array
            of x, y positions, timestamps and polarities respectively)
        - load_data() initialize the data loading method and ensure the following
            members are filled:
            sensor_resolution - the sensor resolution
            has_flow - if this dataset has optic flow
            t0 - timestamp of first event
            tk - timestamp of last event
            num_events - the total number of events
            frame_ts - list of the timestamps of the frames
            num_frames - the number of frames
        - find_ts_index(timestamp) given a timestamp, find the index of
            the corresponding event

    Parameters:
        data_path Path to the file containing the event/image data
        transforms Dict containing the desired augmentations
        sensor_resolution The size of the image sensor from which the events originate
        num_bins The number of bins desired in the voxel grid
        voxel_method Which method should be used to form the voxels.
            Currently supports:
            * "k_events" (new voxels are formed every k events)
            * "t_seconds" (new voxels are formed every t seconds)
            * "between_frames" (all events between frames are taken, requires frames to exist)
            A sliding window width must be given for k_events and t_seconds,
            which determines overlap (no overlap if set to 0). Eg:
            method={'method':'k_events', 'k':10000, 'sliding_window_w':100}
            method={'method':'t_events', 't':0.5, 'sliding_window_t':0.1}
            method={'method':'between_frames'}
            Default is 'between_frames'.
    """

    def get_raw(self, index):
        """
        Get raw frame at index
        """
        raise NotImplementedError

    def get_intensity(self, index):
        """
        Get intensity at index
        """
        raise NotImplementedError

    def get_aolp(self, index):
        """
        Get aolp at index
        """
        raise NotImplementedError

    def get_dolp(self, index):
        """
        Get dolp at index
        """
        raise NotImplementedError

    def get_flow(self, index):
        """
        Get optic flow at index
        """
        raise NotImplementedError

    def get_events(self, idx0, idx1):
        """
        Get events between idx0, idx1
        """
        raise NotImplementedError

    def load_data(self, data_path):
        """
        Perform initialization tasks and ensure essential members are populated.
        Required members are:
            members are filled:
            self.sensor_resolution - the sensor resolution
            self.has_flow - if this dataset has optic flow
            self.t0 - timestamp of first event
            self.tk - timestamp of last event
            self.num_events - the total number of events
            self.frame_ts - list of the timestamps of the frames
            self.num_frames - the number of frames
        """
        raise NotImplementedError

    def find_ts_index(self, timestamp):
        """
        Given a timestamp, find the event index
        """
        raise NotImplementedError


    def __init__(self, data_path, transforms={}, sensor_resolution=None, num_bins=5,
                 voxel_method=None, max_length=None, combined_voxel_channels=True,
                 filter_hot_events=False):
        """
        self.transform applies to event voxels, frames and flow.
        self.vox_transform applies to event voxels only.
        """

        self.num_bins = num_bins
        self.data_path = data_path
        self.combined_voxel_channels = combined_voxel_channels
        self.sensor_resolution = sensor_resolution
        self.data_source_idx = -1
        self.has_raw = False
        self.has_intensity = False
        self.has_aolp = False
        self.has_dolp = False
        self.has_flow = False
        self.channels = self.num_bins if combined_voxel_channels else self.num_bins*2

        self.t0, self.tk, self.num_events, self.frame_ts, self.num_frames = None, None, None, None, None

        self.load_data(data_path)

        # if self.sensor_resolution is None or self.t0 is None or self.tk is None or self.num_events is None or self.frame_ts is None or self.num_frames is None:
        if self.sensor_resolution is None or self.t0 is None or self.tk is None or self.num_events is None:
            raise Exception("Dataloader failed to initialize all required members ({})".format(self.data_path))

        self.num_pixels = self.sensor_resolution[0] * self.sensor_resolution[1]
        self.duration = self.tk - self.t0

        if filter_hot_events:
            secs_for_hot_mask = 0.2
            hot_pix_percent = 0.01
            hot_num = min(self.find_ts_index(secs_for_hot_mask+self.t0), self.num_events)
            xs, ys, ts, ps = self.get_events(0, hot_num)
            self.hot_events_mask = get_hot_event_mask(xs.astype(np.int), ys.astype(np.int), ps, self.sensor_resolution, num_hot=int(self.num_pixels*hot_pix_percent))
            self.hot_events_mask = np.stack([self.hot_events_mask]*self.channels, axis=2).transpose(2, 0, 1)
        else:
            self.hot_events_mask = np.ones([self.channels, *self.sensor_resolution])
        self.hot_events_mask = torch.from_numpy(self.hot_events_mask).float()

        if voxel_method is None:
            voxel_method = {'method': 'between_frames'}
        self.set_voxel_method(voxel_method)

        if 'LegacyNorm' in transforms.keys() and 'RobustNorm' in transforms.keys():
            raise Exception('Cannot specify both LegacyNorm and RobustNorm')

        self.normalize_voxels = False
        for norm in ['RobustNorm', 'LegacyNorm']:
            if norm in transforms.keys():
                vox_transforms_list = [eval(t)(**kwargs) for t, kwargs in transforms.items()]
                del (transforms[norm])
                self.normalize_voxels = True
                self.vox_transform = Compose(vox_transforms_list)
                break

        transforms_list = [eval(t)(**kwargs) for t, kwargs in transforms.items()]

        if len(transforms_list) == 0:
            self.transform = None
        elif len(transforms_list) == 1:
            self.transform = transforms_list[0]
        else:
            self.transform = Compose(transforms_list)
        if not self.normalize_voxels:
            self.vox_transform = self.transform

        # print(self.transform)
        # print(self.vox_transform)
        # exit(0)

        if max_length is not None:
            self.length = min(self.length, max_length + 1)

    def __getitem__(self, index, seed=None):
        """
        Get data at index.
            :param index: index of data
            :param seed: random seed for data augmentation
        """
        assert 0 <= index < self.__len__(), "index {} out of bounds (0 <= x < {})".format(index, self.__len__())
        seed = random.randint(0, 2 ** 32) if seed is None else seed

        idx0, idx1 = self.get_event_indices(index)
        xs, ys, ts, ps = self.get_events(idx0, idx1)
        try:
            ts_0, ts_k = ts[0], ts[-1]
        except:
            ts_0, ts_k = 0, 0
        if len(xs) < 3:
            # print('xs is less than 3')
            voxel = self.get_empty_voxel_grid(self.combined_voxel_channels)
        else:
            xs = torch.from_numpy(xs.astype(np.float32))
            ys = torch.from_numpy(ys.astype(np.float32))
            ts = torch.from_numpy((ts-ts_0).astype(np.float32))
            ps = torch.from_numpy(ps.astype(np.float32))

            # added by Haiyang Mei, to prevent Inf or Nan in the input voxel
            if ts_k - ts_0 == 0:
                voxel = self.get_empty_voxel_grid(self.combined_voxel_channels)
                # print('dt is zero')
            else:
                voxel = self.get_voxel_grid(xs, ys, ts, ps, combined_voxel_channels=self.combined_voxel_channels)

        # events voxel
        voxel = self.transform_voxel(voxel, seed).float()
        dt = ts_k - ts_0
        if dt == 0:
            dt = np.array(0.0)
        # print('dt is:', dt)

        # print("Get voxel: event_t0={}, event_tk={}, image_ts={}".format(ts_0, ts_k, self.frame_ts[index]))
        if self.voxel_method['method'] == 'between_frames':
            # raw
            if self.has_raw:
                raw = self.get_raw(index)
                raw = self.transform_raw(raw, seed)
            else:
                print('no raw')
                raw = torch.zeros((1, self.sensor_resolution[0], self.sensor_resolution[0]))

            # intensity
            if self.has_intensity:
                intensity = self.get_intensity(index)
                intensity = self.transform_intensity(intensity, seed)
            else:
                print('no intensity')
                intensity = torch.zeros((1, int(self.sensor_resolution[0] / 2), int(self.sensor_resolution[0] / 2)))

            # aolp
            if self.has_aolp:
                aolp = self.get_aolp(index)
                aolp = self.transform_aolp(aolp, seed)
            else:
                print('no aolp')
                aolp = torch.zeros((1, int(self.sensor_resolution[0] / 2), int(self.sensor_resolution[0] / 2)))

            # dolp
            if self.has_dolp:
                dolp = self.get_dolp(index)
                dolp = self.transform_dolp(dolp, seed)
            else:
                print('no dolp')
                dolp = torch.zeros((1, int(self.sensor_resolution[0] / 2), int(self.sensor_resolution[0] / 2)))

            # flow
            if self.has_flow:
                flow = self.get_flow(index)
                # convert to displacement (pix)
                # print(dt)
                # print(flow)
                # exit(0)
                # flow = flow / 33333 * dt
                flow = flow / 5000 * dt
                # print(np.max(flow))
                # print(np.min(flow))
                flow = self.transform_flow(flow, seed)
            else:
                flow = torch.zeros((2, intensity.shape[-2], intensity.shape[-1]), dtype=intensity.dtype, device=intensity.device)

            timestamp = torch.tensor(self.frame_ts[index], dtype=torch.float64)
            item = {'raw': raw,
                    'intensity': intensity,
                    'aolp': aolp,
                    'dolp': dolp,
                    'flow': flow,
                    'events': voxel,
                    'timestamp': timestamp,
                    'data_source_idx': self.data_source_idx,
                    'dt': torch.tensor(dt, dtype=torch.float64)}
        else:
            # print("Not between")
            item = {'events': voxel,
                    'timestamp': torch.tensor(ts_k, dtype=torch.float64),
                    'data_source_idx': self.data_source_idx,
                    'dt': torch.tensor(dt, dtype=torch.float64)}
        return item

    def compute_frame_indices(self):
        """
        For each frame, find the start and end indices of the
        time synchronized events
        """
        frame_indices = []
        start_idx = 0
        for ts in self.frame_ts:
            end_index = self.find_ts_index(ts)
            frame_indices.append([start_idx, end_index])
            start_idx = end_index
        return frame_indices

    def compute_timeblock_indices(self):
        """
        For each block of time (using t_events), find the start and
        end indices of the corresponding events
        """
        timeblock_indices = []
        start_idx = 0
        for i in range(self.__len__()):
            start_time = ((self.voxel_method['t'] - self.voxel_method['sliding_window_t']) * i) + self.t0
            end_time = start_time + self.voxel_method['t']
            end_idx = self.find_ts_index(end_time)
            timeblock_indices.append([start_idx, end_idx])
            start_idx = end_idx
        return timeblock_indices

    def compute_k_indices(self):
        """
        For each block of k events, find the start and
        end indices of the corresponding events
        """
        k_indices = []
        start_idx = 0
        for i in range(self.__len__()):
            idx0 = (self.voxel_method['k'] - self.voxel_method['sliding_window_w']) * i
            idx1 = idx0 + self.voxel_method['k']
            k_indices.append([idx0, idx1])
        return k_indices

    def set_voxel_method(self, voxel_method):
        """
        Given the desired method of computing voxels,
        compute the event_indices lookup table and dataset length
        """
        self.voxel_method = voxel_method
        if self.voxel_method['method'] == 'k_events':
            self.length = max(int(self.num_events / (voxel_method['k'] - voxel_method['sliding_window_w'])), 0)
            self.event_indices = self.compute_k_indices()
        elif self.voxel_method['method'] == 't_seconds':
            # print(voxel_method['t'])
            # print(voxel_method['sliding_window_t'])
            self.length = max(int(self.duration / (voxel_method['t'] - voxel_method['sliding_window_t'])), 0)
            print(self.length)
            self.event_indices = self.compute_timeblock_indices()
            # print('ok')
        elif self.voxel_method['method'] == 'between_frames':
            self.length = self.num_frames - 1
            self.event_indices = self.compute_frame_indices()
        else:
            raise Exception("Invalid voxel forming method chosen ({})".format(self.voxel_method))
        if self.length == 0:
            raise Exception("Current voxel generation parameters lead to sequence length of zero")

    def __len__(self):
        return self.length

    def get_event_indices(self, index):
        """
        Get start and end indices of events at index
        """
        idx0, idx1 = self.event_indices[index]
        if not (idx0 >= 0 and idx1 <= self.num_events):
            raise Exception("WARNING: Event indices {},{} out of bounds 0,{}".format(idx0, idx1, self.num_events))
        return idx0, idx1

    def get_empty_voxel_grid(self, combined_voxel_channels=True):
        """Return an empty voxel grid filled with zeros"""
        if combined_voxel_channels:
            size = (self.num_bins, *self.sensor_resolution)
        else:
            size = (2*self.num_bins, *self.sensor_resolution)
        return torch.zeros(size, dtype=torch.float32)

    def get_voxel_grid(self, xs, ys, ts, ps, combined_voxel_channels=True):
        """
        Given events, return voxel grid
        :param xs: tensor containg x coords of events
        :param ys: tensor containg y coords of events
        :param ts: tensor containg t coords of events
        :param ps: tensor containg p coords of events
        :param combined_voxel_channels: if True, create voxel grid merging positive and
            negative events (resulting in NUM_BINS x H x W tensor). Otherwise, create
            voxel grid for positive and negative events separately
            (resulting in 2*NUM_BINS x H x W tensor)
        """
        if combined_voxel_channels:
            # generate voxel grid which has size self.num_bins x H x W
            voxel_grid = events_to_voxel_torch(xs, ys, ts, ps, self.num_bins, sensor_size=self.sensor_resolution)
        else:
            # generate voxel grid which has size 2*self.num_bins x H x W
            voxel_grid = events_to_neg_pos_voxel_torch(xs, ys, ts, ps, self.num_bins,
                                                       sensor_size=self.sensor_resolution)
            voxel_grid = torch.cat([voxel_grid[0], voxel_grid[1]], 0)

        voxel_grid = voxel_grid * self.hot_events_mask

        return voxel_grid

    def transform_voxel(self, voxel, seed):
        """
        Augment voxel and turn into tensor
        """
        if self.vox_transform:
            random.seed(seed)
            voxel = self.vox_transform(voxel)
        return voxel

    def transform_raw(self, raw, seed):
        """
        Augment raw and turn into tensor
        """
        raw = torch.from_numpy(raw).float().unsqueeze(0) / 255
        if self.transform:
            random.seed(seed)
            raw = self.transform(raw)
        return raw

    def transform_intensity(self, intensity, seed):
        """
        Augment intensity and turn into tensor
        """
        intensity = torch.from_numpy(intensity).float().unsqueeze(0) / 255
        if self.transform:
            random.seed(seed)
            intensity = self.transform(intensity, is_polarization=True)
        return intensity

    def transform_aolp(self, aolp, seed):
        """
        Augment aolp and turn into tensor
        """
        aolp = torch.from_numpy(aolp).float().unsqueeze(0) / 255
        if self.transform:
            random.seed(seed)
            aolp = self.transform(aolp, is_polarization=True, is_aolp=True)
        return aolp

    def transform_dolp(self, dolp, seed):
        """
        Augment dolp and turn into tensor
        """
        dolp = torch.from_numpy(dolp).float().unsqueeze(0) / 255
        if self.transform:
            random.seed(seed)
            dolp = self.transform(dolp, is_polarization=True)
        return dolp

    def transform_flow(self, flow, seed):
        """
        Augment flow and turn into tensor
        """
        flow = torch.from_numpy(flow)  # should end up [2 x H x W]
        if self.transform:
            random.seed(seed)
            flow = self.transform(flow, is_polarization=True, is_flow=True)
        return flow


class BaseVoxelDataset_s012_iad(Dataset):
    """
    Dataloader for voxel grids given file containing events.
    Also loads time-synchronized intensities, aolp, and dolp if available.
    Voxel grids are formed on-the-fly.
    For each index, returns a dict containing:
        * frame is a H x W tensor containing the first frame whose
          timestamp >= event tensor
        * events is a C x H x W tensor containing the voxel grid
        * flow is a 2 x H x W tensor containing the flow (displacement) from
          the current frame to the last frame
        * dt is the time spanned by 'events'
        * data_source_idx is the index of the data source (simulated, IJRR, MVSEC etc)
    Subclasses must implement:
        - get_frame(index) method which retrieves the frame at index i
        - get_flow(index) method which retrieves the optic flow at index i
        - get_events(idx0, idx1) method which gets the events between idx0 and idx1
            (in format xs, ys, ts, ps, where each is a np array
            of x, y positions, timestamps and polarities respectively)
        - load_data() initialize the data loading method and ensure the following
            members are filled:
            sensor_resolution - the sensor resolution
            has_flow - if this dataset has optic flow
            t0 - timestamp of first event
            tk - timestamp of last event
            num_events - the total number of events
            frame_ts - list of the timestamps of the frames
            num_frames - the number of frames
        - find_ts_index(timestamp) given a timestamp, find the index of
            the corresponding event

    Parameters:
        data_path Path to the file containing the event/image data
        transforms Dict containing the desired augmentations
        sensor_resolution The size of the image sensor from which the events originate
        num_bins The number of bins desired in the voxel grid
        voxel_method Which method should be used to form the voxels.
            Currently supports:
            * "k_events" (new voxels are formed every k events)
            * "t_seconds" (new voxels are formed every t seconds)
            * "between_frames" (all events between frames are taken, requires frames to exist)
            A sliding window width must be given for k_events and t_seconds,
            which determines overlap (no overlap if set to 0). Eg:
            method={'method':'k_events', 'k':10000, 'sliding_window_w':100}
            method={'method':'t_events', 't':0.5, 'sliding_window_t':0.1}
            method={'method':'between_frames'}
            Default is 'between_frames'.
    """

    def get_s0(self, index):
        """
        Get s0 at index
        """
        raise NotImplementedError

    def get_s1(self, index):
        """
        Get s1 at index
        """
        raise NotImplementedError

    def get_s2(self, index):
        """
        Get s2 at index
        """
        raise NotImplementedError

    def get_intensity(self, index):
        """
        Get intensity at index
        """
        raise NotImplementedError

    def get_aolp(self, index):
        """
        Get aolp at index
        """
        raise NotImplementedError

    def get_dolp(self, index):
        """
        Get dolp at index
        """
        raise NotImplementedError

    def get_flow(self, index):
        """
        Get optic flow at index
        """
        raise NotImplementedError

    def get_events(self, idx0, idx1):
        """
        Get events between idx0, idx1
        """
        raise NotImplementedError

    def load_data(self, data_path):
        """
        Perform initialization tasks and ensure essential members are populated.
        Required members are:
            members are filled:
            self.sensor_resolution - the sensor resolution
            self.has_flow - if this dataset has optic flow
            self.t0 - timestamp of first event
            self.tk - timestamp of last event
            self.num_events - the total number of events
            self.frame_ts - list of the timestamps of the frames
            self.num_frames - the number of frames
        """
        raise NotImplementedError

    def find_ts_index(self, timestamp):
        """
        Given a timestamp, find the event index
        """
        raise NotImplementedError


    def __init__(self, data_path, transforms={}, sensor_resolution=None, num_bins=5,
                 voxel_method=None, max_length=None, combined_voxel_channels=True,
                 filter_hot_events=False):
        """
        self.transform applies to event voxels, frames and flow.
        self.vox_transform applies to event voxels only.
        """

        self.num_bins = num_bins
        self.data_path = data_path
        self.combined_voxel_channels = combined_voxel_channels
        self.sensor_resolution = tuple(sensor_resolution)
        self.data_source_idx = -1
        self.has_s0 = False
        self.has_s1 = False
        self.has_s2 = False
        self.has_intensity = False
        self.has_aolp = False
        self.has_dolp = False
        self.has_flow = False
        self.channels = self.num_bins if combined_voxel_channels else self.num_bins*2

        self.t0, self.tk, self.num_events, self.frame_ts, self.num_frames = None, None, None, None, None

        self.load_data(data_path)

        # if self.sensor_resolution is None or self.t0 is None or self.tk is None or self.num_events is None or self.frame_ts is None or self.num_frames is None:
        if self.sensor_resolution is None or self.t0 is None or self.tk is None or self.num_events is None:
            raise Exception("Dataloader failed to initialize all required members ({})".format(self.data_path))

        self.num_pixels = self.sensor_resolution[0] * self.sensor_resolution[1]
        self.duration = self.tk - self.t0

        if filter_hot_events:
            secs_for_hot_mask = 0.2
            hot_pix_percent = 0.01
            hot_num = min(self.find_ts_index(secs_for_hot_mask+self.t0), self.num_events)
            xs, ys, ts, ps = self.get_events(0, hot_num)
            self.hot_events_mask = get_hot_event_mask(xs.astype(np.int), ys.astype(np.int), ps, self.sensor_resolution, num_hot=int(self.num_pixels*hot_pix_percent))
            self.hot_events_mask = np.stack([self.hot_events_mask]*self.channels, axis=2).transpose(2, 0, 1)
        else:
            self.hot_events_mask = np.ones([self.channels, *self.sensor_resolution])
        self.hot_events_mask = torch.from_numpy(self.hot_events_mask).float()

        if voxel_method is None:
            voxel_method = {'method': 'between_frames'}
        self.set_voxel_method(voxel_method)

        if 'LegacyNorm' in transforms.keys() and 'RobustNorm' in transforms.keys():
            raise Exception('Cannot specify both LegacyNorm and RobustNorm')

        self.normalize_voxels = False
        for norm in ['RobustNorm', 'LegacyNorm']:
            if norm in transforms.keys():
                vox_transforms_list = [eval(t)(**kwargs) for t, kwargs in transforms.items()]
                del (transforms[norm])
                self.normalize_voxels = True
                self.vox_transform = Compose(vox_transforms_list)
                break

        transforms_list = [eval(t)(**kwargs) for t, kwargs in transforms.items()]

        if len(transforms_list) == 0:
            self.transform = None
        elif len(transforms_list) == 1:
            self.transform = transforms_list[0]
        else:
            self.transform = Compose(transforms_list)
        if not self.normalize_voxels:
            self.vox_transform = self.transform

        # print(self.transform)
        # print(self.vox_transform)
        # exit(0)

        if max_length is not None:
            self.length = min(self.length, max_length + 1)

    def __getitem__(self, index, seed=None):
        """
        Get data at index.
            :param index: index of data
            :param seed: random seed for data augmentation
        """
        assert 0 <= index < self.__len__(), "index {} out of bounds (0 <= x < {})".format(index, self.__len__())
        seed = random.randint(0, 2 ** 32) if seed is None else seed

        idx0, idx1 = self.get_event_indices(index)
        xs, ys, ts, ps = self.get_events(idx0, idx1)
        try:
            ts_0, ts_k = ts[0], ts[-1]
        except:
            ts_0, ts_k = 0, 0
        if len(xs) < 3:
            # print('xs is less than 3')
            voxel = self.get_empty_voxel_grid(self.combined_voxel_channels)
        else:
            xs = torch.from_numpy(xs.astype(np.float32))
            ys = torch.from_numpy(ys.astype(np.float32))
            ts = torch.from_numpy((ts-ts_0).astype(np.float32))
            ps = torch.from_numpy(ps.astype(np.float32))

            # added by Haiyang Mei, to prevent Inf or Nan in the input voxel
            if ts_k - ts_0 == 0:
                voxel = self.get_empty_voxel_grid(self.combined_voxel_channels)
                # print('dt is zero')
            else:
                voxel = self.get_voxel_grid(xs, ys, ts, ps, combined_voxel_channels=self.combined_voxel_channels)

        # events voxel
        voxel = self.transform_voxel(voxel, seed).float()
        dt = ts_k - ts_0
        if dt == 0:
            dt = np.array(0.0)
        # print('dt is:', dt)

        # print("Get voxel: event_t0={}, event_tk={}, image_ts={}".format(ts_0, ts_k, self.frame_ts[index]))
        if self.voxel_method['method'] == 'between_frames':
            # s0
            if self.has_s0:
                s0 = self.get_s0(index)
                s0 = self.transform_s0(s0, seed)
            else:
                print('no s0')
                s0 = torch.zeros((1, int(self.sensor_resolution[0] / 2), int(self.sensor_resolution[0] / 2)))

            # s1
            if self.has_s1:
                s1 = self.get_s1(index)
                s1 = self.transform_s1(s1, seed)
            else:
                print('no s1')
                s1 = torch.zeros((1, int(self.sensor_resolution[0] / 2), int(self.sensor_resolution[0] / 2)))

            # s2
            if self.has_s2:
                s2 = self.get_s2(index)
                s2 = self.transform_s2(s2, seed)
            else:
                print('no s2')
                s2 = torch.zeros((1, int(self.sensor_resolution[0] / 2), int(self.sensor_resolution[0] / 2)))

            # intensity
            if self.has_intensity:
                intensity = self.get_intensity(index)
                intensity = self.transform_intensity(intensity, seed)
            else:
                print('no intensity')
                intensity = torch.zeros((1, int(self.sensor_resolution[0] / 2), int(self.sensor_resolution[0] / 2)))

            # aolp
            if self.has_aolp:
                aolp = self.get_aolp(index)
                aolp = self.transform_aolp(aolp, seed)
            else:
                print('no aolp')
                aolp = torch.zeros((1, int(self.sensor_resolution[0] / 2), int(self.sensor_resolution[0] / 2)))

            # dolp
            if self.has_dolp:
                dolp = self.get_dolp(index)
                dolp = self.transform_dolp(dolp, seed)
            else:
                print('no dolp')
                dolp = torch.zeros((1, int(self.sensor_resolution[0] / 2), int(self.sensor_resolution[0] / 2)))

            # flow
            if self.has_flow:
                flow = self.get_flow(index)
                # convert to displacement (pix)
                # print(dt)
                # print(flow)
                # exit(0)
                # flow = flow / 33333 * dt
                flow = flow / 5000 * dt
                # print(np.max(flow))
                # print(np.min(flow))
                flow = self.transform_flow(flow, seed)
            else:
                flow = torch.zeros((2, intensity.shape[-2], intensity.shape[-1]), dtype=intensity.dtype, device=intensity.device)

            timestamp = torch.tensor(self.frame_ts[index], dtype=torch.float64)
            item = {'s0': s0,
                    's1': s1,
                    's2': s2,
                    'intensity': intensity,
                    'aolp': aolp,
                    'dolp': dolp,
                    'flow': flow,
                    'events': voxel,
                    'timestamp': timestamp,
                    'data_source_idx': self.data_source_idx,
                    'dt': torch.tensor(dt, dtype=torch.float64)}
        else:
            # print("Not between")
            item = {'events': voxel,
                    'timestamp': torch.tensor(ts_k, dtype=torch.float64),
                    'data_source_idx': self.data_source_idx,
                    'dt': torch.tensor(dt, dtype=torch.float64)}
        return item

    def compute_frame_indices(self):
        """
        For each frame, find the start and end indices of the
        time synchronized events
        """
        frame_indices = []
        start_idx = 0
        for ts in self.frame_ts:
            end_index = self.find_ts_index(ts)
            frame_indices.append([start_idx, end_index])
            start_idx = end_index
        return frame_indices

    def compute_timeblock_indices(self):
        """
        For each block of time (using t_events), find the start and
        end indices of the corresponding events
        """
        timeblock_indices = []
        start_idx = 0
        for i in range(self.__len__()):
            start_time = ((self.voxel_method['t'] - self.voxel_method['sliding_window_t']) * i) + self.t0
            end_time = start_time + self.voxel_method['t']
            end_idx = self.find_ts_index(end_time)
            timeblock_indices.append([start_idx, end_idx])
            start_idx = end_idx
        return timeblock_indices

    def compute_k_indices(self):
        """
        For each block of k events, find the start and
        end indices of the corresponding events
        """
        k_indices = []
        start_idx = 0
        for i in range(self.__len__()):
            idx0 = (self.voxel_method['k'] - self.voxel_method['sliding_window_w']) * i
            idx1 = idx0 + self.voxel_method['k']
            k_indices.append([idx0, idx1])
        return k_indices

    def set_voxel_method(self, voxel_method):
        """
        Given the desired method of computing voxels,
        compute the event_indices lookup table and dataset length
        """
        self.voxel_method = voxel_method
        if self.voxel_method['method'] == 'k_events':
            self.length = max(int(self.num_events / (voxel_method['k'] - voxel_method['sliding_window_w'])), 0)
            self.event_indices = self.compute_k_indices()
        elif self.voxel_method['method'] == 't_seconds':
            # print(voxel_method['t'])
            # print(voxel_method['sliding_window_t'])
            self.length = max(int(self.duration / (voxel_method['t'] - voxel_method['sliding_window_t'])), 0)
            print(self.length)
            self.event_indices = self.compute_timeblock_indices()
            # print('ok')
        elif self.voxel_method['method'] == 'between_frames':
            self.length = self.num_frames - 1
            self.event_indices = self.compute_frame_indices()
        else:
            raise Exception("Invalid voxel forming method chosen ({})".format(self.voxel_method))
        if self.length == 0:
            raise Exception("Current voxel generation parameters lead to sequence length of zero")

    def __len__(self):
        return self.length

    def get_event_indices(self, index):
        """
        Get start and end indices of events at index
        """
        idx0, idx1 = self.event_indices[index]
        if not (idx0 >= 0 and idx1 <= self.num_events):
            raise Exception("WARNING: Event indices {},{} out of bounds 0,{}".format(idx0, idx1, self.num_events))
        return idx0, idx1

    def get_empty_voxel_grid(self, combined_voxel_channels=True):
        """Return an empty voxel grid filled with zeros"""
        if combined_voxel_channels:
            size = (self.num_bins, *self.sensor_resolution)
        else:
            size = (2*self.num_bins, *self.sensor_resolution)
        return torch.zeros(size, dtype=torch.float32)

    def get_voxel_grid(self, xs, ys, ts, ps, combined_voxel_channels=True):
        """
        Given events, return voxel grid
        :param xs: tensor containg x coords of events
        :param ys: tensor containg y coords of events
        :param ts: tensor containg t coords of events
        :param ps: tensor containg p coords of events
        :param combined_voxel_channels: if True, create voxel grid merging positive and
            negative events (resulting in NUM_BINS x H x W tensor). Otherwise, create
            voxel grid for positive and negative events separately
            (resulting in 2*NUM_BINS x H x W tensor)
        """
        if combined_voxel_channels:
            # generate voxel grid which has size self.num_bins x H x W
            voxel_grid = events_to_voxel_torch(xs, ys, ts, ps, self.num_bins, sensor_size=self.sensor_resolution)
        else:
            # generate voxel grid which has size 2*self.num_bins x H x W
            voxel_grid = events_to_neg_pos_voxel_torch(xs, ys, ts, ps, self.num_bins,
                                                       sensor_size=self.sensor_resolution)
            voxel_grid = torch.cat([voxel_grid[0], voxel_grid[1]], 0)

        voxel_grid = voxel_grid * self.hot_events_mask

        return voxel_grid

    def transform_voxel(self, voxel, seed):
        """
        Augment voxel and turn into tensor
        """
        if self.vox_transform:
            random.seed(seed)
            voxel = self.vox_transform(voxel)
        return voxel

    def transform_s0(self, s0, seed):
        """
        Augment s0 and turn into tensor
        """
        s0 = torch.from_numpy(s0).float().unsqueeze(0) / 255
        if self.transform:
            random.seed(seed)
            s0 = self.transform(s0, is_polarization=True)
        return s0

    def transform_s1(self, s1, seed):
        """
        Augment s1 and turn into tensor
        """
        s1 = torch.from_numpy(s1).float().unsqueeze(0) / 255
        if self.transform:
            random.seed(seed)
            s1 = self.transform(s1, is_polarization=True)
        return s1

    def transform_s2(self, s2, seed):
        """
        Augment s2 and turn into tensor
        """
        s2 = torch.from_numpy(s2).float().unsqueeze(0) / 255
        if self.transform:
            random.seed(seed)
            s2 = self.transform(s2, is_polarization=True)
        return s2

    def transform_intensity(self, intensity, seed):
        """
        Augment intensity and turn into tensor
        """
        intensity = torch.from_numpy(intensity).float().unsqueeze(0) / 255
        if self.transform:
            random.seed(seed)
            intensity = self.transform(intensity, is_polarization=True)
        return intensity

    def transform_aolp(self, aolp, seed):
        """
        Augment aolp and turn into tensor
        """
        aolp = torch.from_numpy(aolp).float().unsqueeze(0) / 255
        if self.transform:
            random.seed(seed)
            aolp = self.transform(aolp, is_polarization=True, is_aolp=True)
        return aolp

    def transform_dolp(self, dolp, seed):
        """
        Augment dolp and turn into tensor
        """
        dolp = torch.from_numpy(dolp).float().unsqueeze(0) / 255
        if self.transform:
            random.seed(seed)
            dolp = self.transform(dolp, is_polarization=True)
        return dolp

    def transform_flow(self, flow, seed):
        """
        Augment flow and turn into tensor
        """
        flow = torch.from_numpy(flow)  # should end up [2 x H x W]
        if self.transform:
            random.seed(seed)
            flow = self.transform(flow, is_polarization=True, is_flow=True)
        return flow


class DynamicH5Dataset(BaseVoxelDataset):
    """
    Dataloader for events saved in the Monash University HDF5 events format
    (see https://github.com/TimoStoff/event_utils for code to convert datasets)
    """

    def get_frame(self, index):
        return self.h5_file['images']['image{:09d}'.format(index)][:]

    def get_flow(self, index):
        return self.h5_file['flow']['flow{:09d}'.format(index)][:]

    def get_events(self, idx0, idx1):
        xs = self.h5_file['events/xs'][idx0:idx1]
        ys = self.h5_file['events/ys'][idx0:idx1]
        ts = self.h5_file['events/ts'][idx0:idx1]
        ps = self.h5_file['events/ps'][idx0:idx1] * 2.0 - 1.0
        return xs, ys, ts, ps

    def load_data(self, data_path):
        try:
            self.h5_file = h5py.File(data_path, 'r')
        except OSError as err:
            print("Couldn't open {}: {}".format(data_path, err))

        if self.sensor_resolution is None:
            self.sensor_resolution = self.h5_file.attrs['sensor_resolution'][0:2]
        else:
            self.sensor_resolution = self.sensor_resolution[0:2]
        print("sensor resolution = {}".format(self.sensor_resolution))
        self.has_flow = 'flow' in self.h5_file.keys() and len(self.h5_file['flow']) > 0
        self.t0 = self.h5_file['events/ts'][0]
        self.tk = self.h5_file['events/ts'][-1]
        self.num_events = self.h5_file.attrs["num_events"]
        self.num_frames = self.h5_file.attrs["num_imgs"]

        self.frame_ts = []
        for img_name in self.h5_file['images']:
            self.frame_ts.append(self.h5_file['images/{}'.format(img_name)].attrs['timestamp'])

        data_source = self.h5_file.attrs.get('source', 'unknown')
        try:
            self.data_source_idx = data_sources.index(data_source)
        except ValueError:
            self.data_source_idx = -1

    def find_ts_index(self, timestamp):
        idx = binary_search_h5_dset(self.h5_file['events/ts'], timestamp)
        return idx

    def compute_frame_indices(self):
        frame_indices = []
        start_idx = 0
        for img_name in self.h5_file['images']:
            end_idx = self.h5_file['images/{}'.format(img_name)].attrs['event_idx']
            frame_indices.append([start_idx, end_idx])
            start_idx = end_idx
        return frame_indices


class DynamicH5Dataset_v2e(BaseVoxelDataset):
    """
    Dataloader for events saved in the v2e HDF5 events format
    """

    def get_frame(self, index):
        return self.h5_file['/frame'][index, :, :]

    def get_flow(self, index):
        return self.h5_file['/frame_flow'][index, :, :, :].transpose(2, 0, 1)
        # return self.h5_file['/flow'][index, :, :, :].transpose(2, 0, 1)

    def get_events(self, idx0, idx1):
        ts = self.h5_file['/events'][idx0:idx1][:, 0]
        xs = self.h5_file['/events'][idx0:idx1][:, 1]
        ys = self.h5_file['/events'][idx0:idx1][:, 2]
        # ps = self.h5_file['/events'][idx0:idx1][:, 3] * 2.0 - 1.0
        ps = self.h5_file['/events'][idx0:idx1][:, 3]
        ps = ps * 2.0 - 1.0
        return xs, ys, ts, ps

    def load_data(self, data_path):
        try:
            self.h5_file = h5py.File(data_path, 'r')
        except OSError as err:
            print("Couldn't open {}: {}".format(data_path, err))

        if self.sensor_resolution is None:
            print("Please specify sensor resolution!")
            raise ValueError
        else:
            print("sensor resolution = {}".format(self.sensor_resolution))
        # self.has_flow = 'frame_flow' in self.h5_file.keys() and len(self.h5_file['/frame_flow']) > 0
        self.has_flow = False
        # self.has_flow = 'flow' in self.h5_file.keys() and len(self.h5_file['/flow']) > 0
        self.t0 = self.h5_file['/events'][0, 0]
        self.tk = self.h5_file['/events'][-1, 0]
        self.num_events = self.h5_file['/events'].shape[0]
        self.num_frames = self.h5_file['/frame'].shape[0]

        self.frame_ts = self.h5_file['/frame_ts']

        # data_source = self.h5_file.attrs.get('source', 'unknown')
        data_source = 'rain'
        try:
            self.data_source_idx = data_sources.index(data_source)
        except ValueError:
            self.data_source_idx = -1

    def find_ts_index(self, timestamp):
        idx = binary_search_h5_dset(self.h5_file['/events'][:, 0], timestamp)
        return idx

    def compute_frame_indices(self):
        frame_indices = []
        start_idx = 0
        for idx in self.h5_file['/frame_idx']:
            end_idx = idx
            frame_indices.append([start_idx, end_idx])
            start_idx = end_idx
        return frame_indices


class DynamicH5Dataset_v2e_90(BaseVoxelDataset):
    """
    Dataloader for events saved in the v2e HDF5 events format
    """

    def get_frame(self, index):
        return self.h5_file['/frame'][index, :, :]

    def get_flow(self, index):
        return self.h5_file['/frame_flow'][index, :, :, :].transpose(2, 0, 1)
        # return self.h5_file['/flow'][index, :, :, :].transpose(2, 0, 1)

    def get_events(self, idx0, idx1):
        ts = self.h5_file['/events'][idx0:idx1][:, 0]
        xs = self.h5_file['/events'][idx0:idx1][:, 1]
        ys = self.h5_file['/events'][idx0:idx1][:, 2]
        ps = self.h5_file['/events'][idx0:idx1][:, 3] * 2.0 - 1.0

        # extract events under 90 degree only
        condition_x = (xs % 2 == 0)
        condition_y = (ys % 2 == 0)
        condition = condition_x * condition_y

        xs = (xs[condition] / 2).astype(np.int32)
        ys = (ys[condition] / 2).astype(np.int32)
        ts = ts[condition]
        ps = ps[condition]

        return xs, ys, ts, ps

    def load_data(self, data_path):
        try:
            self.h5_file = h5py.File(data_path, 'r')
        except OSError as err:
            print("Couldn't open {}: {}".format(data_path, err))

        if self.sensor_resolution is None:
            print("Please specify sensor resolution!")
            raise ValueError
        else:
            print("sensor resolution = {}".format(self.sensor_resolution))
        # self.has_flow = 'frame_flow' in self.h5_file.keys() and len(self.h5_file['/frame_flow']) > 0
        self.has_flow = False
        # self.has_flow = 'flow' in self.h5_file.keys() and len(self.h5_file['/flow']) > 0
        self.t0 = self.h5_file['/events'][0, 0]
        self.tk = self.h5_file['/events'][-1, 0]
        self.num_events = self.h5_file['/events'].shape[0]
        self.num_frames = self.h5_file['/frame'].shape[0]

        self.frame_ts = self.h5_file['/frame_ts']

        # data_source = self.h5_file.attrs.get('source', 'unknown')
        data_source = 'rain'
        try:
            self.data_source_idx = data_sources.index(data_source)
        except ValueError:
            self.data_source_idx = -1

    def find_ts_index(self, timestamp):
        idx = binary_search_h5_dset(self.h5_file['/events'][:, 0], timestamp)
        return idx

    def compute_frame_indices(self):
        frame_indices = []
        start_idx = 0
        for idx in self.h5_file['/frame_idx']:
            end_idx = idx
            frame_indices.append([start_idx, end_idx])
            start_idx = end_idx
        return frame_indices


class DynamicH5Dataset_v2e_45(BaseVoxelDataset):
    """
    Dataloader for events saved in the v2e HDF5 events format
    """

    def get_frame(self, index):
        return self.h5_file['/frame'][index, :, :]

    def get_flow(self, index):
        return self.h5_file['/frame_flow'][index, :, :, :].transpose(2, 0, 1)
        # return self.h5_file['/flow'][index, :, :, :].transpose(2, 0, 1)

    def get_events(self, idx0, idx1):
        ts = self.h5_file['/events'][idx0:idx1][:, 0]
        xs = self.h5_file['/events'][idx0:idx1][:, 1]
        ys = self.h5_file['/events'][idx0:idx1][:, 2]
        ps = self.h5_file['/events'][idx0:idx1][:, 3] * 2.0 - 1.0

        # extract events under 45 degree only
        condition_x = (xs % 2 == 1)
        condition_y = (ys % 2 == 0)
        condition = condition_x * condition_y

        xs = (xs[condition] / 2).astype(np.int32)
        ys = (ys[condition] / 2).astype(np.int32)
        ts = ts[condition]
        ps = ps[condition]

        return xs, ys, ts, ps

    def load_data(self, data_path):
        try:
            self.h5_file = h5py.File(data_path, 'r')
        except OSError as err:
            print("Couldn't open {}: {}".format(data_path, err))

        if self.sensor_resolution is None:
            print("Please specify sensor resolution!")
            raise ValueError
        else:
            print("sensor resolution = {}".format(self.sensor_resolution))
        # self.has_flow = 'frame_flow' in self.h5_file.keys() and len(self.h5_file['/frame_flow']) > 0
        self.has_flow = False
        # self.has_flow = 'flow' in self.h5_file.keys() and len(self.h5_file['/flow']) > 0
        self.t0 = self.h5_file['/events'][0, 0]
        self.tk = self.h5_file['/events'][-1, 0]
        self.num_events = self.h5_file['/events'].shape[0]
        self.num_frames = self.h5_file['/frame'].shape[0]

        self.frame_ts = self.h5_file['/frame_ts']

        # data_source = self.h5_file.attrs.get('source', 'unknown')
        data_source = 'rain'
        try:
            self.data_source_idx = data_sources.index(data_source)
        except ValueError:
            self.data_source_idx = -1

    def find_ts_index(self, timestamp):
        idx = binary_search_h5_dset(self.h5_file['/events'][:, 0], timestamp)
        return idx

    def compute_frame_indices(self):
        frame_indices = []
        start_idx = 0
        for idx in self.h5_file['/frame_idx']:
            end_idx = idx
            frame_indices.append([start_idx, end_idx])
            start_idx = end_idx
        return frame_indices


class DynamicH5Dataset_v2e_135(BaseVoxelDataset):
    """
    Dataloader for events saved in the v2e HDF5 events format
    """

    def get_frame(self, index):
        return self.h5_file['/frame'][index, :, :]

    def get_flow(self, index):
        return self.h5_file['/frame_flow'][index, :, :, :].transpose(2, 0, 1)
        # return self.h5_file['/flow'][index, :, :, :].transpose(2, 0, 1)

    def get_events(self, idx0, idx1):
        ts = self.h5_file['/events'][idx0:idx1][:, 0]
        xs = self.h5_file['/events'][idx0:idx1][:, 1]
        ys = self.h5_file['/events'][idx0:idx1][:, 2]
        ps = self.h5_file['/events'][idx0:idx1][:, 3] * 2.0 - 1.0

        # extract events under 135 degree only
        condition_x = (xs % 2 == 0)
        condition_y = (ys % 2 == 1)
        condition = condition_x * condition_y

        xs = (xs[condition] / 2).astype(np.int32)
        ys = (ys[condition] / 2).astype(np.int32)
        ts = ts[condition]
        ps = ps[condition]

        return xs, ys, ts, ps

    def load_data(self, data_path):
        try:
            self.h5_file = h5py.File(data_path, 'r')
        except OSError as err:
            print("Couldn't open {}: {}".format(data_path, err))

        if self.sensor_resolution is None:
            print("Please specify sensor resolution!")
            raise ValueError
        else:
            print("sensor resolution = {}".format(self.sensor_resolution))
        # self.has_flow = 'frame_flow' in self.h5_file.keys() and len(self.h5_file['/frame_flow']) > 0
        self.has_flow = False
        # self.has_flow = 'flow' in self.h5_file.keys() and len(self.h5_file['/flow']) > 0
        self.t0 = self.h5_file['/events'][0, 0]
        self.tk = self.h5_file['/events'][-1, 0]
        self.num_events = self.h5_file['/events'].shape[0]
        self.num_frames = self.h5_file['/frame'].shape[0]

        self.frame_ts = self.h5_file['/frame_ts']

        # data_source = self.h5_file.attrs.get('source', 'unknown')
        data_source = 'rain'
        try:
            self.data_source_idx = data_sources.index(data_source)
        except ValueError:
            self.data_source_idx = -1

    def find_ts_index(self, timestamp):
        idx = binary_search_h5_dset(self.h5_file['/events'][:, 0], timestamp)
        return idx

    def compute_frame_indices(self):
        frame_indices = []
        start_idx = 0
        for idx in self.h5_file['/frame_idx']:
            end_idx = idx
            frame_indices.append([start_idx, end_idx])
            start_idx = end_idx
        return frame_indices


class DynamicH5Dataset_v2e_0(BaseVoxelDataset):
    """
    Dataloader for events saved in the v2e HDF5 events format
    """

    def get_frame(self, index):
        return self.h5_file['/frame'][index, :, :]

    def get_flow(self, index):
        return self.h5_file['/frame_flow'][index, :, :, :].transpose(2, 0, 1)
        # return self.h5_file['/flow'][index, :, :, :].transpose(2, 0, 1)

    def get_events(self, idx0, idx1):
        ts = self.h5_file['/events'][idx0:idx1][:, 0]
        xs = self.h5_file['/events'][idx0:idx1][:, 1]
        ys = self.h5_file['/events'][idx0:idx1][:, 2]
        ps = self.h5_file['/events'][idx0:idx1][:, 3] * 2.0 - 1.0

        # extract events under 90 degree only
        condition_x = (xs % 2 == 1)
        condition_y = (ys % 2 == 1)
        condition = condition_x * condition_y

        xs = (xs[condition] / 2).astype(np.int32)
        ys = (ys[condition] / 2).astype(np.int32)
        ts = ts[condition]
        ps = ps[condition]

        return xs, ys, ts, ps

    def load_data(self, data_path):
        try:
            self.h5_file = h5py.File(data_path, 'r')
        except OSError as err:
            print("Couldn't open {}: {}".format(data_path, err))

        if self.sensor_resolution is None:
            print("Please specify sensor resolution!")
            raise ValueError
        else:
            print("sensor resolution = {}".format(self.sensor_resolution))
        # self.has_flow = 'frame_flow' in self.h5_file.keys() and len(self.h5_file['/frame_flow']) > 0
        self.has_flow = False
        # self.has_flow = 'flow' in self.h5_file.keys() and len(self.h5_file['/flow']) > 0
        self.t0 = self.h5_file['/events'][0, 0]
        self.tk = self.h5_file['/events'][-1, 0]
        self.num_events = self.h5_file['/events'].shape[0]
        self.num_frames = self.h5_file['/frame'].shape[0]

        self.frame_ts = self.h5_file['/frame_ts']

        # data_source = self.h5_file.attrs.get('source', 'unknown')
        data_source = 'rain'
        try:
            self.data_source_idx = data_sources.index(data_source)
        except ValueError:
            self.data_source_idx = -1

    def find_ts_index(self, timestamp):
        idx = binary_search_h5_dset(self.h5_file['/events'][:, 0], timestamp)
        return idx

    def compute_frame_indices(self):
        frame_indices = []
        start_idx = 0
        for idx in self.h5_file['/frame_idx']:
            end_idx = idx
            frame_indices.append([start_idx, end_idx])
            start_idx = end_idx
        return frame_indices


class DynamicH5Dataset_v2e_p(BaseVoxelDataset_p):
    """
    Dataloader for events saved in the v2e polarization HDF5 events format
    """

    def get_intensity(self, index):
        return self.h5_file['/intensity'][index, :, :]

    def get_aolp(self, index):
        return self.h5_file['/aolp'][index, :, :]

    def get_dolp(self, index):
        return self.h5_file['/dolp'][index, :, :]

    def get_events(self, idx0, idx1):
        ts = self.h5_file['/events'][idx0:idx1][:, 0]
        xs = self.h5_file['/events'][idx0:idx1][:, 1]
        ys = self.h5_file['/events'][idx0:idx1][:, 2]
        ps = self.h5_file['/events'][idx0:idx1][:, 3] * 2.0 - 1.0
        return xs, ys, ts, ps

    def get_flow(self, index):
        return self.h5_file['/flow'][index, :, :, :].transpose(2, 0, 1)

    def load_data(self, data_path):
        try:
            self.h5_file = h5py.File(data_path, 'r')
        except OSError as err:
            print("Couldn't open {}: {}".format(data_path, err))

        self.sensor_resolution = tuple(self.h5_file.attrs['sensor_resolution'])[::-1]

        if self.sensor_resolution is None:
            print("Please specify sensor resolution!")
            raise ValueError
        else:
            print("sensor resolution = {}".format(self.sensor_resolution))
        self.has_intensity = 'intensity' in self.h5_file.keys() and len(self.h5_file['/intensity']) > 0
        self.has_aolp = 'aolp' in self.h5_file.keys() and len(self.h5_file['/aolp']) > 0
        self.has_dolp = 'dolp' in self.h5_file.keys() and len(self.h5_file['/dolp']) > 0
        self.has_flow = 'flow' in self.h5_file.keys() and len(self.h5_file['/flow']) > 0
        self.t0 = self.h5_file['/events'][0, 0]
        self.tk = self.h5_file['/events'][-1, 0]
        self.num_events = self.h5_file['/events'].shape[0]

        # here should be commented if test on real events
        self.num_frames = self.h5_file['/frame'].shape[0]
        self.frame_ts = self.h5_file['/frame_ts']

        data_source = self.h5_file.attrs.get('source', 'rain')
        try:
            self.data_source_idx = data_sources.index(data_source)
        except ValueError:
            self.data_source_idx = -1

    def find_ts_index(self, timestamp):
        idx = binary_search_h5_dset(self.h5_file['/events'][:, 0], timestamp)
        return idx

    # comment for real data
    def compute_frame_indices(self):
        frame_indices = []
        start_idx = 0
        for idx in self.h5_file['/frame_idx']:
            end_idx = idx
            frame_indices.append([start_idx, end_idx])
            start_idx = end_idx
        return frame_indices


class DynamicH5Dataset_v2e_rp(BaseVoxelDataset_rp):
    """
    Dataloader for events saved in the v2e polarization HDF5 events format
    """

    def get_raw(self, index):
        return self.h5_file['/frame'][index, :, :]

    def get_intensity(self, index):
        return self.h5_file['/intensity'][index, :, :]

    def get_aolp(self, index):
        return self.h5_file['/aolp'][index, :, :]

    def get_dolp(self, index):
        return self.h5_file['/dolp'][index, :, :]

    def get_events(self, idx0, idx1):
        ts = self.h5_file['/events'][idx0:idx1][:, 0]
        xs = self.h5_file['/events'][idx0:idx1][:, 1]
        ys = self.h5_file['/events'][idx0:idx1][:, 2]
        ps = self.h5_file['/events'][idx0:idx1][:, 3] * 2.0 - 1.0
        return xs, ys, ts, ps

    def get_flow(self, index):
        return self.h5_file['/flow'][index, :, :, :].transpose(2, 0, 1)

    def load_data(self, data_path):
        try:
            self.h5_file = h5py.File(data_path, 'r')
        except OSError as err:
            print("Couldn't open {}: {}".format(data_path, err))

        self.sensor_resolution = tuple(self.h5_file.attrs['sensor_resolution'])[::-1]

        if self.sensor_resolution is None:
            print("Please specify sensor resolution!")
            raise ValueError
        else:
            print("sensor resolution = {}".format(self.sensor_resolution))
        self.has_raw = 'frame' in self.h5_file.keys() and len(self.h5_file['/frame']) > 0
        self.has_intensity = 'intensity' in self.h5_file.keys() and len(self.h5_file['/intensity']) > 0
        self.has_aolp = 'aolp' in self.h5_file.keys() and len(self.h5_file['/aolp']) > 0
        self.has_dolp = 'dolp' in self.h5_file.keys() and len(self.h5_file['/dolp']) > 0
        self.has_flow = 'flow' in self.h5_file.keys() and len(self.h5_file['/flow']) > 0
        self.t0 = self.h5_file['/events'][0, 0]
        self.tk = self.h5_file['/events'][-1, 0]
        self.num_events = self.h5_file['/events'].shape[0]

        # here should be commented if test on real events
        self.num_frames = self.h5_file['/frame'].shape[0]
        self.frame_ts = self.h5_file['/frame_ts']

        data_source = self.h5_file.attrs.get('source', 'rain')
        try:
            self.data_source_idx = data_sources.index(data_source)
        except ValueError:
            self.data_source_idx = -1

    def find_ts_index(self, timestamp):
        idx = binary_search_h5_dset(self.h5_file['/events'][:, 0], timestamp)
        return idx

    def compute_frame_indices(self):
        frame_indices = []
        start_idx = 0
        for idx in self.h5_file['/frame_idx']:
            end_idx = idx
            frame_indices.append([start_idx, end_idx])
            start_idx = end_idx
        return frame_indices


class DynamicH5Dataset_v2e_s012_iad(BaseVoxelDataset_s012_iad):
    """
    Dataloader for events saved in the v2e polarization HDF5 events format
    s012+iad
    """

    def get_s0(self, index):
        return self.h5_file['/s0'][index, :, :]

    def get_s1(self, index):
        return self.h5_file['/s1'][index, :, :]

    def get_s2(self, index):
        return self.h5_file['/s2'][index, :, :]

    def get_intensity(self, index):
        return self.h5_file['/intensity'][index, :, :]

    def get_aolp(self, index):
        return self.h5_file['/aolp'][index, :, :]

    def get_dolp(self, index):
        return self.h5_file['/dolp'][index, :, :]

    def get_events(self, idx0, idx1):
        ts = self.h5_file['/events'][idx0:idx1][:, 0]
        xs = self.h5_file['/events'][idx0:idx1][:, 1]
        ys = self.h5_file['/events'][idx0:idx1][:, 2]
        ps = self.h5_file['/events'][idx0:idx1][:, 3] * 2.0 - 1.0
        return xs, ys, ts, ps

    def get_flow(self, index):
        return self.h5_file['/flow'][index, :, :, :].transpose(2, 0, 1)

    def load_data(self, data_path):
        try:
            self.h5_file = h5py.File(data_path, 'r')
        except OSError as err:
            print("Couldn't open {}: {}".format(data_path, err))

        if self.sensor_resolution is None:
            print("Please specify sensor resolution!")
            raise ValueError
        else:
            print("sensor resolution = {}".format(self.sensor_resolution))
        self.has_s0 = 's0' in self.h5_file.keys() and len(self.h5_file['/s0']) > 0
        self.has_s1 = 's1' in self.h5_file.keys() and len(self.h5_file['/s1']) > 0
        self.has_s2 = 's2' in self.h5_file.keys() and len(self.h5_file['/s2']) > 0
        self.has_intensity = 'intensity' in self.h5_file.keys() and len(self.h5_file['/intensity']) > 0
        self.has_aolp = 'aolp' in self.h5_file.keys() and len(self.h5_file['/aolp']) > 0
        self.has_dolp = 'dolp' in self.h5_file.keys() and len(self.h5_file['/dolp']) > 0
        self.has_flow = 'flow' in self.h5_file.keys() and len(self.h5_file['/flow']) > 0
        self.t0 = self.h5_file['/events'][0, 0]
        self.tk = self.h5_file['/events'][-1, 0]
        self.num_events = self.h5_file['/events'].shape[0]

        # here should be commented if test on real events
        self.num_frames = self.h5_file['/frame'].shape[0]
        self.frame_ts = self.h5_file['/frame_ts']

        data_source = self.h5_file.attrs.get('source', 'movingcam')
        try:
            self.data_source_idx = data_sources.index(data_source)
        except ValueError:
            self.data_source_idx = -1

    def find_ts_index(self, timestamp):
        idx = binary_search_h5_dset(self.h5_file['/events'][:, 0], timestamp)
        return idx

    def compute_frame_indices(self):
        frame_indices = []
        start_idx = 0
        for idx in self.h5_file['/frame_idx']:
            end_idx = idx
            frame_indices.append([start_idx, end_idx])
            start_idx = end_idx
        return frame_indices


class DynamicH5Dataset_v2e_s012_iad_real_data(BaseVoxelDataset_s012_iad):
    """
    Dataloader for events saved in the v2e polarization HDF5 events format
    s012+iad
    """

    def get_s0(self, index):
        return self.h5_file['/s0'][index, :, :]

    def get_s1(self, index):
        return self.h5_file['/s1'][index, :, :]

    def get_s2(self, index):
        return self.h5_file['/s2'][index, :, :]

    def get_intensity(self, index):
        return self.h5_file['/intensity'][index, :, :]

    def get_aolp(self, index):
        return self.h5_file['/aolp'][index, :, :]

    def get_dolp(self, index):
        return self.h5_file['/dolp'][index, :, :]

    def get_events(self, idx0, idx1):
        ts = self.h5_file['/events'][idx0:idx1][:, 0]
        xs = self.h5_file['/events'][idx0:idx1][:, 1]
        ys = self.h5_file['/events'][idx0:idx1][:, 2]
        ps = self.h5_file['/events'][idx0:idx1][:, 3] * 2.0 - 1.0
        return xs, ys, ts, ps

    def get_flow(self, index):
        return self.h5_file['/flow'][index, :, :, :].transpose(2, 0, 1)

    def load_data(self, data_path):
        try:
            self.h5_file = h5py.File(data_path, 'r')
        except OSError as err:
            print("Couldn't open {}: {}".format(data_path, err))

        if self.sensor_resolution is None:
            print("Please specify sensor resolution!")
            raise ValueError
        else:
            print("sensor resolution = {}".format(self.sensor_resolution))
        self.has_s0 = 's0' in self.h5_file.keys() and len(self.h5_file['/s0']) > 0
        self.has_s1 = 's1' in self.h5_file.keys() and len(self.h5_file['/s1']) > 0
        self.has_s2 = 's2' in self.h5_file.keys() and len(self.h5_file['/s2']) > 0
        self.has_intensity = 'intensity' in self.h5_file.keys() and len(self.h5_file['/intensity']) > 0
        self.has_aolp = 'aolp' in self.h5_file.keys() and len(self.h5_file['/aolp']) > 0
        self.has_dolp = 'dolp' in self.h5_file.keys() and len(self.h5_file['/dolp']) > 0
        self.has_flow = 'flow' in self.h5_file.keys() and len(self.h5_file['/flow']) > 0
        self.t0 = self.h5_file['/events'][0, 0]
        self.tk = self.h5_file['/events'][-1, 0]
        self.num_events = self.h5_file['/events'].shape[0]

        # here should be commented if test on real events
        # self.num_frames = self.h5_file['/frame'].shape[0]
        # self.frame_ts = self.h5_file['/frame_ts']

        data_source = self.h5_file.attrs.get('source', 'movingcam')
        try:
            self.data_source_idx = data_sources.index(data_source)
        except ValueError:
            self.data_source_idx = -1

    def find_ts_index(self, timestamp):
        idx = binary_search_h5_dset(self.h5_file['/events'][:, 0], timestamp)
        return idx

    def compute_frame_indices(self):
        frame_indices = []
        start_idx = 0
        for idx in self.h5_file['/frame_idx']:
            end_idx = idx
            frame_indices.append([start_idx, end_idx])
            start_idx = end_idx
        return frame_indices


class DynamicH5Dataset_v2e_p_real_data(BaseVoxelDataset_p):
    """
    Dataloader for events saved in the v2e polarization HDF5 events format
    """

    def get_intensity(self, index):
        return self.h5_file['/intensity'][index, :, :]

    def get_aolp(self, index):
        return self.h5_file['/aolp'][index, :, :]

    def get_dolp(self, index):
        return self.h5_file['/dolp'][index, :, :]

    def get_events(self, idx0, idx1):
        # training h5 format
        ts = self.h5_file['/events'][idx0:idx1][:, 0]
        xs = self.h5_file['/events'][idx0:idx1][:, 1]
        ys = self.h5_file['/events'][idx0:idx1][:, 2]
        ps = self.h5_file['/events'][idx0:idx1][:, 3] * 2.0 - 1.0
        # Timo format
        # xs = self.h5_file['events/xs'][idx0:idx1]
        # ys = self.h5_file['events/ys'][idx0:idx1]
        # ts = self.h5_file['events/ts'][idx0:idx1]
        # ps = self.h5_file['events/ps'][idx0:idx1] * 2.0 - 1.0
        return xs, ys, ts, ps

    def get_flow(self, index):
        return self.h5_file['/flow'][index, :, :, :].transpose(2, 0, 1)

    def load_data(self, data_path):
        try:
            self.h5_file = h5py.File(data_path, 'r')
        except OSError as err:
            print("Couldn't open {}: {}".format(data_path, err))

        if self.sensor_resolution is None:
            print("Please specify sensor resolution!")
            raise ValueError
        else:
            print("sensor resolution = {}".format(self.sensor_resolution))
        self.has_intensity = 'intensity' in self.h5_file.keys() and len(self.h5_file['/intensity']) > 0
        self.has_aolp = 'aolp' in self.h5_file.keys() and len(self.h5_file['/aolp']) > 0
        self.has_dolp = 'dolp' in self.h5_file.keys() and len(self.h5_file['/dolp']) > 0
        self.has_flow = 'flow' in self.h5_file.keys() and len(self.h5_file['/flow']) > 0
        # training h5 format
        self.t0 = self.h5_file['/events'][0, 0]
        self.tk = self.h5_file['/events'][-1, 0]
        self.num_events = self.h5_file['/events'].shape[0]
        # Timo format
        # self.t0 = self.h5_file['events/ts'][0]
        # self.tk = self.h5_file['events/ts'][-1]
        # self.num_events = self.h5_file.attrs["num_events"]

        data_source = self.h5_file.attrs.get('source', 'real')
        try:
            self.data_source_idx = data_sources.index(data_source)
        except ValueError:
            self.data_source_idx = -1

    def find_ts_index(self, timestamp):
        # training h5 format
        idx = binary_search_h5_dset(self.h5_file['/events'][:, 0], timestamp)
        # Timo format
        # idx = binary_search_h5_dset(self.h5_file['events/ts'], timestamp)
        return idx

    def compute_frame_indices(self):
        frame_indices = []
        start_idx = 0
        for idx in self.h5_file['/frame_idx']:
            end_idx = idx
            frame_indices.append([start_idx, end_idx])
            start_idx = end_idx
        return frame_indices


class DynamicH5Dataset_v2e_p_90(BaseVoxelDataset_p):
    """
    Dataloader for events saved in the v2e polarization HDF5 events format
    """

    def get_intensity(self, index):
        return self.h5_file['/intensity'][index, :, :]
        # return self.h5_file['/frame'][index, :, :]

    def get_aolp(self, index):
        return self.h5_file['/aolp'][index, :, :]
        # return self.h5_file['/frame'][index, :, :]

    def get_dolp(self, index):
        return self.h5_file['/dolp'][index, :, :]
        # return self.h5_file['/frame'][index, :, :]

    def get_events(self, idx0, idx1):
        ts = self.h5_file['/events'][idx0:idx1][:, 0]
        xs = self.h5_file['/events'][idx0:idx1][:, 1]
        ys = self.h5_file['/events'][idx0:idx1][:, 2]
        ps = self.h5_file['/events'][idx0:idx1][:, 3] * 2.0 - 1.0

        # extract events under 90 degree only
        condition_x = (xs % 2 == 0)
        condition_y = (ys % 2 == 0)
        condition = condition_x * condition_y

        xs = (xs[condition] / 2).astype(np.int32)
        ys = (ys[condition] / 2).astype(np.int32)
        ts = ts[condition]
        ps = ps[condition]

        return xs, ys, ts, ps

    def get_flow(self, index):
        return self.h5_file['/flow'][index, :, :, :].transpose(2, 0, 1)

    def load_data(self, data_path):
        try:
            self.h5_file = h5py.File(data_path, 'r')
        except OSError as err:
            print("Couldn't open {}: {}".format(data_path, err))

        self.sensor_resolution = tuple(self.h5_file.attrs['sensor_resolution'])[::-1]

        if self.sensor_resolution is None:
            print("Please specify sensor resolution!")
            raise ValueError
        else:
            print("sensor resolution = {}".format(self.sensor_resolution))
        self.has_intensity = 'intensity' in self.h5_file.keys() and len(self.h5_file['/intensity']) > 0
        self.has_aolp = 'aolp' in self.h5_file.keys() and len(self.h5_file['/aolp']) > 0
        self.has_dolp = 'dolp' in self.h5_file.keys() and len(self.h5_file['/dolp']) > 0
        self.has_flow = 'flow' in self.h5_file.keys() and len(self.h5_file['/flow']) > 0
        self.t0 = self.h5_file['/events'][0, 0]
        self.tk = self.h5_file['/events'][-1, 0]
        self.num_events = self.h5_file['/events'].shape[0]

        self.num_frames = self.h5_file['/frame'].shape[0]
        self.frame_ts = self.h5_file['/frame_ts']

        data_source = self.h5_file.attrs.get('source', 'unknown')
        try:
            self.data_source_idx = data_sources.index(data_source)
        except ValueError:
            self.data_source_idx = -1

    def find_ts_index(self, timestamp):
        idx = binary_search_h5_dset(self.h5_file['/events'][:, 0], timestamp)
        return idx

    def compute_frame_indices(self):
        frame_indices = []
        start_idx = 0
        for idx in self.h5_file['/frame_idx']:
            end_idx = idx
            frame_indices.append([start_idx, end_idx])
            start_idx = end_idx
        return frame_indices


class DynamicH5Dataset_v2e_p_45(BaseVoxelDataset_p):
    """
    Dataloader for events saved in the v2e polarization HDF5 events format
    """

    def get_intensity(self, index):
        return self.h5_file['/intensity'][index, :, :]

    def get_aolp(self, index):
        return self.h5_file['/aolp'][index, :, :]

    def get_dolp(self, index):
        return self.h5_file['/dolp'][index, :, :]

    def get_events(self, idx0, idx1):
        ts = self.h5_file['/events'][idx0:idx1][:, 0]
        xs = self.h5_file['/events'][idx0:idx1][:, 1]
        ys = self.h5_file['/events'][idx0:idx1][:, 2]
        ps = self.h5_file['/events'][idx0:idx1][:, 3] * 2.0 - 1.0

        # extract events under 45 degree only
        condition_x = (xs % 2 == 1)
        condition_y = (ys % 2 == 0)
        condition = condition_x * condition_y

        xs = (xs[condition] / 2).astype(np.int32)
        ys = (ys[condition] / 2).astype(np.int32)
        ts = ts[condition]
        ps = ps[condition]

        return xs, ys, ts, ps

    def get_flow(self, index):
        return self.h5_file['/flow'][index, :, :, :].transpose(2, 0, 1)

    def load_data(self, data_path):
        try:
            self.h5_file = h5py.File(data_path, 'r')
        except OSError as err:
            print("Couldn't open {}: {}".format(data_path, err))

        self.sensor_resolution = tuple(self.h5_file.attrs['sensor_resolution'])[::-1]

        if self.sensor_resolution is None:
            print("Please specify sensor resolution!")
            raise ValueError
        else:
            print("sensor resolution = {}".format(self.sensor_resolution))
        self.has_intensity = 'intensity' in self.h5_file.keys() and len(self.h5_file['/intensity']) > 0
        self.has_aolp = 'aolp' in self.h5_file.keys() and len(self.h5_file['/aolp']) > 0
        self.has_dolp = 'dolp' in self.h5_file.keys() and len(self.h5_file['/dolp']) > 0
        self.has_flow = 'flow' in self.h5_file.keys() and len(self.h5_file['/flow']) > 0
        self.t0 = self.h5_file['/events'][0, 0]
        self.tk = self.h5_file['/events'][-1, 0]
        self.num_events = self.h5_file['/events'].shape[0]

        self.num_frames = self.h5_file['/frame'].shape[0]
        self.frame_ts = self.h5_file['/frame_ts']

        data_source = self.h5_file.attrs.get('source', 'unknown')
        try:
            self.data_source_idx = data_sources.index(data_source)
        except ValueError:
            self.data_source_idx = -1

    def find_ts_index(self, timestamp):
        idx = binary_search_h5_dset(self.h5_file['/events'][:, 0], timestamp)
        return idx

    def compute_frame_indices(self):
        frame_indices = []
        start_idx = 0
        for idx in self.h5_file['/frame_idx']:
            end_idx = idx
            frame_indices.append([start_idx, end_idx])
            start_idx = end_idx
        return frame_indices


class DynamicH5Dataset_v2e_p_135(BaseVoxelDataset_p):
    """
    Dataloader for events saved in the v2e polarization HDF5 events format
    """

    def get_intensity(self, index):
        return self.h5_file['/intensity'][index, :, :]

    def get_aolp(self, index):
        return self.h5_file['/aolp'][index, :, :]

    def get_dolp(self, index):
        return self.h5_file['/dolp'][index, :, :]

    def get_events(self, idx0, idx1):
        ts = self.h5_file['/events'][idx0:idx1][:, 0]
        xs = self.h5_file['/events'][idx0:idx1][:, 1]
        ys = self.h5_file['/events'][idx0:idx1][:, 2]
        ps = self.h5_file['/events'][idx0:idx1][:, 3] * 2.0 - 1.0

        # extract events under 135 degree only
        condition_x = (xs % 2 == 0)
        condition_y = (ys % 2 == 1)
        condition = condition_x * condition_y

        xs = (xs[condition] / 2).astype(np.int32)
        ys = (ys[condition] / 2).astype(np.int32)
        ts = ts[condition]
        ps = ps[condition]

        return xs, ys, ts, ps

    def get_flow(self, index):
        return self.h5_file['/flow'][index, :, :, :].transpose(2, 0, 1)

    def load_data(self, data_path):
        try:
            self.h5_file = h5py.File(data_path, 'r')
        except OSError as err:
            print("Couldn't open {}: {}".format(data_path, err))

        self.sensor_resolution = tuple(self.h5_file.attrs['sensor_resolution'])[::-1]

        if self.sensor_resolution is None:
            print("Please specify sensor resolution!")
            raise ValueError
        else:
            print("sensor resolution = {}".format(self.sensor_resolution))
        self.has_intensity = 'intensity' in self.h5_file.keys() and len(self.h5_file['/intensity']) > 0
        self.has_aolp = 'aolp' in self.h5_file.keys() and len(self.h5_file['/aolp']) > 0
        self.has_dolp = 'dolp' in self.h5_file.keys() and len(self.h5_file['/dolp']) > 0
        self.has_flow = 'flow' in self.h5_file.keys() and len(self.h5_file['/flow']) > 0
        self.t0 = self.h5_file['/events'][0, 0]
        self.tk = self.h5_file['/events'][-1, 0]
        self.num_events = self.h5_file['/events'].shape[0]

        self.num_frames = self.h5_file['/frame'].shape[0]
        self.frame_ts = self.h5_file['/frame_ts']

        data_source = self.h5_file.attrs.get('source', 'unknown')
        try:
            self.data_source_idx = data_sources.index(data_source)
        except ValueError:
            self.data_source_idx = -1

    def find_ts_index(self, timestamp):
        idx = binary_search_h5_dset(self.h5_file['/events'][:, 0], timestamp)
        return idx

    def compute_frame_indices(self):
        frame_indices = []
        start_idx = 0
        for idx in self.h5_file['/frame_idx']:
            end_idx = idx
            frame_indices.append([start_idx, end_idx])
            start_idx = end_idx
        return frame_indices


class DynamicH5Dataset_v2e_p_0(BaseVoxelDataset_p):
    """
    Dataloader for events saved in the v2e polarization HDF5 events format
    """

    def get_intensity(self, index):
        return self.h5_file['/intensity'][index, :, :]

    def get_aolp(self, index):
        return self.h5_file['/aolp'][index, :, :]

    def get_dolp(self, index):
        return self.h5_file['/dolp'][index, :, :]

    def get_events(self, idx0, idx1):
        ts = self.h5_file['/events'][idx0:idx1][:, 0]
        xs = self.h5_file['/events'][idx0:idx1][:, 1]
        ys = self.h5_file['/events'][idx0:idx1][:, 2]
        ps = self.h5_file['/events'][idx0:idx1][:, 3] * 2.0 - 1.0

        # extract events under 0 degree only
        condition_x = (xs % 2 == 1)
        condition_y = (ys % 2 == 1)
        condition = condition_x * condition_y

        xs = (xs[condition] / 2).astype(np.int32)
        ys = (ys[condition] / 2).astype(np.int32)
        ts = ts[condition]
        ps = ps[condition]

        return xs, ys, ts, ps

    def get_flow(self, index):
        return self.h5_file['/flow'][index, :, :, :].transpose(2, 0, 1)

    def load_data(self, data_path):
        try:
            self.h5_file = h5py.File(data_path, 'r')
        except OSError as err:
            print("Couldn't open {}: {}".format(data_path, err))

        self.sensor_resolution = tuple(self.h5_file.attrs['sensor_resolution'])[::-1]

        if self.sensor_resolution is None:
            print("Please specify sensor resolution!")
            raise ValueError
        else:
            print("sensor resolution = {}".format(self.sensor_resolution))
        self.has_intensity = 'intensity' in self.h5_file.keys() and len(self.h5_file['/intensity']) > 0
        self.has_aolp = 'aolp' in self.h5_file.keys() and len(self.h5_file['/aolp']) > 0
        self.has_dolp = 'dolp' in self.h5_file.keys() and len(self.h5_file['/dolp']) > 0
        self.has_flow = 'flow' in self.h5_file.keys() and len(self.h5_file['/flow']) > 0
        self.t0 = self.h5_file['/events'][0, 0]
        self.tk = self.h5_file['/events'][-1, 0]
        self.num_events = self.h5_file['/events'].shape[0]

        self.num_frames = self.h5_file['/frame'].shape[0]
        self.frame_ts = self.h5_file['/frame_ts']

        data_source = self.h5_file.attrs.get('source', 'unknown')
        try:
            self.data_source_idx = data_sources.index(data_source)
        except ValueError:
            self.data_source_idx = -1

    def find_ts_index(self, timestamp):
        idx = binary_search_h5_dset(self.h5_file['/events'][:, 0], timestamp)
        return idx

    def compute_frame_indices(self):
        frame_indices = []
        start_idx = 0
        for idx in self.h5_file['/frame_idx']:
            end_idx = idx
            frame_indices.append([start_idx, end_idx])
            start_idx = end_idx
        return frame_indices


class MemMapDataset(BaseVoxelDataset):
    """
    Dataloader for events saved in the MemMap events format used at RPG.
    (see https://github.com/TimoStoff/event_utils for code to convert datasets)
    """

    def get_frame(self, index):
        frame = self.filehandle['images'][index][:, :, 0]
        return frame

    def get_flow(self, index):
        flow = self.filehandle['optic_flow'][index]
        return flow

    def get_events(self, idx0, idx1):
        xy = self.filehandle["xy"][idx0:idx1]
        xs = xy[:, 0].astype(np.float32)
        ys = xy[:, 1].astype(np.float32)
        ts = self.filehandle["t"][idx0:idx1]
        ps = self.filehandle["p"][idx0:idx1] * 2.0 - 1.0
        return xs, ys, ts, ps

    def load_data(self, data_path, timestamp_fname="timestamps.npy", image_fname="images.npy",
                  optic_flow_fname="optic_flow.npy", optic_flow_stamps_fname="optic_flow_stamps.npy",
                  t_fname="t.npy", xy_fname="xy.npy", p_fname="p.npy"):

        assert os.path.isdir(data_path), '%s is not a valid data_pathectory' % data_path

        data = {}
        self.has_flow = False
        for subroot, _, fnames in sorted(os.walk(data_path)):
            for fname in sorted(fnames):
                path = os.path.join(subroot, fname)
                if fname.endswith(".npy"):
                    if fname.endswith(timestamp_fname):
                        frame_stamps = np.load(path)
                        data["frame_stamps"] = frame_stamps
                    elif fname.endswith(image_fname):
                        data["images"] = np.load(path, mmap_mode="r")
                    elif fname.endswith(optic_flow_fname):
                        data["optic_flow"] = np.load(path, mmap_mode="r")
                        self.has_flow = True
                    elif fname.endswith(optic_flow_stamps_fname):
                        optic_flow_stamps = np.load(path)
                        data["optic_flow_stamps"] = optic_flow_stamps

                    try:
                        handle = np.load(path, mmap_mode="r")
                    except Exception as err:
                        print("Couldn't load {}:".format(path))
                        raise err
                    if fname.endswith(t_fname):  # timestamps
                        data["t"] = handle.squeeze()
                    elif fname.endswith(xy_fname):  # coordinates
                        data["xy"] = handle.squeeze()
                    elif fname.endswith(p_fname):  # polarity
                        data["p"] = handle.squeeze()
            if len(data) > 0:
                data['path'] = subroot
                if "t" not in data:
                    print("Ignoring root {} since no events".format(subroot))
                    continue
                assert (len(data['p']) == len(data['xy']) and len(data['p']) == len(data['t']))

                self.t0, self.tk = data['t'][0], data['t'][-1]
                self.num_events = len(data['p'])
                self.num_frames = len(data['images'])

                self.frame_ts = []
                for ts in data["frame_stamps"]:
                    self.frame_ts.append(ts)
                data["index"] = self.frame_ts

        self.filehandle = data
        self.find_config(data_path)

    def find_ts_index(self, timestamp):
        index = np.searchsorted(self.filehandle["t"], timestamp)
        return index

    def infer_resolution(self):
        if len(self.filehandle["images"]) > 0:
            return self.filehandle["images"][0].shape[:2]
        else:
            print('WARNING: sensor resolution inferred from maximum event coordinates - highly not recommended')
            return [np.max(self.filehandle["xy"][:, 1]) + 1, np.max(self.filehandle["xy"][:, 0]) + 1]

    def find_config(self, data_path):
        if self.sensor_resolution is None:
            config = os.path.join(data_path, "dataset_config.json")
            if os.path.exists(config):
                self.config = read_json(config)
                self.data_source = self.config['data_source']
                self.sensor_resolution = self.config["sensor_resolution"]
            else:
                data_source = 'unknown'
                self.sensor_resolution = self.infer_resolution()
                print("Inferred sensor resolution: {}".format(self.sensor_resolution))


class SequenceDataset(Dataset):
    """Load sequences of time-synchronized {event tensors + frames} from a folder."""
    def __init__(self, data_root, sequence_length, dataset_type='MemMapDataset',
            step_size=None, proba_pause_when_running=0.0,
            proba_pause_when_paused=0.0, normalize_image=False,
            noise_kwargs={}, hot_pixel_kwargs={}, dataset_kwargs={}):
        self.L = sequence_length
        self.step_size = step_size if step_size is not None else self.L
        self.proba_pause_when_running = proba_pause_when_running
        self.proba_pause_when_paused = proba_pause_when_paused
        self.normalize_image = normalize_image
        self.noise_kwargs = noise_kwargs
        self.hot_pixel_kwargs = hot_pixel_kwargs

        assert(self.L > 0)
        assert(self.step_size > 0)

        self.dataset = eval(dataset_type)(data_root, **dataset_kwargs)
        if self.L >= self.dataset.length:
            self.length = 0
        else:
            self.length = (self.dataset.length - self.L) // self.step_size + 1

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        """ Returns a list containing synchronized events <-> frame pairs
            [e_{i-L} <-> I_{i-L},
             e_{i-L+1} <-> I_{i-L+1},
            ...,
            e_{i-1} <-> I_{i-1},
            e_i <-> I_i]
        """
        assert(i >= 0)
        assert(i < self.length)

        # generate a random seed here, that we will pass to the transform function
        # of each item, to make sure all the items in the sequence are transformed
        # in the same way
        seed = random.randint(0, 2**32)

        # data augmentation: add random, virtual "pauses",
        # i.e. zero out random event tensors and repeat the last frame
        sequence = []

        # add the first element (i.e. do not start with a pause)
        k = 0
        j = i * self.step_size
        item = self.dataset.__getitem__(j, seed)
        sequence.append(item)

        paused = False
        for n in range(self.L - 1):

            # decide whether we should make a "pause" at this step
            # the probability of "pause" is conditioned on the previous state (to encourage long sequences)
            u = np.random.rand()
            if paused:
                probability_pause = self.proba_pause_when_paused
            else:
                probability_pause = self.proba_pause_when_running
            paused = (u < probability_pause)

            if paused:
                # add a tensor filled with zeros, paired with the last frame
                # do not increase the counter
                item = self.dataset.__getitem__(j + k, seed)
                item['events'].fill_(0.0)
                if 'flow' in item:
                    item['flow'].fill_(0.0)
                sequence.append(item)
            else:
                # normal case: append the next item to the list
                k += 1
                item = self.dataset.__getitem__(j + k, seed)
                sequence.append(item)
            # add noise
            if self.noise_kwargs:
                item['events'] = add_noise_to_voxel(item['events'], **self.noise_kwargs)

        # add hot pixels
        if self.hot_pixel_kwargs:
            add_hot_pixels_to_sequence_(sequence, **self.hot_pixel_kwargs)

        # normalize image
        if self.normalize_image:
            # normalize_image_sequence_(sequence, key='frame')
            normalize_image_sequence_(sequence, key='intensity')
            # if i == 0:
            #     print('Normalize intensity.')
        return sequence

