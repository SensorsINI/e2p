import numpy as np
from math import fabs


def robust_min(img, p=5):
    return np.percentile(img.ravel(), p)


def robust_max(img, p=95):
    return np.percentile(img.ravel(), p)


def normalize(img, m=10, M=90):
    return np.clip((img - robust_min(img, m)) / (robust_max(img, M) - robust_min(img, m)), 0.0, 1.0)


def first_element_greater_than(values, req_value):
    """Returns the pair (i, values[i]) such that i is the minimum value that satisfies values[i] >= req_value.
    Returns (-1, None) if there is no such i.
    Note: this function assumes that values is a sorted array!"""
    i = np.searchsorted(values, req_value)
    val = values[i] if i < len(values) else None
    return (i, val)


def last_element_less_than(values, req_value):
    """Returns the pair (i, values[i]) such that i is the maximum value that satisfies values[i] <= req_value.
    Returns (-1, None) if there is no such i.
    Note: this function assumes that values is a sorted array!"""
    i = np.searchsorted(values, req_value, side='right') - 1
    val = values[i] if i >= 0 else None
    return (i, val)


def closest_element_to(values, req_value):
    """Returns the tuple (i, values[i], diff) such that i is the closest value to req_value,
    and diff = |values(i) - req_value|
    Note: this function assumes that values is a sorted array!"""
    assert(len(values) > 0)

    i = np.searchsorted(values, req_value, side='left')
    if i > 0 and (i == len(values) or fabs(req_value - values[i - 1]) < fabs(req_value - values[i])):
        idx = i - 1
        val = values[i - 1]
    else:
        idx = i
        val = values[i]

    diff = fabs(val - req_value)
    return (idx, val, diff)




import json
import numpy as np
import cv2 as cv
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
from math import fabs, ceil, floor
from torch.nn import ZeroPad2d
from os.path import join
import torch


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


def optimal_crop_size(max_size, max_subsample_factor, safety_margin=0):
    """ Find the optimal crop size for a given max_size and subsample_factor.
        The optimal crop size is the smallest integer which is greater or equal than max_size,
        while being divisible by 2^max_subsample_factor.
    """
    crop_size = int(pow(2, max_subsample_factor) * ceil(max_size / pow(2, max_subsample_factor)))
    crop_size += safety_margin * pow(2, max_subsample_factor)
    return crop_size


class CropParameters:
    """ Helper class to compute and store useful parameters for pre-processing and post-processing
        of images in and out of E2VID.
        Pre-processing: finding the best image size for the network, and padding the input image with zeros
        Post-processing: Crop the output image back to the original image size
    """

    def __init__(self, width, height, num_encoders, safety_margin=0):

        self.height = height
        self.width = width
        self.num_encoders = num_encoders
        self.width_crop_size = optimal_crop_size(self.width, num_encoders, safety_margin)
        self.height_crop_size = optimal_crop_size(self.height, num_encoders, safety_margin)

        self.padding_top = ceil(0.5 * (self.height_crop_size - self.height))
        self.padding_bottom = floor(0.5 * (self.height_crop_size - self.height))
        self.padding_left = ceil(0.5 * (self.width_crop_size - self.width))
        self.padding_right = floor(0.5 * (self.width_crop_size - self.width))
        self.pad = ZeroPad2d((self.padding_left, self.padding_right, self.padding_top, self.padding_bottom))

        self.cx = floor(self.width_crop_size / 2)
        self.cy = floor(self.height_crop_size / 2)

        self.ix0 = self.cx - floor(self.width / 2)
        self.ix1 = self.cx + ceil(self.width / 2)
        self.iy0 = self.cy - floor(self.height / 2)
        self.iy1 = self.cy + ceil(self.height / 2)

    def crop(self, img):
        return img[..., self.iy0:self.iy1, self.ix0:self.ix1]


def format_power(size):
    power = 1e3
    n = 0
    power_labels = {0: '', 1: 'K', 2: 'M', 3: 'G', 4: 'T'}
    while size > power:
        size /= power
        n += 1
    return size, power_labels[n]


def flow2bgr_np(disp_x, disp_y, max_magnitude=None):
    """
    Convert an optic flow tensor to an RGB color map for visualization
    Code adapted from: https://github.com/ClementPinard/FlowNetPytorch/blob/master/main.py#L339

    :param disp_x: a [H x W] NumPy array containing the X displacement
    :param disp_y: a [H x W] NumPy array containing the Y displacement
    :returns bgr: a [H x W x 3] NumPy array containing a color-coded representation of the flow [0, 255]
    """
    assert(disp_x.shape == disp_y.shape)
    H, W = disp_x.shape

    # X, Y = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W))

    # flow_x = (X - disp_x) * float(W) / 2
    # flow_y = (Y - disp_y) * float(H) / 2
    # magnitude, angle = cv.cartToPolar(flow_x, flow_y)
    # magnitude, angle = cv.cartToPolar(disp_x, disp_y)

    # follow alex zhu color convention https://github.com/daniilidis-group/EV-FlowNet

    flows = np.stack((disp_x, disp_y), axis=2)
    magnitude = np.linalg.norm(flows, axis=2)

    angle = np.arctan2(disp_y, disp_x)
    angle += np.pi
    angle *= 180. / np.pi / 2.
    angle = angle.astype(np.uint8)

    if max_magnitude is None:
        v = np.zeros(magnitude.shape, dtype=np.uint8)
        cv.normalize(src=magnitude, dst=v, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    else:
        v = np.clip(255.0 * magnitude / max_magnitude, 0, 255)
        v = v.astype(np.uint8)

    hsv = np.zeros((H, W, 3), dtype=np.uint8)
    hsv[..., 1] = 255
    hsv[..., 0] = angle
    hsv[..., 2] = v
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    return bgr


def recursive_clone(tensor):
    """
    Assumes tensor is a torch.tensor with 'clone()' method, possibly
    inside nested iterable.
    E.g., tensor = [(pytorch_tensor, pytorch_tensor), ...]
    """
    if hasattr(tensor, 'clone'):
        return tensor.clone()
    try:
        return type(tensor)(recursive_clone(t) for t in tensor)
    except TypeError:
        print('{} is not iterable and has no clone() method.'.format(tensor))


def get_height_width(data_loader):
    for d in data_loader:
        return d['events'].shape[-2:]  # d['events'] is a ... x H x W voxel grid 


def torch2cv2(image):
    """convert torch tensor to format compatible with cv2.imwrite"""
    image = torch.squeeze(image)  # H x W
    image = image.cpu().numpy()
    image = np.clip(image, 0, 1)
    return (image * 255).astype(np.uint8)

# added by Haiyang Mei
def torch2numpy(image):
    """convert torch tensor to numpy"""
    image = torch.squeeze(image)  # H x W
    image = image.cpu().numpy()
    return image

# added by Haiyang Mei
def numpy2cv2(image):
    """convert numpy to format compatible with cv2.imwrite"""
    image = np.clip(image, 0, 1)
    return (image * 255).astype(np.uint8)

def append_timestamp(path, description, timestamp):
    with open(path, 'a') as f:
        f.write('{} {:.15f}\n'.format(description, timestamp))


def setup_output_folder(output_folder):
    """
    Ensure existence of output_folder and overwrite output_folder/timestamps.txt file.
    Returns path to output_folder/timestamps.txt
    """
    ensure_dir(output_folder)
    print('Saving to: {}'.format(output_folder))
    timestamps_path = join(output_folder, 'timestamps.txt')
    open(timestamps_path, 'w').close()  # overwrite with emptiness
    return timestamps_path

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()
        
    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]
    
    def result(self):
        return dict(self._data.average)
