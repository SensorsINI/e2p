"""
 @Time    : 01.04.22 15:47
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : txt_img_to_h5.py
 @Function:
 
"""
import glob
import argparse
import os
import cv2
import h5py
import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from data_formats.event_packagers import *


def get_sensor_size(txt_path):
    try:
        header = pd.read_csv(txt_path, delim_whitespace=True, header=None, names=['width', 'height'],
                             dtype={'width': np.int, 'height': np.int},
                             nrows=1)
        width, height = header.values[0]
        sensor_size = [height, width]
    except:
        sensor_size = None
        print('Warning: could not read sensor size from first line of {}'.format(txt_path))
    return sensor_size


def extract_img(img_txt_paths, img_path, output_path, zero_timestamps=False,
                packager=hdf5_packager):
    ep = packager(output_path)
    first_ts = -1
    t0 = -1

    # compute sensor size
    # sensor_size = get_sensor_size(img_path)
    # Extract events to h5
    # ep.set_data_available(num_images, num_flow=0)
    total_num_pos, total_num_neg, img_cnt, last_ts = 0, 0, 0, 0

    chunksize = 1
    # save images
    iterator = pd.read_csv(img_txt_paths, delim_whitespace=True, header=None,
                           names=['img_ts', 'images'],
                           dtype={'img_ts': np.float64, 'images': str},
                           engine='c',
                           skiprows=1, chunksize=chunksize, nrows=None, memory_map=True)
    #    skiprows=1, nrows=None, memory_map=True)

    print(iterator)
    # read images
    for i, ts_window in enumerate(iterator):
        image_ts_file = ts_window.values
        img_ts = image_ts_file[:, 0]
        img_name = image_ts_file[:, 1]
        # print('img_ts: x')
        # print(img_ts)
        # print('img_name: x')
        # print(img_name)

        # for j, img_names in enumerate(img_name):
        #     for k, img_ts_idx in enumerate(img_ts):
        #         # print('img_ts_idx: ')
        #         # print(img_ts_idx)
        #         # print('img_cnt: ')
        #         # print(img_cnt)
        # print(str(img_path))
        # print(img_name[0])
        image = cv2.imread(os.path.join(str(img_path), img_name[0]), cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
        # sensor_size = image.shape
        ep.package_image(image, img_ts[0], img_cnt)
        img_cnt += 1
    return img_cnt, ep


def extract_txt(sensor_size, ep, img_cnt, txt_path, output_path, zero_timestamps=False,
                packager=hdf5_packager):
    first_ts = -1
    t0 = -1
    if not os.path.exists(txt_path):
        print("{} does not exist!".format(txt_path))
        return

    # compute sensor size
    # sensor_size = get_sensor_size(txt_path)
    # Extract events to h5
    # ep.set_data_available(num_images, num_flow=0)
    total_num_pos, total_num_neg, last_ts = 0, 0, 0

    chunksize = 100000
    iterator = pd.read_csv(txt_path, delim_whitespace=True, header=None,
                           names=['t', 'x', 'y', 'pol'],
                           dtype={'t': np.float64, 'x': np.int16, 'y': np.int16, 'pol': np.int16},
                           engine='c',
                           skiprows=1, chunksize=chunksize, nrows=None, memory_map=True)

    for i, event_window in enumerate(iterator):
        events = event_window.values
        ts = events[:, 0].astype(np.float64)
        xs = events[:, 1].astype(np.int16)
        ys = events[:, 2].astype(np.int16)
        ps = events[:, 3]

        # test
        # xs = events[:, 0].astype(np.float64)
        # ys = events[:, 1].astype(np.int16)
        # ts = events[:, 2].astype(np.int16)
        # ps = events[:, 3]

        ps[ps < 0] = 0  # should be [0 or 1]
        ps = ps.astype(bool)

        # print(ts)
        if first_ts == -1:
            first_ts = ts[0]

        if zero_timestamps:
            ts -= first_ts
        last_ts = ts[-1]
        # if sensor_size is None or sensor_size[0] < max(ys) or sensor_size[1] < max(xs):
        #     sensor_size = [max(ys)+1, max(xs)+1]
        #     print("Sensor size inferred from events as {}".format(sensor_size))

        sum_ps = sum(ps)
        total_num_pos += sum_ps
        total_num_neg += len(ps) - sum_ps
        ep.package_events(xs, ys, ts, ps)
        if i % 10 == 9:
            print('Events written: {} M'.format((total_num_pos + total_num_neg) / 1e6))
    print('Events written: {} M'.format((total_num_pos + total_num_neg) / 1e6))
    # define sensor size here
    print("Sensor size [h={}, w={}]".format(sensor_size[0], sensor_size[1]))
    t0 = 0 if zero_timestamps else first_ts
    ep.add_metadata(total_num_pos, total_num_neg, last_ts - t0, t0, last_ts, img_cnt, num_flow=0,
                    sensor_size=sensor_size)


def extract_txt_imgs(sensor_size, img_txt_paths, img_paths, txt_paths, output_dir, zero_timestamps=False):
    out_path = os.path.join(output_dir, "events.h5")
    # out_path = output_dir
    for path in img_txt_paths:
        img_cnt, ep = extract_img(path, img_paths, out_path, zero_timestamps=zero_timestamps)
    for path in txt_paths:
        filename = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(output_dir, "{}.h5".format(filename))
        print("Extracting {} to {}".format(path, out_path))
        extract_txt(sensor_size, ep, img_cnt, path, out_path, zero_timestamps=zero_timestamps)


if __name__ == "__main__":
    """
    Tool for converting txt events to an efficient HDF5 format that can be speedily
    accessed by python code.
    Input path can be single file or directory containing files.
    Individual input event files can be txt or zip with format matching
    https://github.com/uzh-rpg/rpg_e2vid:
      width height
      t1 x1 y1 p1
      t2 x2 y2 p2
      t3 x3 y3 p3
      ...
    i.e. first line of file is sensor size width first and height second.
    This script only does events -> h5, not images or anything else (yet).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="txt file to extract or directory containing txt files")
    parser.add_argument("--img_path_txt", help="image file to extract or directory containing image files")
    parser.add_argument("--img_path", help="image file to extract or directory containing image files")
    parser.add_argument("--output_dir", default="/tmp/extracted_data", help="Folder where to extract the data")
    parser.add_argument('--zero_timestamps', action='store_true',
                        help='If true, timestamps will be offset to start at 0')
    parser.add_argument('--sensor_size_x')
    parser.add_argument('--sensor_size_y')
    args = parser.parse_args()

    print('Data will be extracted in folder: {}'.format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if os.path.isdir(args.path):
        txt_paths = sorted(list(glob.glob(os.path.join(args.path, "*.txt")))
                           + list(glob.glob(os.path.join(args.path, "*.zip"))))
    else:
        txt_paths = [args.path]

    sensor_size = [int(args.sensor_size_y), int(args.sensor_size_x)]
    img_txt_paths = [args.img_path_txt]
    img_paths = args.img_path
    extract_txt_imgs(sensor_size, img_txt_paths, img_paths, txt_paths, args.output_dir,
                     zero_timestamps=args.zero_timestamps)