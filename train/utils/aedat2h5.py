"""
 @Time    : 23.06.22 11:00
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : aedat2h5.py
 @Function:
 
"""
import os

from dv import LegacyAedatFile
import numpy as np
import h5py

# name = 'Davis346B-2022-06-22T14-31-55-0500-INLX0008-0'
# input_root = '/home/mhy/data/beetles/22June-20220623T122254Z-001/22June'
# output_root = '/home/mhy/data/beetles/22June-20220623T122254Z-001/22June_h5'

name = 'Davis346B-2022-09-11T20-42-43+0200-00000000-0-Glass'
input_root = '/home/mhy/firenet-pdavis/data/aedat'
output_root = '/home/mhy/firenet-pdavis/data/aedat_h5'
if not os.path.exists(output_root):
    os.makedirs(output_root)

input_path = os.path.join(input_root, name + '.aedat')
output_path = os.path.join(output_root, name + '.h5')

i = 0
events = []

output = h5py.File(output_path, 'w')
output.create_dataset(name='/events', shape=(0, 4), maxshape=(None, 4), dtype='uint32', compression='gzip')
output.close()

with LegacyAedatFile(input_path) as f:
    for event in f:
        x = event.x
        y = event.y
        polarity = 1 if event.polarity else 0
        timestamp = event.timestamp

        four = [timestamp, x, y, polarity]

        events.append(four)

        i += 1
        if i % 20000 == 0:
            print(i)
            events = np.vstack(events).astype(np.uint32)
            output = h5py.File(output_path, 'a')
            data = output['/events']
            data.resize(data.shape[0] + events.shape[0], axis=0)
            data[-events.shape[0]:] = events
            output.close()
            events = []

    # for the remaining data
    print(i)
    events = np.vstack(events).astype(np.uint32)
    output = h5py.File(output_path, 'a')
    data = output['/events']
    data.resize(data.shape[0] + events.shape[0], axis=0)
    data[-events.shape[0]:] = events
    output.close()

print('Succeed!')
