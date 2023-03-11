"""
 @Time    : 20.04.22 18:25
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : add_flow.py
 @Function:
 
"""
import os
import sys
sys.path.append('..')
import h5py
import numpy as np
from tqdm import tqdm

# input_path = '/home/mhy/firenet-pdavis/data/train_p/subject07_group2_time1_p.h5'
# output_path = '/home/mhy/firenet-pdavis/data/train_p/subject07_group2_time1_pf.h5'
# flow_dir = '/home/mhy/firenet-pdavis/data/train_p/flow_npy_subject07_group2_time1_p'
input_path = '/home/mhy/firenet-pdavis/data/test_p/subject09_group2_time1_p.h5'
output_path = '/home/mhy/firenet-pdavis/data/test_p/subject09_group2_time1_pf.h5'
flow_dir = '/home/mhy/firenet-pdavis/data/test_p/flow_npy_subject09_group2_time1_p'

flow_list = os.listdir(flow_dir)
flow_list = sorted(flow_list)

flows = []
for flow_name in tqdm(flow_list):
    flow_path = os.path.join(flow_dir, flow_name)
    flow = np.load(flow_path)
    flows.append(flow)

flows = np.stack(flows, axis=0)

f_input = h5py.File(input_path, 'r')

f_output = h5py.File(output_path, 'w')
f_output.create_dataset('/events', data=f_input['/events'], chunks=True)
f_output.create_dataset('/frame', data=f_input['/frame'], chunks=True)
f_output.create_dataset('/frame_idx', data=f_input['/frame_idx'], chunks=True)
f_output.create_dataset('/frame_ts', data=f_input['/frame_ts'], chunks=True)

f_output.create_dataset('/intensity', data=f_input['intensity'], chunks=True)
f_output.create_dataset('/aolp', data=f_input['aolp'], chunks=True)
f_output.create_dataset('/dolp', data=f_input['dolp'], chunks=True)

f_output.create_dataset('/flow', data=flows, chunks=True)

f_output.attrs['sensor_resolution'] = (256, 256)

f_output.close()
print('Add Flow Succeed!')
