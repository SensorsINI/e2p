"""
 @Time    : 02.04.22 11:07
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : h5disp.py
 @Function:
 
"""
import os
import cv2
import h5py
import numpy as np

# dir = '/home/mhy/v2e/output/subject09_group1_time1'
# name = 'e.h5'

# path = os.path.join(dir, name)

# path = '/home/mhy/firenet-pdavis/data/raw_demosaicing_polarization_5s_iad/004103_iad.h5'
# path = '/home/mhy/aedat2pvideo/aedat/real/UIUC_183_121_196_134_RPM_800-events.h5'
path = '/home/mhy/firenet-pdavis/data/real/real-01.h5'
# path = '/home/mhy/firenet-pdavis/data/raw_demosaicing_polarization_5s_iad/004000_iad.h5'

# path = '/home/mhy/data/beetles/22June-20220623T122254Z-001/22June_h5/Davis346B-2022-06-22T14-22-51-0500-INLX0008-0.h5'

f = h5py.File(path, 'r')
for key in f.keys():
    print(f[key].name)
    print(f[key].shape)
    print(f[key].dtype)
for item in f.attrs.items():
    print(item)

print(f['/frame_idx'][:10])
print(f['/frame_ts'][:10])
print(f['/events'][:10])
#
# print(f['/frame_idx'][-15:])
# print(f['/frame_ts'][-15:])
# print(f['/events'][-100:])
#
# print(np.max(f['/intensity'][100]))



# print('-----------------------------')
# print(f['/events'].shape)
# print('-----------------------------')
# print(f['/events'][0, :])
# print(f['/events'][1, :])
# print(f['/events'][2, :])
# print(f['/events'][-1, :])

# cv2.imshow('image', f['/frame'][100, :])
# cv2.waitKey()
# print(f['/frame_idx'][:])
# print(f['/frame_ts'][:])

# print(f.attrs.get('source', 'unknown'))
