"""
 @Time    : 20.10.22 17:34
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : compute_vfe.py
 @Function:
 
"""
import h5py

# list_path = '/home/mhy/firenet-pdavis/data/movingcam/test_5s.txt'
# list_path = '/home/mhy/firenet-pdavis/data/movingcam/train_5s.txt'
# list_path = '/home/mhy/firenet-pdavis/data/movingcam/train_test_5s.txt'

# list_path = '/home/mhy/firenet-pdavis/data/movingcam/test_real2.txt'
# list_path = '/home/mhy/firenet-pdavis/data/movingcam/train_real2.txt'
list_path = '/home/mhy/firenet-pdavis/data/movingcam/train_test_real2.txt'

with open(list_path, 'r') as f:
    list = [line.strip() for line in f]

print(list_path)
print('Videos:\t%d'%len(list))

frames = 0
events = 0
for path in list:
    f = h5py.File(path, 'r')
    x = f['frame'].shape[0]
    frames += x

    y = f['events'].shape[0]
    events += y

print('Frames:\t%.3f K'%(frames/1e3))
print('Events:\t%.6f M'%(events/1e6))

print('Nice!')
