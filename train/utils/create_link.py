"""
 @Time    : 6/21/22 21:36
 @Author  : Haiyang Mei
 @E-mail  : mhy666@mail.dlut.edu.cn
 
 @Project : firenet-pdavis
 @File    : create_link.py
 @Function:
 
"""
import os

# txt_path = '~/firenet-pdavis/data/movingcam/test_pff.txt'
# src_dir = '~/firenet-pdavis/data/h5'

# txt_path = '/root/data/movingcam/all_iad.txt'
# src_dir = '/root/data/iad'

txt_path = '/root/data/movingcam/new_iad_all.txt'
src_dir = '/root/data/new_iad'

list = []
with open(txt_path) as f:
    line = f.readline()
    if line[:-1].endswith('.h5'):
        list.append(line[:-1])
    while line:
        line = f.readline()
        if line[:-1].endswith('.h5'):
            list.append(line[:-1])
print(list)

for dst_path in list:
    file_name = dst_path.split('/')[-1]

    dst_dir = dst_path[:-9]
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    src_path = os.path.join(src_dir, file_name)

    call_with_args = 'ln -s {} {}'.format(src_path, dst_path)

    print(call_with_args)

    os.system(call_with_args)

print('Succeed!')
