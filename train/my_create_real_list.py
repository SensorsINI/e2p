"""
 @Time    : 17.10.22 19:03
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : my_create_real_list.py
 @Function:
 
"""
import os

root = './data/new/real'
txt_path = './data/new/real.txt'

print(f'creating {txt_path} with h5 file list....')

list = sorted([x for x in os.listdir(root) if x.endswith('.h5')])
paths = []
for name in list:
    # path = os.path.join(root, name).replace('/home/mhy/firenet-pdavis', '.')
    print(name)
    paths.append(name)

with open(txt_path, 'w') as f:
    n=0
    for x in paths:
        f.write(f"{x}\n")
        n+=1

print(f'Nice! - done creating {txt_path} with {n} h5 files')
