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

list = sorted([x for x in os.listdir(root) if x.endswith('.h5')])
paths = []
for name in list:
    # path = os.path.join(root, name).replace('/home/mhy/firenet-pdavis', '.')
    paths.append(path)

with open(txt_path, 'w') as f:
    for x in paths:
        f.write(f"{x}\n")

print('Nice!')
