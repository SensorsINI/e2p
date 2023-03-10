"""
 @Time    : 18.10.22 10:00
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : rename.py
 @Function:
 
"""
import os

root = '/home/mhy/firenet-pdavis/data/real'

list = os.listdir(root)

for i, name in enumerate(list):
    old_name = os.path.join(root, name)
    new_name = os.path.join(root, 'real-%02d'%i + '.aedat')

    os.rename(old_name, new_name)

print('Nice!')
