"""
 @Time    : 22.03.22 11:33
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : split1.py
 @Function:
 
"""
import os
from tqdm import tqdm

txt_dir = '/home/mhy/firenet-pdavis/data/v2e/txt'

txt = 'subject09_group3_time4'

txt_path = os.path.join(txt_dir, txt + '.txt')
i_path = os.path.join(txt_dir, txt + '_i.txt')

f = open(txt_path, 'r')
lines = f.readlines()

# i = ['346 260\n']
i = ['1224 1024\n']

for line in tqdm(lines[6:]):
    timestamp, x, y, polarity = line.split()
    if int(polarity) == 0:
        polarity = '-1'

    new_x = x
    new_y = y
    new_line = timestamp + '\t' + new_x + '\t' + new_y + '\t' + polarity + '\n'

    i.append(new_line)

with open(i_path, 'w') as f:
    for line in i:
        f.write(line)

print('Succeed!')
