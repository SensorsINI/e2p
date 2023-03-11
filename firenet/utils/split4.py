"""
 @Time    : 2/14/22 22:44
 @Author  : Haiyang Mei
 @E-mail  : mhy666@mail.dlut.edu.cn
 
 @Project : firenet-pdavis
 @File    : split4.py
 @Function:
 
"""
import os
from tqdm import tqdm

txt_dir = '/home/mhy/firenet-pdavis/data/v2e/txt'

txt = 'subject09_group2_time1'

txt_path = os.path.join(txt_dir, txt + '.txt')
i0_path = os.path.join(txt_dir, txt + '_i0.txt')
i45_path = os.path.join(txt_dir, txt + '_i45.txt')
i90_path = os.path.join(txt_dir, txt + '_i90.txt')
i135_path = os.path.join(txt_dir, txt + '_i135.txt')

f = open(txt_path, 'r')
lines = f.readlines()

i0 = ['173 130\n']
i45 = ['173 130\n']
i90 = ['173 130\n']
i135 = ['173 130\n']

for line in tqdm(lines[6:]):
    # print(line)
    # print(len(line))
    timestamp, x, y, polarity = line.split( )
    if int(polarity) == 0:
        polarity = '-1'

    new_x = str(int(int(x) / 2))
    new_y = str(int(int(y) / 2))
    new_line = timestamp + '\t' + new_x + '\t' + new_y + '\t' + polarity + '\n'

    # print(line)
    # print(new_line)
    # exit(0)

    if int(x) % 2:
        # i0 or i45
        if int(y) % 2:
            # line = line.replace('\t' + str(x) + '\t', '\t' + str(int(int(x) / 2)) + '\t')
            # line = line.replace('\t' + str(y) + '\t', '\t' + str(int(int(y) / 2)) + '\t')
            i0.append(new_line)
        else:
            # line = line.replace('\t' + str(x) + '\t', '\t' + str(int(int(x) / 2)) + '\t')
            # line = line.replace('\t' + str(y) + '\t', '\t' + str(int(int(y) / 2)) + '\t')
            i45.append(new_line)
    else:
        # i90 or i135
        if int(y) % 2:
            # line = line.replace('\t' + str(x) + '\t', '\t' + str(int(int(x) / 2)) + '\t')
            # line = line.replace('\t' + str(y) + '\t', '\t' + str(int(int(y) / 2)) + '\t')
            i135.append(new_line)
        else:
            # line = line.replace('\t' + str(x) + '\t', '\t' + str(int(int(x) / 2)) + '\t')
            # line = line.replace('\t' + str(y) + '\t', '\t' + str(int(int(y) / 2)) + '\t')
            i90.append(new_line)

with open(i0_path, 'w') as f:
    for i in i0:
        f.write(i)
with open(i45_path, 'w') as f:
    for i in i45:
        f.write(i)
with open(i90_path, 'w') as f:
    for i in i90:
        f.write(i)
with open(i135_path, 'w') as f:
    for i in i135:
        f.write(i)

print('Succeed!')
