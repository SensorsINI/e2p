"""
 @Time    : 17.10.22 15:37
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : my_extract_events.py
 @Function:
 
"""
import os
from tqdm import tqdm
from dv import LegacyAedatFile

root = 'data/E2PD/new'

aedat_list = sorted([x for x in os.listdir(root) if x.endswith('.aedat')])
for name in tqdm(aedat_list):
    aedat_path = os.path.join(root, name)
    events_path = os.path.join(root, name[:-6] + '.txt')

    events = []

    with LegacyAedatFile(aedat_path) as f:
        for event in f:
            x = event.x
            y = event.y
            # for flip
            x = 345 - x
            y = 259 - y
            polarity = 1 if event.polarity else 0
            timestamp = event.timestamp

            four = [timestamp, x, y, polarity]

            events.append(four)

    print(len(events))
    with open(events_path, 'w') as f:
        for event in events:
            f.write(f"{event[0]}\t{event[1]}\t{event[2]}\t{event[3]}\n")

print('Succeed!')
