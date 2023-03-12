"""
 @Time    : 17.10.22 15:28
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : my_extract_aps.py
 @Function:
 
"""
import os
from tqdm import tqdm

root = './data/new/real'

aedat_list = sorted([x for x in os.listdir(root) if x.endswith('.aedat')])

for name in tqdm(aedat_list):
    aedat_path = os.path.join(root, name)

    call_with_args = 'bash ../jaer/dvs-slice-avi-writer.sh -width=346 -height=260 -writeapsframes=true -writedvsframes=false -aechip=eu.seebetter.ini.chips.davis.Davis346B -format=PNG -framerate=10 -showoutput=false -writetimecodefile=true {}'.format(aedat_path)

    print(call_with_args)

    os.system(call_with_args)

print('Succeed!')
