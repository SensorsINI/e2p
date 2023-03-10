"""
 @Time    : 20.03.22 22:16
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : concat_iad.py
 @Function:
 
"""
# import os
# import sys
# sys.path.append('..')
# from misc import check_mkdir
# import cv2
# from tqdm import tqdm
# import numpy as np
#
# root = '/home/mhy/firenet-pdavis/output/FireNet_retrained'
# name = 'mhy3_f12p'
#
# dir = os.path.join(root, name)
#
# dir_intensity = os.path.join(dir, 'intensity')
# dir_aolp = os.path.join(dir, 'aolp')
# dir_dolp = os.path.join(dir, 'dolp')
# dir_concat = os.path.join(dir, 'concat')
# check_mkdir(dir_concat)
#
# intensity_list = [x for x in os.listdir(dir_intensity) if x.endswith('.png')]
# intensity_list.sort(key=lambda x: int(x.split('_')[1]))
#
# for i in tqdm(range(len(intensity_list))):
#     intensity = cv2.imread(os.path.join(dir_intensity, intensity_list[i]), cv2.IMREAD_GRAYSCALE)
#     aolp = cv2.imread(os.path.join(dir_aolp, intensity_list[i][:-6] + '_aolp.png'), cv2.IMREAD_GRAYSCALE)
#     dolp = cv2.imread(os.path.join(dir_dolp, intensity_list[i][:-6] + '_dolp.png'), cv2.IMREAD_GRAYSCALE)
#
#     mask = np.where(dolp[:, :] >= 12.75, 255, 0).astype(np.uint8)
#     aolp_masked = np.where(mask == 255, aolp, 0).astype(np.uint8)
#
#     concat = cv2.hconcat([intensity, aolp_masked, dolp])
#
#     cv2.imwrite(os.path.join(dir_concat, str(i) + '.png'), concat)
#
# cv2.destroyAllWindows()
#
# print('Succeed!')

# import os
# import sys
# sys.path.append('..')
# from misc import check_mkdir
# import cv2
# from tqdm import tqdm
# import numpy as np
#
# root = '/home/mhy/firenet-pdavis/output/FireNet_retrained'
# name = 'mhy4_p'
# # root = '/home/mhy/firenet-pdavis/output'
# # name = 'mhy5'
# # set = 'subject09_group2_time1_pf'
#
# dir = os.path.join(root, name)
#
# dir_intensity = os.path.join(dir, 'intensity')
# dir_aolp = os.path.join(dir, 'aolp')
# dir_dolp = os.path.join(dir, 'dolp')
# # dir_intensity = os.path.join(dir, 'intensity_' + set)
# # dir_aolp = os.path.join(dir, 'aolp_' + set)
# # dir_dolp = os.path.join(dir, 'dolp_' + set)
# dir_concat = os.path.join(dir, 'concat')
# check_mkdir(dir_concat)
#
# intensity_list = [x for x in os.listdir(dir_intensity) if x.endswith('.png')]
# intensity_list.sort(key=lambda x: int(x.split('_')[1][:-4]))
#
# for i in tqdm(range(len(intensity_list))):
#     intensity = cv2.imread(os.path.join(dir_intensity, intensity_list[i]), cv2.IMREAD_GRAYSCALE)
#     aolp = cv2.imread(os.path.join(dir_aolp, intensity_list[i].replace('intensity', 'aolp')), cv2.IMREAD_GRAYSCALE)
#     dolp = cv2.imread(os.path.join(dir_dolp, intensity_list[i].replace('intensity', 'dolp')), cv2.IMREAD_GRAYSCALE)
#
#     # mask = np.where(dolp[:, :] >= 12.75, 255, 0).astype(np.uint8)
#     # aolp_masked = np.where(mask == 255, aolp, 0).astype(np.uint8)
#
#     concat = cv2.hconcat([intensity, aolp, dolp])
#
#     cv2.imwrite(os.path.join(dir_concat, str(i) + '.png'), concat)
#
# cv2.destroyAllWindows()
#
# print('Succeed!')

# for gt
import os
import sys
sys.path.append('..')
from misc import check_mkdir
import cv2
from tqdm import tqdm
import numpy as np

dir = '/home/mhy/firenet-pdavis/data/test_p'
set = 'subject09_group2_time1_pf'

dir_intensity = os.path.join(dir, 'intensity_' + set)
dir_aolp = os.path.join(dir, 'aolp_' + set)
dir_dolp = os.path.join(dir, 'dolp_' + set)
dir_concat = os.path.join(dir, 'concat_' + set)
check_mkdir(dir_concat)

intensity_list = sorted([x for x in os.listdir(dir_intensity) if x.endswith('.png')])

for name in tqdm(intensity_list):
    intensity = cv2.imread(os.path.join(dir_intensity, name), cv2.IMREAD_GRAYSCALE)
    aolp = cv2.imread(os.path.join(dir_aolp, name), cv2.IMREAD_GRAYSCALE)
    dolp = cv2.imread(os.path.join(dir_dolp, name), cv2.IMREAD_GRAYSCALE)

    concat = cv2.hconcat([intensity, aolp, dolp])

    cv2.imwrite(os.path.join(dir_concat, name), concat)

cv2.destroyAllWindows()

print('Succeed!')
