"""
 @Time    : 2/16/22 00:32
 @Author  : Haiyang Mei
 @E-mail  : mhy666@mail.dlut.edu.cn
 
 @Project : firenet-pdavis
 @File    : f32v.py
 @Function:
 
"""
import os
import cv2
from tqdm import tqdm

rate = 30
# size = (519, 260)
size = (519, 130)

name = 'subject09_group2_time1_f42p'

name2 = 'subject01_group1_time1_f12p'

root1 = os.path.join('/home/mhy/firenet-pdavis/output/FireNet', name)
root2 = os.path.join('/home/mhy/firenet-pdavis/output/FireNet', name2)
root3 = '/home/mhy/firenet-pdavis/output/FireNet'

dir_name_intensity = 'intensity'
dir_name_aolp = 'aolp'
dir_name_dolp = 'dolp'

intensity_dir1 = os.path.join(root1, dir_name_intensity)
intensity_dir2 = os.path.join(root2, dir_name_intensity)

intensity_list1 = [x for x in os.listdir(intensity_dir1) if x.endswith('.png')]
intensity_list1.sort(key=lambda x:int(x.split('_')[1].split('.')[0]))
# intensity_list2 = [x for x in os.listdir(intensity_dir2) if x.endswith('.png')]
# intensity_list2.sort()

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# fourcc = cv2.VideoWriter_fourcc(*'xvid')
video = cv2.VideoWriter(os.path.join(root3, name + '.avi'), fourcc, rate, size, 0)

for i in tqdm(range(len(intensity_list1))):
    intensity1 = cv2.imread(os.path.join(root1, dir_name_intensity, intensity_list1[i]), cv2.IMREAD_GRAYSCALE)
    aolp1 = cv2.imread(os.path.join(root1, dir_name_aolp, intensity_list1[i][:-6] + '_aolp.png'), cv2.IMREAD_GRAYSCALE)
    dolp1 = cv2.imread(os.path.join(root1, dir_name_dolp, intensity_list1[i][:-6] + '_dolp.png'), cv2.IMREAD_GRAYSCALE)

    # print(intensity1.shape)
    # print(aolp1.shape)
    # print(dolp1.shape)
    image1 = cv2.hconcat([intensity1, aolp1, dolp1])
    # print(image1.shape)

    # intensity2 = cv2.imread(os.path.join(root2, dir_name_intensity, intensity_list2[i]))
    # aolp2 = cv2.imread(os.path.join(root2, dir_name_aolp, intensity_list2[i][:-6] + '_aolp.png'))
    # dolp2 = cv2.imread(os.path.join(root2, dir_name_dolp, intensity_list2[i][:-6] + '_dolp.png'))
    #
    # image2 = cv2.hconcat([intensity2, aolp2, dolp2])
    #
    # image = cv2.vconcat([image1, image2])

    # video.write(image)
    video.write(image1)

cv2.destroyAllWindows()
video.release()

print('Succeed!')