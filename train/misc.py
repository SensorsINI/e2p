"""
 @Time    : 2/14/22 16:30
 @Author  : Haiyang Mei
 @E-mail  : mhy666@mail.dlut.edu.cn
 
 @Project : firenet-pdavis
 @File    : misc.py
 @Function:
 
"""
import os

def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
