"""
 @Time    : 20.09.22 21:27
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : model_v.py
 @Function:
 
"""
import cv2
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base.base_model import BaseModel
from .submodules_v import *

from .legacy import FireNet_legacy


class V0(BaseModel):
    """
    predict s0, s1, and s2
    """
    def __init__(self, num_bins=5, base_num_channels=32, kernel_size=3, num_recurrent_units=1, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        self.num_recurrent_units = num_recurrent_units
        # s0
        self.H_s0 = Average(self.num_bins, base_num_channels)
        self.G_s0 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R_s0 = ResidualBlock(base_num_channels)
        self.P_s0 = nn.Conv2d(base_num_channels, 1, 1, 1, 0)

        # s1
        self.H_s1 = Contrast(self.num_bins, base_num_channels)
        self.G_s1 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R_s1 = ResidualBlock(base_num_channels)
        self.P_s1 = nn.Conv2d(base_num_channels, 1, 1, 1, 0)

        # s2
        self.H_s2 = Contrast(self.num_bins, base_num_channels)
        self.G_s2 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R_s2 = ResidualBlock(base_num_channels)
        self.P_s2 = nn.Conv2d(base_num_channels, 1, 1, 1, 0)

        self.reset_states_s0()
        self.reset_states_s1()
        self.reset_states_s2()

    def reset_states_s0(self):
        self._states_s0 = [None] * self.num_recurrent_units

    def reset_states_s1(self):
        self._states_s1 = [None] * self.num_recurrent_units

    def reset_states_s2(self):
        self._states_s2 = [None] * self.num_recurrent_units

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W raw event tensor
        :return: N x 1 x H/2 x W/2 s0, s1, and s2
        """
        # split input
        x_90 = x[:, :, 0::2, 0::2]
        x_45 = x[:, :, 0::2, 1::2]
        x_135 = x[:, :, 1::2, 0::2]
        x_0 = x[:, :, 1::2, 1::2]

        # s0
        x_s0 = self.H_s0(x_0, x_90)
        x_s0 = self.G_s0(x_s0, self._states_s0[0])
        self._states_s0[0] = x_s0
        x_s0 = self.R_s0(x_s0)
        s0 = self.P_s0(x_s0)

        # s1
        x_s1 = self.H_s1(x_0, x_90)
        x_s1 = self.G_s1(x_s1, self._states_s1[0])
        self._states_s1[0] = x_s1
        x_s1 = self.R_s1(x_s1)
        s1 = self.P_s1(x_s1)

        # s2
        x_s2 = self.H_s2(x_45, x_135)
        x_s2 = self.G_s2(x_s2, self._states_s2[0])
        self._states_s2[0] = x_s2
        x_s2 = self.R_s2(x_s2)
        s2 = self.P_s2(x_s2)

        return {
            's0': s0,
            's1': (s1 + 1) / 2,
            's2': (s2 + 1) / 2
        }


class V1(BaseModel):
    """
    predict iad
    """
    def __init__(self, num_bins=5, base_num_channels=32, kernel_size=3, num_recurrent_units=1, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        self.num_recurrent_units = num_recurrent_units
        # i
        self.H_i = nn.Conv2d(num_bins * 4, base_num_channels, 1, 1, 0)
        self.G_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R_i = ResidualBlock(base_num_channels)
        self.P_i = nn.Conv2d(base_num_channels, 1, 1, 1, 0)

        # a
        self.H_a = nn.Conv2d(num_bins * 4, base_num_channels, 1, 1, 0)
        self.G_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R_a = ResidualBlock(base_num_channels)
        self.P_a = nn.Conv2d(base_num_channels, 1, 1, 1, 0)

        # d
        self.H_d = nn.Conv2d(num_bins * 4, base_num_channels, 1, 1, 0)
        self.G_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R_d = ResidualBlock(base_num_channels)
        self.P_d = nn.Conv2d(base_num_channels, 1, 1, 1, 0)

        self.reset_states_i()
        self.reset_states_a()
        self.reset_states_d()

    def reset_states_i(self):
        self._states_i = [None] * self.num_recurrent_units

    def reset_states_a(self):
        self._states_a = [None] * self.num_recurrent_units

    def reset_states_d(self):
        self._states_d = [None] * self.num_recurrent_units

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W raw event tensor
        :return: N x 1 x H/2 x W/2 iad
        """
        # split input
        x_90 = x[:, :, 0::2, 0::2]
        x_45 = x[:, :, 0::2, 1::2]
        x_135 = x[:, :, 1::2, 0::2]
        x_0 = x[:, :, 1::2, 1::2]

        x_four = torch.cat((x_90, x_45, x_135, x_0), 1)

        # i
        x_i = self.H_i(x_four)
        x_i = self.G_i(x_i, self._states_i[0])
        self._states_i[0] = x_i
        x_i = self.R_i(x_i)
        i = self.P_i(x_i)

        # a
        x_a = self.H_a(x_four)
        x_a = self.G_a(x_a, self._states_a[0])
        self._states_a[0] = x_a
        x_a = self.R_a(x_a)
        a = self.P_a(x_a)

        # d
        x_d = self.H_d(x_four)
        x_d = self.G_d(x_d, self._states_d[0])
        self._states_d[0] = x_d
        x_d = self.R_d(x_d)
        d = self.P_d(x_d)

        return {
            'i': i,
            'a': a,
            'd': d
        }


class V2(BaseModel):
    """
    predict s012 and iad
    """
    def __init__(self, num_bins=5, base_num_channels=32, kernel_size=3, num_recurrent_units=2, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        self.num_recurrent_units = num_recurrent_units
        # s0
        self.H_s0 = nn.Conv2d(num_bins, base_num_channels, 3, 1, 1)
        self.G1_s0 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_s0 = ResidualBlock(base_num_channels)
        self.G2_s0 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_s0 = ResidualBlock(base_num_channels)
        self.P_s0 = nn.Conv2d(base_num_channels, 1, 1, 1, 0)

        # s1
        self.H_s1 = nn.Conv2d(num_bins, base_num_channels, 3, 1, 1)
        self.G1_s1 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_s1 = ResidualBlock(base_num_channels)
        self.G2_s1 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_s1 = ResidualBlock(base_num_channels)
        self.P_s1 = nn.Conv2d(base_num_channels, 1, 1, 1, 0)

        # s2
        self.H_s2 = nn.Conv2d(num_bins, base_num_channels, 3, 1, 1)
        self.G1_s2 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_s2 = ResidualBlock(base_num_channels)
        self.G2_s2 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_s2 = ResidualBlock(base_num_channels)
        self.P_s2 = nn.Conv2d(base_num_channels, 1, 1, 1, 0)

        self.reset_states_s0()
        self.reset_states_s1()
        self.reset_states_s2()

        self.E_i = nn.Sequential(nn.Conv2d(base_num_channels, base_num_channels, 3, 1, 1), nn.ReLU(),
                                 nn.Conv2d(base_num_channels, 1, 3, 1, 1))
        self.E_a = nn.Sequential(nn.Conv2d(base_num_channels * 2, base_num_channels, 3, 1, 1), nn.ReLU(),
                                 nn.Conv2d(base_num_channels, 1, 3, 1, 1))
        self.E_d = nn.Sequential(nn.Conv2d(base_num_channels * 3, base_num_channels, 3, 1, 1), nn.ReLU(),
                                 nn.Conv2d(base_num_channels, 1, 3, 1, 1))

    def reset_states_s0(self):
        self._states_s0 = [None] * self.num_recurrent_units

    def reset_states_s1(self):
        self._states_s1 = [None] * self.num_recurrent_units

    def reset_states_s2(self):
        self._states_s2 = [None] * self.num_recurrent_units

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W raw event tensor
        :return: N x 1 x H/2 x W/2 s0, s1, and s2
        """
        # split input
        x_90 = x[:, :, 0::2, 0::2]
        x_45 = x[:, :, 0::2, 1::2]
        x_135 = x[:, :, 1::2, 0::2]
        x_0 = x[:, :, 1::2, 1::2]

        # s0
        x_s0 = self.H_s0((x_0 + x_90) / 2)
        x_s0 = self.G1_s0(x_s0, self._states_s0[0])
        self._states_s0[0] = x_s0
        x_s0 = self.R1_s0(x_s0)
        x_s0 = self.G2_s0(x_s0, self._states_s0[1])
        self._states_s0[1] = x_s0
        x_s0 = self.R2_s0(x_s0)
        s0 = self.P_s0(x_s0)

        # s1
        x_s1 = self.H_s1(x_0 - x_90)
        x_s1 = self.G1_s1(x_s1, self._states_s1[0])
        self._states_s1[0] = x_s1
        x_s1 = self.R1_s1(x_s1)
        x_s1 = self.G2_s1(x_s1, self._states_s1[1])
        self._states_s1[1] = x_s1
        x_s1 = self.R2_s1(x_s1)
        s1 = self.P_s1(x_s1)

        # s2
        x_s2 = self.H_s2(x_45 - x_135)
        x_s2 = self.G1_s2(x_s2, self._states_s2[0])
        self._states_s2[0] = x_s2
        x_s2 = self.R1_s2(x_s2)
        x_s2 = self.G2_s2(x_s2, self._states_s2[1])
        self._states_s2[1] = x_s2
        x_s2 = self.R2_s2(x_s2)
        s2 = self.P_s2(x_s2)

        s0_c = torch.clip(s0, 0, 2)
        s1_c = torch.clip(s1, -1, 1)
        s2_c = torch.clip(s2, -1, 1)

        # i
        o_i = s0_c / 2
        e_i = self.E_i(x_s0)
        i = torch.clip(o_i + e_i, 0, 1)

        # a
        o_a = 0.5 * torch.arctan2(s2_c, s1_c)
        e_a = self.E_a(torch.cat((x_s1, x_s2), 1))
        a = torch.clip(o_a + e_a, 0, 1)

        # d
        o_d = torch.div(torch.sqrt(torch.square(s1_c) + torch.square(s2_c)), s0_c + 1e-4)
        e_d = self.E_d(torch.cat((x_s0, x_s1, x_s2), 1))
        d = torch.clip(o_d + e_d, 0, 1)

        return {
            'i': i,
            'a': a,
            'd': d,
            's0': s0_c / 2,
            's1': (s1_c + 1) / 2,
            's2': (s2_c + 1) / 2
        }


class V3(BaseModel):
    """
    predict iad
    """
    def __init__(self, num_bins=5, base_num_channels=32, kernel_size=3, num_recurrent_units=2, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        self.num_recurrent_units = num_recurrent_units
        # i
        self.H_i = nn.Conv2d(num_bins * 4, base_num_channels, 1, 1, 0)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_i = ResidualBlock(base_num_channels)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_i = ResidualBlock(base_num_channels)
        self.P_i = nn.Conv2d(base_num_channels, 1, 1, 1, 0)

        # a
        self.H_a = nn.Conv2d(num_bins * 4, base_num_channels, 1, 1, 0)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_a = ResidualBlock(base_num_channels)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_a = ResidualBlock(base_num_channels)
        self.P_a = nn.Conv2d(base_num_channels, 1, 1, 1, 0)

        # d
        self.H_d = nn.Conv2d(num_bins * 4, base_num_channels, 1, 1, 0)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_d = ResidualBlock(base_num_channels)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_d = ResidualBlock(base_num_channels)
        self.P_d = nn.Conv2d(base_num_channels, 1, 1, 1, 0)

        self.reset_states_i()
        self.reset_states_a()
        self.reset_states_d()

    def reset_states_i(self):
        self._states_i = [None] * self.num_recurrent_units

    def reset_states_a(self):
        self._states_a = [None] * self.num_recurrent_units

    def reset_states_d(self):
        self._states_d = [None] * self.num_recurrent_units

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W raw event tensor
        :return: N x 1 x H/2 x W/2 iad
        """
        # split input
        x_90 = x[:, :, 0::2, 0::2]
        x_45 = x[:, :, 0::2, 1::2]
        x_135 = x[:, :, 1::2, 0::2]
        x_0 = x[:, :, 1::2, 1::2]

        x_four = torch.cat((x_90, x_45, x_135, x_0), 1)

        # i
        x_i = self.H_i(x_four)
        x_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_i
        x_i = self.R1_i(x_i)
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i
        x_i = self.R2_i(x_i)
        i = self.P_i(x_i)

        # a
        x_a = self.H_a(x_four)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a
        x_a = self.R1_a(x_a)
        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a
        x_a = self.R2_a(x_a)
        a = self.P_a(x_a)

        # d
        x_d = self.H_d(x_four)
        x_d = self.G1_d(x_d, self._states_d[0])
        self._states_d[0] = x_d
        x_d = self.R1_d(x_d)
        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d
        x_d = self.R2_d(x_d)
        d = self.P_d(x_d)

        return {
            'i': i,
            'a': a,
            'd': d
        }
