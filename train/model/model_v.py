"""
 @Time    : 06.10.22 10:30
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : model_v.py
 @Function:
 
"""
import math

import cv2
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base.base_model import BaseModel
from .submodules_v import *

from train.model.legacy import FireNet_legacy
from train.model.model_original import E2VIDRecurrent
from train.utils.henri_compatible import make_henri_compatible
from train.model import model_original as model_arch


class V0(BaseModel):
    """
    FireNet + Conv i, a, d
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, num_recurrent_units=2, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        self.num_recurrent_units = num_recurrent_units

        self.head = nn.Sequential(nn.Conv2d(self.num_bins, base_num_channels, 2, 2, 0), nn.ReLU())
        self.G1 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1 = ResidualBlock(base_num_channels)
        self.G2 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2 = ResidualBlock(base_num_channels)
        self.pred_i = nn.Conv2d(base_num_channels, 1, 1, 1, 0)
        self.pred_a = nn.Conv2d(base_num_channels, 1, 1, 1, 0)
        self.pred_d = nn.Conv2d(base_num_channels, 1, 1, 1, 0)

        self.reset_states()

    def reset_states(self):
        self._states = [None] * self.num_recurrent_units

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W event tensor
        :return: N x 1 x H/2 x W/2 i, a, d
        """
        x = self.head(x)
        x = self.G1(x, self._states[0])
        self._states[0] = x
        x = self.R1(x)
        x = self.G2(x, self._states[1])
        self._states[1] = x
        x = self.R2(x)
        i = self.pred_i(x)
        a = self.pred_a(x)
        d = self.pred_d(x)

        return {
            'i': i,
            'a': a,
            'd': d
        }


class V1(BaseModel):
    """
    Three FireNet branches for i, a, and d, respectively.
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, num_recurrent_units=2, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        self.num_recurrent_units = num_recurrent_units

        self.head_i = nn.Sequential(nn.Conv2d(self.num_bins * 4, base_num_channels, 3, 1, 1), nn.ReLU())
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_i = ResidualBlock(base_num_channels)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_i = ResidualBlock(base_num_channels)
        self.pred_i = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.head_a = nn.Sequential(nn.Conv2d(self.num_bins * 4, base_num_channels, 3, 1, 1), nn.ReLU())
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_a = ResidualBlock(base_num_channels)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_a = ResidualBlock(base_num_channels)
        self.pred_a = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.head_d = nn.Sequential(nn.Conv2d(self.num_bins * 4, base_num_channels, 3, 1, 1), nn.ReLU())
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_d = ResidualBlock(base_num_channels)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_d = ResidualBlock(base_num_channels)
        self.pred_d = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

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
        :param x: N x num_input_channels x H x W event tensor
        :return: N x 1 x H/2 x W/2 i, a, d
        """
        # split input
        x_90 = x[:, :, 0::2, 0::2]
        x_45 = x[:, :, 0::2, 1::2]
        x_135 = x[:, :, 1::2, 0::2]
        x_0 = x[:, :, 1::2, 1::2]

        x_four = torch.cat((x_90, x_45, x_135, x_0), 1)

        # i
        x_i = self.head_i(x_four)
        x_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_i
        x_i = self.R1_i(x_i)
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i
        x_i = self.R2_i(x_i)
        i = self.pred_i(x_i)

        # a
        x_a = self.head_a(x_four)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a
        x_a = self.R1_a(x_a)
        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a
        x_a = self.R2_a(x_a)
        a = self.pred_a(x_a)

        # d
        x_d = self.head_d(x_four)
        x_d = self.G1_d(x_d, self._states_d[0])
        self._states_d[0] = x_d
        x_d = self.R1_d(x_d)
        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d
        x_d = self.R2_d(x_d)
        d = self.pred_d(x_d)

        return {
            'i': torch.clip(i, 0, 1),
            'a': torch.clip(a, 0, 1),
            'd': torch.clip(d, 0, 1)
        }


class V2(BaseModel):
    """
    Four FireNet branches for the four channels.
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, num_recurrent_units=2, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        self.num_recurrent_units = num_recurrent_units

        self.head_i90 = nn.Sequential(nn.Conv2d(self.num_bins, base_num_channels, 3, 1, 1), nn.ReLU())
        self.G1_i90 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_i90 = ResidualBlock(base_num_channels)
        self.G2_i90 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_i90 = ResidualBlock(base_num_channels)
        self.pred_i90 = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.head_i45 = nn.Sequential(nn.Conv2d(self.num_bins, base_num_channels, 3, 1, 1), nn.ReLU())
        self.G1_i45 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_i45 = ResidualBlock(base_num_channels)
        self.G2_i45 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_i45 = ResidualBlock(base_num_channels)
        self.pred_i45 = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.head_i135 = nn.Sequential(nn.Conv2d(self.num_bins, base_num_channels, 3, 1, 1), nn.ReLU())
        self.G1_i135 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_i135 = ResidualBlock(base_num_channels)
        self.G2_i135 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_i135 = ResidualBlock(base_num_channels)
        self.pred_i135 = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.head_i0 = nn.Sequential(nn.Conv2d(self.num_bins, base_num_channels, 3, 1, 1), nn.ReLU())
        self.G1_i0 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_i0 = ResidualBlock(base_num_channels)
        self.G2_i0 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_i0 = ResidualBlock(base_num_channels)
        self.pred_i0 = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.reset_states_i90()
        self.reset_states_i45()
        self.reset_states_i135()
        self.reset_states_i0()

        self.sigmoid = nn.Sigmoid()

    def reset_states_i90(self):
        self._states_i90 = [None] * self.num_recurrent_units

    def reset_states_i45(self):
        self._states_i45 = [None] * self.num_recurrent_units

    def reset_states_i135(self):
        self._states_i135 = [None] * self.num_recurrent_units

    def reset_states_i0(self):
        self._states_i0 = [None] * self.num_recurrent_units

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W event tensor
        :return: N x 1 x H/2 x W/2 i, a, d
        """
        # split input
        x_i90 = x[:, :, 0::2, 0::2]
        x_i45 = x[:, :, 0::2, 1::2]
        x_i135 = x[:, :, 1::2, 0::2]
        x_i0 = x[:, :, 1::2, 1::2]

        # i90
        x_i90 = self.head_i90(x_i90)
        x_i90 = self.G1_i90(x_i90, self._states_i90[0])
        self._states_i90[0] = x_i90
        x_i90 = self.R1_i90(x_i90)
        x_i90 = self.G2_i90(x_i90, self._states_i90[1])
        self._states_i90[1] = x_i90
        x_i90 = self.R2_i90(x_i90)
        i90 = self.pred_i90(x_i90)

        # i45
        x_i45 = self.head_i45(x_i45)
        x_i45 = self.G1_i45(x_i45, self._states_i45[0])
        self._states_i45[0] = x_i45
        x_i45 = self.R1_i45(x_i45)
        x_i45 = self.G2_i45(x_i45, self._states_i45[1])
        self._states_i45[1] = x_i45
        x_i45 = self.R2_i45(x_i45)
        i45 = self.pred_i45(x_i45)

        # i135
        x_i135 = self.head_i135(x_i135)
        x_i135 = self.G1_i135(x_i135, self._states_i135[0])
        self._states_i135[0] = x_i135
        x_i135 = self.R1_i135(x_i135)
        x_i135 = self.G2_i135(x_i135, self._states_i135[1])
        self._states_i135[1] = x_i135
        x_i135 = self.R2_i135(x_i135)
        i135 = self.pred_i135(x_i135)

        # i0
        x_i0 = self.head_i0(x_i0)
        x_i0 = self.G1_i0(x_i0, self._states_i0[0])
        self._states_i0[0] = x_i0
        x_i0 = self.R1_i0(x_i0)
        x_i0 = self.G2_i0(x_i0, self._states_i0[1])
        self._states_i0[1] = x_i0
        x_i0 = self.R2_i0(x_i0)
        i0 = self.pred_i0(x_i0)

        # iad
        # i90 = torch.clip(i90, 0, 1)
        # i45 = torch.clip(i45, 0, 1)
        # i135 = torch.clip(i135, 0, 1)
        # i0 = torch.clip(i0, 0, 1)
        i90 = self.sigmoid(i90)
        i45 = self.sigmoid(i45)
        i135 = self.sigmoid(i135)
        i0 = self.sigmoid(i0)

        s0 = i0 + i90
        s1 = i0 - i90
        s2 = i45 - i135

        i = s0 / 2
        a = 0.5 * torch.arctan2(s2, s1 + 1e-6)
        a = (a + 0.5 * math.pi) / math.pi
        d = torch.div(torch.sqrt(torch.square(s1) + torch.square(s2)), s0 + 1e-6)

        return {
            'i90': i90,
            'i45': i45,
            'i135': i135,
            'i0': i0,
            'i': torch.clip(i, 0, 1),
            'a': torch.clip(a, 0, 1),
            'd': torch.clip(d, 0, 1)
        }


class V3(BaseModel):
    """
    Four FireNet branches for the four channels + ad branches for refinement
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, num_recurrent_units=2, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        self.num_recurrent_units = num_recurrent_units

        self.head_i90 = nn.Sequential(nn.Conv2d(self.num_bins, base_num_channels, 3, 1, 1), nn.ReLU())
        self.G1_i90 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i90 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_i90 = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.head_i45 = nn.Sequential(nn.Conv2d(self.num_bins, base_num_channels, 3, 1, 1), nn.ReLU())
        self.G1_i45 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i45 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_i45 = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.head_i135 = nn.Sequential(nn.Conv2d(self.num_bins, base_num_channels, 3, 1, 1), nn.ReLU())
        self.G1_i135 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i135 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_i135 = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.head_i0 = nn.Sequential(nn.Conv2d(self.num_bins, base_num_channels, 3, 1, 1), nn.ReLU())
        self.G1_i0 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i0 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_i0 = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.jr1 = JR(base_num_channels)
        self.jr2 = JR(base_num_channels)

        self.head_a = RPPP(self.num_bins, base_num_channels)
        self.G_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R_a = ResidualBlock(base_num_channels)
        self.pred_a = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.head_d = RPPP(self.num_bins, base_num_channels)
        self.G_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R_d = ResidualBlock(base_num_channels)
        self.pred_d = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.reset_states_i90()
        self.reset_states_i45()
        self.reset_states_i135()
        self.reset_states_i0()
        self.reset_states_a()
        self.reset_states_d()

        self.sigmoid = nn.Sigmoid()

    def reset_states_i90(self):
        self._states_i90 = [None] * self.num_recurrent_units

    def reset_states_i45(self):
        self._states_i45 = [None] * self.num_recurrent_units

    def reset_states_i135(self):
        self._states_i135 = [None] * self.num_recurrent_units

    def reset_states_i0(self):
        self._states_i0 = [None] * self.num_recurrent_units

    def reset_states_a(self):
        self._states_a = [None] * 1

    def reset_states_d(self):
        self._states_d = [None] * 1

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W event tensor
        :return: N x 1 x H/2 x W/2 i, a, d
        """
        # split input
        x_i90 = x[:, :, 0::2, 0::2]
        x_i45 = x[:, :, 0::2, 1::2]
        x_i135 = x[:, :, 1::2, 0::2]
        x_i0 = x[:, :, 1::2, 1::2]

        # first stage
        x_i90 = self.head_i90(x_i90)
        x_i90 = self.G1_i90(x_i90, self._states_i90[0])
        self._states_i90[0] = x_i90

        x_i45 = self.head_i45(x_i45)
        x_i45 = self.G1_i45(x_i45, self._states_i45[0])
        self._states_i45[0] = x_i45

        x_i135 = self.head_i135(x_i135)
        x_i135 = self.G1_i135(x_i135, self._states_i135[0])
        self._states_i135[0] = x_i135

        x_i0 = self.head_i0(x_i0)
        x_i0 = self.G1_i0(x_i0, self._states_i0[0])
        self._states_i0[0] = x_i0

        x_i90, x_i45, x_i135, x_i0 = self.jr1(x_i90, x_i45, x_i135, x_i0)

        # second stage
        x_i90 = self.G2_i90(x_i90, self._states_i90[1])
        self._states_i90[1] = x_i90

        x_i45 = self.G2_i45(x_i45, self._states_i45[1])
        self._states_i45[1] = x_i45

        x_i135 = self.G2_i135(x_i135, self._states_i135[1])
        self._states_i135[1] = x_i135

        x_i0 = self.G2_i0(x_i0, self._states_i0[1])
        self._states_i0[1] = x_i0

        x_i90, x_i45, x_i135, x_i0 = self.jr2(x_i90, x_i45, x_i135, x_i0)

        # direction prediction
        i90 = self.pred_i90(x_i90)
        i45 = self.pred_i45(x_i45)
        i135 = self.pred_i135(x_i135)
        i0 = self.pred_i0(x_i0)

        # sigmoid
        i90 = self.sigmoid(i90)
        i45 = self.sigmoid(i45)
        i135 = self.sigmoid(i135)
        i0 = self.sigmoid(i0)

        # i
        i = (i90 + i0) / 2

        # ad0
        s0 = i0 + i90
        s1 = i0 - i90
        s2 = i45 - i135

        a0 = 0.5 * torch.arctan2(s2, s1)
        a0 = (a0 + 0.5 * math.pi) / math.pi

        d0 = torch.div(torch.sqrt(torch.square(s1) + torch.square(s2)), s0 + 1e-6)
        d0 = torch.clip(d0, 0, 1)

        # a1
        x_a = self.head_a(x)
        x_a = self.G_a(x_a, self._states_a[0])
        self._states_a[0] = x_a
        x_a = self.R_a(x_a)
        a1 = self.pred_a(x_a)

        # a
        a = (a0 + a1) / 2
        a = torch.clip(a, 0, 1)

        # d1
        x_d = self.head_d(x)
        x_d = self.G_d(x_d, self._states_d[0])
        self._states_d[0] = x_d
        x_d = self.R_d(x_d)
        d1 = self.pred_d(x_d)

        # d
        d = (d0 + d1) / 2
        d = torch.clip(d, 0, 1)

        return {
            'i90': i90,
            'i45': i45,
            'i135': i135,
            'i0': i0,
            'i': i,
            'a': a,
            'd': d
        }


class V3_BN(BaseModel):
    """
    Four FireNet branches for the four channels + ad branches for refinement + batch normalization
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, num_recurrent_units=2, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        self.num_recurrent_units = num_recurrent_units

        self.head_i90 = nn.Sequential(nn.Conv2d(self.num_bins, base_num_channels, 3, 1, 1),
                                      nn.BatchNorm2d(base_num_channels), nn.ReLU())
        self.G1_i90 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i90 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_i90 = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.head_i45 = nn.Sequential(nn.Conv2d(self.num_bins, base_num_channels, 3, 1, 1),
                                      nn.BatchNorm2d(base_num_channels), nn.ReLU())
        self.G1_i45 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i45 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_i45 = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.head_i135 = nn.Sequential(nn.Conv2d(self.num_bins, base_num_channels, 3, 1, 1),
                                       nn.BatchNorm2d(base_num_channels), nn.ReLU())
        self.G1_i135 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i135 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_i135 = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.head_i0 = nn.Sequential(nn.Conv2d(self.num_bins, base_num_channels, 3, 1, 1),
                                     nn.BatchNorm2d(base_num_channels), nn.ReLU())
        self.G1_i0 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i0 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_i0 = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.jr1 = JR_BN(base_num_channels)
        self.jr2 = JR_BN(base_num_channels)

        self.head_a = RPPP_BN(self.num_bins, base_num_channels)
        self.G_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R_a = ResidualBlock_BN(base_num_channels)
        self.pred_a = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.head_d = RPPP_BN(self.num_bins, base_num_channels)
        self.G_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R_d = ResidualBlock_BN(base_num_channels)
        self.pred_d = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.reset_states_i90()
        self.reset_states_i45()
        self.reset_states_i135()
        self.reset_states_i0()
        self.reset_states_a()
        self.reset_states_d()

        self.sigmoid = nn.Sigmoid()

    def reset_states_i90(self):
        self._states_i90 = [None] * self.num_recurrent_units

    def reset_states_i45(self):
        self._states_i45 = [None] * self.num_recurrent_units

    def reset_states_i135(self):
        self._states_i135 = [None] * self.num_recurrent_units

    def reset_states_i0(self):
        self._states_i0 = [None] * self.num_recurrent_units

    def reset_states_a(self):
        self._states_a = [None] * 1

    def reset_states_d(self):
        self._states_d = [None] * 1

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W event tensor
        :return: N x 1 x H/2 x W/2 i, a, d
        """
        # split input
        x_i90 = x[:, :, 0::2, 0::2]
        x_i45 = x[:, :, 0::2, 1::2]
        x_i135 = x[:, :, 1::2, 0::2]
        x_i0 = x[:, :, 1::2, 1::2]

        # first stage
        x_i90 = self.head_i90(x_i90)
        x_i90 = self.G1_i90(x_i90, self._states_i90[0])
        self._states_i90[0] = x_i90

        x_i45 = self.head_i45(x_i45)
        x_i45 = self.G1_i45(x_i45, self._states_i45[0])
        self._states_i45[0] = x_i45

        x_i135 = self.head_i135(x_i135)
        x_i135 = self.G1_i135(x_i135, self._states_i135[0])
        self._states_i135[0] = x_i135

        x_i0 = self.head_i0(x_i0)
        x_i0 = self.G1_i0(x_i0, self._states_i0[0])
        self._states_i0[0] = x_i0

        x_i90, x_i45, x_i135, x_i0 = self.jr1(x_i90, x_i45, x_i135, x_i0)

        # second stage
        x_i90 = self.G2_i90(x_i90, self._states_i90[1])
        self._states_i90[1] = x_i90

        x_i45 = self.G2_i45(x_i45, self._states_i45[1])
        self._states_i45[1] = x_i45

        x_i135 = self.G2_i135(x_i135, self._states_i135[1])
        self._states_i135[1] = x_i135

        x_i0 = self.G2_i0(x_i0, self._states_i0[1])
        self._states_i0[1] = x_i0

        x_i90, x_i45, x_i135, x_i0 = self.jr2(x_i90, x_i45, x_i135, x_i0)

        # direction prediction
        i90 = self.pred_i90(x_i90)
        i45 = self.pred_i45(x_i45)
        i135 = self.pred_i135(x_i135)
        i0 = self.pred_i0(x_i0)

        # sigmoid
        i90 = self.sigmoid(i90)
        i45 = self.sigmoid(i45)
        i135 = self.sigmoid(i135)
        i0 = self.sigmoid(i0)

        # i
        i = (i90 + i0) / 2

        # ad0
        s0 = i0 + i90
        s1 = i0 - i90
        s2 = i45 - i135

        a0 = 0.5 * torch.arctan2(s2, s1)
        a0 = (a0 + 0.5 * math.pi) / math.pi

        d0 = torch.div(torch.sqrt(torch.square(s1) + torch.square(s2)), s0 + 1e-6)
        d0 = torch.clip(d0, 0, 1)

        # a1
        x_a = self.head_a(x)
        x_a = self.G_a(x_a, self._states_a[0])
        self._states_a[0] = x_a
        x_a = self.R_a(x_a)
        a1 = self.pred_a(x_a)

        # a
        a = (a0 + a1) / 2
        a = torch.clip(a, 0, 1)

        # d1
        x_d = self.head_d(x)
        x_d = self.G_d(x_d, self._states_d[0])
        self._states_d[0] = x_d
        x_d = self.R_d(x_d)
        d1 = self.pred_d(x_d)

        # d
        d = (d0 + d1) / 2
        d = torch.clip(d, 0, 1)

        return {
            'i90': i90,
            'i45': i45,
            'i135': i135,
            'i0': i0,
            'i': i,
            'a': a,
            'd': d
            # 'i': i,
            # 'a': a0,
            # 'd': d0
        }


class V4(BaseModel):
    """
    Four FireNet branches for the four channels + ad branches (two gr blocks) for refinement + batch normalization
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, num_recurrent_units=2, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        self.num_recurrent_units = num_recurrent_units

        self.head_i90 = nn.Sequential(nn.Conv2d(self.num_bins, base_num_channels, 3, 1, 1),
                                      nn.BatchNorm2d(base_num_channels), nn.ReLU())
        self.G1_i90 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i90 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_i90 = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.head_i45 = nn.Sequential(nn.Conv2d(self.num_bins, base_num_channels, 3, 1, 1),
                                      nn.BatchNorm2d(base_num_channels), nn.ReLU())
        self.G1_i45 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i45 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_i45 = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.head_i135 = nn.Sequential(nn.Conv2d(self.num_bins, base_num_channels, 3, 1, 1),
                                       nn.BatchNorm2d(base_num_channels), nn.ReLU())
        self.G1_i135 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i135 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_i135 = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.head_i0 = nn.Sequential(nn.Conv2d(self.num_bins, base_num_channels, 3, 1, 1),
                                     nn.BatchNorm2d(base_num_channels), nn.ReLU())
        self.G1_i0 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i0 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_i0 = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.jr1 = JR_BN(base_num_channels)
        self.jr2 = JR_BN(base_num_channels)

        self.head_a = RPPP_BN(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_a = ResidualBlock_BN(base_num_channels)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_a = ResidualBlock_BN(base_num_channels)
        self.pred_a = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.head_d = RPPP_BN(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_d = ResidualBlock_BN(base_num_channels)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_d = ResidualBlock_BN(base_num_channels)
        self.pred_d = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.reset_states_i90()
        self.reset_states_i45()
        self.reset_states_i135()
        self.reset_states_i0()
        self.reset_states_a()
        self.reset_states_d()

        self.sigmoid = nn.Sigmoid()

    def reset_states_i90(self):
        self._states_i90 = [None] * self.num_recurrent_units

    def reset_states_i45(self):
        self._states_i45 = [None] * self.num_recurrent_units

    def reset_states_i135(self):
        self._states_i135 = [None] * self.num_recurrent_units

    def reset_states_i0(self):
        self._states_i0 = [None] * self.num_recurrent_units

    def reset_states_a(self):
        self._states_a = [None] * self.num_recurrent_units

    def reset_states_d(self):
        self._states_d = [None] * self.num_recurrent_units

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W event tensor
        :return: N x 1 x H/2 x W/2 i, a, d
        """
        # split input
        x_i90 = x[:, :, 0::2, 0::2]
        x_i45 = x[:, :, 0::2, 1::2]
        x_i135 = x[:, :, 1::2, 0::2]
        x_i0 = x[:, :, 1::2, 1::2]

        # first stage
        x_i90 = self.head_i90(x_i90)
        x_i90 = self.G1_i90(x_i90, self._states_i90[0])
        self._states_i90[0] = x_i90

        x_i45 = self.head_i45(x_i45)
        x_i45 = self.G1_i45(x_i45, self._states_i45[0])
        self._states_i45[0] = x_i45

        x_i135 = self.head_i135(x_i135)
        x_i135 = self.G1_i135(x_i135, self._states_i135[0])
        self._states_i135[0] = x_i135

        x_i0 = self.head_i0(x_i0)
        x_i0 = self.G1_i0(x_i0, self._states_i0[0])
        self._states_i0[0] = x_i0

        x_i90, x_i45, x_i135, x_i0 = self.jr1(x_i90, x_i45, x_i135, x_i0)

        # second stage
        x_i90 = self.G2_i90(x_i90, self._states_i90[1])
        self._states_i90[1] = x_i90

        x_i45 = self.G2_i45(x_i45, self._states_i45[1])
        self._states_i45[1] = x_i45

        x_i135 = self.G2_i135(x_i135, self._states_i135[1])
        self._states_i135[1] = x_i135

        x_i0 = self.G2_i0(x_i0, self._states_i0[1])
        self._states_i0[1] = x_i0

        x_i90, x_i45, x_i135, x_i0 = self.jr2(x_i90, x_i45, x_i135, x_i0)

        # direction prediction
        i90 = self.pred_i90(x_i90)
        i45 = self.pred_i45(x_i45)
        i135 = self.pred_i135(x_i135)
        i0 = self.pred_i0(x_i0)

        # sigmoid
        i90 = self.sigmoid(i90)
        i45 = self.sigmoid(i45)
        i135 = self.sigmoid(i135)
        i0 = self.sigmoid(i0)

        # i
        i = (i90 + i0) / 2

        # ad0
        s0 = i0 + i90
        s1 = i0 - i90
        s2 = i45 - i135

        a0 = 0.5 * torch.arctan2(s2, s1)
        a0 = (a0 + 0.5 * math.pi) / math.pi

        d0 = torch.div(torch.sqrt(torch.square(s1) + torch.square(s2)), s0 + 1e-6)
        d0 = torch.clip(d0, 0, 1)

        # a1
        x_a = self.head_a(x)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a
        x_a = self.R1_a(x_a)
        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a
        x_a = self.R2_a(x_a)
        a1 = self.pred_a(x_a)

        # a
        a = (a0 + a1) / 2
        a = torch.clip(a, 0, 1)

        # d1
        x_d = self.head_d(x)
        x_d = self.G1_d(x_d, self._states_d[0])
        self._states_d[0] = x_d
        x_d = self.R1_d(x_d)
        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d
        x_d = self.R2_d(x_d)
        d1 = self.pred_d(x_d)

        # d
        d = (d0 + d1) / 2
        d = torch.clip(d, 0, 1)

        return {
            'i90': i90,
            'i45': i45,
            'i135': i135,
            'i0': i0,
            'i': i,
            'a': a,
            'd': d
        }


class V5(BaseModel):
    """
    Four FireNet branches for the four channels + ad branches (two gr blocks) for refinement + batch normalization
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, num_recurrent_units=2, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        self.num_recurrent_units = num_recurrent_units

        self.head_i90 = nn.Sequential(nn.Conv2d(self.num_bins, base_num_channels, 3, 1, 1),
                                      nn.BatchNorm2d(base_num_channels), nn.ReLU())
        self.G1_i90 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i90 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_i90 = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.head_i45 = nn.Sequential(nn.Conv2d(self.num_bins, base_num_channels, 3, 1, 1),
                                      nn.BatchNorm2d(base_num_channels), nn.ReLU())
        self.G1_i45 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i45 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_i45 = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.head_i135 = nn.Sequential(nn.Conv2d(self.num_bins, base_num_channels, 3, 1, 1),
                                       nn.BatchNorm2d(base_num_channels), nn.ReLU())
        self.G1_i135 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i135 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_i135 = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.head_i0 = nn.Sequential(nn.Conv2d(self.num_bins, base_num_channels, 3, 1, 1),
                                     nn.BatchNorm2d(base_num_channels), nn.ReLU())
        self.G1_i0 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i0 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_i0 = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.jr1 = JR_BN(base_num_channels)
        self.jr2 = JR_BN(base_num_channels)

        self.head_a = RPPP_BN(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_a = ResidualBlock_BN(base_num_channels)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_a = ResidualBlock_BN(base_num_channels)
        self.pred_a = nn.Conv2d(base_num_channels, 1, 3, 1, 1)
        self.output_a = nn.Conv2d(1, 1, 5, 1, 2)

        self.head_d = RPPP_BN(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_d = ResidualBlock_BN(base_num_channels)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_d = ResidualBlock_BN(base_num_channels)
        self.pred_d = nn.Conv2d(base_num_channels, 1, 3, 1, 1)
        self.output_d = nn.Conv2d(1, 1, 5, 1, 2)

        self.reset_states_i90()
        self.reset_states_i45()
        self.reset_states_i135()
        self.reset_states_i0()
        self.reset_states_a()
        self.reset_states_d()

        self.sigmoid = nn.Sigmoid()

    def reset_states_i90(self):
        self._states_i90 = [None] * self.num_recurrent_units

    def reset_states_i45(self):
        self._states_i45 = [None] * self.num_recurrent_units

    def reset_states_i135(self):
        self._states_i135 = [None] * self.num_recurrent_units

    def reset_states_i0(self):
        self._states_i0 = [None] * self.num_recurrent_units

    def reset_states_a(self):
        self._states_a = [None] * self.num_recurrent_units

    def reset_states_d(self):
        self._states_d = [None] * self.num_recurrent_units

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W event tensor
        :return: N x 1 x H/2 x W/2 i, a, d
        """
        # split input
        x_i90 = x[:, :, 0::2, 0::2]
        x_i45 = x[:, :, 0::2, 1::2]
        x_i135 = x[:, :, 1::2, 0::2]
        x_i0 = x[:, :, 1::2, 1::2]

        # first stage
        x_i90 = self.head_i90(x_i90)
        x_i90 = self.G1_i90(x_i90, self._states_i90[0])
        self._states_i90[0] = x_i90

        x_i45 = self.head_i45(x_i45)
        x_i45 = self.G1_i45(x_i45, self._states_i45[0])
        self._states_i45[0] = x_i45

        x_i135 = self.head_i135(x_i135)
        x_i135 = self.G1_i135(x_i135, self._states_i135[0])
        self._states_i135[0] = x_i135

        x_i0 = self.head_i0(x_i0)
        x_i0 = self.G1_i0(x_i0, self._states_i0[0])
        self._states_i0[0] = x_i0

        x_i90, x_i45, x_i135, x_i0 = self.jr1(x_i90, x_i45, x_i135, x_i0)

        # second stage
        x_i90 = self.G2_i90(x_i90, self._states_i90[1])
        self._states_i90[1] = x_i90

        x_i45 = self.G2_i45(x_i45, self._states_i45[1])
        self._states_i45[1] = x_i45

        x_i135 = self.G2_i135(x_i135, self._states_i135[1])
        self._states_i135[1] = x_i135

        x_i0 = self.G2_i0(x_i0, self._states_i0[1])
        self._states_i0[1] = x_i0

        x_i90, x_i45, x_i135, x_i0 = self.jr2(x_i90, x_i45, x_i135, x_i0)

        # direction prediction
        i90 = self.pred_i90(x_i90)
        i45 = self.pred_i45(x_i45)
        i135 = self.pred_i135(x_i135)
        i0 = self.pred_i0(x_i0)

        # sigmoid
        i90 = self.sigmoid(i90)
        i45 = self.sigmoid(i45)
        i135 = self.sigmoid(i135)
        i0 = self.sigmoid(i0)

        # i
        i = (i90 + i0) / 2

        # ad0
        s0 = i0 + i90
        s1 = i0 - i90
        s2 = i45 - i135

        a0 = 0.5 * torch.arctan2(s2, s1)
        a0 = (a0 + 0.5 * math.pi) / math.pi

        d0 = torch.div(torch.sqrt(torch.square(s1) + torch.square(s2)), s0 + 1e-6)
        d0 = torch.clip(d0, 0, 1)

        # a1
        x_a = self.head_a(x)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a
        x_a = self.R1_a(x_a)
        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a
        x_a = self.R2_a(x_a)
        a1 = self.pred_a(x_a)

        # a
        a = a0 + a1
        a = self.output_a(a)
        a = self.sigmoid(a)

        # d1
        x_d = self.head_d(x)
        x_d = self.G1_d(x_d, self._states_d[0])
        self._states_d[0] = x_d
        x_d = self.R1_d(x_d)
        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d
        x_d = self.R2_d(x_d)
        d1 = self.pred_d(x_d)

        # d
        d = d0 + d1
        d = self.output_d(d)
        d = self.sigmoid(d)

        return {
            'i90': i90,
            'i45': i45,
            'i135': i135,
            'i0': i0,
            'i': i,
            'a': a0,
            'd': d0
        }


class V5_CLIP(BaseModel):
    """
    Four FireNet branches for the four channels + ad branches (two gr blocks) for refinement + batch normalization
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, num_recurrent_units=2, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        self.num_recurrent_units = num_recurrent_units

        self.head_i90 = nn.Sequential(nn.Conv2d(self.num_bins, base_num_channels, 3, 1, 1),
                                      nn.BatchNorm2d(base_num_channels), nn.ReLU())
        self.G1_i90 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i90 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_i90 = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.head_i45 = nn.Sequential(nn.Conv2d(self.num_bins, base_num_channels, 3, 1, 1),
                                      nn.BatchNorm2d(base_num_channels), nn.ReLU())
        self.G1_i45 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i45 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_i45 = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.head_i135 = nn.Sequential(nn.Conv2d(self.num_bins, base_num_channels, 3, 1, 1),
                                       nn.BatchNorm2d(base_num_channels), nn.ReLU())
        self.G1_i135 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i135 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_i135 = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.head_i0 = nn.Sequential(nn.Conv2d(self.num_bins, base_num_channels, 3, 1, 1),
                                     nn.BatchNorm2d(base_num_channels), nn.ReLU())
        self.G1_i0 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i0 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_i0 = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.jr1 = JR_BN(base_num_channels)
        self.jr2 = JR_BN(base_num_channels)

        self.head_a = RPPP_BN(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_a = ResidualBlock_BN(base_num_channels)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_a = ResidualBlock_BN(base_num_channels)
        self.pred_a = nn.Conv2d(base_num_channels, 1, 3, 1, 1)
        self.output_a = nn.Conv2d(1, 1, 5, 1, 2)

        self.head_d = RPPP_BN(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_d = ResidualBlock_BN(base_num_channels)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_d = ResidualBlock_BN(base_num_channels)
        self.pred_d = nn.Conv2d(base_num_channels, 1, 3, 1, 1)
        self.output_d = nn.Conv2d(1, 1, 5, 1, 2)

        self.reset_states_i90()
        self.reset_states_i45()
        self.reset_states_i135()
        self.reset_states_i0()
        self.reset_states_a()
        self.reset_states_d()

        self.sigmoid = nn.Sigmoid()

    def reset_states_i90(self):
        self._states_i90 = [None] * self.num_recurrent_units

    def reset_states_i45(self):
        self._states_i45 = [None] * self.num_recurrent_units

    def reset_states_i135(self):
        self._states_i135 = [None] * self.num_recurrent_units

    def reset_states_i0(self):
        self._states_i0 = [None] * self.num_recurrent_units

    def reset_states_a(self):
        self._states_a = [None] * self.num_recurrent_units

    def reset_states_d(self):
        self._states_d = [None] * self.num_recurrent_units

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W event tensor
        :return: N x 1 x H/2 x W/2 i, a, d
        """
        # split input
        x_i90 = x[:, :, 0::2, 0::2]
        x_i45 = x[:, :, 0::2, 1::2]
        x_i135 = x[:, :, 1::2, 0::2]
        x_i0 = x[:, :, 1::2, 1::2]

        # first stage
        x_i90 = self.head_i90(x_i90)
        x_i90 = self.G1_i90(x_i90, self._states_i90[0])
        self._states_i90[0] = x_i90

        x_i45 = self.head_i45(x_i45)
        x_i45 = self.G1_i45(x_i45, self._states_i45[0])
        self._states_i45[0] = x_i45

        x_i135 = self.head_i135(x_i135)
        x_i135 = self.G1_i135(x_i135, self._states_i135[0])
        self._states_i135[0] = x_i135

        x_i0 = self.head_i0(x_i0)
        x_i0 = self.G1_i0(x_i0, self._states_i0[0])
        self._states_i0[0] = x_i0

        x_i90, x_i45, x_i135, x_i0 = self.jr1(x_i90, x_i45, x_i135, x_i0)

        # second stage
        x_i90 = self.G2_i90(x_i90, self._states_i90[1])
        self._states_i90[1] = x_i90

        x_i45 = self.G2_i45(x_i45, self._states_i45[1])
        self._states_i45[1] = x_i45

        x_i135 = self.G2_i135(x_i135, self._states_i135[1])
        self._states_i135[1] = x_i135

        x_i0 = self.G2_i0(x_i0, self._states_i0[1])
        self._states_i0[1] = x_i0

        x_i90, x_i45, x_i135, x_i0 = self.jr2(x_i90, x_i45, x_i135, x_i0)

        # direction prediction
        i90 = self.pred_i90(x_i90)
        i45 = self.pred_i45(x_i45)
        i135 = self.pred_i135(x_i135)
        i0 = self.pred_i0(x_i0)

        # sigmoid
        i90 = self.sigmoid(i90)
        i45 = self.sigmoid(i45)
        i135 = self.sigmoid(i135)
        i0 = self.sigmoid(i0)

        # i
        i = (i90 + i0) / 2

        # ad0
        s0 = i0 + i90
        s1 = i0 - i90
        s2 = i45 - i135

        a0 = 0.5 * torch.arctan2(s2, s1)
        a0 = (a0 + 0.5 * math.pi) / math.pi

        d0 = torch.div(torch.sqrt(torch.square(s1) + torch.square(s2)), s0 + 1e-6)
        d0 = torch.clip(d0, 0, 1)

        # a1
        x_a = self.head_a(x)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a
        x_a = self.R1_a(x_a)
        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a
        x_a = self.R2_a(x_a)
        a1 = self.pred_a(x_a)

        # a
        a = a0 + a1
        a = self.output_a(a)
        a = torch.clip(a, 0, 1)

        # d1
        x_d = self.head_d(x)
        x_d = self.G1_d(x_d, self._states_d[0])
        self._states_d[0] = x_d
        x_d = self.R1_d(x_d)
        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d
        x_d = self.R2_d(x_d)
        d1 = self.pred_d(x_d)

        # d
        d = d0 + d1
        d = self.output_d(d)
        d = torch.clip(d, 0, 1)

        return {
            'i90': i90,
            'i45': i45,
            'i135': i135,
            'i0': i0,
            'i': i,
            'a': a,
            'd': d
        }


class V5_I(BaseModel):
    """
    Four FireNet branches for the four channels + ad branches (two gr blocks) for refinement + batch normalization
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, num_recurrent_units=2, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        self.num_recurrent_units = num_recurrent_units

        self.head_i90 = nn.Sequential(nn.Conv2d(self.num_bins, base_num_channels, 3, 1, 1),
                                      nn.BatchNorm2d(base_num_channels), nn.ReLU())
        self.G1_i90 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i90 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_i90 = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.head_i45 = nn.Sequential(nn.Conv2d(self.num_bins, base_num_channels, 3, 1, 1),
                                      nn.BatchNorm2d(base_num_channels), nn.ReLU())
        self.G1_i45 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i45 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_i45 = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.head_i135 = nn.Sequential(nn.Conv2d(self.num_bins, base_num_channels, 3, 1, 1),
                                       nn.BatchNorm2d(base_num_channels), nn.ReLU())
        self.G1_i135 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i135 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_i135 = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.head_i0 = nn.Sequential(nn.Conv2d(self.num_bins, base_num_channels, 3, 1, 1),
                                     nn.BatchNorm2d(base_num_channels), nn.ReLU())
        self.G1_i0 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i0 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_i0 = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.jr1 = JR_BN(base_num_channels)
        self.jr2 = JR_BN(base_num_channels)

        self.output_i = nn.Conv2d(1, 1, 5, 1, 2)

        self.head_a = RPPP_BN(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_a = ResidualBlock_BN(base_num_channels)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_a = ResidualBlock_BN(base_num_channels)
        self.pred_a = nn.Conv2d(base_num_channels, 1, 3, 1, 1)
        self.output_a = nn.Conv2d(1, 1, 5, 1, 2)

        self.head_d = RPPP_BN(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_d = ResidualBlock_BN(base_num_channels)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_d = ResidualBlock_BN(base_num_channels)
        self.pred_d = nn.Conv2d(base_num_channels, 1, 3, 1, 1)
        self.output_d = nn.Conv2d(1, 1, 5, 1, 2)

        self.reset_states_i90()
        self.reset_states_i45()
        self.reset_states_i135()
        self.reset_states_i0()
        self.reset_states_a()
        self.reset_states_d()

        self.sigmoid = nn.Sigmoid()

    def reset_states_i90(self):
        self._states_i90 = [None] * self.num_recurrent_units

    def reset_states_i45(self):
        self._states_i45 = [None] * self.num_recurrent_units

    def reset_states_i135(self):
        self._states_i135 = [None] * self.num_recurrent_units

    def reset_states_i0(self):
        self._states_i0 = [None] * self.num_recurrent_units

    def reset_states_a(self):
        self._states_a = [None] * self.num_recurrent_units

    def reset_states_d(self):
        self._states_d = [None] * self.num_recurrent_units

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W event tensor
        :return: N x 1 x H/2 x W/2 i, a, d
        """
        # split input
        x_i90 = x[:, :, 0::2, 0::2]
        x_i45 = x[:, :, 0::2, 1::2]
        x_i135 = x[:, :, 1::2, 0::2]
        x_i0 = x[:, :, 1::2, 1::2]

        # first stage
        x_i90 = self.head_i90(x_i90)
        x_i90 = self.G1_i90(x_i90, self._states_i90[0])
        self._states_i90[0] = x_i90

        x_i45 = self.head_i45(x_i45)
        x_i45 = self.G1_i45(x_i45, self._states_i45[0])
        self._states_i45[0] = x_i45

        x_i135 = self.head_i135(x_i135)
        x_i135 = self.G1_i135(x_i135, self._states_i135[0])
        self._states_i135[0] = x_i135

        x_i0 = self.head_i0(x_i0)
        x_i0 = self.G1_i0(x_i0, self._states_i0[0])
        self._states_i0[0] = x_i0

        x_i90, x_i45, x_i135, x_i0 = self.jr1(x_i90, x_i45, x_i135, x_i0)

        # second stage
        x_i90 = self.G2_i90(x_i90, self._states_i90[1])
        self._states_i90[1] = x_i90

        x_i45 = self.G2_i45(x_i45, self._states_i45[1])
        self._states_i45[1] = x_i45

        x_i135 = self.G2_i135(x_i135, self._states_i135[1])
        self._states_i135[1] = x_i135

        x_i0 = self.G2_i0(x_i0, self._states_i0[1])
        self._states_i0[1] = x_i0

        x_i90, x_i45, x_i135, x_i0 = self.jr2(x_i90, x_i45, x_i135, x_i0)

        # direction prediction
        i90 = self.pred_i90(x_i90)
        i45 = self.pred_i45(x_i45)
        i135 = self.pred_i135(x_i135)
        i0 = self.pred_i0(x_i0)

        # sigmoid
        i90 = self.sigmoid(i90)
        i45 = self.sigmoid(i45)
        i135 = self.sigmoid(i135)
        i0 = self.sigmoid(i0)

        # i
        i = (i90 + i0) / 2
        i = self.output_i(i)
        i = self.sigmoid(i)

        # ad0
        s0 = i0 + i90
        s1 = i0 - i90
        s2 = i45 - i135

        a0 = 0.5 * torch.arctan2(s2, s1)
        a0 = (a0 + 0.5 * math.pi) / math.pi

        d0 = torch.div(torch.sqrt(torch.square(s1) + torch.square(s2)), s0 + 1e-6)
        d0 = torch.clip(d0, 0, 1)

        # a1
        x_a = self.head_a(x)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a
        x_a = self.R1_a(x_a)
        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a
        x_a = self.R2_a(x_a)
        a1 = self.pred_a(x_a)

        # a
        a = a0 + a1
        a = self.output_a(a)
        a = self.sigmoid(a)

        # d1
        x_d = self.head_d(x)
        x_d = self.G1_d(x_d, self._states_d[0])
        self._states_d[0] = x_d
        x_d = self.R1_d(x_d)
        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d
        x_d = self.R2_d(x_d)
        d1 = self.pred_d(x_d)

        # d
        d = d0 + d1
        d = self.output_d(d)
        d = self.sigmoid(d)

        return {
            'i90': i90,
            'i45': i45,
            'i135': i135,
            'i0': i0,
            'i': i,
            'a': a,
            'd': d
        }


class V6(BaseModel):
    """
    Four FireNet branches for the four channels + ad branches (two gr blocks) for refinement
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, num_recurrent_units=2, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        self.num_recurrent_units = num_recurrent_units

        self.head_i90 = nn.Sequential(nn.Conv2d(self.num_bins, base_num_channels, 3, 1, 1), nn.ReLU())
        self.G1_i90 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i90 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_i90 = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.head_i45 = nn.Sequential(nn.Conv2d(self.num_bins, base_num_channels, 3, 1, 1), nn.ReLU())
        self.G1_i45 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i45 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_i45 = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.head_i135 = nn.Sequential(nn.Conv2d(self.num_bins, base_num_channels, 3, 1, 1), nn.ReLU())
        self.G1_i135 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i135 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_i135 = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.head_i0 = nn.Sequential(nn.Conv2d(self.num_bins, base_num_channels, 3, 1, 1), nn.ReLU())
        self.G1_i0 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i0 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_i0 = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.jr1 = JR(base_num_channels)
        self.jr2 = JR(base_num_channels)

        self.head_a = RPPP(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_a = ResidualBlock(base_num_channels)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_a = ResidualBlock(base_num_channels)
        self.pred_a = nn.Conv2d(base_num_channels, 1, 3, 1, 1)
        self.output_a = nn.Conv2d(1, 1, 5, 1, 2)

        self.head_d = RPPP(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_d = ResidualBlock(base_num_channels)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_d = ResidualBlock(base_num_channels)
        self.pred_d = nn.Conv2d(base_num_channels, 1, 3, 1, 1)
        self.output_d = nn.Conv2d(1, 1, 5, 1, 2)

        self.reset_states_i90()
        self.reset_states_i45()
        self.reset_states_i135()
        self.reset_states_i0()
        self.reset_states_a()
        self.reset_states_d()

        self.sigmoid = nn.Sigmoid()

    def reset_states_i90(self):
        self._states_i90 = [None] * self.num_recurrent_units

    def reset_states_i45(self):
        self._states_i45 = [None] * self.num_recurrent_units

    def reset_states_i135(self):
        self._states_i135 = [None] * self.num_recurrent_units

    def reset_states_i0(self):
        self._states_i0 = [None] * self.num_recurrent_units

    def reset_states_a(self):
        self._states_a = [None] * self.num_recurrent_units

    def reset_states_d(self):
        self._states_d = [None] * self.num_recurrent_units

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W event tensor
        :return: N x 1 x H/2 x W/2 i, a, d
        """
        # split input
        x_i90 = x[:, :, 0::2, 0::2]
        x_i45 = x[:, :, 0::2, 1::2]
        x_i135 = x[:, :, 1::2, 0::2]
        x_i0 = x[:, :, 1::2, 1::2]

        # first stage
        x_i90 = self.head_i90(x_i90)
        x_i90 = self.G1_i90(x_i90, self._states_i90[0])
        self._states_i90[0] = x_i90

        x_i45 = self.head_i45(x_i45)
        x_i45 = self.G1_i45(x_i45, self._states_i45[0])
        self._states_i45[0] = x_i45

        x_i135 = self.head_i135(x_i135)
        x_i135 = self.G1_i135(x_i135, self._states_i135[0])
        self._states_i135[0] = x_i135

        x_i0 = self.head_i0(x_i0)
        x_i0 = self.G1_i0(x_i0, self._states_i0[0])
        self._states_i0[0] = x_i0

        x_i90, x_i45, x_i135, x_i0 = self.jr1(x_i90, x_i45, x_i135, x_i0)

        # second stage
        x_i90 = self.G2_i90(x_i90, self._states_i90[1])
        self._states_i90[1] = x_i90

        x_i45 = self.G2_i45(x_i45, self._states_i45[1])
        self._states_i45[1] = x_i45

        x_i135 = self.G2_i135(x_i135, self._states_i135[1])
        self._states_i135[1] = x_i135

        x_i0 = self.G2_i0(x_i0, self._states_i0[1])
        self._states_i0[1] = x_i0

        x_i90, x_i45, x_i135, x_i0 = self.jr2(x_i90, x_i45, x_i135, x_i0)

        # direction prediction
        i90 = self.pred_i90(x_i90)
        i45 = self.pred_i45(x_i45)
        i135 = self.pred_i135(x_i135)
        i0 = self.pred_i0(x_i0)

        # i
        i = (i90 + i0) / 2

        # clip
        i90 = torch.clip(i90, 0, 1)
        i45 = torch.clip(i45, 0, 1)
        i135 = torch.clip(i135, 0, 1)
        i0 = torch.clip(i0, 0, 1)

        # ad0
        s0 = i0 + i90
        s1 = i0 - i90
        s2 = i45 - i135

        a0 = 0.5 * torch.arctan2(s2, s1)
        a0 = (a0 + 0.5 * math.pi) / math.pi

        d0 = torch.div(torch.sqrt(torch.square(s1) + torch.square(s2)), s0 + 1e-6)
        d0 = torch.clip(d0, 0, 1)

        # a1
        x_a = self.head_a(x)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a
        x_a = self.R1_a(x_a)
        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a
        x_a = self.R2_a(x_a)
        a1 = self.pred_a(x_a)

        # a
        a = a0 + a1
        a = self.output_a(a)

        # d1
        x_d = self.head_d(x)
        x_d = self.G1_d(x_d, self._states_d[0])
        self._states_d[0] = x_d
        x_d = self.R1_d(x_d)
        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d
        x_d = self.R2_d(x_d)
        d1 = self.pred_d(x_d)

        # d
        d = d0 + d1
        d = self.output_d(d)

        return {
            'i90': i90,
            'i45': i45,
            'i135': i135,
            'i0': i0,
            'i': i,
            'a': a,
            'd': d
        }


class V7(BaseModel):
    """
    Three branches for iad.
    """
    def __init__(self, num_bins=5, base_num_channels=32, kernel_size=3, num_recurrent_units=2, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        self.num_recurrent_units = num_recurrent_units

        self.head_i = RPPP(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_i = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.head_a = RPPP(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_a = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.head_d = RPPP(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_d = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.jr1 = JR_IAD(base_num_channels)
        self.jr2 = JR_IAD(base_num_channels)

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
        :param x: N x num_input_channels x H x W event tensor
        :return: N x 1 x H/2 x W/2 i, a, d
        """
        # first stage
        x_i = self.head_i(x)
        x_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_i

        x_a = self.head_a(x)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a

        x_d = self.head_d(x)
        x_d = self.G1_d(x_d, self._states_d[0])
        self._states_d[0] = x_d

        x_i, x_a, x_d = self.jr1(x_i, x_a, x_d)

        # second stage
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i

        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a

        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d

        x_i, x_a, x_d = self.jr2(x_i, x_a, x_d)

        # prediction
        i = self.pred_i(x_i)
        a = self.pred_a(x_a)
        d = self.pred_d(x_d)

        return {
            'i': i,
            'a': a,
            'd': d
        }


class V8(BaseModel):
    """
    Three branches for iad + firenet.
    """
    def __init__(self, num_bins=5, base_num_channels=32, kernel_size=3, num_recurrent_units=2, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        self.num_recurrent_units = num_recurrent_units

        self.head_i = RPPP(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_i = nn.Conv2d(base_num_channels * 2, 1, 3, 1, 1)

        self.head_a = RPPP(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_a = nn.Conv2d(base_num_channels * 2, 1, 3, 1, 1)

        self.head_d = RPPP(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_d = nn.Conv2d(base_num_channels * 2, 1, 3, 1, 1)

        self.jr1 = JR_IAD(base_num_channels)
        self.jr2 = JR_IAD(base_num_channels)

        self.reset_states_i()
        self.reset_states_a()
        self.reset_states_d()

        checkpoint = torch.load('./ckpt/firenet_1000.pth.tar')
        checkpoint = make_henri_compatible(checkpoint, '')
        checkpoint['config']['arch']['type'] = 'FireNet_legacy'

        self.firenet_i90 = checkpoint['config'].init_obj('arch', model_arch)
        self.firenet_i90.load_state_dict(checkpoint['state_dict'])
        self.firenet_i90.reset_states()
        print('load firenet weights succeed!')

        self.firenet_i45 = checkpoint['config'].init_obj('arch', model_arch)
        self.firenet_i45.load_state_dict(checkpoint['state_dict'])
        self.firenet_i45.reset_states()
        print('load firenet weights succeed!')

        self.firenet_i135 = checkpoint['config'].init_obj('arch', model_arch)
        self.firenet_i135.load_state_dict(checkpoint['state_dict'])
        self.firenet_i135.reset_states()
        print('load firenet weights succeed!')

        self.firenet_i0 = checkpoint['config'].init_obj('arch', model_arch)
        self.firenet_i0.load_state_dict(checkpoint['state_dict'])
        self.firenet_i0.reset_states()
        print('load firenet weights succeed!')

        self.conv_i_f = nn.Sequential(nn.Conv2d(1, base_num_channels, 3, 1, 1), nn.ReLU())
        self.conv_a_f = nn.Sequential(nn.Conv2d(1, base_num_channels, 3, 1, 1), nn.ReLU())
        self.conv_d_f = nn.Sequential(nn.Conv2d(1, base_num_channels, 3, 1, 1), nn.ReLU())

    def reset_states_i(self):
        self._states_i = [None] * self.num_recurrent_units

    def reset_states_a(self):
        self._states_a = [None] * self.num_recurrent_units

    def reset_states_d(self):
        self._states_d = [None] * self.num_recurrent_units

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W event tensor
        :return: N x 1 x H/2 x W/2 i, a, d
        """
        # first stage
        x_i = self.head_i(x)
        x_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_i

        x_a = self.head_a(x)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a

        x_d = self.head_d(x)
        x_d = self.G1_d(x_d, self._states_d[0])
        self._states_d[0] = x_d

        x_i, x_a, x_d = self.jr1(x_i, x_a, x_d)

        # second stage
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i

        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a

        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d

        x_i, x_a, x_d = self.jr2(x_i, x_a, x_d)

        # firenet
        x_i90 = x[:, :, 0::2, 0::2]
        x_i45 = x[:, :, 0::2, 1::2]
        x_i135 = x[:, :, 1::2, 0::2]
        x_i0 = x[:, :, 1::2, 1::2]

        i90 = torch.clip(self.firenet_i90(x_i90)['image'].detach(), 0, 1)
        i45 = torch.clip(self.firenet_i45(x_i45)['image'].detach(), 0, 1)
        i135 = torch.clip(self.firenet_i135(x_i135)['image'].detach(), 0, 1)
        i0 = torch.clip(self.firenet_i0(x_i0)['image'].detach(), 0, 1)

        s0 = i0 + i90
        s1 = i0 - i90
        s2 = i45 - i135

        i_f = s0 / 2

        a_f = 0.5 * torch.arctan2(s2, s1)
        a_f = (a_f + 0.5 * math.pi) / math.pi

        d_f = torch.div(torch.sqrt(torch.square(s1) + torch.square(s2)), s0 + 1e-6)
        d_f = torch.clip(d_f, 0, 1)

        # prediction
        i = self.pred_i(torch.cat((x_i, self.conv_i_f(i_f)), 1))
        a = self.pred_a(torch.cat((x_a, self.conv_a_f(a_f)), 1))
        d = self.pred_d(torch.cat((x_d, self.conv_d_f(d_f)), 1))

        return {
            # 'i': i,
            # 'a': a,
            # 'd': d
            'i': i_f,
            'a': a_f,
            'd': d_f
        }


class V9(BaseModel):
    """
    Three branches for iad + firenet.
    """
    def __init__(self, num_bins=5, base_num_channels=32, kernel_size=3, num_recurrent_units=2, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        self.num_recurrent_units = num_recurrent_units

        self.head_i = RPPP(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_i = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.head_a = RPPP(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_a = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.head_d = RPPP(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_d = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.jr1 = JR_IAD(base_num_channels)
        self.jr2 = JR_IAD(base_num_channels)

        self.reset_states_i()
        self.reset_states_a()
        self.reset_states_d()

        checkpoint = torch.load('./ckpt/firenet_1000.pth.tar')
        checkpoint = make_henri_compatible(checkpoint, '')
        checkpoint['config']['arch']['type'] = 'FireNet_legacy'

        self.firenet_i90 = checkpoint['config'].init_obj('arch', model_arch)
        self.firenet_i90.load_state_dict(checkpoint['state_dict'])
        self.firenet_i90.reset_states()
        print('load firenet weights succeed!')

        self.firenet_i45 = checkpoint['config'].init_obj('arch', model_arch)
        self.firenet_i45.load_state_dict(checkpoint['state_dict'])
        self.firenet_i45.reset_states()
        print('load firenet weights succeed!')

        self.firenet_i135 = checkpoint['config'].init_obj('arch', model_arch)
        self.firenet_i135.load_state_dict(checkpoint['state_dict'])
        self.firenet_i135.reset_states()
        print('load firenet weights succeed!')

        self.firenet_i0 = checkpoint['config'].init_obj('arch', model_arch)
        self.firenet_i0.load_state_dict(checkpoint['state_dict'])
        self.firenet_i0.reset_states()
        print('load firenet weights succeed!')

        self.conv_i_f = nn.Sequential(nn.Conv2d(1, base_num_channels, 3, 1, 1), nn.ReLU())
        self.conv_a_f = nn.Sequential(nn.Conv2d(1, base_num_channels, 3, 1, 1), nn.ReLU())
        self.conv_d_f = nn.Sequential(nn.Conv2d(1, base_num_channels, 3, 1, 1), nn.ReLU())

    def reset_states_i(self):
        self._states_i = [None] * self.num_recurrent_units

    def reset_states_a(self):
        self._states_a = [None] * self.num_recurrent_units

    def reset_states_d(self):
        self._states_d = [None] * self.num_recurrent_units

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W event tensor
        :return: N x 1 x H/2 x W/2 i, a, d
        """
        # first stage
        x_i = self.head_i(x)
        x_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_i

        x_a = self.head_a(x)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a

        x_d = self.head_d(x)
        x_d = self.G1_d(x_d, self._states_d[0])
        self._states_d[0] = x_d

        x_i, x_a, x_d = self.jr1(x_i, x_a, x_d)

        # firenet
        x_i90 = x[:, :, 0::2, 0::2]
        x_i45 = x[:, :, 0::2, 1::2]
        x_i135 = x[:, :, 1::2, 0::2]
        x_i0 = x[:, :, 1::2, 1::2]

        i90 = torch.clip(self.firenet_i90(x_i90)['image'].detach(), 0, 1)
        i45 = torch.clip(self.firenet_i45(x_i45)['image'].detach(), 0, 1)
        i135 = torch.clip(self.firenet_i135(x_i135)['image'].detach(), 0, 1)
        i0 = torch.clip(self.firenet_i0(x_i0)['image'].detach(), 0, 1)

        s0 = i0 + i90
        s1 = i0 - i90
        s2 = i45 - i135

        i_f = s0 / 2

        a_f = 0.5 * torch.arctan2(s2, s1)
        a_f = (a_f + 0.5 * math.pi) / math.pi

        d_f = torch.div(torch.sqrt(torch.square(s1) + torch.square(s2)), s0 + 1e-6)
        d_f = torch.clip(d_f, 0, 1)

        x_i = x_i + self.conv_i_f(i_f)
        x_a = x_a + self.conv_a_f(a_f)
        x_d = x_d + self.conv_d_f(d_f)

        # second stage
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i

        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a

        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d

        x_i, x_a, x_d = self.jr2(x_i, x_a, x_d)

        # prediction
        i = self.pred_i(x_i)
        a = self.pred_a(x_a)
        d = self.pred_d(x_d)

        return {
            'i': i,
            'a': a,
            'd': d
        }


class V10(BaseModel):
    """
    Three branches for iad + firenet.
    """
    def __init__(self, num_bins=5, base_num_channels=32, kernel_size=3, num_recurrent_units=2, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        self.num_recurrent_units = num_recurrent_units

        self.head_i = RPPP(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_i = nn.Conv2d(base_num_channels * 2, 1, 3, 1, 1)

        self.head_a = RPPP(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_a = nn.Conv2d(base_num_channels * 2, 1, 3, 1, 1)

        self.head_d = RPPP(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_d = nn.Conv2d(base_num_channels * 2, 1, 3, 1, 1)

        self.jr1 = JR_IAD(base_num_channels)
        self.jr2 = JR_IAD(base_num_channels)

        self.reset_states_i()
        self.reset_states_a()
        self.reset_states_d()

        checkpoint = torch.load('./ckpt/firenet_1000.pth.tar')
        checkpoint = make_henri_compatible(checkpoint, '')
        checkpoint['config']['arch']['type'] = 'FireNet_legacy'

        self.firenet_i90 = checkpoint['config'].init_obj('arch', model_arch)
        self.firenet_i90.load_state_dict(checkpoint['state_dict'])
        self.firenet_i90.eval()
        for param in self.firenet_i90.parameters():
            param.requires_grad = False
        self.firenet_i90.reset_states()
        print('load firenet_i90 weights succeed!')

        self.firenet_i45 = checkpoint['config'].init_obj('arch', model_arch)
        self.firenet_i45.load_state_dict(checkpoint['state_dict'])
        self.firenet_i45.eval()
        for param in self.firenet_i45.parameters():
            param.requires_grad = False
        self.firenet_i45.reset_states()
        print('load firenet_i45 weights succeed!')

        self.firenet_i135 = checkpoint['config'].init_obj('arch', model_arch)
        self.firenet_i135.load_state_dict(checkpoint['state_dict'])
        self.firenet_i135.eval()
        for param in self.firenet_i135.parameters():
            param.requires_grad = False
        self.firenet_i135.reset_states()
        print('load firenet_i135 weights succeed!')

        self.firenet_i0 = checkpoint['config'].init_obj('arch', model_arch)
        self.firenet_i0.load_state_dict(checkpoint['state_dict'])
        self.firenet_i0.eval()
        for param in self.firenet_i0.parameters():
            param.requires_grad = False
        self.firenet_i0.reset_states()
        print('load firenet_i0 weights succeed!')

        self.conv_i_f = nn.Sequential(nn.Conv2d(1, base_num_channels, 3, 1, 1), nn.ReLU())
        self.conv_a_f = nn.Sequential(nn.Conv2d(1, base_num_channels, 3, 1, 1), nn.ReLU())
        self.conv_d_f = nn.Sequential(nn.Conv2d(1, base_num_channels, 3, 1, 1), nn.ReLU())

    def reset_states_i(self):
        self._states_i = [None] * self.num_recurrent_units

    def reset_states_a(self):
        self._states_a = [None] * self.num_recurrent_units

    def reset_states_d(self):
        self._states_d = [None] * self.num_recurrent_units

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W event tensor
        :return: N x 1 x H/2 x W/2 i, a, d
        """
        # first stage
        x_i = self.head_i(x)
        x_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_i

        x_a = self.head_a(x)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a

        x_d = self.head_d(x)
        x_d = self.G1_d(x_d, self._states_d[0])
        self._states_d[0] = x_d

        x_i, x_a, x_d = self.jr1(x_i, x_a, x_d)

        # second stage
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i

        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a

        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d

        x_i, x_a, x_d = self.jr2(x_i, x_a, x_d)

        # firenet
        x_i90 = x[:, :, 0::2, 0::2]
        x_i45 = x[:, :, 0::2, 1::2]
        x_i135 = x[:, :, 1::2, 0::2]
        x_i0 = x[:, :, 1::2, 1::2]

        i90 = torch.clip(self.firenet_i90(x_i90)['image'].detach(), 0, 1)
        i45 = torch.clip(self.firenet_i45(x_i45)['image'].detach(), 0, 1)
        i135 = torch.clip(self.firenet_i135(x_i135)['image'].detach(), 0, 1)
        i0 = torch.clip(self.firenet_i0(x_i0)['image'].detach(), 0, 1)

        s0 = i0 + i90
        s1 = i0 - i90
        s2 = i45 - i135

        i_f = s0 / 2

        a_f = 0.5 * torch.arctan2(s2, s1)
        a_f = (a_f + 0.5 * math.pi) / math.pi

        d_f = torch.div(torch.sqrt(torch.square(s1) + torch.square(s2)), s0 + 1e-6)
        d_f = torch.clip(d_f, 0, 1)

        # prediction
        i = self.pred_i(torch.cat((x_i, self.conv_i_f(i_f)), 1))
        a = self.pred_a(torch.cat((x_a, self.conv_a_f(a_f)), 1))
        d = self.pred_d(torch.cat((x_d, self.conv_d_f(d_f)), 1))

        return {
            'i_f': i_f,
            'a_f': a_f,
            'd_f': d_f,
            'i': i,
            'a': a,
            'd': d
        }


class V11(BaseModel):
    """
    Three branches for iad + firenet, esa for features fusion.
    """
    def __init__(self, num_bins=5, base_num_channels=32, kernel_size=3, num_recurrent_units=2, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        self.num_recurrent_units = num_recurrent_units

        self.head_i = RPPP(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.fusion_weight_i = Fusion_Weight(base_num_channels)
        self.pred_i = nn.Conv2d(base_num_channels * 2, 1, 3, 1, 1)

        self.head_a = RPPP(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.fusion_weight_a = Fusion_Weight(base_num_channels)
        self.pred_a = nn.Conv2d(base_num_channels * 2, 1, 3, 1, 1)

        self.head_d = RPPP(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.fusion_weight_d = Fusion_Weight(base_num_channels)
        self.pred_d = nn.Conv2d(base_num_channels * 2, 1, 3, 1, 1)

        self.jr1 = JR_IAD(base_num_channels)
        self.jr2 = JR_IAD(base_num_channels)

        self.reset_states_i()
        self.reset_states_a()
        self.reset_states_d()

        checkpoint = torch.load('./ckpt/firenet_1000.pth.tar')
        checkpoint = make_henri_compatible(checkpoint, '')
        checkpoint['config']['arch']['type'] = 'FireNet_legacy'

        self.firenet_i90 = checkpoint['config'].init_obj('arch', model_arch)
        self.firenet_i90.load_state_dict(checkpoint['state_dict'])
        self.firenet_i90.eval()
        for param in self.firenet_i90.parameters():
            param.requires_grad = False
        self.firenet_i90.reset_states()
        print('load firenet_i90 weights succeed!')

        self.firenet_i45 = checkpoint['config'].init_obj('arch', model_arch)
        self.firenet_i45.load_state_dict(checkpoint['state_dict'])
        self.firenet_i45.eval()
        for param in self.firenet_i45.parameters():
            param.requires_grad = False
        self.firenet_i45.reset_states()
        print('load firenet_i45 weights succeed!')

        self.firenet_i135 = checkpoint['config'].init_obj('arch', model_arch)
        self.firenet_i135.load_state_dict(checkpoint['state_dict'])
        self.firenet_i135.eval()
        for param in self.firenet_i135.parameters():
            param.requires_grad = False
        self.firenet_i135.reset_states()
        print('load firenet_i135 weights succeed!')

        self.firenet_i0 = checkpoint['config'].init_obj('arch', model_arch)
        self.firenet_i0.load_state_dict(checkpoint['state_dict'])
        self.firenet_i0.eval()
        for param in self.firenet_i0.parameters():
            param.requires_grad = False
        self.firenet_i0.reset_states()
        print('load firenet_i0 weights succeed!')

        self.conv_i_f = nn.Sequential(nn.Conv2d(1, base_num_channels, 3, 1, 1), nn.ReLU(True))
        self.conv_a_f = nn.Sequential(nn.Conv2d(1, base_num_channels, 3, 1, 1), nn.ReLU(True))
        self.conv_d_f = nn.Sequential(nn.Conv2d(1, base_num_channels, 3, 1, 1), nn.ReLU(True))

    def reset_states_i(self):
        self._states_i = [None] * self.num_recurrent_units

    def reset_states_a(self):
        self._states_a = [None] * self.num_recurrent_units

    def reset_states_d(self):
        self._states_d = [None] * self.num_recurrent_units

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W event tensor
        :return: N x 1 x H/2 x W/2 i, a, d
        """
        # first stage
        x_i = self.head_i(x)
        x_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_i

        x_a = self.head_a(x)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a

        x_d = self.head_d(x)
        x_d = self.G1_d(x_d, self._states_d[0])
        self._states_d[0] = x_d

        x_i, x_a, x_d = self.jr1(x_i, x_a, x_d)

        # second stage
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i

        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a

        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d

        x_i, x_a, x_d = self.jr2(x_i, x_a, x_d)

        # firenet
        x_i90 = x[:, :, 0::2, 0::2]
        x_i45 = x[:, :, 0::2, 1::2]
        x_i135 = x[:, :, 1::2, 0::2]
        x_i0 = x[:, :, 1::2, 1::2]

        i90 = torch.clip(self.firenet_i90(x_i90)['image'].detach(), 0, 1)
        i45 = torch.clip(self.firenet_i45(x_i45)['image'].detach(), 0, 1)
        i135 = torch.clip(self.firenet_i135(x_i135)['image'].detach(), 0, 1)
        i0 = torch.clip(self.firenet_i0(x_i0)['image'].detach(), 0, 1)

        s0 = i0 + i90
        s1 = i0 - i90
        s2 = i45 - i135

        i_f = s0 / 2

        a_f = 0.5 * torch.arctan2(s2, s1)
        a_f = (a_f + 0.5 * math.pi) / math.pi

        d_f = torch.div(torch.sqrt(torch.square(s1) + torch.square(s2)), s0 + 1e-6)
        d_f = torch.clip(d_f, 0, 1)

        # prediction
        x_i_f = self.conv_i_f(i_f)
        x_i_c = torch.cat((x_i, x_i_f), 1)
        fusion_weight_i = self.fusion_weight_i(x_i_c)
        x_i_w = x_i_c * fusion_weight_i
        i = self.pred_i(x_i_w)

        x_a_f = self.conv_a_f(a_f)
        x_a_c = torch.cat((x_a, x_a_f), 1)
        fusion_weight_a = self.fusion_weight_a(x_a_c)
        x_a_w = x_a_c * fusion_weight_a
        a = self.pred_a(x_a_w)

        x_d_f = self.conv_d_f(d_f)
        x_d_c = torch.cat((x_d, x_d_f), 1)
        fusion_weight_d = self.fusion_weight_d(x_d_c)
        x_d_w = x_d_c * fusion_weight_d
        d = self.pred_d(x_d_w)

        return {
            'i_f': i_f,
            'a_f': a_f,
            'd_f': d_f,
            'i': i,
            'a': a,
            'd': d
        }


class V11_WOF(BaseModel):
    """
    Three branches for iad, no firenet branch.
    """
    def __init__(self, num_bins=5, base_num_channels=32, kernel_size=3, num_recurrent_units=2, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        self.num_recurrent_units = num_recurrent_units

        self.head_i = RPPP(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.fusion_weight_i = Fusion_Weight(int(base_num_channels / 2))
        self.pred_i = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.head_a = RPPP(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.fusion_weight_a = Fusion_Weight(int(base_num_channels / 2))
        self.pred_a = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.head_d = RPPP(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.fusion_weight_d = Fusion_Weight(int(base_num_channels / 2))
        self.pred_d = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.jr1 = JR_IAD(base_num_channels)
        self.jr2 = JR_IAD(base_num_channels)

        self.reset_states_i()
        self.reset_states_a()
        self.reset_states_d()

        # checkpoint = torch.load('./ckpt/firenet_1000.pth.tar')
        # checkpoint = make_henri_compatible(checkpoint, '')
        # checkpoint['config']['arch']['type'] = 'FireNet_legacy'
        #
        # self.firenet_i90 = checkpoint['config'].init_obj('arch', model_arch)
        # self.firenet_i90.load_state_dict(checkpoint['state_dict'])
        # self.firenet_i90.eval()
        # for param in self.firenet_i90.parameters():
        #     param.requires_grad = False
        # self.firenet_i90.reset_states()
        # print('load firenet_i90 weights succeed!')
        #
        # self.firenet_i45 = checkpoint['config'].init_obj('arch', model_arch)
        # self.firenet_i45.load_state_dict(checkpoint['state_dict'])
        # self.firenet_i45.eval()
        # for param in self.firenet_i45.parameters():
        #     param.requires_grad = False
        # self.firenet_i45.reset_states()
        # print('load firenet_i45 weights succeed!')
        #
        # self.firenet_i135 = checkpoint['config'].init_obj('arch', model_arch)
        # self.firenet_i135.load_state_dict(checkpoint['state_dict'])
        # self.firenet_i135.eval()
        # for param in self.firenet_i135.parameters():
        #     param.requires_grad = False
        # self.firenet_i135.reset_states()
        # print('load firenet_i135 weights succeed!')
        #
        # self.firenet_i0 = checkpoint['config'].init_obj('arch', model_arch)
        # self.firenet_i0.load_state_dict(checkpoint['state_dict'])
        # self.firenet_i0.eval()
        # for param in self.firenet_i0.parameters():
        #     param.requires_grad = False
        # self.firenet_i0.reset_states()
        # print('load firenet_i0 weights succeed!')
        #
        # self.conv_i_f = nn.Sequential(nn.Conv2d(1, base_num_channels, 3, 1, 1), nn.ReLU(True))
        # self.conv_a_f = nn.Sequential(nn.Conv2d(1, base_num_channels, 3, 1, 1), nn.ReLU(True))
        # self.conv_d_f = nn.Sequential(nn.Conv2d(1, base_num_channels, 3, 1, 1), nn.ReLU(True))

    def reset_states_i(self):
        self._states_i = [None] * self.num_recurrent_units

    def reset_states_a(self):
        self._states_a = [None] * self.num_recurrent_units

    def reset_states_d(self):
        self._states_d = [None] * self.num_recurrent_units

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W event tensor
        :return: N x 1 x H/2 x W/2 i, a, d
        """
        # first stage
        x_i = self.head_i(x)
        x_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_i

        x_a = self.head_a(x)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a

        x_d = self.head_d(x)
        x_d = self.G1_d(x_d, self._states_d[0])
        self._states_d[0] = x_d

        x_i, x_a, x_d = self.jr1(x_i, x_a, x_d)

        # second stage
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i

        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a

        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d

        x_i, x_a, x_d = self.jr2(x_i, x_a, x_d)

        # firenet
        # x_i90 = x[:, :, 0::2, 0::2]
        # x_i45 = x[:, :, 0::2, 1::2]
        # x_i135 = x[:, :, 1::2, 0::2]
        # x_i0 = x[:, :, 1::2, 1::2]
        #
        # i90 = torch.clip(self.firenet_i90(x_i90)['image'].detach(), 0, 1)
        # i45 = torch.clip(self.firenet_i45(x_i45)['image'].detach(), 0, 1)
        # i135 = torch.clip(self.firenet_i135(x_i135)['image'].detach(), 0, 1)
        # i0 = torch.clip(self.firenet_i0(x_i0)['image'].detach(), 0, 1)
        #
        # s0 = i0 + i90
        # s1 = i0 - i90
        # s2 = i45 - i135
        #
        # i_f = s0 / 2
        #
        # a_f = 0.5 * torch.arctan2(s2, s1)
        # a_f = (a_f + 0.5 * math.pi) / math.pi
        #
        # d_f = torch.div(torch.sqrt(torch.square(s1) + torch.square(s2)), s0 + 1e-6)
        # d_f = torch.clip(d_f, 0, 1)

        # prediction
        # x_i_f = self.conv_i_f(i_f)
        # x_i_c = torch.cat((x_i, x_i_f), 1)
        x_i_c = x_i
        fusion_weight_i = self.fusion_weight_i(x_i_c)
        x_i_w = x_i_c * fusion_weight_i
        i = self.pred_i(x_i_w)

        # x_a_f = self.conv_a_f(a_f)
        # x_a_c = torch.cat((x_a, x_a_f), 1)
        x_a_c = x_a
        fusion_weight_a = self.fusion_weight_a(x_a_c)
        x_a_w = x_a_c * fusion_weight_a
        a = self.pred_a(x_a_w)

        # x_d_f = self.conv_d_f(d_f)
        # x_d_c = torch.cat((x_d, x_d_f), 1)
        x_d_c = x_d
        fusion_weight_d = self.fusion_weight_d(x_d_c)
        x_d_w = x_d_c * fusion_weight_d
        d = self.pred_d(x_d_w)

        return {
            # 'i_f': i_f,
            # 'a_f': a_f,
            # 'd_f': d_f,
            'i': i,
            'a': a,
            'd': d
        }


class V11_E(BaseModel):
    """
    Three branches for iad, no firenet branch.
    """
    def __init__(self, num_bins=5, base_num_channels=32, kernel_size=3, num_recurrent_units=2, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        self.num_recurrent_units = num_recurrent_units

        self.head_i = RPPP(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.fusion_weight1_i = Fusion_Weight(int(base_num_channels / 2))
        self.fusion_weight2_i = Fusion_Weight(int(base_num_channels / 2))
        self.pred_i = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.head_a = RPPP(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.fusion_weight1_a = Fusion_Weight(int(base_num_channels / 2))
        self.fusion_weight2_a = Fusion_Weight(int(base_num_channels / 2))
        self.pred_a = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.head_d = RPPP(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.fusion_weight1_d = Fusion_Weight(int(base_num_channels / 2))
        self.fusion_weight2_d = Fusion_Weight(int(base_num_channels / 2))
        self.pred_d = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.jr1 = JR_IAD(base_num_channels)
        self.jr2 = JR_IAD(base_num_channels)

        self.reset_states_i()
        self.reset_states_a()
        self.reset_states_d()

        # checkpoint = torch.load('./ckpt/firenet_1000.pth.tar')
        # checkpoint = make_henri_compatible(checkpoint, '')
        # checkpoint['config']['arch']['type'] = 'FireNet_legacy'
        #
        # self.firenet_i90 = checkpoint['config'].init_obj('arch', model_arch)
        # self.firenet_i90.load_state_dict(checkpoint['state_dict'])
        # self.firenet_i90.eval()
        # for param in self.firenet_i90.parameters():
        #     param.requires_grad = False
        # self.firenet_i90.reset_states()
        # print('load firenet_i90 weights succeed!')
        #
        # self.firenet_i45 = checkpoint['config'].init_obj('arch', model_arch)
        # self.firenet_i45.load_state_dict(checkpoint['state_dict'])
        # self.firenet_i45.eval()
        # for param in self.firenet_i45.parameters():
        #     param.requires_grad = False
        # self.firenet_i45.reset_states()
        # print('load firenet_i45 weights succeed!')
        #
        # self.firenet_i135 = checkpoint['config'].init_obj('arch', model_arch)
        # self.firenet_i135.load_state_dict(checkpoint['state_dict'])
        # self.firenet_i135.eval()
        # for param in self.firenet_i135.parameters():
        #     param.requires_grad = False
        # self.firenet_i135.reset_states()
        # print('load firenet_i135 weights succeed!')
        #
        # self.firenet_i0 = checkpoint['config'].init_obj('arch', model_arch)
        # self.firenet_i0.load_state_dict(checkpoint['state_dict'])
        # self.firenet_i0.eval()
        # for param in self.firenet_i0.parameters():
        #     param.requires_grad = False
        # self.firenet_i0.reset_states()
        # print('load firenet_i0 weights succeed!')
        #
        # self.conv_i_f = nn.Sequential(nn.Conv2d(1, base_num_channels, 3, 1, 1), nn.ReLU(True))
        # self.conv_a_f = nn.Sequential(nn.Conv2d(1, base_num_channels, 3, 1, 1), nn.ReLU(True))
        # self.conv_d_f = nn.Sequential(nn.Conv2d(1, base_num_channels, 3, 1, 1), nn.ReLU(True))

    def reset_states_i(self):
        self._states_i = [None] * self.num_recurrent_units

    def reset_states_a(self):
        self._states_a = [None] * self.num_recurrent_units

    def reset_states_d(self):
        self._states_d = [None] * self.num_recurrent_units

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W event tensor
        :return: N x 1 x H/2 x W/2 i, a, d
        """
        # first stage
        x_i = self.head_i(x)
        x_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_i

        x_a = self.head_a(x)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a

        x_d = self.head_d(x)
        x_d = self.G1_d(x_d, self._states_d[0])
        self._states_d[0] = x_d

        x_i, x_a, x_d = self.jr1(x_i, x_a, x_d)

        # e1
        x_i_c = x_i
        fusion_weight_i = self.fusion_weight1_i(x_i_c)
        x_i_w = x_i_c * fusion_weight_i

        x_a_c = x_a
        fusion_weight_a = self.fusion_weight1_a(x_a_c)
        x_a_w = x_a_c * fusion_weight_a

        x_d_c = x_d
        fusion_weight_d = self.fusion_weight1_d(x_d_c)
        x_d_w = x_d_c * fusion_weight_d

        # second stage
        x_i = self.G2_i(x_i_w, self._states_i[1])
        self._states_i[1] = x_i

        x_a = self.G2_a(x_a_w, self._states_a[1])
        self._states_a[1] = x_a

        x_d = self.G2_d(x_d_w, self._states_d[1])
        self._states_d[1] = x_d

        x_i, x_a, x_d = self.jr2(x_i, x_a, x_d)

        # e2
        x_i_c = x_i
        fusion_weight_i = self.fusion_weight2_i(x_i_c)
        x_i_w = x_i_c * fusion_weight_i
        i = self.pred_i(x_i_w)

        x_a_c = x_a
        fusion_weight_a = self.fusion_weight2_a(x_a_c)
        x_a_w = x_a_c * fusion_weight_a
        a = self.pred_a(x_a_w)

        x_d_c = x_d
        fusion_weight_d = self.fusion_weight2_d(x_d_c)
        x_d_w = x_d_c * fusion_weight_d
        d = self.pred_d(x_d_w)

        return {
            'i': i,
            'a': a,
            'd': d
        }


class V11_WOE(BaseModel):
    """
    Three branches for iad, no firenet branch.
    """
    def __init__(self, num_bins=5, base_num_channels=32, kernel_size=3, num_recurrent_units=2, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        self.num_recurrent_units = num_recurrent_units

        self.head_i = RPPP(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        # self.fusion_weight1_i = Fusion_Weight(int(base_num_channels / 2))
        # self.fusion_weight2_i = Fusion_Weight(int(base_num_channels / 2))
        self.pred_i = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.head_a = RPPP(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        # self.fusion_weight1_a = Fusion_Weight(int(base_num_channels / 2))
        # self.fusion_weight2_a = Fusion_Weight(int(base_num_channels / 2))
        self.pred_a = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.head_d = RPPP(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        # self.fusion_weight1_d = Fusion_Weight(int(base_num_channels / 2))
        # self.fusion_weight2_d = Fusion_Weight(int(base_num_channels / 2))
        self.pred_d = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.jr1 = JR_IAD(base_num_channels)
        self.jr2 = JR_IAD(base_num_channels)

        self.reset_states_i()
        self.reset_states_a()
        self.reset_states_d()

        # checkpoint = torch.load('./ckpt/firenet_1000.pth.tar')
        # checkpoint = make_henri_compatible(checkpoint, '')
        # checkpoint['config']['arch']['type'] = 'FireNet_legacy'
        #
        # self.firenet_i90 = checkpoint['config'].init_obj('arch', model_arch)
        # self.firenet_i90.load_state_dict(checkpoint['state_dict'])
        # self.firenet_i90.eval()
        # for param in self.firenet_i90.parameters():
        #     param.requires_grad = False
        # self.firenet_i90.reset_states()
        # print('load firenet_i90 weights succeed!')
        #
        # self.firenet_i45 = checkpoint['config'].init_obj('arch', model_arch)
        # self.firenet_i45.load_state_dict(checkpoint['state_dict'])
        # self.firenet_i45.eval()
        # for param in self.firenet_i45.parameters():
        #     param.requires_grad = False
        # self.firenet_i45.reset_states()
        # print('load firenet_i45 weights succeed!')
        #
        # self.firenet_i135 = checkpoint['config'].init_obj('arch', model_arch)
        # self.firenet_i135.load_state_dict(checkpoint['state_dict'])
        # self.firenet_i135.eval()
        # for param in self.firenet_i135.parameters():
        #     param.requires_grad = False
        # self.firenet_i135.reset_states()
        # print('load firenet_i135 weights succeed!')
        #
        # self.firenet_i0 = checkpoint['config'].init_obj('arch', model_arch)
        # self.firenet_i0.load_state_dict(checkpoint['state_dict'])
        # self.firenet_i0.eval()
        # for param in self.firenet_i0.parameters():
        #     param.requires_grad = False
        # self.firenet_i0.reset_states()
        # print('load firenet_i0 weights succeed!')
        #
        # self.conv_i_f = nn.Sequential(nn.Conv2d(1, base_num_channels, 3, 1, 1), nn.ReLU(True))
        # self.conv_a_f = nn.Sequential(nn.Conv2d(1, base_num_channels, 3, 1, 1), nn.ReLU(True))
        # self.conv_d_f = nn.Sequential(nn.Conv2d(1, base_num_channels, 3, 1, 1), nn.ReLU(True))

    def reset_states_i(self):
        self._states_i = [None] * self.num_recurrent_units

    def reset_states_a(self):
        self._states_a = [None] * self.num_recurrent_units

    def reset_states_d(self):
        self._states_d = [None] * self.num_recurrent_units

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W event tensor
        :return: N x 1 x H/2 x W/2 i, a, d
        """
        # first stage
        x_i = self.head_i(x)
        x_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_i

        x_a = self.head_a(x)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a

        x_d = self.head_d(x)
        x_d = self.G1_d(x_d, self._states_d[0])
        self._states_d[0] = x_d

        x_i, x_a, x_d = self.jr1(x_i, x_a, x_d)

        # e1
        # x_i_c = x_i
        # fusion_weight_i = self.fusion_weight1_i(x_i_c)
        # x_i_w = x_i_c * fusion_weight_i
        #
        # x_a_c = x_a
        # fusion_weight_a = self.fusion_weight1_a(x_a_c)
        # x_a_w = x_a_c * fusion_weight_a
        #
        # x_d_c = x_d
        # fusion_weight_d = self.fusion_weight1_d(x_d_c)
        # x_d_w = x_d_c * fusion_weight_d

        # second stage
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i

        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a

        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d

        x_i, x_a, x_d = self.jr2(x_i, x_a, x_d)

        # e2
        # x_i_c = x_i
        # fusion_weight_i = self.fusion_weight2_i(x_i_c)
        # x_i_w = x_i_c * fusion_weight_i
        i = self.pred_i(x_i)

        # x_a_c = x_a
        # fusion_weight_a = self.fusion_weight2_a(x_a_c)
        # x_a_w = x_a_c * fusion_weight_a
        a = self.pred_a(x_a)

        # x_d_c = x_d
        # fusion_weight_d = self.fusion_weight2_d(x_d_c)
        # x_d_w = x_d_c * fusion_weight_d
        d = self.pred_d(x_d)

        return {
            'i': i,
            'a': a,
            'd': d
        }


class V12(BaseModel):
    """
    Three branches for iad + one trainable firenet.
    """
    def __init__(self, num_bins=5, base_num_channels=32, kernel_size=3, num_recurrent_units=2, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        self.num_recurrent_units = num_recurrent_units

        self.head_i = RPPP(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.fusion_weight_i = Fusion_Weight(base_num_channels)
        self.pred_i = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.head_a = RPPP(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.fusion_weight_a = Fusion_Weight(base_num_channels)
        self.pred_a = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.head_d = RPPP(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.fusion_weight_d = Fusion_Weight(base_num_channels)
        self.pred_d = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.jr1 = JR_IAD(base_num_channels)
        self.jr2 = JR_IAD(base_num_channels)

        self.reset_states_i()
        self.reset_states_a()
        self.reset_states_d()

        self.head_i_shared = nn.Sequential(nn.Conv2d(self.num_bins, base_num_channels, 3, 1, 1), nn.ReLU())
        self.G1_i_shared = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_i_shared = ResidualBlock(base_num_channels)
        self.G2_i_shared = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_i_shared = ResidualBlock(base_num_channels)
        self.pred_i_shared = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.reset_states_i_shared()

        self.sigmoid = nn.Sigmoid()

    def reset_states_i(self):
        self._states_i = [None] * self.num_recurrent_units

    def reset_states_a(self):
        self._states_a = [None] * self.num_recurrent_units

    def reset_states_d(self):
        self._states_d = [None] * self.num_recurrent_units

    def reset_states_i_shared(self):
        self._states_i_shared = [None] * self.num_recurrent_units * 4

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W event tensor
        :return: N x 1 x H/2 x W/2 i, a, d
        """
        # first stage
        x_i = self.head_i(x)
        x_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_i

        x_a = self.head_a(x)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a

        x_d = self.head_d(x)
        x_d = self.G1_d(x_d, self._states_d[0])
        self._states_d[0] = x_d

        x_i, x_a, x_d = self.jr1(x_i, x_a, x_d)

        # second stage
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i

        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a

        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d

        x_i, x_a, x_d = self.jr2(x_i, x_a, x_d)

        i = self.pred_i(x_i)
        a = self.pred_a(x_a)
        d = self.pred_d(x_d)

        # i = torch.clip(i, 0, 1)
        # a = torch.clip(a, 0, 1)
        # d = torch.clip(d, 0, 1)
        i = self.sigmoid(i)
        a = self.sigmoid(a)
        d = self.sigmoid(d)

        # firenet
        x_i90 = x[:, :, 0::2, 0::2]
        x_i45 = x[:, :, 0::2, 1::2]
        x_i135 = x[:, :, 1::2, 0::2]
        x_i0 = x[:, :, 1::2, 1::2]

        x_i90 = self.head_i_shared(x_i90)
        x_i90 = self.G1_i_shared(x_i90, self._states_i_shared[0])
        self._states_i_shared[0] = x_i90
        x_i90 = self.R1_i_shared(x_i90)
        x_i90 = self.G2_i_shared(x_i90, self._states_i_shared[1])
        self._states_i_shared[1] = x_i90
        x_i90 = self.R2_i_shared(x_i90)
        i90 = self.pred_i_shared(x_i90)

        x_i45 = self.head_i_shared(x_i45)
        x_i45 = self.G1_i_shared(x_i45, self._states_i_shared[2])
        self._states_i_shared[2] = x_i45
        x_i45 = self.R1_i_shared(x_i45)
        x_i45 = self.G2_i_shared(x_i45, self._states_i_shared[3])
        self._states_i_shared[3] = x_i45
        x_i45 = self.R2_i_shared(x_i45)
        i45 = self.pred_i_shared(x_i45)

        x_i135 = self.head_i_shared(x_i135)
        x_i135 = self.G1_i_shared(x_i135, self._states_i_shared[4])
        self._states_i_shared[4] = x_i135
        x_i135 = self.R1_i_shared(x_i135)
        x_i135 = self.G2_i_shared(x_i135, self._states_i_shared[5])
        self._states_i_shared[5] = x_i135
        x_i135 = self.R2_i_shared(x_i135)
        i135 = self.pred_i_shared(x_i135)

        x_i0 = self.head_i_shared(x_i0)
        x_i0 = self.G1_i_shared(x_i0, self._states_i_shared[6])
        self._states_i_shared[6] = x_i0
        x_i0 = self.R1_i_shared(x_i0)
        x_i0 = self.G2_i_shared(x_i0, self._states_i_shared[7])
        self._states_i_shared[7] = x_i0
        x_i0 = self.R2_i_shared(x_i0)
        i0 = self.pred_i_shared(x_i0)

        # i90 = torch.clip(i90, 0, 1)
        # i45 = torch.clip(i45, 0, 1)
        # i135 = torch.clip(i135, 0, 1)
        # i0 = torch.clip(i0, 0, 1)
        i90 = self.sigmoid(i90)
        i45 = self.sigmoid(i45)
        i135 = self.sigmoid(i135)
        i0 = self.sigmoid(i0)

        s0 = i0 + i90
        s1 = i0 - i90
        s2 = i45 - i135

        i_f = s0 / 2

        a_f = 0.5 * torch.arctan2(s2, s1)
        a_f = (a_f + 0.5 * math.pi) / math.pi

        d_f = torch.div(torch.sqrt(torch.square(s1) + torch.square(s2)), s0 + 1e-6)
        d_f = torch.clip(d_f, 0, 1)

        i_mean = (i + i_f) / 2
        a_mean = (a + a_f) / 2
        d_mean = (d + d_f) / 2

        return {
            'i90': i90,
            'i45': i45,
            'i135': i135,
            'i0': i0,
            'i': i_mean,
            'a': a_mean,
            'd': d_mean
        }


class PE2VNet(BaseModel):
    """
    Polarization Events to Video Network
    """
    def __init__(self, num_bins=5, base_num_channels=32, kernel_size=3, num_recurrent_units=2, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        self.num_recurrent_units = num_recurrent_units

        self.head_i = RPPP(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.fusion_weight_i = Fusion_Weight(base_num_channels)
        self.pred_i = nn.Conv2d(base_num_channels * 2, 1, 3, 1, 1)

        self.head_a = RPPP(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.fusion_weight_a = Fusion_Weight(base_num_channels)
        self.pred_a = nn.Conv2d(base_num_channels * 2, 1, 3, 1, 1)

        self.head_d = RPPP(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.fusion_weight_d = Fusion_Weight(base_num_channels)
        self.pred_d = nn.Conv2d(base_num_channels * 2, 1, 3, 1, 1)

        self.jr1 = JR_IAD(base_num_channels)
        self.jr2 = JR_IAD(base_num_channels)

        self.reset_states_i()
        self.reset_states_a()
        self.reset_states_d()

        checkpoint = torch.load('./ckpt/firenet_1000.pth.tar')
        checkpoint = make_henri_compatible(checkpoint, '')
        checkpoint['config']['arch']['type'] = 'FireNet_legacy'

        self.firenet_i90 = checkpoint['config'].init_obj('arch', model_arch)
        self.firenet_i90.load_state_dict(checkpoint['state_dict'])
        self.firenet_i90.eval()
        for param in self.firenet_i90.parameters():
            param.requires_grad = False
        self.firenet_i90.reset_states()
        print('load firenet_i90 weights succeed!')

        self.firenet_i45 = checkpoint['config'].init_obj('arch', model_arch)
        self.firenet_i45.load_state_dict(checkpoint['state_dict'])
        self.firenet_i45.eval()
        for param in self.firenet_i45.parameters():
            param.requires_grad = False
        self.firenet_i45.reset_states()
        print('load firenet_i45 weights succeed!')

        self.firenet_i135 = checkpoint['config'].init_obj('arch', model_arch)
        self.firenet_i135.load_state_dict(checkpoint['state_dict'])
        self.firenet_i135.eval()
        for param in self.firenet_i135.parameters():
            param.requires_grad = False
        self.firenet_i135.reset_states()
        print('load firenet_i135 weights succeed!')

        self.firenet_i0 = checkpoint['config'].init_obj('arch', model_arch)
        self.firenet_i0.load_state_dict(checkpoint['state_dict'])
        self.firenet_i0.eval()
        for param in self.firenet_i0.parameters():
            param.requires_grad = False
        self.firenet_i0.reset_states()
        print('load firenet_i0 weights succeed!')

        self.conv_i_f = nn.Sequential(nn.Conv2d(1, base_num_channels, 3, 1, 1), nn.ReLU(True))
        self.conv_a_f = nn.Sequential(nn.Conv2d(1, base_num_channels, 3, 1, 1), nn.ReLU(True))
        self.conv_d_f = nn.Sequential(nn.Conv2d(1, base_num_channels, 3, 1, 1), nn.ReLU(True))

    def reset_states_i(self):
        self._states_i = [None] * self.num_recurrent_units

    def reset_states_a(self):
        self._states_a = [None] * self.num_recurrent_units

    def reset_states_d(self):
        self._states_d = [None] * self.num_recurrent_units

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W event tensor
        :return: N x 1 x H/2 x W/2 i, a, d
        """
        # first stage
        x_i = self.head_i(x)
        x_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_i

        x_a = self.head_a(x)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a

        x_d = self.head_d(x)
        x_d = self.G1_d(x_d, self._states_d[0])
        self._states_d[0] = x_d

        x_i, x_a, x_d = self.jr1(x_i, x_a, x_d)

        # second stage
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i

        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a

        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d

        x_i, x_a, x_d = self.jr2(x_i, x_a, x_d)

        # firenet
        x_i90 = x[:, :, 0::2, 0::2]
        x_i45 = x[:, :, 0::2, 1::2]
        x_i135 = x[:, :, 1::2, 0::2]
        x_i0 = x[:, :, 1::2, 1::2]

        i90 = torch.clip(self.firenet_i90(x_i90)['image'].detach(), 0, 1)
        i45 = torch.clip(self.firenet_i45(x_i45)['image'].detach(), 0, 1)
        i135 = torch.clip(self.firenet_i135(x_i135)['image'].detach(), 0, 1)
        i0 = torch.clip(self.firenet_i0(x_i0)['image'].detach(), 0, 1)

        s0 = i0 + i90
        s1 = i0 - i90
        s2 = i45 - i135

        i_f = s0 / 2

        a_f = 0.5 * torch.arctan2(s2, s1)
        a_f = (a_f + 0.5 * math.pi) / math.pi

        d_f = torch.div(torch.sqrt(torch.square(s1) + torch.square(s2)), s0 + 1e-6)
        d_f = torch.clip(d_f, 0, 1)

        # prediction
        x_i_f = self.conv_i_f(i_f)
        x_i_c = torch.cat((x_i, x_i_f), 1)
        fusion_weight_i = self.fusion_weight_i(x_i_c)
        x_i_w = x_i_c * fusion_weight_i
        i = self.pred_i(x_i_w)

        x_a_f = self.conv_a_f(a_f)
        x_a_c = torch.cat((x_a, x_a_f), 1)
        fusion_weight_a = self.fusion_weight_a(x_a_c)
        x_a_w = x_a_c * fusion_weight_a
        a = self.pred_a(x_a_w)

        x_d_f = self.conv_d_f(d_f)
        x_d_c = torch.cat((x_d, x_d_f), 1)
        fusion_weight_d = self.fusion_weight_d(x_d_c)
        x_d_w = x_d_c * fusion_weight_d
        d = self.pred_d(x_d_w)

        return {
            'i_f': i_f,
            'a_f': a_f,
            'd_f': d_f,
            'i': i,
            'a': a,
            'd': d
        }


class V13(BaseModel):
    """
    Polarization Events to Video Network
    early fusion
    """
    def __init__(self, num_bins=5, base_num_channels=32, kernel_size=3, num_recurrent_units=1, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        self.num_recurrent_units = num_recurrent_units

        self.head_i = RPPP(self.num_bins, base_num_channels)
        self.fusion_weight_i = Fusion_Weight(base_num_channels)
        self.G_i = ConvGRU(base_num_channels * 2, base_num_channels * 2, kernel_size)
        self.pred_i = nn.Conv2d(base_num_channels * 2, 1, 1, 1, 0)

        self.head_a = RPPP(self.num_bins, base_num_channels)
        self.fusion_weight_a = Fusion_Weight(base_num_channels)
        self.G_a = ConvGRU(base_num_channels * 2, base_num_channels * 2, kernel_size)
        self.pred_a = nn.Conv2d(base_num_channels * 2, 1, 1, 1, 0)

        self.head_d = RPPP(self.num_bins, base_num_channels)
        self.fusion_weight_d = Fusion_Weight(base_num_channels)
        self.G_d = ConvGRU(base_num_channels * 2, base_num_channels * 2, kernel_size)
        self.pred_d = nn.Conv2d(base_num_channels * 2, 1, 1, 1, 0)

        self.jr = JR_IAD(base_num_channels * 2)

        self.reset_states_i()
        self.reset_states_a()
        self.reset_states_d()

        checkpoint = torch.load('./ckpt/firenet_1000.pth.tar')
        checkpoint = make_henri_compatible(checkpoint, '')
        checkpoint['config']['arch']['type'] = 'FireNet_legacy'

        self.firenet_i90 = checkpoint['config'].init_obj('arch', model_arch)
        self.firenet_i90.load_state_dict(checkpoint['state_dict'])
        self.firenet_i90.eval()
        for param in self.firenet_i90.parameters():
            param.requires_grad = False
        self.firenet_i90.reset_states()
        print('load firenet_i90 weights succeed!')

        self.firenet_i45 = checkpoint['config'].init_obj('arch', model_arch)
        self.firenet_i45.load_state_dict(checkpoint['state_dict'])
        self.firenet_i45.eval()
        for param in self.firenet_i45.parameters():
            param.requires_grad = False
        self.firenet_i45.reset_states()
        print('load firenet_i45 weights succeed!')

        self.firenet_i135 = checkpoint['config'].init_obj('arch', model_arch)
        self.firenet_i135.load_state_dict(checkpoint['state_dict'])
        self.firenet_i135.eval()
        for param in self.firenet_i135.parameters():
            param.requires_grad = False
        self.firenet_i135.reset_states()
        print('load firenet_i135 weights succeed!')

        self.firenet_i0 = checkpoint['config'].init_obj('arch', model_arch)
        self.firenet_i0.load_state_dict(checkpoint['state_dict'])
        self.firenet_i0.eval()
        for param in self.firenet_i0.parameters():
            param.requires_grad = False
        self.firenet_i0.reset_states()
        print('load firenet_i0 weights succeed!')

        self.conv_i_f = nn.Sequential(nn.Conv2d(1, base_num_channels, 3, 1, 1), nn.ReLU(True))
        self.conv_a_f = nn.Sequential(nn.Conv2d(1, base_num_channels, 3, 1, 1), nn.ReLU(True))
        self.conv_d_f = nn.Sequential(nn.Conv2d(1, base_num_channels, 3, 1, 1), nn.ReLU(True))

    def reset_states_i(self):
        self._states_i = [None] * self.num_recurrent_units

    def reset_states_a(self):
        self._states_a = [None] * self.num_recurrent_units

    def reset_states_d(self):
        self._states_d = [None] * self.num_recurrent_units

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W event tensor
        :return: N x 1 x H/2 x W/2 i, a, d
        """
        # head
        x_i = self.head_i(x)
        x_a = self.head_a(x)
        x_d = self.head_d(x)

        # firenet
        x_i90 = x[:, :, 0::2, 0::2]
        x_i45 = x[:, :, 0::2, 1::2]
        x_i135 = x[:, :, 1::2, 0::2]
        x_i0 = x[:, :, 1::2, 1::2]

        i90 = torch.clip(self.firenet_i90(x_i90)['image'].detach(), 0, 1)
        i45 = torch.clip(self.firenet_i45(x_i45)['image'].detach(), 0, 1)
        i135 = torch.clip(self.firenet_i135(x_i135)['image'].detach(), 0, 1)
        i0 = torch.clip(self.firenet_i0(x_i0)['image'].detach(), 0, 1)

        s0 = i0 + i90
        s1 = i0 - i90
        s2 = i45 - i135

        i_f = s0 / 2
        x_i_f = self.conv_i_f(i_f)

        a_f = 0.5 * torch.arctan2(s2, s1)
        a_f = (a_f + 0.5 * math.pi) / math.pi
        x_a_f = self.conv_a_f(a_f)

        d_f = torch.div(torch.sqrt(torch.square(s1) + torch.square(s2)), s0 + 1e-6)
        d_f = torch.clip(d_f, 0, 1)
        x_d_f = self.conv_d_f(d_f)

        # fusion
        x_i_c = torch.cat((x_i, x_i_f), 1)
        fusion_weight_i = self.fusion_weight_i(x_i_c)
        x_i_w = x_i_c * fusion_weight_i

        x_a_c = torch.cat((x_a, x_a_f), 1)
        fusion_weight_a = self.fusion_weight_a(x_a_c)
        x_a_w = x_a_c * fusion_weight_a

        x_d_c = torch.cat((x_d, x_d_f), 1)
        fusion_weight_d = self.fusion_weight_d(x_d_c)
        x_d_w = x_d_c * fusion_weight_d

        # temporal and spatial
        x_i = self.G_i(x_i_w, self._states_i[0])
        self._states_i[0] = x_i

        x_a = self.G_a(x_a_w, self._states_a[0])
        self._states_a[0] = x_a

        x_d = self.G_d(x_d_w, self._states_d[0])
        self._states_d[0] = x_d

        x_i, x_a, x_d = self.jr(x_i, x_a, x_d)

        # prediction
        i = self.pred_i(x_i)
        a = self.pred_a(x_a)
        d = self.pred_d(x_d)

        return {
            'i_f': i_f,
            'a_f': a_f,
            'd_f': d_f,
            'i': i,
            'a': a,
            'd': d
        }


class V14(BaseModel):
    """
    Polarization Events to Video Network
    three naive firenet branches
    """
    def __init__(self, num_bins=5, base_num_channels=32, kernel_size=3, num_recurrent_units=2, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        self.num_recurrent_units = num_recurrent_units

        self.head_i = nn.Conv2d(self.num_bins, base_num_channels, 2, 2, 0)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_i = ResidualBlock(base_num_channels)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_i = ResidualBlock(base_num_channels)
        self.pred_i = nn.Conv2d(base_num_channels, 1, 1, 1, 0)

        self.head_a = nn.Conv2d(self.num_bins, base_num_channels, 2, 2, 0)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_a = ResidualBlock(base_num_channels)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_a = ResidualBlock(base_num_channels)
        self.pred_a = nn.Conv2d(base_num_channels, 1, 1, 1, 0)

        self.head_d = nn.Conv2d(self.num_bins, base_num_channels, 2, 2, 0)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_d = ResidualBlock(base_num_channels)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_d = ResidualBlock(base_num_channels)
        self.pred_d = nn.Conv2d(base_num_channels, 1, 1, 1, 0)

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
        :param x: N x num_input_channels x H x W event tensor
        :return: N x 1 x H/2 x W/2 i, a, d
        """
        # i
        x_i = self.head_i(x)
        x_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_i
        x_i = self.R1_i(x_i)
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i
        x_i = self.R2_i(x_i)
        i = self.pred_i(x_i)

        # a
        x_a = self.head_a(x)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a
        x_a = self.R1_a(x_a)
        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a
        x_a = self.R2_a(x_a)
        a = self.pred_a(x_a)

        # d
        x_d = self.head_d(x)
        x_d = self.G1_d(x_d, self._states_d[0])
        self._states_d[0] = x_d
        x_d = self.R1_d(x_d)
        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d
        x_d = self.R2_d(x_d)
        d = self.pred_d(x_d)

        return {
            'i': i,
            'a': a,
            'd': d
        }


class V15(BaseModel):
    """
    Three branches for iad.
    """
    def __init__(self, num_bins=5, base_num_channels=32, kernel_size=3, num_recurrent_units=2, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)

        half_num_channels = int(base_num_channels / 2)
        self.num_bins = num_bins
        self.num_recurrent_units = num_recurrent_units

        self.head_i = RPPP(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.fusion_weight1_i = Fusion_Weight(half_num_channels)
        self.fusion_weight2_i = Fusion_Weight(half_num_channels)
        self.pred_i = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.head_a = RPPP(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.fusion_weight1_a = Fusion_Weight(half_num_channels)
        self.fusion_weight2_a = Fusion_Weight(half_num_channels)
        self.pred_a = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.head_d = RPPP(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.fusion_weight1_d = Fusion_Weight(half_num_channels)
        self.fusion_weight2_d = Fusion_Weight(half_num_channels)
        self.pred_d = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.jr1 = JR_IAD(base_num_channels)
        self.jr2 = JR_IAD(base_num_channels)

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
        :param x: N x num_input_channels x H x W event tensor
        :return: N x 1 x H/2 x W/2 i, a, d
        """
        # first stage
        x_i = self.head_i(x)
        x_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_i

        x_a = self.head_a(x)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a

        x_d = self.head_d(x)
        x_d = self.G1_d(x_d, self._states_d[0])
        self._states_d[0] = x_d

        x_i, x_a, x_d = self.jr1(x_i, x_a, x_d)

        # enhancement 1
        fusion_weight1_i = self.fusion_weight1_i(x_i)
        x_i = x_i * fusion_weight1_i

        fusion_weight1_a = self.fusion_weight1_a(x_a)
        x_a = x_a * fusion_weight1_a

        fusion_weight1_d = self.fusion_weight1_d(x_d)
        x_d = x_d * fusion_weight1_d

        # second stage
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i

        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a

        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d

        x_i, x_a, x_d = self.jr2(x_i, x_a, x_d)

        # enhancement 2
        fusion_weight2_i = self.fusion_weight2_i(x_i)
        x_i = x_i * fusion_weight2_i

        fusion_weight2_a = self.fusion_weight2_a(x_a)
        x_a = x_a * fusion_weight2_a

        fusion_weight2_d = self.fusion_weight2_d(x_d)
        x_d = x_d * fusion_weight2_d

        # prediction
        i = self.pred_i(x_i)
        a = self.pred_a(x_a)
        d = self.pred_d(x_d)

        return {
            'i': i,
            'a': a,
            'd': d
        }


class V16(BaseModel):
    """
    Three branches for iad.
    """
    def __init__(self, num_bins=5, base_num_channels=32, kernel_size=3, num_recurrent_units=2, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)

        self.num_bins = num_bins
        self.num_recurrent_units = num_recurrent_units

        self.head_i = RPPP(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_i = ResidualBlock(base_num_channels)
        self.R2_i = ResidualBlock(base_num_channels)
        self.pred_i = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.head_a = RPPP(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_a = ResidualBlock(base_num_channels)
        self.R2_a = ResidualBlock(base_num_channels)
        self.pred_a = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.head_d = RPPP(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_d = ResidualBlock(base_num_channels)
        self.R2_d = ResidualBlock(base_num_channels)
        self.pred_d = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.fusion_weight1 = Fusion_Weight_IAD(base_num_channels)
        self.fusion_weight2 = Fusion_Weight_IAD(base_num_channels)

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
        :param x: N x num_input_channels x H x W event tensor
        :return: N x 1 x H/2 x W/2 i, a, d
        """
        # first stage
        x_i = self.head_i(x)
        x_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_i
        x_i = self.R1_i(x_i)

        x_a = self.head_a(x)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a
        x_a = self.R1_a(x_a)

        x_d = self.head_d(x)
        x_d = self.G1_d(x_d, self._states_d[0])
        self._states_d[0] = x_d
        x_d = self.R1_d(x_d)

        x_iad = torch.cat((x_i, x_a, x_d), 1)
        fusion_weight1 = self.fusion_weight1(x_iad)
        x_iad_w1 = x_iad * fusion_weight1
        x_i, x_a, x_d = torch.chunk(x_iad_w1, 3, dim=1)

        # second stage
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i
        x_i = self.R2_i(x_i)

        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a
        x_a = self.R2_a(x_a)

        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d
        x_d = self.R2_d(x_d)

        x_iad = torch.cat((x_i, x_a, x_d), 1)
        fusion_weight2 = self.fusion_weight2(x_iad)
        x_iad_w2 = x_iad * fusion_weight2
        x_i, x_a, x_d = torch.chunk(x_iad_w2, 3, dim=1)

        # prediction
        i = self.pred_i(x_i)
        a = self.pred_a(x_a)
        d = self.pred_d(x_d)

        return {
            'i': i,
            'a': a,
            'd': d,
            'wi': fusion_weight2[:, 0:32, :, :],
            'wa': fusion_weight2[:, 32:64, :, :],
            'wd': fusion_weight2[:, 64:96, :, :]
        }



class V16_B(BaseModel):
    """
    Three branches for iad.
    Ablation study: base
    """
    def __init__(self, num_bins=5, base_num_channels=32, kernel_size=3, num_recurrent_units=2, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)

        self.num_bins = num_bins
        self.num_recurrent_units = num_recurrent_units

        self.head_i = nn.Sequential(nn.Conv2d(self.num_bins, base_num_channels, 2, 2, 0), nn.ReLU(True))
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_i = ResidualBlock(base_num_channels)
        self.R2_i = ResidualBlock(base_num_channels)
        self.pred_i = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.head_a = nn.Sequential(nn.Conv2d(self.num_bins, base_num_channels, 2, 2, 0), nn.ReLU(True))
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_a = ResidualBlock(base_num_channels)
        self.R2_a = ResidualBlock(base_num_channels)
        self.pred_a = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.head_d = nn.Sequential(nn.Conv2d(self.num_bins, base_num_channels, 2, 2, 0), nn.ReLU(True))
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_d = ResidualBlock(base_num_channels)
        self.R2_d = ResidualBlock(base_num_channels)
        self.pred_d = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        # self.fusion_weight1 = Fusion_Weight_IAD(base_num_channels)
        # self.fusion_weight2 = Fusion_Weight_IAD(base_num_channels)

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
        :param x: N x num_input_channels x H x W event tensor
        :return: N x 1 x H/2 x W/2 i, a, d
        """
        # first stage
        x_i = self.head_i(x)
        x_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_i
        x_i = self.R1_i(x_i)

        x_a = self.head_a(x)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a
        x_a = self.R1_a(x_a)

        x_d = self.head_d(x)
        x_d = self.G1_d(x_d, self._states_d[0])
        self._states_d[0] = x_d
        x_d = self.R1_d(x_d)

        # x_iad = torch.cat((x_i, x_a, x_d), 1)
        # fusion_weight1 = self.fusion_weight1(x_iad)
        # x_iad_w1 = x_iad * fusion_weight1
        # x_i, x_a, x_d = torch.chunk(x_iad_w1, 3, dim=1)

        # second stage
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i
        x_i = self.R2_i(x_i)

        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a
        x_a = self.R2_a(x_a)

        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d
        x_d = self.R2_d(x_d)

        # x_iad = torch.cat((x_i, x_a, x_d), 1)
        # fusion_weight2 = self.fusion_weight2(x_iad)
        # x_iad_w2 = x_iad * fusion_weight2
        # x_i, x_a, x_d = torch.chunk(x_iad_w2, 3, dim=1)

        # prediction
        i = self.pred_i(x_i)
        a = self.pred_a(x_a)
        d = self.pred_d(x_d)

        return {
            'i': i,
            'a': a,
            'd': d
        }


class V16_B_I(BaseModel):
    """
    Three branches for iad.
    Ablation study: base, four branches for the four directions intensity
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, num_recurrent_units=2, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)

        self.num_bins = num_bins
        self.num_recurrent_units = num_recurrent_units

        self.head_i90 = nn.Sequential(nn.Conv2d(self.num_bins, base_num_channels, 3, 1, 1), nn.ReLU(True))
        self.G1_i90 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i90 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_i90 = ResidualBlock(base_num_channels)
        self.R2_i90 = ResidualBlock(base_num_channels)
        self.pred_i90 = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.head_i45 = nn.Sequential(nn.Conv2d(self.num_bins, base_num_channels, 3, 1, 1), nn.ReLU(True))
        self.G1_i45 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i45 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_i45 = ResidualBlock(base_num_channels)
        self.R2_i45 = ResidualBlock(base_num_channels)
        self.pred_i45 = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.head_i135 = nn.Sequential(nn.Conv2d(self.num_bins, base_num_channels, 3, 1, 1), nn.ReLU(True))
        self.G1_i135 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i135 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_i135 = ResidualBlock(base_num_channels)
        self.R2_i135 = ResidualBlock(base_num_channels)
        self.pred_i135 = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.head_i0 = nn.Sequential(nn.Conv2d(self.num_bins, base_num_channels, 3, 1, 1), nn.ReLU(True))
        self.G1_i0 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i0 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_i0 = ResidualBlock(base_num_channels)
        self.R2_i0 = ResidualBlock(base_num_channels)
        self.pred_i0 = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        # self.fusion_weight1 = Fusion_Weight_IAD(base_num_channels)
        # self.fusion_weight2 = Fusion_Weight_IAD(base_num_channels)

        self.reset_states_i90()
        self.reset_states_i45()
        self.reset_states_i135()
        self.reset_states_i0()

    def reset_states_i90(self):
        self._states_i90 = [None] * self.num_recurrent_units

    def reset_states_i45(self):
        self._states_i45 = [None] * self.num_recurrent_units

    def reset_states_i135(self):
        self._states_i135 = [None] * self.num_recurrent_units

    def reset_states_i0(self):
        self._states_i0 = [None] * self.num_recurrent_units

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W event tensor
        :return: N x 1 x H/2 x W/2 i, a, d
        """
        x_90 = x[:, :, 0::2, 0::2]
        x_45 = x[:, :, 0::2, 1::2]
        x_135 = x[:, :, 1::2, 0::2]
        x_0 = x[:, :, 1::2, 1::2]

        # first stage
        x_i90 = self.head_i90(x_90)
        x_i90 = self.G1_i90(x_i90, self._states_i90[0])
        self._states_i90[0] = x_i90
        x_i90 = self.R1_i90(x_i90)

        x_i45 = self.head_i45(x_45)
        x_i45 = self.G1_i45(x_i45, self._states_i45[0])
        self._states_i45[0] = x_i45
        x_i45 = self.R1_i45(x_i45)

        x_i135 = self.head_i135(x_135)
        x_i135 = self.G1_i135(x_i135, self._states_i135[0])
        self._states_i135[0] = x_i135
        x_i135 = self.R1_i135(x_i135)

        x_i0 = self.head_i0(x_0)
        x_i0 = self.G1_i0(x_i0, self._states_i0[0])
        self._states_i0[0] = x_i0
        x_i0 = self.R1_i0(x_i0)

        # x_iad = torch.cat((x_i, x_a, x_d), 1)
        # fusion_weight1 = self.fusion_weight1(x_iad)
        # x_iad_w1 = x_iad * fusion_weight1
        # x_i, x_a, x_d = torch.chunk(x_iad_w1, 3, dim=1)

        # second stage
        x_i90 = self.G2_i90(x_i90, self._states_i90[1])
        self._states_i90[1] = x_i90
        x_i90 = self.R2_i90(x_i90)

        x_i45 = self.G2_i45(x_i45, self._states_i45[1])
        self._states_i45[1] = x_i45
        x_i45 = self.R2_i45(x_i45)

        x_i135 = self.G2_i135(x_i135, self._states_i135[1])
        self._states_i135[1] = x_i135
        x_i135 = self.R2_i135(x_i135)

        x_i0 = self.G2_i0(x_i0, self._states_i0[1])
        self._states_i0[1] = x_i0
        x_i0 = self.R2_i0(x_i0)

        # x_iad = torch.cat((x_i, x_a, x_d), 1)
        # fusion_weight2 = self.fusion_weight2(x_iad)
        # x_iad_w2 = x_iad * fusion_weight2
        # x_i, x_a, x_d = torch.chunk(x_iad_w2, 3, dim=1)

        # prediction
        i90 = self.pred_i90(x_i90)
        i45 = self.pred_i45(x_i45)
        i135 = self.pred_i135(x_i135)
        i0 = self.pred_i0(x_i0)

        return {
            'i90': i90,
            'i45': i45,
            'i135': i135,
            'i0': i0
        }


class V16_B_ONE(BaseModel):
    """
    One branch for iad.
    Ablation study: one branch for iad
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, num_recurrent_units=2, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)

        self.num_bins = num_bins
        self.num_recurrent_units = num_recurrent_units

        self.head_i = nn.Sequential(nn.Conv2d(self.num_bins, base_num_channels, 2, 2, 0), nn.ReLU(True))
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_i = ResidualBlock(base_num_channels)
        self.R2_i = ResidualBlock(base_num_channels)
        self.pred_i = nn.Conv2d(base_num_channels, 1, 3, 1, 1)
        self.pred_a = nn.Conv2d(base_num_channels, 1, 3, 1, 1)
        self.pred_d = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        # for keep the training code unchanged
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
        :param x: N x num_input_channels x H x W event tensor
        :return: N x 1 x H/2 x W/2 i, a, d
        """
        # first stage
        x_i = self.head_i(x)
        x_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_i
        x_i = self.R1_i(x_i)

        # second stage
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i
        x_i = self.R2_i(x_i)

        # prediction
        i = self.pred_i(x_i)
        a = self.pred_a(x_i)
        d = self.pred_d(x_i)

        return {
            'i': i,
            'a': a,
            'd': d
        }


class V16_B_K4(BaseModel):
    """
    Three branches for iad.
    Ablation study: base + head kernel is 4
    """
    def __init__(self, num_bins=5, base_num_channels=32, kernel_size=3, num_recurrent_units=2, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)

        self.num_bins = num_bins
        self.num_recurrent_units = num_recurrent_units

        self.head_i = nn.Sequential(nn.Conv2d(self.num_bins, base_num_channels, 4, 2, 1), nn.ReLU(True))
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_i = ResidualBlock(base_num_channels)
        self.R2_i = ResidualBlock(base_num_channels)
        self.pred_i = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.head_a = nn.Sequential(nn.Conv2d(self.num_bins, base_num_channels, 4, 2, 1), nn.ReLU(True))
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_a = ResidualBlock(base_num_channels)
        self.R2_a = ResidualBlock(base_num_channels)
        self.pred_a = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.head_d = nn.Sequential(nn.Conv2d(self.num_bins, base_num_channels, 4, 2, 1), nn.ReLU(True))
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_d = ResidualBlock(base_num_channels)
        self.R2_d = ResidualBlock(base_num_channels)
        self.pred_d = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        # self.fusion_weight1 = Fusion_Weight_IAD(base_num_channels)
        # self.fusion_weight2 = Fusion_Weight_IAD(base_num_channels)

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
        :param x: N x num_input_channels x H x W event tensor
        :return: N x 1 x H/2 x W/2 i, a, d
        """
        # first stage
        x_i = self.head_i(x)
        x_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_i
        x_i = self.R1_i(x_i)

        x_a = self.head_a(x)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a
        x_a = self.R1_a(x_a)

        x_d = self.head_d(x)
        x_d = self.G1_d(x_d, self._states_d[0])
        self._states_d[0] = x_d
        x_d = self.R1_d(x_d)

        # x_iad = torch.cat((x_i, x_a, x_d), 1)
        # fusion_weight1 = self.fusion_weight1(x_iad)
        # x_iad_w1 = x_iad * fusion_weight1
        # x_i, x_a, x_d = torch.chunk(x_iad_w1, 3, dim=1)

        # second stage
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i
        x_i = self.R2_i(x_i)

        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a
        x_a = self.R2_a(x_a)

        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d
        x_d = self.R2_d(x_d)

        # x_iad = torch.cat((x_i, x_a, x_d), 1)
        # fusion_weight2 = self.fusion_weight2(x_iad)
        # x_iad_w2 = x_iad * fusion_weight2
        # x_i, x_a, x_d = torch.chunk(x_iad_w2, 3, dim=1)

        # prediction
        i = self.pred_i(x_i)
        a = self.pred_a(x_a)
        d = self.pred_d(x_d)

        return {
            'i': i,
            'a': a,
            'd': d
        }


class V16_BR(BaseModel):
    """
    Three branches for iad.
    Ablation study: base + rppp
    """
    def __init__(self, num_bins=5, base_num_channels=32, kernel_size=3, num_recurrent_units=2, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)

        self.num_bins = num_bins
        self.num_recurrent_units = num_recurrent_units

        self.head_i = RPPP(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_i = ResidualBlock(base_num_channels)
        self.R2_i = ResidualBlock(base_num_channels)
        self.pred_i = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.head_a = RPPP(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_a = ResidualBlock(base_num_channels)
        self.R2_a = ResidualBlock(base_num_channels)
        self.pred_a = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.head_d = RPPP(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_d = ResidualBlock(base_num_channels)
        self.R2_d = ResidualBlock(base_num_channels)
        self.pred_d = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        # self.fusion_weight1 = Fusion_Weight_IAD(base_num_channels)
        # self.fusion_weight2 = Fusion_Weight_IAD(base_num_channels)

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
        :param x: N x num_input_channels x H x W event tensor
        :return: N x 1 x H/2 x W/2 i, a, d
        """
        # first stage
        x_i = self.head_i(x)
        x_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_i
        x_i = self.R1_i(x_i)

        x_a = self.head_a(x)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a
        x_a = self.R1_a(x_a)

        x_d = self.head_d(x)
        x_d = self.G1_d(x_d, self._states_d[0])
        self._states_d[0] = x_d
        x_d = self.R1_d(x_d)

        # x_iad = torch.cat((x_i, x_a, x_d), 1)
        # fusion_weight1 = self.fusion_weight1(x_iad)
        # x_iad_w1 = x_iad * fusion_weight1
        # x_i, x_a, x_d = torch.chunk(x_iad_w1, 3, dim=1)

        # second stage
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i
        x_i = self.R2_i(x_i)

        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a
        x_a = self.R2_a(x_a)

        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d
        x_d = self.R2_d(x_d)

        # x_iad = torch.cat((x_i, x_a, x_d), 1)
        # fusion_weight2 = self.fusion_weight2(x_iad)
        # x_iad_w2 = x_iad * fusion_weight2
        # x_i, x_a, x_d = torch.chunk(x_iad_w2, 3, dim=1)

        # prediction
        i = self.pred_i(x_i)
        a = self.pred_a(x_a)
        d = self.pred_d(x_d)

        return {
            'i': i,
            'a': a,
            'd': d
        }


class V16_BRA(BaseModel):
    """
    Three branches for iad.
    Ablation study: base + rppp + ae
    """
    def __init__(self, num_bins=5, base_num_channels=32, kernel_size=3, num_recurrent_units=2, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)

        half_num_channels = int(base_num_channels / 2)
        self.num_bins = num_bins
        self.num_recurrent_units = num_recurrent_units

        self.head_i = RPPP(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_i = ResidualBlock(base_num_channels)
        self.R2_i = ResidualBlock(base_num_channels)
        self.fusion_weight1_i = Fusion_Weight(half_num_channels)
        self.fusion_weight2_i = Fusion_Weight(half_num_channels)
        self.pred_i = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.head_a = RPPP(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_a = ResidualBlock(base_num_channels)
        self.R2_a = ResidualBlock(base_num_channels)
        self.fusion_weight1_a = Fusion_Weight(half_num_channels)
        self.fusion_weight2_a = Fusion_Weight(half_num_channels)
        self.pred_a = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.head_d = RPPP(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_d = ResidualBlock(base_num_channels)
        self.R2_d = ResidualBlock(base_num_channels)
        self.fusion_weight1_d = Fusion_Weight(half_num_channels)
        self.fusion_weight2_d = Fusion_Weight(half_num_channels)
        self.pred_d = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

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
        :param x: N x num_input_channels x H x W event tensor
        :return: N x 1 x H/2 x W/2 i, a, d
        """
        # first stage
        x_i = self.head_i(x)
        x_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_i
        x_i = self.R1_i(x_i)

        x_a = self.head_a(x)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a
        x_a = self.R1_a(x_a)

        x_d = self.head_d(x)
        x_d = self.G1_d(x_d, self._states_d[0])
        self._states_d[0] = x_d
        x_d = self.R1_d(x_d)

        # enhancement 1
        fusion_weight1_i = self.fusion_weight1_i(x_i)
        x_i = x_i * fusion_weight1_i

        fusion_weight1_a = self.fusion_weight1_a(x_a)
        x_a = x_a * fusion_weight1_a

        fusion_weight1_d = self.fusion_weight1_d(x_d)
        x_d = x_d * fusion_weight1_d

        # second stage
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i
        x_i = self.R2_i(x_i)

        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a
        x_a = self.R2_a(x_a)

        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d
        x_d = self.R2_d(x_d)

        # enhancement 2
        fusion_weight2_i = self.fusion_weight2_i(x_i)
        x_i = x_i * fusion_weight2_i

        fusion_weight2_a = self.fusion_weight2_a(x_a)
        x_a = x_a * fusion_weight2_a

        fusion_weight2_d = self.fusion_weight2_d(x_d)
        x_d = x_d * fusion_weight2_d

        # prediction
        i = self.pred_i(x_i)
        a = self.pred_a(x_a)
        d = self.pred_d(x_d)

        return {
            'i': i,
            'a': a,
            'd': d
        }


class V16_BC(BaseModel):
    """
    Three branches for iad.
    Ablation study: base + cdae
    """
    def __init__(self, num_bins=5, base_num_channels=32, kernel_size=3, num_recurrent_units=2, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)

        self.num_bins = num_bins
        self.num_recurrent_units = num_recurrent_units

        self.head_i = nn.Sequential(nn.Conv2d(self.num_bins, base_num_channels, 2, 2, 0), nn.ReLU(True))
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_i = ResidualBlock(base_num_channels)
        self.R2_i = ResidualBlock(base_num_channels)
        self.pred_i = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.head_a = nn.Sequential(nn.Conv2d(self.num_bins, base_num_channels, 2, 2, 0), nn.ReLU(True))
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_a = ResidualBlock(base_num_channels)
        self.R2_a = ResidualBlock(base_num_channels)
        self.pred_a = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.head_d = nn.Sequential(nn.Conv2d(self.num_bins, base_num_channels, 2, 2, 0), nn.ReLU(True))
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_d = ResidualBlock(base_num_channels)
        self.R2_d = ResidualBlock(base_num_channels)
        self.pred_d = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.fusion_weight1 = Fusion_Weight_IAD(base_num_channels)
        self.fusion_weight2 = Fusion_Weight_IAD(base_num_channels)

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
        :param x: N x num_input_channels x H x W event tensor
        :return: N x 1 x H/2 x W/2 i, a, d
        """
        # first stage
        x_i = self.head_i(x)
        x_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_i
        x_i = self.R1_i(x_i)

        x_a = self.head_a(x)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a
        x_a = self.R1_a(x_a)

        x_d = self.head_d(x)
        x_d = self.G1_d(x_d, self._states_d[0])
        self._states_d[0] = x_d
        x_d = self.R1_d(x_d)

        x_iad = torch.cat((x_i, x_a, x_d), 1)
        fusion_weight1 = self.fusion_weight1(x_iad)
        x_iad_w1 = x_iad * fusion_weight1
        x_i, x_a, x_d = torch.chunk(x_iad_w1, 3, dim=1)

        # second stage
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i
        x_i = self.R2_i(x_i)

        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a
        x_a = self.R2_a(x_a)

        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d
        x_d = self.R2_d(x_d)

        x_iad = torch.cat((x_i, x_a, x_d), 1)
        fusion_weight2 = self.fusion_weight2(x_iad)
        x_iad_w2 = x_iad * fusion_weight2
        x_i, x_a, x_d = torch.chunk(x_iad_w2, 3, dim=1)

        # prediction
        i = self.pred_i(x_i)
        a = self.pred_a(x_a)
        d = self.pred_d(x_d)

        return {
            'i': i,
            'a': a,
            'd': d
        }


class V17(BaseModel):
    """
    Three branches for iad.
    """
    def __init__(self, num_bins=5, base_num_channels=32, kernel_size=3, num_recurrent_units=2, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)

        self.num_bins = num_bins
        self.num_recurrent_units = num_recurrent_units

        self.head_i = nn.Sequential(nn.AvgPool2d((2, 2), stride=2),
                                    nn.Conv2d(self.num_bins, base_num_channels, 3, 1, 1), nn.ReLU(True))
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_i = ResidualBlock(base_num_channels)
        self.R2_i = ResidualBlock(base_num_channels)
        self.pred_i = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.head_a = RPPP(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_a = ResidualBlock(base_num_channels)
        self.R2_a = ResidualBlock(base_num_channels)
        self.pred_a = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.head_d = RPPP(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_d = ResidualBlock(base_num_channels)
        self.R2_d = ResidualBlock(base_num_channels)
        self.pred_d = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.fusion_weight1 = Fusion_Weight_IAD(base_num_channels)
        self.fusion_weight2 = Fusion_Weight_IAD(base_num_channels)

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
        :param x: N x num_input_channels x H x W event tensor
        :return: N x 1 x H/2 x W/2 i, a, d
        """
        # first stage
        x_i = self.head_i(x)
        x_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_i
        x_i = self.R1_i(x_i)

        x_a = self.head_a(x)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a
        x_a = self.R1_a(x_a)

        x_d = self.head_d(x)
        x_d = self.G1_d(x_d, self._states_d[0])
        self._states_d[0] = x_d
        x_d = self.R1_d(x_d)

        x_iad = torch.cat((x_i, x_a, x_d), 1)
        fusion_weight1 = self.fusion_weight1(x_iad)
        x_iad_w1 = x_iad * fusion_weight1
        x_i, x_a, x_d = torch.chunk(x_iad_w1, 3, dim=1)

        # second stage
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i
        x_i = self.R2_i(x_i)

        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a
        x_a = self.R2_a(x_a)

        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d
        x_d = self.R2_d(x_d)

        x_iad = torch.cat((x_i, x_a, x_d), 1)
        fusion_weight2 = self.fusion_weight2(x_iad)
        x_iad_w2 = x_iad * fusion_weight2
        x_i, x_a, x_d = torch.chunk(x_iad_w2, 3, dim=1)

        # prediction
        i = self.pred_i(x_i)
        a = self.pred_a(x_a)
        d = self.pred_d(x_d)

        return {
            'i': i,
            'a': a,
            'd': d
        }


class V18(BaseModel):
    """
    Three branches for iad.
    """
    def __init__(self, num_bins=5, base_num_channels=32, kernel_size=3, num_recurrent_units=2, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)

        self.num_bins = num_bins
        self.num_recurrent_units = num_recurrent_units

        self.head_i = RPPP(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_i = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.head_a = RPPP(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_a = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.head_d = RPPP(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_d = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.jr1 = JR_IAD(base_num_channels)
        self.jr2 = JR_IAD(base_num_channels)

        self.fusion_weight1 = Fusion_Weight_IAD(base_num_channels)
        self.fusion_weight2 = Fusion_Weight_IAD(base_num_channels)

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
        :param x: N x num_input_channels x H x W event tensor
        :return: N x 1 x H/2 x W/2 i, a, d
        """
        # first stage
        x_i = self.head_i(x)
        x_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_i

        x_a = self.head_a(x)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a

        x_d = self.head_d(x)
        x_d = self.G1_d(x_d, self._states_d[0])
        self._states_d[0] = x_d

        x_i, x_a, x_d = self.jr1(x_i, x_a, x_d)

        x_iad = torch.cat((x_i, x_a, x_d), 1)
        fusion_weight1 = self.fusion_weight1(x_iad)
        x_iad_w1 = x_iad * fusion_weight1
        x_i, x_a, x_d = torch.chunk(x_iad_w1, 3, dim=1)

        # second stage
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i

        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a

        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d

        x_i, x_a, x_d = self.jr2(x_i, x_a, x_d)

        x_iad = torch.cat((x_i, x_a, x_d), 1)
        fusion_weight2 = self.fusion_weight2(x_iad)
        x_iad_w2 = x_iad * fusion_weight2
        x_i, x_a, x_d = torch.chunk(x_iad_w2, 3, dim=1)

        # prediction
        i = self.pred_i(x_i)
        a = self.pred_a(x_a)
        d = self.pred_d(x_d)

        return {
            'i': i,
            'a': a,
            'd': d
        }


class V19(BaseModel):
    """
    Three branches for iad.
    """
    def __init__(self, num_bins=5, base_num_channels=32, kernel_size=3, num_recurrent_units=2, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)

        self.num_bins = num_bins
        self.num_recurrent_units = num_recurrent_units

        self.head_i = RPPP(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_i = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.head_a = RPPP(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_a = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.head_d = RPPP(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_d = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.jr1 = JR_IAD(base_num_channels, n_layer=1)
        self.jr2 = JR_IAD(base_num_channels, n_layer=1)

        self.fusion_weight1 = Fusion_Weight_IAD(base_num_channels)
        self.fusion_weight2 = Fusion_Weight_IAD(base_num_channels)

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
        :param x: N x num_input_channels x H x W event tensor
        :return: N x 1 x H/2 x W/2 i, a, d
        """
        # first stage
        x_i = self.head_i(x)
        x_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_i

        x_a = self.head_a(x)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a

        x_d = self.head_d(x)
        x_d = self.G1_d(x_d, self._states_d[0])
        self._states_d[0] = x_d

        x_i, x_a, x_d = self.jr1(x_i, x_a, x_d)

        x_iad = torch.cat((x_i, x_a, x_d), 1)
        fusion_weight1 = self.fusion_weight1(x_iad)
        x_iad_w1 = x_iad * fusion_weight1
        x_i, x_a, x_d = torch.chunk(x_iad_w1, 3, dim=1)

        # second stage
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i

        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a

        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d

        x_i, x_a, x_d = self.jr2(x_i, x_a, x_d)

        x_iad = torch.cat((x_i, x_a, x_d), 1)
        fusion_weight2 = self.fusion_weight2(x_iad)
        x_iad_w2 = x_iad * fusion_weight2
        x_i, x_a, x_d = torch.chunk(x_iad_w2, 3, dim=1)

        # prediction
        i = self.pred_i(x_i)
        a = self.pred_a(x_a)
        d = self.pred_d(x_d)

        return {
            'i': i,
            'a': a,
            'd': d
        }


class E2P(BaseModel):
    """
    Three branches for iad.
    """
    def __init__(self, num_bins=5, base_num_channels=32, kernel_size=3, num_recurrent_units=2, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)

        self.num_bins = num_bins
        self.num_recurrent_units = num_recurrent_units

        self.head_i = RPPP(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_i = ResidualBlock(base_num_channels)
        self.R2_i = ResidualBlock(base_num_channels)
        self.pred_i = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.head_a = RPPP(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_a = ResidualBlock(base_num_channels)
        self.R2_a = ResidualBlock(base_num_channels)
        self.pred_a = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.head_d = RPPP(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_d = ResidualBlock(base_num_channels)
        self.R2_d = ResidualBlock(base_num_channels)
        self.pred_d = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.fusion_weight1 = Fusion_Weight_IAD(base_num_channels)
        self.fusion_weight2 = Fusion_Weight_IAD(base_num_channels)

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
        :param x: N x num_input_channels x H x W event tensor
        :return: N x 1 x H/2 x W/2 i, a, d
        """
        # first stage
        x_i = self.head_i(x)
        x_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_i
        x_i = self.R1_i(x_i)

        x_a = self.head_a(x)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a
        x_a = self.R1_a(x_a)

        x_d = self.head_d(x)
        x_d = self.G1_d(x_d, self._states_d[0])
        self._states_d[0] = x_d
        x_d = self.R1_d(x_d)

        x_iad = torch.cat((x_i, x_a, x_d), 1)
        fusion_weight1 = self.fusion_weight1(x_iad)
        x_iad_w1 = x_iad * fusion_weight1
        x_i, x_a, x_d = torch.chunk(x_iad_w1, 3, dim=1)

        # second stage
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i
        x_i = self.R2_i(x_i)

        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a
        x_a = self.R2_a(x_a)

        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d
        x_d = self.R2_d(x_d)

        x_iad = torch.cat((x_i, x_a, x_d), 1)
        fusion_weight2 = self.fusion_weight2(x_iad)
        x_iad_w2 = x_iad * fusion_weight2
        x_i, x_a, x_d = torch.chunk(x_iad_w2, 3, dim=1)

        # prediction
        i = self.pred_i(x_i)
        a = self.pred_a(x_a)
        d = self.pred_d(x_d)

        return {
            'i': i,
            'a': a,
            'd': d
        }

class E2PSC(BaseModel):
    """
    Three branches for iad.
    """
    def __init__(self, num_bins=5, base_num_channels=32, kernel_size=3, num_recurrent_units=2, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)

        self.num_bins = num_bins
        self.num_recurrent_units = num_recurrent_units

        self.head_i = RPPP(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_i = ResidualBlock(base_num_channels)
        self.R2_i = ResidualBlock(base_num_channels)
        self.pred_i = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.head_a = RPPP(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_a = ResidualBlock(base_num_channels)
        self.R2_a = ResidualBlock(base_num_channels)
        self.pred_a = nn.Conv2d(base_num_channels, 2, 3, 1, 1)

        self.head_d = RPPP(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_d = ResidualBlock(base_num_channels)
        self.R2_d = ResidualBlock(base_num_channels)
        self.pred_d = nn.Conv2d(base_num_channels, 1, 3, 1, 1)

        self.fusion_weight1 = Fusion_Weight_IAD(base_num_channels)
        self.fusion_weight2 = Fusion_Weight_IAD(base_num_channels)

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
        :param x: N x num_input_channels x H x W event tensor
        :return: N x 1 x H/2 x W/2 i, a, d
        """
        # first stage
        x_i = self.head_i(x)
        x_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_i
        x_i = self.R1_i(x_i)

        x_a = self.head_a(x)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a
        x_a = self.R1_a(x_a)

        x_d = self.head_d(x)
        x_d = self.G1_d(x_d, self._states_d[0])
        self._states_d[0] = x_d
        x_d = self.R1_d(x_d)

        x_iad = torch.cat((x_i, x_a, x_d), 1)
        fusion_weight1 = self.fusion_weight1(x_iad)
        x_iad_w1 = x_iad * fusion_weight1
        x_i, x_a, x_d = torch.chunk(x_iad_w1, 3, dim=1)

        # second stage
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i
        x_i = self.R2_i(x_i)

        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a
        x_a = self.R2_a(x_a)

        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d
        x_d = self.R2_d(x_d)

        x_iad = torch.cat((x_i, x_a, x_d), 1)
        fusion_weight2 = self.fusion_weight2(x_iad)
        x_iad_w2 = x_iad * fusion_weight2
        x_i, x_a, x_d = torch.chunk(x_iad_w2, 3, dim=1)

        # prediction
        i = self.pred_i(x_i)
        a = self.pred_a(x_a)
        d = self.pred_d(x_d)

        return {
            'i': i,
            'a': a,
            'd': d
        }
