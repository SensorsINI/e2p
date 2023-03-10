"""
 @Time    : 17.06.22 19:27
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : model_mhy.py
 @Function:
 
"""
import cv2
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_util import CropParameters, recursive_clone, copy_states
from .base.base_model import BaseModel
from .unet import UNetFlow, WNet, UNetFlowNoRecur, UNetRecurrent, UNet
from .submodules import *
from utils.color_utils import merge_channels_into_color_image
from .legacy import FireNet_legacy
from dct import *


class FireNet(BaseModel):
    """
    Refactored version of model from the paper: "Fast Image Reconstruction with an Event Camera", Scheerlinck et. al., 2019.
    The model is essentially a lighter version of E2VID, which runs faster (~2-3x faster) and has considerably less parameters (~200x less).
    However, the reconstructions are not as high quality as E2VID: they suffer from smearing artefacts, and initialization takes longer.
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        padding = kernel_size // 2
        self.head = ConvLayer(self.num_bins, base_num_channels, kernel_size, padding=padding)
        self.G1 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1 = ResidualBlock(base_num_channels, base_num_channels)
        self.G2 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2 = ResidualBlock(base_num_channels, base_num_channels)
        self.pred = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
        self.reset_states()

    @property
    def states(self):
        return copy_states(self._states)

    @states.setter
    def states(self, states):
        self._states = states

    def reset_states(self):
        self._states = [None] * self.num_recurrent_units

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W event tensor
        :return: N x num_output_channels x H x W image
        """
        x = self.head(x)
        x = self.G1(x, self._states[0])
        self._states[0] = x
        x = self.R1(x)
        x = self.G2(x, self._states[1])
        self._states[1] = x
        x = self.R2(x)
        return {'image': self.pred(x)}


class M1(BaseModel):
    """
    Raw2p.
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        padding = kernel_size // 2
        self.head = ConvLayer(self.num_bins, base_num_channels, kernel_size, padding=padding)
        self.G1 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1 = ResidualBlock(base_num_channels, base_num_channels)
        self.G2 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2 = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=4, stride=2, padding=1, activation=None)
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=4, stride=2, padding=1, activation=None)
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=4, stride=2, padding=1, activation=None)
        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
        self.reset_states()

    @property
    def states(self):
        return copy_states(self._states)

    @states.setter
    def states(self, states):
        self._states = states

    def reset_states(self):
        self._states = [None] * self.num_recurrent_units

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W event tensor
        :return: N x num_output_channels x H x W image
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


class M8(BaseModel):
    """
    Split input and use four separate branches to estimate four-direction intensities.
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        padding = kernel_size // 2
        # 90
        self.head_90 = ConvLayer(self.num_bins, base_num_channels, kernel_size, padding=padding)
        self.G1_90 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_90 = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_90 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_90 = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_90 = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        # 90
        self.head_45 = ConvLayer(self.num_bins, base_num_channels, kernel_size, padding=padding)
        self.G1_45 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_45 = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_45 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_45 = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_45 = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        # 135
        self.head_135 = ConvLayer(self.num_bins, base_num_channels, kernel_size, padding=padding)
        self.G1_135 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_135 = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_135 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_135 = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_135 = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        # 0
        self.head_0 = ConvLayer(self.num_bins, base_num_channels, kernel_size, padding=padding)
        self.G1_0 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_0 = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_0 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_0 = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_0 = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
        self.reset_states_90()
        self.reset_states_45()
        self.reset_states_135()
        self.reset_states_0()

    # 90
    @property
    def states_90(self):
        return copy_states(self._states_90)
    @states_90.setter
    def states_90(self, states):
        self._states_90 = states
    def reset_states_90(self):
        self._states_90 = [None] * self.num_recurrent_units

    # 45
    @property
    def states_45(self):
        return copy_states(self._states_45)
    @states_45.setter
    def states_45(self, states):
        self._states_45 = states
    def reset_states_45(self):
        self._states_45 = [None] * self.num_recurrent_units

    # 135
    @property
    def states_135(self):
        return copy_states(self._states_135)
    @states_135.setter
    def states_135(self, states):
        self._states_135 = states
    def reset_states_135(self):
        self._states_135 = [None] * self.num_recurrent_units

    # 0
    @property
    def states_0(self):
        return copy_states(self._states_0)
    @states_0.setter
    def states_0(self, states):
        self._states_0 = states
    def reset_states_0(self):
        self._states_0 = [None] * self.num_recurrent_units

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W event tensor
        :return: N x num_output_channels x H/2 x W/2 four intensity images
        """
        # split input
        x_90 = x[:, :, 0::2, 0::2]
        x_45 = x[:, :, 0::2, 1::2]
        x_135 = x[:, :, 1::2, 0::2]
        x_0 = x[:, :, 1::2, 1::2]

        # 90 branch
        x_90 = self.head_90(x_90)
        x_90 = self.G1_90(x_90, self._states_90[0])
        self._states_90[0] = x_90
        x_90 = self.R1_90(x_90)
        x_90 = self.G2_90(x_90, self._states_90[1])
        self._states_90[1] = x_90
        x_90 = self.R2_90(x_90)
        i_90 = self.pred_90(x_90)

        # 45 branch
        x_45 = self.head_45(x_45)
        x_45 = self.G1_45(x_45, self._states_45[0])
        self._states_45[0] = x_45
        x_45 = self.R1_45(x_45)
        x_45 = self.G2_45(x_45, self._states_45[1])
        self._states_45[1] = x_45
        x_45 = self.R2_45(x_45)
        i_45 = self.pred_45(x_45)

        # 135 branch
        x_135 = self.head_135(x_135)
        x_135 = self.G1_135(x_135, self._states_135[0])
        self._states_135[0] = x_135
        x_135 = self.R1_135(x_135)
        x_135 = self.G2_135(x_135, self._states_135[1])
        self._states_135[1] = x_135
        x_135 = self.R2_135(x_135)
        i_135 = self.pred_135(x_135)

        # 0 branch
        x_0 = self.head_0(x_0)
        x_0 = self.G1_0(x_0, self._states_0[0])
        self._states_0[0] = x_0
        x_0 = self.R1_0(x_0)
        x_0 = self.G2_0(x_0, self._states_0[1])
        self._states_0[1] = x_0
        x_0 = self.R2_0(x_0)
        i_0 = self.pred_0(x_0)

        return {
            'i_90': i_90,
            'i_45': i_45,
            'i_135': i_135,
            'i_0': i_0,
        }


class M9(BaseModel):
    """
    Split input and use four separate branches to estimate four-direction intensities, then estimate Stocks Parameters.
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        padding = kernel_size // 2
        # 90
        self.head_90 = ConvLayer(self.num_bins, base_num_channels, kernel_size, padding=padding)
        self.G1_90 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_90 = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_90 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_90 = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_90 = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        # 45
        self.head_45 = ConvLayer(self.num_bins, base_num_channels, kernel_size, padding=padding)
        self.G1_45 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_45 = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_45 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_45 = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_45 = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        # 135
        self.head_135 = ConvLayer(self.num_bins, base_num_channels, kernel_size, padding=padding)
        self.G1_135 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_135 = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_135 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_135 = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_135 = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        # 0
        self.head_0 = ConvLayer(self.num_bins, base_num_channels, kernel_size, padding=padding)
        self.G1_0 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_0 = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_0 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_0 = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_0 = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)

        # S0
        self.G3_s0 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R3_s0 = ResidualBlock(base_num_channels, base_num_channels)
        self.G4_s0 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R4_s0 = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_s0 = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        # S1
        self.G3_s1 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R3_s1 = ResidualBlock(base_num_channels, base_num_channels)
        self.G4_s1 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R4_s1 = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_s1 = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        # S2
        self.G3_s2 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R3_s2 = ResidualBlock(base_num_channels, base_num_channels)
        self.G4_s2 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R4_s2 = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_s2 = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
        self.reset_states_90()
        self.reset_states_45()
        self.reset_states_135()
        self.reset_states_0()
        self.reset_states_s0()
        self.reset_states_s1()
        self.reset_states_s2()

    # 90
    def reset_states_90(self):
        self._states_90 = [None] * self.num_recurrent_units

    # 45
    def reset_states_45(self):
        self._states_45 = [None] * self.num_recurrent_units

    # 135
    def reset_states_135(self):
        self._states_135 = [None] * self.num_recurrent_units

    # 0
    def reset_states_0(self):
        self._states_0 = [None] * self.num_recurrent_units

    # s0
    def reset_states_s0(self):
        self._states_s0 = [None] * self.num_recurrent_units

    # s1
    def reset_states_s1(self):
        self._states_s1 = [None] * self.num_recurrent_units

    # s2
    def reset_states_s2(self):
        self._states_s2 = [None] * self.num_recurrent_units

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W event tensor
        :return: N x num_output_channels x H/2 x W/2 four intensity images
        """
        # split input
        x_90 = x[:, :, 0::2, 0::2]
        x_45 = x[:, :, 0::2, 1::2]
        x_135 = x[:, :, 1::2, 0::2]
        x_0 = x[:, :, 1::2, 1::2]

        # 90 branch
        x_90 = self.head_90(x_90)
        x_90 = self.G1_90(x_90, self._states_90[0])
        self._states_90[0] = x_90
        x_90 = self.R1_90(x_90)
        x_90 = self.G2_90(x_90, self._states_90[1])
        self._states_90[1] = x_90
        x_90 = self.R2_90(x_90)
        i_90 = self.pred_90(x_90)

        # 45 branch
        x_45 = self.head_45(x_45)
        x_45 = self.G1_45(x_45, self._states_45[0])
        self._states_45[0] = x_45
        x_45 = self.R1_45(x_45)
        x_45 = self.G2_45(x_45, self._states_45[1])
        self._states_45[1] = x_45
        x_45 = self.R2_45(x_45)
        i_45 = self.pred_45(x_45)

        # 135 branch
        x_135 = self.head_135(x_135)
        x_135 = self.G1_135(x_135, self._states_135[0])
        self._states_135[0] = x_135
        x_135 = self.R1_135(x_135)
        x_135 = self.G2_135(x_135, self._states_135[1])
        self._states_135[1] = x_135
        x_135 = self.R2_135(x_135)
        i_135 = self.pred_135(x_135)

        # 0 branch
        x_0 = self.head_0(x_0)
        x_0 = self.G1_0(x_0, self._states_0[0])
        self._states_0[0] = x_0
        x_0 = self.R1_0(x_0)
        x_0 = self.G2_0(x_0, self._states_0[1])
        self._states_0[1] = x_0
        x_0 = self.R2_0(x_0)
        i_0 = self.pred_0(x_0)

        # s0 branch
        x_s0 = (x_0 + x_90 + x_45 + x_135) / 2
        x_s0 = self.G3_s0(x_s0, self._states_s0[0])
        self._states_s0[0] = x_s0
        x_s0 = self.R3_s0(x_s0)
        x_s0 = self.G4_s0(x_s0, self._states_s0[1])
        self._states_s0[1] = x_s0
        x_s0 = self.R4_s0(x_s0)
        s0 = self.pred_s0(x_s0)

        # s1 branch
        x_s1 = x_0 - x_90
        x_s1 = self.G3_s1(x_s1, self._states_s1[0])
        self._states_s1[0] = x_s1
        x_s1 = self.R3_s1(x_s1)
        x_s1 = self.G4_s1(x_s1, self._states_s1[1])
        self._states_s1[1] = x_s1
        x_s1 = self.R4_s1(x_s1)
        s1 = self.pred_s1(x_s1)

        # s2 branch
        x_s2 = x_45 - x_135
        x_s2 = self.G3_s2(x_s2, self._states_s2[0])
        self._states_s2[0] = x_s2
        x_s2 = self.R3_s2(x_s2)
        x_s2 = self.G4_s2(x_s2, self._states_s2[1])
        self._states_s2[1] = x_s2
        x_s2 = self.R4_s2(x_s2)
        s2 = self.pred_s2(x_s2)

        return {
            'i_90': i_90,
            'i_45': i_45,
            'i_135': i_135,
            'i_0': i_0,
            's0': s0,
            's1': s1,
            's2': s2,
        }


class M10(BaseModel):
    """
    Split input and use four separate branches to estimate four-direction intensities, then estimate Stocks Parameters,
    and finally predict aolp and dolp.
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        padding = kernel_size // 2
        # 90
        self.head_90 = ConvLayer(self.num_bins, base_num_channels, kernel_size, padding=padding)
        self.G1_90 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_90 = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_90 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_90 = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_90 = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        # 45
        self.head_45 = ConvLayer(self.num_bins, base_num_channels, kernel_size, padding=padding)
        self.G1_45 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_45 = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_45 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_45 = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_45 = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        # 135
        self.head_135 = ConvLayer(self.num_bins, base_num_channels, kernel_size, padding=padding)
        self.G1_135 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_135 = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_135 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_135 = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_135 = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        # 0
        self.head_0 = ConvLayer(self.num_bins, base_num_channels, kernel_size, padding=padding)
        self.G1_0 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_0 = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_0 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_0 = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_0 = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)

        # S0
        self.G3_s0 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R3_s0 = ResidualBlock(base_num_channels, base_num_channels)
        self.G4_s0 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R4_s0 = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_s0 = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        # S1
        self.G3_s1 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R3_s1 = ResidualBlock(base_num_channels, base_num_channels)
        self.G4_s1 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R4_s1 = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_s1 = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        # S2
        self.G3_s2 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R3_s2 = ResidualBlock(base_num_channels, base_num_channels)
        self.G4_s2 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R4_s2 = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_s2 = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)

        # Intensity
        self.G5_intensity = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R5_intensity = ResidualBlock(base_num_channels, base_num_channels)
        self.G6_intensity = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R6_intensity = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_intensity = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        # AoLP
        self.G5_aolp = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R5_aolp = ResidualBlock(base_num_channels, base_num_channels)
        self.G6_aolp = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R6_aolp = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_aolp = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        # DoLP
        self.G5_dolp = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R5_dolp = ResidualBlock(base_num_channels, base_num_channels)
        self.G6_dolp = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R6_dolp = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_dolp = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
        self.reset_states_90()
        self.reset_states_45()
        self.reset_states_135()
        self.reset_states_0()
        self.reset_states_s0()
        self.reset_states_s1()
        self.reset_states_s2()
        self.reset_states_intensity()
        self.reset_states_aolp()
        self.reset_states_dolp()

    # 90
    def reset_states_90(self):
        self._states_90 = [None] * self.num_recurrent_units

    # 45
    def reset_states_45(self):
        self._states_45 = [None] * self.num_recurrent_units

    # 135
    def reset_states_135(self):
        self._states_135 = [None] * self.num_recurrent_units

    # 0
    def reset_states_0(self):
        self._states_0 = [None] * self.num_recurrent_units

    # s0
    def reset_states_s0(self):
        self._states_s0 = [None] * self.num_recurrent_units

    # s1
    def reset_states_s1(self):
        self._states_s1 = [None] * self.num_recurrent_units

    # s2
    def reset_states_s2(self):
        self._states_s2 = [None] * self.num_recurrent_units

    # intensity
    def reset_states_intensity(self):
        self._states_intensity = [None] * self.num_recurrent_units

    # aolp
    def reset_states_aolp(self):
        self._states_aolp = [None] * self.num_recurrent_units

    # dolp
    def reset_states_dolp(self):
        self._states_dolp = [None] * self.num_recurrent_units

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W event tensor
        :return: N x num_output_channels x H/2 x W/2 four intensity images
        """
        # split input
        x_90 = x[:, :, 0::2, 0::2]
        x_45 = x[:, :, 0::2, 1::2]
        x_135 = x[:, :, 1::2, 0::2]
        x_0 = x[:, :, 1::2, 1::2]

        # 90 branch
        x_90 = self.head_90(x_90)
        x_90 = self.G1_90(x_90, self._states_90[0])
        self._states_90[0] = x_90
        x_90 = self.R1_90(x_90)
        x_90 = self.G2_90(x_90, self._states_90[1])
        self._states_90[1] = x_90
        x_90 = self.R2_90(x_90)
        i_90 = self.pred_90(x_90)
        # 45 branch
        x_45 = self.head_45(x_45)
        x_45 = self.G1_45(x_45, self._states_45[0])
        self._states_45[0] = x_45
        x_45 = self.R1_45(x_45)
        x_45 = self.G2_45(x_45, self._states_45[1])
        self._states_45[1] = x_45
        x_45 = self.R2_45(x_45)
        i_45 = self.pred_45(x_45)
        # 135 branch
        x_135 = self.head_135(x_135)
        x_135 = self.G1_135(x_135, self._states_135[0])
        self._states_135[0] = x_135
        x_135 = self.R1_135(x_135)
        x_135 = self.G2_135(x_135, self._states_135[1])
        self._states_135[1] = x_135
        x_135 = self.R2_135(x_135)
        i_135 = self.pred_135(x_135)
        # 0 branch
        x_0 = self.head_0(x_0)
        x_0 = self.G1_0(x_0, self._states_0[0])
        self._states_0[0] = x_0
        x_0 = self.R1_0(x_0)
        x_0 = self.G2_0(x_0, self._states_0[1])
        self._states_0[1] = x_0
        x_0 = self.R2_0(x_0)
        i_0 = self.pred_0(x_0)

        # s0 branch
        x_s0 = (x_0 + x_90 + x_45 + x_135) / 2
        x_s0 = self.G3_s0(x_s0, self._states_s0[0])
        self._states_s0[0] = x_s0
        x_s0 = self.R3_s0(x_s0)
        x_s0 = self.G4_s0(x_s0, self._states_s0[1])
        self._states_s0[1] = x_s0
        x_s0 = self.R4_s0(x_s0)
        s0 = self.pred_s0(x_s0)
        # s1 branch
        x_s1 = x_0 - x_90
        x_s1 = self.G3_s1(x_s1, self._states_s1[0])
        self._states_s1[0] = x_s1
        x_s1 = self.R3_s1(x_s1)
        x_s1 = self.G4_s1(x_s1, self._states_s1[1])
        self._states_s1[1] = x_s1
        x_s1 = self.R4_s1(x_s1)
        s1 = self.pred_s1(x_s1)
        # s2 branch
        x_s2 = x_45 - x_135
        x_s2 = self.G3_s2(x_s2, self._states_s2[0])
        self._states_s2[0] = x_s2
        x_s2 = self.R3_s2(x_s2)
        x_s2 = self.G4_s2(x_s2, self._states_s2[1])
        self._states_s2[1] = x_s2
        x_s2 = self.R4_s2(x_s2)
        s2 = self.pred_s2(x_s2)

        # intensity branch
        x_intensity = x_s0 / 2
        x_intensity = self.G5_intensity(x_intensity, self._states_intensity[0])
        self._states_intensity[0] = x_intensity
        x_intensity = self.R5_intensity(x_intensity)
        x_intensity = self.G6_intensity(x_intensity, self._states_intensity[1])
        self._states_intensity[1] = x_intensity
        x_intensity = self.R6_intensity(x_intensity)
        intensity = self.pred_intensity(x_intensity)
        # aolp branch
        x_aolp = 0.5 * torch.arctan2(x_s2, x_s1)
        x_aolp = self.G5_aolp(x_aolp, self._states_aolp[0])
        self._states_aolp[0] = x_aolp
        x_aolp = self.R5_aolp(x_aolp)
        x_aolp = self.G6_aolp(x_aolp, self._states_aolp[1])
        self._states_aolp[1] = x_aolp
        x_aolp = self.R6_aolp(x_aolp)
        aolp = self.pred_aolp(x_aolp)
        # dolp branch
        x_dolp = torch.full_like(x_s0, fill_value=0)
        mask = (x_s0 != 0)
        x_dolp[mask] = torch.div(torch.sqrt(torch.square(x_s1[mask]) + torch.square(x_s2[mask])), x_s0[mask])
        x_dolp = self.G5_dolp(x_dolp, self._states_dolp[0])
        self._states_dolp[0] = x_dolp
        x_dolp = self.R5_dolp(x_dolp)
        x_dolp = self.G6_dolp(x_dolp, self._states_dolp[1])
        self._states_dolp[1] = x_dolp
        x_dolp = self.R6_dolp(x_dolp)
        dolp = self.pred_dolp(x_dolp)

        return {
            'i_90': i_90,
            'i_45': i_45,
            'i_135': i_135,
            'i_0': i_0,
            's0': s0,
            's1': s1,
            's2': s2,
            'intensity': intensity,
            'aolp': torch.sigmoid(aolp),
            'dolp': torch.sigmoid(dolp)
        }


class M12(BaseModel):
    """
    Split input and use four separate branches to estimate four-direction intensities, then estimate Stocks Parameters.
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        padding = kernel_size // 2
        # 90
        self.head_90 = ConvLayer(self.num_bins, base_num_channels, kernel_size, padding=padding)
        self.G1_90 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_90 = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_90 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_90 = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_90 = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        # 45
        self.head_45 = ConvLayer(self.num_bins, base_num_channels, kernel_size, padding=padding)
        self.G1_45 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_45 = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_45 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_45 = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_45 = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        # 135
        self.head_135 = ConvLayer(self.num_bins, base_num_channels, kernel_size, padding=padding)
        self.G1_135 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_135 = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_135 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_135 = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_135 = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        # 0
        self.head_0 = ConvLayer(self.num_bins, base_num_channels, kernel_size, padding=padding)
        self.G1_0 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_0 = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_0 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_0 = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_0 = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)

        # S0
        self.G3_s0 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R3_s0 = ResidualBlock(base_num_channels, base_num_channels)
        self.G4_s0 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R4_s0 = ResidualBlock(base_num_channels, base_num_channels)
        self.G5_s0 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R5_s0 = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_s0 = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        # S1
        self.G3_s1 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R3_s1 = ResidualBlock(base_num_channels, base_num_channels)
        self.G4_s1 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R4_s1 = ResidualBlock(base_num_channels, base_num_channels)
        self.G5_s1 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R5_s1 = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_s1 = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        # S2
        self.G3_s2 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R3_s2 = ResidualBlock(base_num_channels, base_num_channels)
        self.G4_s2 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R4_s2 = ResidualBlock(base_num_channels, base_num_channels)
        self.G5_s2 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R5_s2 = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_s2 = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
        self.reset_states_90()
        self.reset_states_45()
        self.reset_states_135()
        self.reset_states_0()
        self.reset_states_s0()
        self.reset_states_s1()
        self.reset_states_s2()

    # 90
    def reset_states_90(self):
        self._states_90 = [None] * self.num_recurrent_units

    # 45
    def reset_states_45(self):
        self._states_45 = [None] * self.num_recurrent_units

    # 135
    def reset_states_135(self):
        self._states_135 = [None] * self.num_recurrent_units

    # 0
    def reset_states_0(self):
        self._states_0 = [None] * self.num_recurrent_units

    # s0
    def reset_states_s0(self):
        self._states_s0 = [None] * 3

    # s1
    def reset_states_s1(self):
        self._states_s1 = [None] * 3

    # s2
    def reset_states_s2(self):
        self._states_s2 = [None] * 3

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W event tensor
        :return: N x num_output_channels x H/2 x W/2 four intensity images
        """
        # split input
        x_90 = x[:, :, 0::2, 0::2]
        x_45 = x[:, :, 0::2, 1::2]
        x_135 = x[:, :, 1::2, 0::2]
        x_0 = x[:, :, 1::2, 1::2]

        # 90 branch
        x_90 = self.head_90(x_90)
        x_90 = self.G1_90(x_90, self._states_90[0])
        self._states_90[0] = x_90
        x_90 = self.R1_90(x_90)
        x_90 = self.G2_90(x_90, self._states_90[1])
        self._states_90[1] = x_90
        x_90 = self.R2_90(x_90)
        i_90 = self.pred_90(x_90)

        # 45 branch
        x_45 = self.head_45(x_45)
        x_45 = self.G1_45(x_45, self._states_45[0])
        self._states_45[0] = x_45
        x_45 = self.R1_45(x_45)
        x_45 = self.G2_45(x_45, self._states_45[1])
        self._states_45[1] = x_45
        x_45 = self.R2_45(x_45)
        i_45 = self.pred_45(x_45)

        # 135 branch
        x_135 = self.head_135(x_135)
        x_135 = self.G1_135(x_135, self._states_135[0])
        self._states_135[0] = x_135
        x_135 = self.R1_135(x_135)
        x_135 = self.G2_135(x_135, self._states_135[1])
        self._states_135[1] = x_135
        x_135 = self.R2_135(x_135)
        i_135 = self.pred_135(x_135)

        # 0 branch
        x_0 = self.head_0(x_0)
        x_0 = self.G1_0(x_0, self._states_0[0])
        self._states_0[0] = x_0
        x_0 = self.R1_0(x_0)
        x_0 = self.G2_0(x_0, self._states_0[1])
        self._states_0[1] = x_0
        x_0 = self.R2_0(x_0)
        i_0 = self.pred_0(x_0)

        # s0 branch
        x_s0 = (x_0 + x_90 + x_45 + x_135) / 2
        x_s0 = self.G3_s0(x_s0, self._states_s0[0])
        self._states_s0[0] = x_s0
        x_s0 = self.R3_s0(x_s0)
        x_s0 = self.G4_s0(x_s0, self._states_s0[1])
        self._states_s0[1] = x_s0
        x_s0 = self.R4_s0(x_s0)
        x_s0 = self.G5_s0(x_s0, self._states_s0[2])
        self._states_s0[2] = x_s0
        x_s0 = self.R5_s0(x_s0)
        s0 = self.pred_s0(x_s0)

        # s1 branch
        x_s1 = x_0 - x_90
        x_s1 = self.G3_s1(x_s1, self._states_s1[0])
        self._states_s1[0] = x_s1
        x_s1 = self.R3_s1(x_s1)
        x_s1 = self.G4_s1(x_s1, self._states_s1[1])
        self._states_s1[1] = x_s1
        x_s1 = self.R4_s1(x_s1)
        x_s1 = self.G5_s1(x_s1, self._states_s1[2])
        self._states_s1[2] = x_s1
        x_s1 = self.R5_s1(x_s1)
        s1 = self.pred_s1(x_s1)

        # s2 branch
        x_s2 = x_45 - x_135
        x_s2 = self.G3_s2(x_s2, self._states_s2[0])
        self._states_s2[0] = x_s2
        x_s2 = self.R3_s2(x_s2)
        x_s2 = self.G4_s2(x_s2, self._states_s2[1])
        self._states_s2[1] = x_s2
        x_s2 = self.R4_s2(x_s2)
        x_s2 = self.G5_s2(x_s2, self._states_s2[2])
        self._states_s2[2] = x_s2
        x_s2 = self.R5_s2(x_s2)
        s2 = self.pred_s2(x_s2)

        return {
            'i_90': i_90,
            'i_45': i_45,
            'i_135': i_135,
            'i_0': i_0,
            's0': s0,
            's1': s1,
            's2': s2,
        }


class M13(BaseModel):
    """
    Split input and use four separate branches to estimate four-direction intensities, then estimate Stocks Parameters.
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        padding = kernel_size // 2

        self.head = ConvLayer(self.num_bins * 4, base_num_channels, kernel_size, padding=padding)
        self.G1 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1 = ResidualBlock(base_num_channels, base_num_channels)
        self.G2 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2 = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_90 = ConvLayer(4, out_channels=1, kernel_size=5, padding=2, activation=None)
        self.pred_45 = ConvLayer(4, out_channels=1, kernel_size=5, padding=2, activation=None)
        self.pred_135 = ConvLayer(4, out_channels=1, kernel_size=5, padding=2, activation=None)
        self.pred_0 = ConvLayer(4, out_channels=1, kernel_size=5, padding=2, activation=None)
        self.pred_s0 = ConvLayer(4, out_channels=1, kernel_size=5, padding=2, activation=None)
        self.pred_s1 = ConvLayer(4, out_channels=1, kernel_size=5, padding=2, activation=None)
        self.pred_s2 = ConvLayer(4, out_channels=1, kernel_size=5, padding=2, activation=None)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
        self.reset_states_i()

    def reset_states_i(self):
        self._states_i = [None] * self.num_recurrent_units


    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W event tensor
        :return: N x num_output_channels x H/2 x W/2 four intensity images
        """
        # split input
        x_90 = x[:, :, 0::2, 0::2]
        x_45 = x[:, :, 0::2, 1::2]
        x_135 = x[:, :, 1::2, 0::2]
        x_0 = x[:, :, 1::2, 1::2]

        x = torch.cat([x_90, x_45, x_135, x_0], 1)

        x = self.head(x)
        x = self.G1(x, self._states_i[0])
        self._states_i[0] = x
        x = self.R1(x)
        x = self.G2(x, self._states_i[1])
        self._states_i[1] = x
        x = self.R2(x)

        x_90 = x[:, :4, :, :]
        x_45 = x[:, 4:8, :, :]
        x_135 = x[:, 8:12, :, :]
        x_0 = x[:, 12:, :, :]

        i_90 = self.pred_90(x_90)
        i_45 = self.pred_45(x_45)
        i_135 = self.pred_135(x_135)
        i_0 = self.pred_0(x_0)

        s0 = self.pred_s0((x_90 + x_45 + x_135 + x_0) / 2)
        s1 = self.pred_s1(x_0 - x_90)
        s2 = self.pred_s2(x_45 - x_135)

        return {
            'i_90': i_90,
            'i_45': i_45,
            'i_135': i_135,
            'i_0': i_0,
            's0': s0,
            's1': s1,
            's2': s2,
        }


class M14(BaseModel):
    """
    Split input and use one branch to estimate four-direction intensities concurrently, then estimate Stocks Parameters.
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        padding = kernel_size // 2
        # intensity
        self.head = ConvLayer(self.num_bins * 4, base_num_channels, kernel_size, padding=padding)
        self.G1 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1 = ResidualBlock(base_num_channels, base_num_channels)
        self.G2 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2 = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_90 = ConvLayer(4, out_channels=1, kernel_size=5, padding=2, activation='relu')
        self.pred_45 = ConvLayer(4, out_channels=1, kernel_size=5, padding=2, activation='relu')
        self.pred_135 = ConvLayer(4, out_channels=1, kernel_size=5, padding=2, activation='relu')
        self.pred_0 = ConvLayer(4, out_channels=1, kernel_size=5, padding=2, activation='relu')
        # s0
        base_num_channels_s = base_num_channels // 4
        self.G3_s0 = ConvGRU(base_num_channels_s, base_num_channels_s, kernel_size)
        self.R3_s0 = ResidualBlock(base_num_channels_s, base_num_channels_s)
        self.G4_s0 = ConvGRU(base_num_channels_s, base_num_channels_s, kernel_size)
        self.R4_s0 = ResidualBlock(base_num_channels_s, base_num_channels_s)
        self.pred_s0 = ConvLayer(4, out_channels=1, kernel_size=5, padding=2, activation='relu')
        # s1
        self.G3_s1 = ConvGRU(base_num_channels_s, base_num_channels_s, kernel_size)
        self.R3_s1 = ResidualBlock(base_num_channels_s, base_num_channels_s)
        self.G4_s1 = ConvGRU(base_num_channels_s, base_num_channels_s, kernel_size)
        self.R4_s1 = ResidualBlock(base_num_channels_s, base_num_channels_s)
        self.pred_s1 = ConvLayer(4, out_channels=1, kernel_size=5, padding=2, activation=None)
        # s2
        self.G3_s2 = ConvGRU(base_num_channels_s, base_num_channels_s, kernel_size)
        self.R3_s2 = ResidualBlock(base_num_channels_s, base_num_channels_s)
        self.G4_s2 = ConvGRU(base_num_channels_s, base_num_channels_s, kernel_size)
        self.R4_s2 = ResidualBlock(base_num_channels_s, base_num_channels_s)
        self.pred_s2 = ConvLayer(4, out_channels=1, kernel_size=5, padding=2, activation=None)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
        self.reset_states_i()
        self.reset_states_s0()
        self.reset_states_s1()
        self.reset_states_s2()

    def reset_states_i(self):
        self._states_i = [None] * self.num_recurrent_units

    def reset_states_s0(self):
        self._states_s0 = [None] * self.num_recurrent_units

    def reset_states_s1(self):
        self._states_s1 = [None] * self.num_recurrent_units

    def reset_states_s2(self):
        self._states_s2 = [None] * self.num_recurrent_units

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W event tensor
        :return: N x num_output_channels x H/2 x W/2 four intensity images and three stock parameters
        """
        # split input
        x_90 = x[:, :, 0::2, 0::2]
        x_45 = x[:, :, 0::2, 1::2]
        x_135 = x[:, :, 1::2, 0::2]
        x_0 = x[:, :, 1::2, 1::2]

        x = torch.cat([x_90, x_45, x_135, x_0], 1)

        x = self.head(x)
        x = self.G1(x, self._states_i[0])
        self._states_i[0] = x
        x = self.R1(x)
        x = self.G2(x, self._states_i[1])
        self._states_i[1] = x
        x = self.R2(x)

        x_90 = x[:, :4, :, :]
        x_45 = x[:, 4:8, :, :]
        x_135 = x[:, 8:12, :, :]
        x_0 = x[:, 12:, :, :]

        i_90 = self.pred_90(x_90)
        i_45 = self.pred_45(x_45)
        i_135 = self.pred_135(x_135)
        i_0 = self.pred_0(x_0)

        # s0
        x_s0 = (x_90 + x_45 + x_135 + x_0) / 2
        x_s0 = self.G3_s0(x_s0, self._states_s0[0])
        self._states_s0[0] = x_s0
        x_s0 = self.R3_s0(x_s0)
        x_s0 = self.G4_s0(x_s0, self._states_s0[1])
        self._states_s0[1] = x_s0
        x_s0 = self.R4_s0(x_s0)
        s0 = self.pred_s0(x_s0)

        # s1
        x_s1 = x_0 - x_90
        x_s1 = self.G3_s1(x_s1, self._states_s1[0])
        self._states_s1[0] = x_s1
        x_s1 = self.R3_s1(x_s1)
        x_s1 = self.G4_s1(x_s1, self._states_s1[1])
        self._states_s1[1] = x_s1
        x_s1 = self.R4_s1(x_s1)
        s1 = self.pred_s1(x_s1)

        # s2
        x_s2 = x_45 - x_135
        x_s2 = self.G3_s2(x_s2, self._states_s2[0])
        self._states_s2[0] = x_s2
        x_s2 = self.R3_s2(x_s2)
        x_s2 = self.G4_s2(x_s2, self._states_s2[1])
        self._states_s2[1] = x_s2
        x_s2 = self.R4_s2(x_s2)
        s2 = self.pred_s2(x_s2)

        return {
            'i_90': i_90,
            'i_45': i_45,
            'i_135': i_135,
            'i_0': i_0,
            's0': s0,
            's1': s1,
            's2': s2,
        }


class M15(BaseModel):
    """
    Split input and use one branch to estimate four-direction intensities concurrently, then estimate Stocks Parameters.
    Using CC attention to enlarge the receptive field.
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        padding = kernel_size // 2
        # intensity
        self.head = ConvLayer(self.num_bins * 4, base_num_channels, kernel_size, padding=padding)
        self.G1 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1 = ResidualBlock(base_num_channels, base_num_channels)
        self.CCA_i = CrissCrossAttention(16)
        self.G2 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2 = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_90 = ConvLayer(4, out_channels=1, kernel_size=5, padding=2, activation='relu')
        self.pred_45 = ConvLayer(4, out_channels=1, kernel_size=5, padding=2, activation='relu')
        self.pred_135 = ConvLayer(4, out_channels=1, kernel_size=5, padding=2, activation='relu')
        self.pred_0 = ConvLayer(4, out_channels=1, kernel_size=5, padding=2, activation='relu')
        # s0
        base_num_channels_s = base_num_channels // 4
        self.G3_s0 = ConvGRU(base_num_channels_s, base_num_channels_s, kernel_size)
        self.R3_s0 = ResidualBlock(base_num_channels_s, base_num_channels_s)
        self.CCA_s0 = CrissCrossAttention(4)
        self.G4_s0 = ConvGRU(base_num_channels_s, base_num_channels_s, kernel_size)
        self.R4_s0 = ResidualBlock(base_num_channels_s, base_num_channels_s)
        self.pred_s0 = ConvLayer(4, out_channels=1, kernel_size=5, padding=2, activation='relu')
        # s1
        self.G3_s1 = ConvGRU(base_num_channels_s, base_num_channels_s, kernel_size)
        self.R3_s1 = ResidualBlock(base_num_channels_s, base_num_channels_s)
        self.CCA_s1 = CrissCrossAttention(4)
        self.G4_s1 = ConvGRU(base_num_channels_s, base_num_channels_s, kernel_size)
        self.R4_s1 = ResidualBlock(base_num_channels_s, base_num_channels_s)
        self.pred_s1 = ConvLayer(4, out_channels=1, kernel_size=5, padding=2, activation=None)
        # s2
        self.G3_s2 = ConvGRU(base_num_channels_s, base_num_channels_s, kernel_size)
        self.R3_s2 = ResidualBlock(base_num_channels_s, base_num_channels_s)
        self.CCA_s2 = CrissCrossAttention(4)
        self.G4_s2 = ConvGRU(base_num_channels_s, base_num_channels_s, kernel_size)
        self.R4_s2 = ResidualBlock(base_num_channels_s, base_num_channels_s)
        self.pred_s2 = ConvLayer(4, out_channels=1, kernel_size=5, padding=2, activation=None)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
        self.reset_states_i()
        self.reset_states_s0()
        self.reset_states_s1()
        self.reset_states_s2()

    def reset_states_i(self):
        self._states_i = [None] * self.num_recurrent_units

    def reset_states_s0(self):
        self._states_s0 = [None] * self.num_recurrent_units

    def reset_states_s1(self):
        self._states_s1 = [None] * self.num_recurrent_units

    def reset_states_s2(self):
        self._states_s2 = [None] * self.num_recurrent_units

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W event tensor
        :return: N x num_output_channels x H/2 x W/2 four intensity images and three stock parameters
        """
        # split input
        x_90 = x[:, :, 0::2, 0::2]
        x_45 = x[:, :, 0::2, 1::2]
        x_135 = x[:, :, 1::2, 0::2]
        x_0 = x[:, :, 1::2, 1::2]

        x = torch.cat([x_90, x_45, x_135, x_0], 1)

        x = self.head(x)
        x = self.G1(x, self._states_i[0])
        self._states_i[0] = x
        x = self.R1(x)
        x = self.CCA_i(x)
        x = self.G2(x, self._states_i[1])
        self._states_i[1] = x
        x = self.R2(x)

        x_90 = x[:, :4, :, :]
        x_45 = x[:, 4:8, :, :]
        x_135 = x[:, 8:12, :, :]
        x_0 = x[:, 12:, :, :]

        i_90 = self.pred_90(x_90)
        i_45 = self.pred_45(x_45)
        i_135 = self.pred_135(x_135)
        i_0 = self.pred_0(x_0)

        # s0
        x_s0 = (x_90 + x_45 + x_135 + x_0) / 2
        x_s0 = self.G3_s0(x_s0, self._states_s0[0])
        self._states_s0[0] = x_s0
        x_s0 = self.R3_s0(x_s0)
        x_s0 = self.CCA_s0(x_s0)
        x_s0 = self.G4_s0(x_s0, self._states_s0[1])
        self._states_s0[1] = x_s0
        x_s0 = self.R4_s0(x_s0)
        s0 = self.pred_s0(x_s0)

        # s1
        x_s1 = x_0 - x_90
        x_s1 = self.G3_s1(x_s1, self._states_s1[0])
        self._states_s1[0] = x_s1
        x_s1 = self.R3_s1(x_s1)
        x_s1 = self.CCA_s1(x_s1)
        x_s1 = self.G4_s1(x_s1, self._states_s1[1])
        self._states_s1[1] = x_s1
        x_s1 = self.R4_s1(x_s1)
        s1 = self.pred_s1(x_s1)

        # s2
        x_s2 = x_45 - x_135
        x_s2 = self.G3_s2(x_s2, self._states_s2[0])
        self._states_s2[0] = x_s2
        x_s2 = self.R3_s2(x_s2)
        x_s2 = self.CCA_s2(x_s2)
        x_s2 = self.G4_s2(x_s2, self._states_s2[1])
        self._states_s2[1] = x_s2
        x_s2 = self.R4_s2(x_s2)
        s2 = self.pred_s2(x_s2)

        return {
            'i_90': i_90,
            'i_45': i_45,
            'i_135': i_135,
            'i_0': i_0,
            's0': s0,
            's1': s1,
            's2': s2,
        }


class M16(BaseModel):
    """
    Split input and use one branch to estimate four-direction intensities concurrently, then estimate Stocks Parameters.
    Using spatially separable convolutions in the residual blocks.
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        padding = kernel_size // 2
        # intensity
        self.head = ConvLayer(self.num_bins * 4, base_num_channels, kernel_size, padding=padding)
        self.G1 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1 = ResidualBlock_SSC(base_num_channels)
        self.G2 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2 = ResidualBlock_SSC(base_num_channels)
        self.pred_90 = ConvLayer(4, out_channels=1, kernel_size=5, padding=2, activation='relu')
        self.pred_45 = ConvLayer(4, out_channels=1, kernel_size=5, padding=2, activation='relu')
        self.pred_135 = ConvLayer(4, out_channels=1, kernel_size=5, padding=2, activation='relu')
        self.pred_0 = ConvLayer(4, out_channels=1, kernel_size=5, padding=2, activation='relu')
        # s0
        base_num_channels_s = base_num_channels // 4
        self.G3_s0 = ConvGRU(base_num_channels_s, base_num_channels_s, kernel_size)
        self.R3_s0 = ResidualBlock_SSC(base_num_channels_s)
        self.G4_s0 = ConvGRU(base_num_channels_s, base_num_channels_s, kernel_size)
        self.R4_s0 = ResidualBlock_SSC(base_num_channels_s)
        self.pred_s0 = ConvLayer(4, out_channels=1, kernel_size=5, padding=2, activation='relu')
        # s1
        self.G3_s1 = ConvGRU(base_num_channels_s, base_num_channels_s, kernel_size)
        self.R3_s1 = ResidualBlock_SSC(base_num_channels_s)
        self.G4_s1 = ConvGRU(base_num_channels_s, base_num_channels_s, kernel_size)
        self.R4_s1 = ResidualBlock_SSC(base_num_channels_s)
        self.pred_s1 = ConvLayer(4, out_channels=1, kernel_size=5, padding=2, activation=None)
        # s2
        self.G3_s2 = ConvGRU(base_num_channels_s, base_num_channels_s, kernel_size)
        self.R3_s2 = ResidualBlock_SSC(base_num_channels_s)
        self.G4_s2 = ConvGRU(base_num_channels_s, base_num_channels_s, kernel_size)
        self.R4_s2 = ResidualBlock_SSC(base_num_channels_s)
        self.pred_s2 = ConvLayer(4, out_channels=1, kernel_size=5, padding=2, activation=None)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
        self.reset_states_i()
        self.reset_states_s0()
        self.reset_states_s1()
        self.reset_states_s2()

    def reset_states_i(self):
        self._states_i = [None] * self.num_recurrent_units

    def reset_states_s0(self):
        self._states_s0 = [None] * self.num_recurrent_units

    def reset_states_s1(self):
        self._states_s1 = [None] * self.num_recurrent_units

    def reset_states_s2(self):
        self._states_s2 = [None] * self.num_recurrent_units

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W event tensor
        :return: N x num_output_channels x H/2 x W/2 four intensity images and three stock parameters
        """
        # split input
        x_90 = x[:, :, 0::2, 0::2]
        x_45 = x[:, :, 0::2, 1::2]
        x_135 = x[:, :, 1::2, 0::2]
        x_0 = x[:, :, 1::2, 1::2]

        x = torch.cat([x_90, x_45, x_135, x_0], 1)

        x = self.head(x)
        x = self.G1(x, self._states_i[0])
        self._states_i[0] = x
        x = self.R1(x)
        x = self.R1(x)
        x = self.G2(x, self._states_i[1])
        self._states_i[1] = x
        x = self.R2(x)
        x = self.R2(x)

        x_90 = x[:, :4, :, :]
        x_45 = x[:, 4:8, :, :]
        x_135 = x[:, 8:12, :, :]
        x_0 = x[:, 12:, :, :]

        i_90 = self.pred_90(x_90)
        i_45 = self.pred_45(x_45)
        i_135 = self.pred_135(x_135)
        i_0 = self.pred_0(x_0)

        # s0
        x_s0 = (x_90 + x_45 + x_135 + x_0) / 2
        x_s0 = self.G3_s0(x_s0, self._states_s0[0])
        self._states_s0[0] = x_s0
        x_s0 = self.R3_s0(x_s0)
        x_s0 = self.R3_s0(x_s0)
        x_s0 = self.G4_s0(x_s0, self._states_s0[1])
        self._states_s0[1] = x_s0
        x_s0 = self.R4_s0(x_s0)
        x_s0 = self.R4_s0(x_s0)
        s0 = self.pred_s0(x_s0)

        # s1
        x_s1 = x_0 - x_90
        x_s1 = self.G3_s1(x_s1, self._states_s1[0])
        self._states_s1[0] = x_s1
        x_s1 = self.R3_s1(x_s1)
        x_s1 = self.R3_s1(x_s1)
        x_s1 = self.G4_s1(x_s1, self._states_s1[1])
        self._states_s1[1] = x_s1
        x_s1 = self.R4_s1(x_s1)
        x_s1 = self.R4_s1(x_s1)
        s1 = self.pred_s1(x_s1)

        # s2
        x_s2 = x_45 - x_135
        x_s2 = self.G3_s2(x_s2, self._states_s2[0])
        self._states_s2[0] = x_s2
        x_s2 = self.R3_s2(x_s2)
        x_s2 = self.R3_s2(x_s2)
        x_s2 = self.G4_s2(x_s2, self._states_s2[1])
        self._states_s2[1] = x_s2
        x_s2 = self.R4_s2(x_s2)
        x_s2 = self.R4_s2(x_s2)
        s2 = self.pred_s2(x_s2)

        return {
            'i_90': i_90,
            'i_45': i_45,
            'i_135': i_135,
            'i_0': i_0,
            's0': s0,
            's1': s1,
            's2': s2,
        }


class M17(BaseModel):
    """
    Split input and use one branch to estimate four-direction intensities concurrently, then estimate Stocks Parameters.
    Using CC attention to enlarge the receptive field.
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        padding = kernel_size // 2
        # intensity
        self.head = ConvLayer(self.num_bins * 4, base_num_channels, kernel_size, padding=padding)
        self.G1 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1 = ResidualBlock_SE(base_num_channels, base_num_channels)
        self.G2 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2 = ResidualBlock_SE(base_num_channels, base_num_channels)
        self.pred_90 = ConvLayer(16, out_channels=1, kernel_size=5, padding=2, activation='relu')
        self.pred_45 = ConvLayer(16, out_channels=1, kernel_size=5, padding=2, activation='relu')
        self.pred_135 = ConvLayer(16, out_channels=1, kernel_size=5, padding=2, activation='relu')
        self.pred_0 = ConvLayer(16, out_channels=1, kernel_size=5, padding=2, activation='relu')
        # s0
        base_num_channels_s = base_num_channels // 4
        self.G3_s0 = ConvGRU(base_num_channels_s, base_num_channels_s, kernel_size)
        self.R3_s0 = ResidualBlock_SE(base_num_channels_s, base_num_channels_s)
        self.G4_s0 = ConvGRU(base_num_channels_s, base_num_channels_s, kernel_size)
        self.R4_s0 = ResidualBlock_SE(base_num_channels_s, base_num_channels_s)
        self.pred_s0 = ConvLayer(16, out_channels=1, kernel_size=5, padding=2, activation='relu')
        # s1
        self.G3_s1 = ConvGRU(base_num_channels_s, base_num_channels_s, kernel_size)
        self.R3_s1 = ResidualBlock_SE(base_num_channels_s, base_num_channels_s)
        self.G4_s1 = ConvGRU(base_num_channels_s, base_num_channels_s, kernel_size)
        self.R4_s1 = ResidualBlock_SE(base_num_channels_s, base_num_channels_s)
        self.pred_s1 = ConvLayer(16, out_channels=1, kernel_size=5, padding=2, activation=None)
        # s2
        self.G3_s2 = ConvGRU(base_num_channels_s, base_num_channels_s, kernel_size)
        self.R3_s2 = ResidualBlock_SE(base_num_channels_s, base_num_channels_s)
        self.G4_s2 = ConvGRU(base_num_channels_s, base_num_channels_s, kernel_size)
        self.R4_s2 = ResidualBlock_SE(base_num_channels_s, base_num_channels_s)
        self.pred_s2 = ConvLayer(16, out_channels=1, kernel_size=5, padding=2, activation=None)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
        self.reset_states_i()
        self.reset_states_s0()
        self.reset_states_s1()
        self.reset_states_s2()

    def reset_states_i(self):
        self._states_i = [None] * self.num_recurrent_units

    def reset_states_s0(self):
        self._states_s0 = [None] * self.num_recurrent_units

    def reset_states_s1(self):
        self._states_s1 = [None] * self.num_recurrent_units

    def reset_states_s2(self):
        self._states_s2 = [None] * self.num_recurrent_units

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W event tensor
        :return: N x num_output_channels x H/2 x W/2 four intensity images and three stock parameters
        """
        # split input
        x_90 = x[:, :, 0::2, 0::2]
        x_45 = x[:, :, 0::2, 1::2]
        x_135 = x[:, :, 1::2, 0::2]
        x_0 = x[:, :, 1::2, 1::2]

        x = torch.cat([x_90, x_45, x_135, x_0], 1)

        x = self.head(x)
        x = self.G1(x, self._states_i[0])
        self._states_i[0] = x
        x = self.R1(x)
        x = self.R1(x)
        x = self.G2(x, self._states_i[1])
        self._states_i[1] = x
        x = self.R2(x)
        x = self.R2(x)

        x_90 = x[:, :16, :, :]
        x_45 = x[:, 16:32, :, :]
        x_135 = x[:, 32:48, :, :]
        x_0 = x[:, 48:, :, :]

        i_90 = self.pred_90(x_90)
        i_45 = self.pred_45(x_45)
        i_135 = self.pred_135(x_135)
        i_0 = self.pred_0(x_0)

        # s0
        x_s0 = (x_90 + x_45 + x_135 + x_0) / 2
        x_s0 = self.G3_s0(x_s0, self._states_s0[0])
        self._states_s0[0] = x_s0
        x_s0 = self.R3_s0(x_s0)
        x_s0 = self.R3_s0(x_s0)
        x_s0 = self.G4_s0(x_s0, self._states_s0[1])
        self._states_s0[1] = x_s0
        x_s0 = self.R4_s0(x_s0)
        x_s0 = self.R4_s0(x_s0)
        s0 = self.pred_s0(x_s0)

        # s1
        x_s1 = x_0 - x_90
        x_s1 = self.G3_s1(x_s1, self._states_s1[0])
        self._states_s1[0] = x_s1
        x_s1 = self.R3_s1(x_s1)
        x_s1 = self.R3_s1(x_s1)
        x_s1 = self.G4_s1(x_s1, self._states_s1[1])
        self._states_s1[1] = x_s1
        x_s1 = self.R4_s1(x_s1)
        x_s1 = self.R4_s1(x_s1)
        s1 = self.pred_s1(x_s1)

        # s2
        x_s2 = x_45 - x_135
        x_s2 = self.G3_s2(x_s2, self._states_s2[0])
        self._states_s2[0] = x_s2
        x_s2 = self.R3_s2(x_s2)
        x_s2 = self.R3_s2(x_s2)
        x_s2 = self.G4_s2(x_s2, self._states_s2[1])
        self._states_s2[1] = x_s2
        x_s2 = self.R4_s2(x_s2)
        x_s2 = self.R4_s2(x_s2)
        s2 = self.pred_s2(x_s2)

        return {
            'i_90': i_90,
            'i_45': i_45,
            'i_135': i_135,
            'i_0': i_0,
            's0': s0,
            's1': s1,
            's2': s2,
        }


class M18(BaseModel):
    """
    Split input and use four separate branches to estimate four-direction intensities, then estimate Stocks Parameters.
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        padding = kernel_size // 2
        # 90
        self.head_90 = ConvLayer(self.num_bins, base_num_channels, kernel_size, padding=padding)
        self.G1_90 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_90 = ResidualBlock_SSC(base_num_channels)
        self.G2_90 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_90 = ResidualBlock_SSC(base_num_channels)
        self.pred_90 = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        # 45
        self.head_45 = ConvLayer(self.num_bins, base_num_channels, kernel_size, padding=padding)
        self.G1_45 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_45 = ResidualBlock_SSC(base_num_channels)
        self.G2_45 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_45 = ResidualBlock_SSC(base_num_channels)
        self.pred_45 = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        # 135
        self.head_135 = ConvLayer(self.num_bins, base_num_channels, kernel_size, padding=padding)
        self.G1_135 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_135 = ResidualBlock_SSC(base_num_channels)
        self.G2_135 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_135 = ResidualBlock_SSC(base_num_channels)
        self.pred_135 = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        # 0
        self.head_0 = ConvLayer(self.num_bins, base_num_channels, kernel_size, padding=padding)
        self.G1_0 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_0 = ResidualBlock_SSC(base_num_channels)
        self.G2_0 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_0 = ResidualBlock_SSC(base_num_channels)
        self.pred_0 = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)

        # S0
        self.G3_s0 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R3_s0 = ResidualBlock_SSC(base_num_channels)
        self.G4_s0 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R4_s0 = ResidualBlock_SSC(base_num_channels)
        self.G5_s0 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R5_s0 = ResidualBlock_SSC(base_num_channels)
        self.pred_s0 = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        # S1
        self.G3_s1 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R3_s1 = ResidualBlock_SSC(base_num_channels)
        self.G4_s1 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R4_s1 = ResidualBlock_SSC(base_num_channels)
        self.G5_s1 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R5_s1 = ResidualBlock_SSC(base_num_channels)
        self.pred_s1 = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        # S2
        self.G3_s2 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R3_s2 = ResidualBlock_SSC(base_num_channels)
        self.G4_s2 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R4_s2 = ResidualBlock_SSC(base_num_channels)
        self.G5_s2 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R5_s2 = ResidualBlock_SSC(base_num_channels)
        self.pred_s2 = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
        self.reset_states_90()
        self.reset_states_45()
        self.reset_states_135()
        self.reset_states_0()
        self.reset_states_s0()
        self.reset_states_s1()
        self.reset_states_s2()

    # 90
    def reset_states_90(self):
        self._states_90 = [None] * self.num_recurrent_units

    # 45
    def reset_states_45(self):
        self._states_45 = [None] * self.num_recurrent_units

    # 135
    def reset_states_135(self):
        self._states_135 = [None] * self.num_recurrent_units

    # 0
    def reset_states_0(self):
        self._states_0 = [None] * self.num_recurrent_units

    # s0
    def reset_states_s0(self):
        self._states_s0 = [None] * 3

    # s1
    def reset_states_s1(self):
        self._states_s1 = [None] * 3

    # s2
    def reset_states_s2(self):
        self._states_s2 = [None] * 3

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W event tensor
        :return: N x num_output_channels x H/2 x W/2 four intensity images
        """
        # split input
        x_90 = x[:, :, 0::2, 0::2]
        x_45 = x[:, :, 0::2, 1::2]
        x_135 = x[:, :, 1::2, 0::2]
        x_0 = x[:, :, 1::2, 1::2]

        # 90 branch
        x_90 = self.head_90(x_90)
        x_90 = self.G1_90(x_90, self._states_90[0])
        self._states_90[0] = x_90
        x_90 = self.R1_90(x_90)
        x_90 = self.G2_90(x_90, self._states_90[1])
        self._states_90[1] = x_90
        x_90 = self.R2_90(x_90)
        i_90 = self.pred_90(x_90)

        # 45 branch
        x_45 = self.head_45(x_45)
        x_45 = self.G1_45(x_45, self._states_45[0])
        self._states_45[0] = x_45
        x_45 = self.R1_45(x_45)
        x_45 = self.G2_45(x_45, self._states_45[1])
        self._states_45[1] = x_45
        x_45 = self.R2_45(x_45)
        i_45 = self.pred_45(x_45)

        # 135 branch
        x_135 = self.head_135(x_135)
        x_135 = self.G1_135(x_135, self._states_135[0])
        self._states_135[0] = x_135
        x_135 = self.R1_135(x_135)
        x_135 = self.G2_135(x_135, self._states_135[1])
        self._states_135[1] = x_135
        x_135 = self.R2_135(x_135)
        i_135 = self.pred_135(x_135)

        # 0 branch
        x_0 = self.head_0(x_0)
        x_0 = self.G1_0(x_0, self._states_0[0])
        self._states_0[0] = x_0
        x_0 = self.R1_0(x_0)
        x_0 = self.G2_0(x_0, self._states_0[1])
        self._states_0[1] = x_0
        x_0 = self.R2_0(x_0)
        i_0 = self.pred_0(x_0)

        # s0 branch
        x_s0 = (x_0 + x_90 + x_45 + x_135) / 2
        x_s0 = self.G3_s0(x_s0, self._states_s0[0])
        self._states_s0[0] = x_s0
        x_s0 = self.R3_s0(x_s0)
        x_s0 = self.G4_s0(x_s0, self._states_s0[1])
        self._states_s0[1] = x_s0
        x_s0 = self.R4_s0(x_s0)
        x_s0 = self.G5_s0(x_s0, self._states_s0[2])
        self._states_s0[2] = x_s0
        x_s0 = self.R5_s0(x_s0)
        s0 = self.pred_s0(x_s0)

        # s1 branch
        x_s1 = x_0 - x_90
        x_s1 = self.G3_s1(x_s1, self._states_s1[0])
        self._states_s1[0] = x_s1
        x_s1 = self.R3_s1(x_s1)
        x_s1 = self.G4_s1(x_s1, self._states_s1[1])
        self._states_s1[1] = x_s1
        x_s1 = self.R4_s1(x_s1)
        x_s1 = self.G5_s1(x_s1, self._states_s1[2])
        self._states_s1[2] = x_s1
        x_s1 = self.R5_s1(x_s1)
        s1 = self.pred_s1(x_s1)

        # s2 branch
        x_s2 = x_45 - x_135
        x_s2 = self.G3_s2(x_s2, self._states_s2[0])
        self._states_s2[0] = x_s2
        x_s2 = self.R3_s2(x_s2)
        x_s2 = self.G4_s2(x_s2, self._states_s2[1])
        self._states_s2[1] = x_s2
        x_s2 = self.R4_s2(x_s2)
        x_s2 = self.G5_s2(x_s2, self._states_s2[2])
        self._states_s2[2] = x_s2
        x_s2 = self.R5_s2(x_s2)
        s2 = self.pred_s2(x_s2)

        return {
            'i_90': i_90,
            'i_45': i_45,
            'i_135': i_135,
            'i_0': i_0,
            's0': s0,
            's1': s1,
            's2': s2,
        }


class M19(BaseModel):
    """
    Split input and use four separate branches to estimate four-direction intensities, then estimate Stocks Parameters.
    There is no intermediate supervision.
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        padding = kernel_size // 2
        # 90
        self.head_90 = ConvLayer(self.num_bins, base_num_channels, kernel_size, padding=padding)
        self.G1_90 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_90 = ResidualBlock_SSC(base_num_channels)
        self.G2_90 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_90 = ResidualBlock_SSC(base_num_channels)
        # 45
        self.head_45 = ConvLayer(self.num_bins, base_num_channels, kernel_size, padding=padding)
        self.G1_45 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_45 = ResidualBlock_SSC(base_num_channels)
        self.G2_45 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_45 = ResidualBlock_SSC(base_num_channels)
        # 135
        self.head_135 = ConvLayer(self.num_bins, base_num_channels, kernel_size, padding=padding)
        self.G1_135 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_135 = ResidualBlock_SSC(base_num_channels)
        self.G2_135 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_135 = ResidualBlock_SSC(base_num_channels)
        # 0
        self.head_0 = ConvLayer(self.num_bins, base_num_channels, kernel_size, padding=padding)
        self.G1_0 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_0 = ResidualBlock_SSC(base_num_channels)
        self.G2_0 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_0 = ResidualBlock_SSC(base_num_channels)

        # S0
        self.G3_s0 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R3_s0 = ResidualBlock_SSC(base_num_channels)
        self.G4_s0 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R4_s0 = ResidualBlock_SSC(base_num_channels)
        self.G5_s0 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R5_s0 = ResidualBlock_SSC(base_num_channels)
        self.pred_s0 = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        # S1
        self.G3_s1 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R3_s1 = ResidualBlock_SSC(base_num_channels)
        self.G4_s1 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R4_s1 = ResidualBlock_SSC(base_num_channels)
        self.G5_s1 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R5_s1 = ResidualBlock_SSC(base_num_channels)
        self.pred_s1 = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        # S2
        self.G3_s2 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R3_s2 = ResidualBlock_SSC(base_num_channels)
        self.G4_s2 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R4_s2 = ResidualBlock_SSC(base_num_channels)
        self.G5_s2 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R5_s2 = ResidualBlock_SSC(base_num_channels)
        self.pred_s2 = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
        self.reset_states_90()
        self.reset_states_45()
        self.reset_states_135()
        self.reset_states_0()
        self.reset_states_s0()
        self.reset_states_s1()
        self.reset_states_s2()

    # 90
    def reset_states_90(self):
        self._states_90 = [None] * self.num_recurrent_units

    # 45
    def reset_states_45(self):
        self._states_45 = [None] * self.num_recurrent_units

    # 135
    def reset_states_135(self):
        self._states_135 = [None] * self.num_recurrent_units

    # 0
    def reset_states_0(self):
        self._states_0 = [None] * self.num_recurrent_units

    # s0
    def reset_states_s0(self):
        self._states_s0 = [None] * 3

    # s1
    def reset_states_s1(self):
        self._states_s1 = [None] * 3

    # s2
    def reset_states_s2(self):
        self._states_s2 = [None] * 3

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W event tensor
        :return: N x num_output_channels x H/2 x W/2 four intensity images
        """
        # split input
        x_90 = x[:, :, 0::2, 0::2]
        x_45 = x[:, :, 0::2, 1::2]
        x_135 = x[:, :, 1::2, 0::2]
        x_0 = x[:, :, 1::2, 1::2]

        # 90 branch
        x_90 = self.head_90(x_90)
        x_90 = self.G1_90(x_90, self._states_90[0])
        self._states_90[0] = x_90
        x_90 = self.R1_90(x_90)
        x_90 = self.G2_90(x_90, self._states_90[1])
        self._states_90[1] = x_90
        x_90 = self.R2_90(x_90)

        # 45 branch
        x_45 = self.head_45(x_45)
        x_45 = self.G1_45(x_45, self._states_45[0])
        self._states_45[0] = x_45
        x_45 = self.R1_45(x_45)
        x_45 = self.G2_45(x_45, self._states_45[1])
        self._states_45[1] = x_45
        x_45 = self.R2_45(x_45)

        # 135 branch
        x_135 = self.head_135(x_135)
        x_135 = self.G1_135(x_135, self._states_135[0])
        self._states_135[0] = x_135
        x_135 = self.R1_135(x_135)
        x_135 = self.G2_135(x_135, self._states_135[1])
        self._states_135[1] = x_135
        x_135 = self.R2_135(x_135)

        # 0 branch
        x_0 = self.head_0(x_0)
        x_0 = self.G1_0(x_0, self._states_0[0])
        self._states_0[0] = x_0
        x_0 = self.R1_0(x_0)
        x_0 = self.G2_0(x_0, self._states_0[1])
        self._states_0[1] = x_0
        x_0 = self.R2_0(x_0)

        # s0 branch
        x_s0 = (x_0 + x_90 + x_45 + x_135) / 2
        x_s0 = self.G3_s0(x_s0, self._states_s0[0])
        self._states_s0[0] = x_s0
        x_s0 = self.R3_s0(x_s0)
        x_s0 = self.G4_s0(x_s0, self._states_s0[1])
        self._states_s0[1] = x_s0
        x_s0 = self.R4_s0(x_s0)
        x_s0 = self.G5_s0(x_s0, self._states_s0[2])
        self._states_s0[2] = x_s0
        x_s0 = self.R5_s0(x_s0)
        s0 = self.pred_s0(x_s0)

        # s1 branch
        x_s1 = x_0 - x_90
        x_s1 = self.G3_s1(x_s1, self._states_s1[0])
        self._states_s1[0] = x_s1
        x_s1 = self.R3_s1(x_s1)
        x_s1 = self.G4_s1(x_s1, self._states_s1[1])
        self._states_s1[1] = x_s1
        x_s1 = self.R4_s1(x_s1)
        x_s1 = self.G5_s1(x_s1, self._states_s1[2])
        self._states_s1[2] = x_s1
        x_s1 = self.R5_s1(x_s1)
        s1 = self.pred_s1(x_s1)

        # s2 branch
        x_s2 = x_45 - x_135
        x_s2 = self.G3_s2(x_s2, self._states_s2[0])
        self._states_s2[0] = x_s2
        x_s2 = self.R3_s2(x_s2)
        x_s2 = self.G4_s2(x_s2, self._states_s2[1])
        self._states_s2[1] = x_s2
        x_s2 = self.R4_s2(x_s2)
        x_s2 = self.G5_s2(x_s2, self._states_s2[2])
        self._states_s2[2] = x_s2
        x_s2 = self.R5_s2(x_s2)
        s2 = self.pred_s2(x_s2)

        return {
            'i_90': s0,
            'i_45': s0,
            'i_135': s0,
            'i_0': s0,
            's0': s0,
            's1': s1,
            's2': s2,
        }


class M20(BaseModel):
    """
    Split input and use four separate branches to estimate four-direction intensities, then estimate polarization.
    There is no intermediate supervision.
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        padding = kernel_size // 2

        base_num_channels = base_num_channels // 4
        # 90
        self.head_90 = ConvLayer(self.num_bins, base_num_channels, kernel_size, padding=padding)
        self.G1_90 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_90 = ResidualBlock_SSC(base_num_channels)
        self.G2_90 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_90 = ResidualBlock_SSC(base_num_channels)
        # 45
        self.head_45 = ConvLayer(self.num_bins, base_num_channels, kernel_size, padding=padding)
        self.G1_45 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_45 = ResidualBlock_SSC(base_num_channels)
        self.G2_45 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_45 = ResidualBlock_SSC(base_num_channels)
        # 135
        self.head_135 = ConvLayer(self.num_bins, base_num_channels, kernel_size, padding=padding)
        self.G1_135 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_135 = ResidualBlock_SSC(base_num_channels)
        self.G2_135 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_135 = ResidualBlock_SSC(base_num_channels)
        # 0
        self.head_0 = ConvLayer(self.num_bins, base_num_channels, kernel_size, padding=padding)
        self.G1_0 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_0 = ResidualBlock_SSC(base_num_channels)
        self.G2_0 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_0 = ResidualBlock_SSC(base_num_channels)

        base_num_channels = base_num_channels * 4
        # i
        self.G3_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R3_i = ResidualBlock_SSC(base_num_channels)
        self.G4_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R4_i = ResidualBlock_SSC(base_num_channels)
        self.G5_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R5_i = ResidualBlock_SSC(base_num_channels)
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation='relu')
        # a
        self.G3_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R3_a = ResidualBlock_SSC(base_num_channels)
        self.G4_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R4_a = ResidualBlock_SSC(base_num_channels)
        self.G5_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R5_a = ResidualBlock_SSC(base_num_channels)
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation='relu')
        # d
        self.G3_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R3_d = ResidualBlock_SSC(base_num_channels)
        self.G4_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R4_d = ResidualBlock_SSC(base_num_channels)
        self.G5_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R5_d = ResidualBlock_SSC(base_num_channels)
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation='relu')

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
        self.reset_states_90()
        self.reset_states_45()
        self.reset_states_135()
        self.reset_states_0()
        self.reset_states_i()
        self.reset_states_a()
        self.reset_states_d()

    # 90
    def reset_states_90(self):
        self._states_90 = [None] * self.num_recurrent_units

    # 45
    def reset_states_45(self):
        self._states_45 = [None] * self.num_recurrent_units

    # 135
    def reset_states_135(self):
        self._states_135 = [None] * self.num_recurrent_units

    # 0
    def reset_states_0(self):
        self._states_0 = [None] * self.num_recurrent_units

    # i
    def reset_states_i(self):
        self._states_i = [None] * 3

    # a
    def reset_states_a(self):
        self._states_a = [None] * 3

    # d
    def reset_states_d(self):
        self._states_d = [None] * 3

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W event tensor
        :return: N x num_output_channels x H/2 x W/2 four intensity images
        """
        # split input
        x_90 = x[:, :, 0::2, 0::2]
        x_45 = x[:, :, 0::2, 1::2]
        x_135 = x[:, :, 1::2, 0::2]
        x_0 = x[:, :, 1::2, 1::2]

        # 90 branch
        x_90 = self.head_90(x_90)
        x_90 = self.G1_90(x_90, self._states_90[0])
        self._states_90[0] = x_90
        x_90 = self.R1_90(x_90)
        x_90 = self.G2_90(x_90, self._states_90[1])
        self._states_90[1] = x_90
        x_90 = self.R2_90(x_90)

        # 45 branch
        x_45 = self.head_45(x_45)
        x_45 = self.G1_45(x_45, self._states_45[0])
        self._states_45[0] = x_45
        x_45 = self.R1_45(x_45)
        x_45 = self.G2_45(x_45, self._states_45[1])
        self._states_45[1] = x_45
        x_45 = self.R2_45(x_45)

        # 135 branch
        x_135 = self.head_135(x_135)
        x_135 = self.G1_135(x_135, self._states_135[0])
        self._states_135[0] = x_135
        x_135 = self.R1_135(x_135)
        x_135 = self.G2_135(x_135, self._states_135[1])
        self._states_135[1] = x_135
        x_135 = self.R2_135(x_135)

        # 0 branch
        x_0 = self.head_0(x_0)
        x_0 = self.G1_0(x_0, self._states_0[0])
        self._states_0[0] = x_0
        x_0 = self.R1_0(x_0)
        x_0 = self.G2_0(x_0, self._states_0[1])
        self._states_0[1] = x_0
        x_0 = self.R2_0(x_0)

        x_iad = torch.cat([x_90, x_45, x_135, x_0], 1)

        # i branch
        x_i = self.G3_i(x_iad, self._states_i[0])
        self._states_i[0] = x_i
        x_i = self.R3_i(x_i)
        x_i = self.G4_i(x_i, self._states_i[1])
        self._states_i[1] = x_i
        x_i = self.R4_i(x_i)
        x_i = self.G5_i(x_i, self._states_i[2])
        self._states_i[2] = x_i
        x_i = self.R5_i(x_i)
        i = self.pred_i(x_i)

        # a branch
        x_a = self.G3_a(x_iad, self._states_a[0])
        self._states_a[0] = x_a
        x_a = self.R3_a(x_a)
        x_a = self.G4_a(x_a, self._states_a[1])
        self._states_a[1] = x_a
        x_a = self.R4_a(x_a)
        x_a = self.G5_a(x_a, self._states_a[2])
        self._states_a[2] = x_a
        x_a = self.R5_a(x_a)
        a = self.pred_a(x_a)

        # d branch
        x_d = self.G3_d(x_iad, self._states_d[0])
        self._states_d[0] = x_d
        x_d = self.R3_d(x_d)
        x_d = self.G4_d(x_d, self._states_d[1])
        self._states_d[1] = x_d
        x_d = self.R4_d(x_d)
        x_d = self.G5_d(x_d, self._states_d[2])
        self._states_d[2] = x_d
        x_d = self.R5_d(x_d)
        d = self.pred_d(x_d)

        return {
            'i': i,
            'a': a,
            'd': d,
        }


class M21(BaseModel):
    """
    Split input and use four separate branches to estimate four-direction intensities, then estimate polarization.
    There is no intermediate supervision.
    Output sigmoid.
    Using dolp to attention aolp.
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        padding = kernel_size // 2

        base_num_channels = base_num_channels // 4
        # 90
        self.head_90 = ConvLayer(self.num_bins, base_num_channels, kernel_size, padding=padding)
        self.G1_90 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_90 = ResidualBlock_SSC(base_num_channels)
        self.G2_90 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_90 = ResidualBlock_SSC(base_num_channels)
        # 45
        self.head_45 = ConvLayer(self.num_bins, base_num_channels, kernel_size, padding=padding)
        self.G1_45 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_45 = ResidualBlock_SSC(base_num_channels)
        self.G2_45 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_45 = ResidualBlock_SSC(base_num_channels)
        # 135
        self.head_135 = ConvLayer(self.num_bins, base_num_channels, kernel_size, padding=padding)
        self.G1_135 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_135 = ResidualBlock_SSC(base_num_channels)
        self.G2_135 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_135 = ResidualBlock_SSC(base_num_channels)
        # 0
        self.head_0 = ConvLayer(self.num_bins, base_num_channels, kernel_size, padding=padding)
        self.G1_0 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_0 = ResidualBlock_SSC(base_num_channels)
        self.G2_0 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_0 = ResidualBlock_SSC(base_num_channels)

        base_num_channels = base_num_channels * 4
        # i
        self.G3_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R3_i = ResidualBlock_SSC(base_num_channels)
        self.G4_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R4_i = ResidualBlock_SSC(base_num_channels)
        self.G5_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R5_i = ResidualBlock_SSC(base_num_channels)
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation='sigmoid')
        # a
        self.G3_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R3_a = ResidualBlock_SSC(base_num_channels)
        self.G4_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R4_a = ResidualBlock_SSC(base_num_channels)
        self.G5_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R5_a = ResidualBlock_SSC(base_num_channels)
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation='sigmoid')
        # d
        self.G3_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R3_d = ResidualBlock_SSC(base_num_channels)
        self.G4_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R4_d = ResidualBlock_SSC(base_num_channels)
        self.G5_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R5_d = ResidualBlock_SSC(base_num_channels)
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation='sigmoid')

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
        self.reset_states_90()
        self.reset_states_45()
        self.reset_states_135()
        self.reset_states_0()
        self.reset_states_i()
        self.reset_states_a()
        self.reset_states_d()

    # 90
    def reset_states_90(self):
        self._states_90 = [None] * self.num_recurrent_units

    # 45
    def reset_states_45(self):
        self._states_45 = [None] * self.num_recurrent_units

    # 135
    def reset_states_135(self):
        self._states_135 = [None] * self.num_recurrent_units

    # 0
    def reset_states_0(self):
        self._states_0 = [None] * self.num_recurrent_units

    # i
    def reset_states_i(self):
        self._states_i = [None] * 3

    # a
    def reset_states_a(self):
        self._states_a = [None] * 3

    # d
    def reset_states_d(self):
        self._states_d = [None] * 3

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W event tensor
        :return: N x num_output_channels x H/2 x W/2 four intensity images
        """
        # split input
        x_90 = x[:, :, 0::2, 0::2]
        x_45 = x[:, :, 0::2, 1::2]
        x_135 = x[:, :, 1::2, 0::2]
        x_0 = x[:, :, 1::2, 1::2]

        # 90 branch
        x_90 = self.head_90(x_90)
        x_90 = self.G1_90(x_90, self._states_90[0])
        self._states_90[0] = x_90
        x_90 = self.R1_90(x_90)
        x_90 = self.G2_90(x_90, self._states_90[1])
        self._states_90[1] = x_90
        x_90 = self.R2_90(x_90)

        # 45 branch
        x_45 = self.head_45(x_45)
        x_45 = self.G1_45(x_45, self._states_45[0])
        self._states_45[0] = x_45
        x_45 = self.R1_45(x_45)
        x_45 = self.G2_45(x_45, self._states_45[1])
        self._states_45[1] = x_45
        x_45 = self.R2_45(x_45)

        # 135 branch
        x_135 = self.head_135(x_135)
        x_135 = self.G1_135(x_135, self._states_135[0])
        self._states_135[0] = x_135
        x_135 = self.R1_135(x_135)
        x_135 = self.G2_135(x_135, self._states_135[1])
        self._states_135[1] = x_135
        x_135 = self.R2_135(x_135)

        # 0 branch
        x_0 = self.head_0(x_0)
        x_0 = self.G1_0(x_0, self._states_0[0])
        self._states_0[0] = x_0
        x_0 = self.R1_0(x_0)
        x_0 = self.G2_0(x_0, self._states_0[1])
        self._states_0[1] = x_0
        x_0 = self.R2_0(x_0)

        x_iad = torch.cat([x_90, x_45, x_135, x_0], 1)

        # i branch
        x_i = self.G3_i(x_iad, self._states_i[0])
        self._states_i[0] = x_i
        x_i = self.R3_i(x_i)
        x_i = self.G4_i(x_i, self._states_i[1])
        self._states_i[1] = x_i
        x_i = self.R4_i(x_i)
        x_i = self.G5_i(x_i, self._states_i[2])
        self._states_i[2] = x_i
        x_i = self.R5_i(x_i)
        i = self.pred_i(x_i)

        # d branch
        x_d = self.G3_d(x_iad, self._states_d[0])
        self._states_d[0] = x_d
        x_d = self.R3_d(x_d)
        x_d = self.G4_d(x_d, self._states_d[1])
        self._states_d[1] = x_d
        x_d = self.R4_d(x_d)
        x_d = self.G5_d(x_d, self._states_d[2])
        self._states_d[2] = x_d
        x_d = self.R5_d(x_d)
        d = self.pred_d(x_d)

        # a branch
        x_a = x_iad * d
        x_a = self.G3_a(x_a, self._states_a[0])
        self._states_a[0] = x_a
        x_a = self.R3_a(x_a)
        x_a = self.G4_a(x_a, self._states_a[1])
        self._states_a[1] = x_a
        x_a = self.R4_a(x_a)
        x_a = self.G5_a(x_a, self._states_a[2])
        self._states_a[2] = x_a
        x_a = self.R5_a(x_a)
        a = self.pred_a(x_a)

        return {
            'i': i,
            'a': a,
            'd': d,
        }


class M22(BaseModel):
    """
    Split input and use four separate branches to estimate four-direction intensities, then estimate polarization.
    There has intermediate supervision.
    Output sigmoid.
    Using dolp to attention aolp.
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        padding = kernel_size // 2

        base_num_channels = base_num_channels // 4
        # 90
        self.head_90 = ConvLayer(self.num_bins, base_num_channels, kernel_size, padding=padding)
        self.G1_90 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_90 = ResidualBlock_SSC(base_num_channels)
        self.G2_90 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_90 = ResidualBlock_SSC(base_num_channels)
        self.pred_90 = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        # 45
        self.head_45 = ConvLayer(self.num_bins, base_num_channels, kernel_size, padding=padding)
        self.G1_45 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_45 = ResidualBlock_SSC(base_num_channels)
        self.G2_45 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_45 = ResidualBlock_SSC(base_num_channels)
        self.pred_45 = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        # 135
        self.head_135 = ConvLayer(self.num_bins, base_num_channels, kernel_size, padding=padding)
        self.G1_135 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_135 = ResidualBlock_SSC(base_num_channels)
        self.G2_135 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_135 = ResidualBlock_SSC(base_num_channels)
        self.pred_135 = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        # 0
        self.head_0 = ConvLayer(self.num_bins, base_num_channels, kernel_size, padding=padding)
        self.G1_0 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_0 = ResidualBlock_SSC(base_num_channels)
        self.G2_0 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_0 = ResidualBlock_SSC(base_num_channels)
        self.pred_0 = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)

        base_num_channels = base_num_channels * 4
        # i
        self.G3_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R3_i = ResidualBlock_SSC(base_num_channels)
        self.G4_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R4_i = ResidualBlock_SSC(base_num_channels)
        self.G5_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R5_i = ResidualBlock_SSC(base_num_channels)
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        # a
        self.G3_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R3_a = ResidualBlock_SSC(base_num_channels)
        self.G4_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R4_a = ResidualBlock_SSC(base_num_channels)
        self.G5_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R5_a = ResidualBlock_SSC(base_num_channels)
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        # d
        self.G3_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R3_d = ResidualBlock_SSC(base_num_channels)
        self.G4_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R4_d = ResidualBlock_SSC(base_num_channels)
        self.G5_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R5_d = ResidualBlock_SSC(base_num_channels)
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation='sigmoid')

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
        self.reset_states_90()
        self.reset_states_45()
        self.reset_states_135()
        self.reset_states_0()
        self.reset_states_i()
        self.reset_states_a()
        self.reset_states_d()

    # 90
    def reset_states_90(self):
        self._states_90 = [None] * self.num_recurrent_units

    # 45
    def reset_states_45(self):
        self._states_45 = [None] * self.num_recurrent_units

    # 135
    def reset_states_135(self):
        self._states_135 = [None] * self.num_recurrent_units

    # 0
    def reset_states_0(self):
        self._states_0 = [None] * self.num_recurrent_units

    # i
    def reset_states_i(self):
        self._states_i = [None] * 3

    # a
    def reset_states_a(self):
        self._states_a = [None] * 3

    # d
    def reset_states_d(self):
        self._states_d = [None] * 3

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W event tensor
        :return: N x num_output_channels x H/2 x W/2 four intensity images
        """
        # split input
        x_90 = x[:, :, 0::2, 0::2]
        x_45 = x[:, :, 0::2, 1::2]
        x_135 = x[:, :, 1::2, 0::2]
        x_0 = x[:, :, 1::2, 1::2]

        # 90 branch
        x_90 = self.head_90(x_90)
        x_90 = self.G1_90(x_90, self._states_90[0])
        self._states_90[0] = x_90
        x_90 = self.R1_90(x_90)
        x_90 = self.G2_90(x_90, self._states_90[1])
        self._states_90[1] = x_90
        x_90 = self.R2_90(x_90)
        i_90 = self.pred_90(x_90)

        # 45 branch
        x_45 = self.head_45(x_45)
        x_45 = self.G1_45(x_45, self._states_45[0])
        self._states_45[0] = x_45
        x_45 = self.R1_45(x_45)
        x_45 = self.G2_45(x_45, self._states_45[1])
        self._states_45[1] = x_45
        x_45 = self.R2_45(x_45)
        i_45 = self.pred_45(x_45)

        # 135 branch
        x_135 = self.head_135(x_135)
        x_135 = self.G1_135(x_135, self._states_135[0])
        self._states_135[0] = x_135
        x_135 = self.R1_135(x_135)
        x_135 = self.G2_135(x_135, self._states_135[1])
        self._states_135[1] = x_135
        x_135 = self.R2_135(x_135)
        i_135 = self.pred_135(x_135)

        # 0 branch
        x_0 = self.head_0(x_0)
        x_0 = self.G1_0(x_0, self._states_0[0])
        self._states_0[0] = x_0
        x_0 = self.R1_0(x_0)
        x_0 = self.G2_0(x_0, self._states_0[1])
        self._states_0[1] = x_0
        x_0 = self.R2_0(x_0)
        i_0 = self.pred_0(x_0)

        x_iad = torch.cat([x_90, x_45, x_135, x_0], 1)

        # i branch
        x_i = self.G3_i(x_iad, self._states_i[0])
        self._states_i[0] = x_i
        x_i = self.R3_i(x_i)
        x_i = self.G4_i(x_i, self._states_i[1])
        self._states_i[1] = x_i
        x_i = self.R4_i(x_i)
        x_i = self.G5_i(x_i, self._states_i[2])
        self._states_i[2] = x_i
        x_i = self.R5_i(x_i)
        i = self.pred_i(x_i)

        # d branch
        x_d = self.G3_d(x_iad, self._states_d[0])
        self._states_d[0] = x_d
        x_d = self.R3_d(x_d)
        x_d = self.G4_d(x_d, self._states_d[1])
        self._states_d[1] = x_d
        x_d = self.R4_d(x_d)
        x_d = self.G5_d(x_d, self._states_d[2])
        self._states_d[2] = x_d
        x_d = self.R5_d(x_d)
        d = self.pred_d(x_d)

        # a branch
        x_a = x_iad * d
        x_a = self.G3_a(x_a, self._states_a[0])
        self._states_a[0] = x_a
        x_a = self.R3_a(x_a)
        x_a = self.G4_a(x_a, self._states_a[1])
        self._states_a[1] = x_a
        x_a = self.R4_a(x_a)
        x_a = self.G5_a(x_a, self._states_a[2])
        self._states_a[2] = x_a
        x_a = self.R5_a(x_a)
        a = self.pred_a(x_a)

        return {
            'i_90': i_90,
            'i_45': i_45,
            'i_135': i_135,
            'i_0': i_0,
            'i': i,
            'a': a,
            'd': d,
        }


class M24(BaseModel):
    """
    first conv is 2x2 with stride of 2, three branches to estimate iad respectively
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        padding = kernel_size // 2
        # i
        self.head_i = ConvLayer(self.num_bins, base_num_channels, kernel_size=2, stride=2, padding=0)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_i = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_i = ResidualBlock(base_num_channels, base_num_channels)
        self.G3_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R3_i = ResidualBlock(base_num_channels, base_num_channels)
        self.G4_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R4_i = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        # a
        self.head_a = ConvLayer(self.num_bins, base_num_channels, kernel_size=2, stride=2, padding=0)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_a = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_a = ResidualBlock(base_num_channels, base_num_channels)
        self.G3_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R3_a = ResidualBlock(base_num_channels, base_num_channels)
        self.G4_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R4_a = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        # d
        self.head_d = ConvLayer(self.num_bins, base_num_channels, kernel_size=2, stride=2, padding=0)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_d = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_d = ResidualBlock(base_num_channels, base_num_channels)
        self.G3_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R3_d = ResidualBlock(base_num_channels, base_num_channels)
        self.G4_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R4_d = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 4
        self.reset_states_i()
        self.reset_states_a()
        self.reset_states_d()

    # i
    def reset_states_i(self):
        self._states_i = [None] * self.num_recurrent_units

    # a
    def reset_states_a(self):
        self._states_a = [None] * self.num_recurrent_units

    # d
    def reset_states_d(self):
        self._states_d = [None] * self.num_recurrent_units

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W event tensor
        :return: N x num_output_channels x H/2 x W/2 iad
        """

        # i branch
        x_i = self.head_i(x)
        x_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_i
        x_i = self.R1_i(x_i)
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i
        x_i = self.R2_i(x_i)
        x_i = self.G3_i(x_i, self._states_i[2])
        self._states_i[2] = x_i
        x_i = self.R3_i(x_i)
        x_i = self.G4_i(x_i, self._states_i[3])
        self._states_i[3] = x_i
        x_i = self.R4_i(x_i)
        i = self.pred_i(x_i)

        # a branch
        x_a = self.head_a(x)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a
        x_a = self.R1_a(x_a)
        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a
        x_a = self.R2_a(x_a)
        x_a = self.G3_a(x_a, self._states_a[2])
        self._states_a[2] = x_a
        x_a = self.R3_a(x_a)
        x_a = self.G4_a(x_a, self._states_a[3])
        self._states_a[3] = x_a
        x_a = self.R4_a(x_a)
        a = self.pred_a(x_a)

        # d branch
        x_d = self.head_d(x)
        x_d = self.G1_d(x_d, self._states_d[0])
        self._states_d[0] = x_d
        x_d = self.R1_d(x_d)
        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d
        x_d = self.R2_d(x_d)
        x_d = self.G3_d(x_d, self._states_d[2])
        self._states_d[2] = x_d
        x_d = self.R3_d(x_d)
        x_d = self.G4_d(x_d, self._states_d[3])
        self._states_d[3] = x_d
        x_d = self.R4_d(x_d)
        d = self.pred_d(x_d)

        return {
            'i': i,
            'a': a,
            'd': d
        }


class M24_S(BaseModel):
    """
    first conv is 2x2 with stride of 2, three branches to estimate iad respectively
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        padding = kernel_size // 2
        # s0
        self.head_s0 = ConvLayer(self.num_bins, base_num_channels, kernel_size=2, stride=2, padding=0)
        self.G1_s0 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_s0 = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_s0 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_s0 = ResidualBlock(base_num_channels, base_num_channels)
        self.G3_s0 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R3_s0 = ResidualBlock(base_num_channels, base_num_channels)
        self.G4_s0 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R4_s0 = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_s0 = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        # s1
        self.head_s1 = ConvLayer(self.num_bins, base_num_channels, kernel_size=2, stride=2, padding=0)
        self.G1_s1 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_s1 = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_s1 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_s1 = ResidualBlock(base_num_channels, base_num_channels)
        self.G3_s1 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R3_s1 = ResidualBlock(base_num_channels, base_num_channels)
        self.G4_s1 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R4_s1 = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_s1 = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        # s2
        self.head_s2 = ConvLayer(self.num_bins, base_num_channels, kernel_size=2, stride=2, padding=0)
        self.G1_s2 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_s2 = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_s2 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_s2 = ResidualBlock(base_num_channels, base_num_channels)
        self.G3_s2 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R3_s2 = ResidualBlock(base_num_channels, base_num_channels)
        self.G4_s2 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R4_s2 = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_s2 = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 4
        self.reset_states_s0()
        self.reset_states_s1()
        self.reset_states_s2()

    # s0
    def reset_states_s0(self):
        self._states_s0 = [None] * self.num_recurrent_units

    # s1
    def reset_states_s1(self):
        self._states_s1 = [None] * self.num_recurrent_units

    # s2
    def reset_states_s2(self):
        self._states_s2 = [None] * self.num_recurrent_units

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W event tensor
        :return: N x num_output_channels x H/2 x W/2 s0 s1 s2
        """

        # s0 branch
        x_s0 = x.clone()
        x_s0[:, :, 0::2, 1::2] = 0
        x_s0[:, :, 1::2, 0::2] = 0
        x_s0 = self.head_s0(x_s0)
        x_s0 = self.G1_s0(x_s0, self._states_s0[0])
        self._states_s0[0] = x_s0
        x_s0 = self.R1_s0(x_s0)
        x_s0 = self.G2_s0(x_s0, self._states_s0[1])
        self._states_s0[1] = x_s0
        x_s0 = self.R2_s0(x_s0)
        x_s0 = self.G3_s0(x_s0, self._states_s0[2])
        self._states_s0[2] = x_s0
        x_s0 = self.R3_s0(x_s0)
        x_s0 = self.G4_s0(x_s0, self._states_s0[3])
        self._states_s0[3] = x_s0
        x_s0 = self.R4_s0(x_s0)
        s0 = self.pred_s0(x_s0)

        # s1 branch
        x_s1 = x.clone()
        x_s1[:, :, 0::2, 1::2] = 0
        x_s1[:, :, 1::2, 0::2] = 0
        x_s1 = self.head_s1(x_s1)
        x_s1 = self.G1_s1(x_s1, self._states_s1[0])
        self._states_s1[0] = x_s1
        x_s1 = self.R1_s1(x_s1)
        x_s1 = self.G2_s1(x_s1, self._states_s1[1])
        self._states_s1[1] = x_s1
        x_s1 = self.R2_s1(x_s1)
        x_s1 = self.G3_s1(x_s1, self._states_s1[2])
        self._states_s1[2] = x_s1
        x_s1 = self.R3_s1(x_s1)
        x_s1 = self.G4_s1(x_s1, self._states_s1[3])
        self._states_s1[3] = x_s1
        x_s1 = self.R4_s1(x_s1)
        s1 = self.pred_s1(x_s1)

        # s2 branch
        x_s2 = x.clone()
        x_s2[:, :, 0::2, 0::2] = 0
        x_s2[:, :, 1::2, 1::2] = 0
        x_s2 = self.head_s2(x_s2)
        x_s2 = self.G1_s2(x_s2, self._states_s2[0])
        self._states_s2[0] = x_s2
        x_s2 = self.R1_s2(x_s2)
        x_s2 = self.G2_s2(x_s2, self._states_s2[1])
        self._states_s2[1] = x_s2
        x_s2 = self.R2_s2(x_s2)
        x_s2 = self.G3_s2(x_s2, self._states_s2[2])
        self._states_s2[2] = x_s2
        x_s2 = self.R3_s2(x_s2)
        x_s2 = self.G4_s2(x_s2, self._states_s2[3])
        self._states_s2[3] = x_s2
        x_s2 = self.R4_s2(x_s2)
        s2 = self.pred_s2(x_s2)

        return {
            's0': s0,
            's1': s1,
            's2': s2
        }


class M25(BaseModel):
    """
    Split input and use four separate branches to estimate four-direction intensities, then estimate Stocks Parameters.
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        padding = kernel_size // 2
        # 90
        self.head_90 = ConvLayer(self.num_bins, base_num_channels, kernel_size, padding=padding)
        self.G1_90 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_90 = ResidualBlock_SSC(base_num_channels)
        self.G2_90 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_90 = ResidualBlock_SSC(base_num_channels)
        self.pred_90 = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        # 45
        self.head_45 = ConvLayer(self.num_bins, base_num_channels, kernel_size, padding=padding)
        self.G1_45 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_45 = ResidualBlock_SSC(base_num_channels)
        self.G2_45 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_45 = ResidualBlock_SSC(base_num_channels)
        self.pred_45 = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        # 135
        self.head_135 = ConvLayer(self.num_bins, base_num_channels, kernel_size, padding=padding)
        self.G1_135 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_135 = ResidualBlock_SSC(base_num_channels)
        self.G2_135 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_135 = ResidualBlock_SSC(base_num_channels)
        self.pred_135 = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        # 0
        self.head_0 = ConvLayer(self.num_bins, base_num_channels, kernel_size, padding=padding)
        self.G1_0 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_0 = ResidualBlock_SSC(base_num_channels)
        self.G2_0 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_0 = ResidualBlock_SSC(base_num_channels)
        self.pred_0 = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)

        # S0
        self.G3_s0 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R3_s0 = ResidualBlock_SSC(base_num_channels)
        self.G4_s0 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R4_s0 = ResidualBlock_SSC(base_num_channels)
        self.G5_s0 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R5_s0 = ResidualBlock_SSC(base_num_channels)
        self.pred_s0 = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        # S1
        self.G3_s1 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R3_s1 = ResidualBlock_SSC(base_num_channels)
        self.G4_s1 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R4_s1 = ResidualBlock_SSC(base_num_channels)
        self.G5_s1 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R5_s1 = ResidualBlock_SSC(base_num_channels)
        self.pred_s1 = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation='tanh')
        # S2
        self.G3_s2 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R3_s2 = ResidualBlock_SSC(base_num_channels)
        self.G4_s2 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R4_s2 = ResidualBlock_SSC(base_num_channels)
        self.G5_s2 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R5_s2 = ResidualBlock_SSC(base_num_channels)
        self.pred_s2 = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation='tanh')

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
        self.reset_states_90()
        self.reset_states_45()
        self.reset_states_135()
        self.reset_states_0()
        self.reset_states_s0()
        self.reset_states_s1()
        self.reset_states_s2()

    # 90
    def reset_states_90(self):
        self._states_90 = [None] * self.num_recurrent_units

    # 45
    def reset_states_45(self):
        self._states_45 = [None] * self.num_recurrent_units

    # 135
    def reset_states_135(self):
        self._states_135 = [None] * self.num_recurrent_units

    # 0
    def reset_states_0(self):
        self._states_0 = [None] * self.num_recurrent_units

    # s0
    def reset_states_s0(self):
        self._states_s0 = [None] * 3

    # s1
    def reset_states_s1(self):
        self._states_s1 = [None] * 3

    # s2
    def reset_states_s2(self):
        self._states_s2 = [None] * 3

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W event tensor
        :return: N x num_output_channels x H/2 x W/2 s0, s1, and s2
        """
        # split input
        x_90 = x[:, :, 0::2, 0::2]
        x_45 = x[:, :, 0::2, 1::2]
        x_135 = x[:, :, 1::2, 0::2]
        x_0 = x[:, :, 1::2, 1::2]

        # 90 branch
        x_90 = self.head_90(x_90)
        x_90 = self.G1_90(x_90, self._states_90[0])
        self._states_90[0] = x_90
        x_90 = self.R1_90(x_90)
        x_90 = self.G2_90(x_90, self._states_90[1])
        self._states_90[1] = x_90
        x_90 = self.R2_90(x_90)
        i_90 = self.pred_90(x_90)

        # 45 branch
        x_45 = self.head_45(x_45)
        x_45 = self.G1_45(x_45, self._states_45[0])
        self._states_45[0] = x_45
        x_45 = self.R1_45(x_45)
        x_45 = self.G2_45(x_45, self._states_45[1])
        self._states_45[1] = x_45
        x_45 = self.R2_45(x_45)
        i_45 = self.pred_45(x_45)

        # 135 branch
        x_135 = self.head_135(x_135)
        x_135 = self.G1_135(x_135, self._states_135[0])
        self._states_135[0] = x_135
        x_135 = self.R1_135(x_135)
        x_135 = self.G2_135(x_135, self._states_135[1])
        self._states_135[1] = x_135
        x_135 = self.R2_135(x_135)
        i_135 = self.pred_135(x_135)

        # 0 branch
        x_0 = self.head_0(x_0)
        x_0 = self.G1_0(x_0, self._states_0[0])
        self._states_0[0] = x_0
        x_0 = self.R1_0(x_0)
        x_0 = self.G2_0(x_0, self._states_0[1])
        self._states_0[1] = x_0
        x_0 = self.R2_0(x_0)
        i_0 = self.pred_0(x_0)

        # s0 branch
        x_s0 = (x_0 + x_90 + x_45 + x_135) / 2
        x_s0 = self.G3_s0(x_s0, self._states_s0[0])
        self._states_s0[0] = x_s0
        x_s0 = self.R3_s0(x_s0)
        x_s0 = self.G4_s0(x_s0, self._states_s0[1])
        self._states_s0[1] = x_s0
        x_s0 = self.R4_s0(x_s0)
        x_s0 = self.G5_s0(x_s0, self._states_s0[2])
        self._states_s0[2] = x_s0
        x_s0 = self.R5_s0(x_s0)
        s0 = self.pred_s0(x_s0)

        # s1 branch
        x_s1 = x_0 - x_90
        x_s1 = self.G3_s1(x_s1, self._states_s1[0])
        self._states_s1[0] = x_s1
        x_s1 = self.R3_s1(x_s1)
        x_s1 = self.G4_s1(x_s1, self._states_s1[1])
        self._states_s1[1] = x_s1
        x_s1 = self.R4_s1(x_s1)
        x_s1 = self.G5_s1(x_s1, self._states_s1[2])
        self._states_s1[2] = x_s1
        x_s1 = self.R5_s1(x_s1)
        s1 = self.pred_s1(x_s1)

        # s2 branch
        x_s2 = x_45 - x_135
        x_s2 = self.G3_s2(x_s2, self._states_s2[0])
        self._states_s2[0] = x_s2
        x_s2 = self.R3_s2(x_s2)
        x_s2 = self.G4_s2(x_s2, self._states_s2[1])
        self._states_s2[1] = x_s2
        x_s2 = self.R4_s2(x_s2)
        x_s2 = self.G5_s2(x_s2, self._states_s2[2])
        self._states_s2[2] = x_s2
        x_s2 = self.R5_s2(x_s2)
        s2 = self.pred_s2(x_s2)

        return {
            'i_90': i_90,
            'i_45': i_45,
            'i_135': i_135,
            'i_0': i_0,
            's0': s0,
            's1': s1,
            's2': s2,
        }


class M26(BaseModel):
    """
    Split input and use four separate branches to estimate four-direction intensities, then estimate Stocks Parameters,
    and finally predict intensity, aolp, and dolp.
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        padding = kernel_size // 2
        # 90
        self.head_90 = ConvLayer(self.num_bins, base_num_channels, kernel_size, padding=padding)
        self.G1_90 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_90 = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_90 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_90 = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_90 = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        # 45
        self.head_45 = ConvLayer(self.num_bins, base_num_channels, kernel_size, padding=padding)
        self.G1_45 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_45 = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_45 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_45 = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_45 = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        # 135
        self.head_135 = ConvLayer(self.num_bins, base_num_channels, kernel_size, padding=padding)
        self.G1_135 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_135 = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_135 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_135 = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_135 = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        # 0
        self.head_0 = ConvLayer(self.num_bins, base_num_channels, kernel_size, padding=padding)
        self.G1_0 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_0 = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_0 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_0 = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_0 = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)

        # S0
        self.G3_s0 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R3_s0 = ResidualBlock(base_num_channels, base_num_channels)
        self.G4_s0 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R4_s0 = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_s0 = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        # S1
        self.G3_s1 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R3_s1 = ResidualBlock(base_num_channels, base_num_channels)
        self.G4_s1 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R4_s1 = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_s1 = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation='tanh')
        # S2
        self.G3_s2 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R3_s2 = ResidualBlock(base_num_channels, base_num_channels)
        self.G4_s2 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R4_s2 = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_s2 = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation='tanh')

        # Intensity
        self.G5_intensity = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R5_intensity = ResidualBlock(base_num_channels, base_num_channels)
        self.G6_intensity = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R6_intensity = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_intensity = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        # AoLP
        self.G5_aolp = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R5_aolp = ResidualBlock(base_num_channels, base_num_channels)
        self.G6_aolp = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R6_aolp = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_aolp = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation='tanh')
        # DoLP
        self.G5_dolp = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R5_dolp = ResidualBlock(base_num_channels, base_num_channels)
        self.G6_dolp = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R6_dolp = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_dolp = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation='sigmoid')

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
        self.reset_states_90()
        self.reset_states_45()
        self.reset_states_135()
        self.reset_states_0()
        self.reset_states_s0()
        self.reset_states_s1()
        self.reset_states_s2()
        self.reset_states_i()
        self.reset_states_a()
        self.reset_states_d()

    # 90
    def reset_states_90(self):
        self._states_90 = [None] * self.num_recurrent_units

    # 45
    def reset_states_45(self):
        self._states_45 = [None] * self.num_recurrent_units

    # 135
    def reset_states_135(self):
        self._states_135 = [None] * self.num_recurrent_units

    # 0
    def reset_states_0(self):
        self._states_0 = [None] * self.num_recurrent_units

    # s0
    def reset_states_s0(self):
        self._states_s0 = [None] * self.num_recurrent_units

    # s1
    def reset_states_s1(self):
        self._states_s1 = [None] * self.num_recurrent_units

    # s2
    def reset_states_s2(self):
        self._states_s2 = [None] * self.num_recurrent_units

    # intensity
    def reset_states_i(self):
        self._states_i = [None] * self.num_recurrent_units

    # aolp
    def reset_states_a(self):
        self._states_a = [None] * self.num_recurrent_units

    # dolp
    def reset_states_d(self):
        self._states_d = [None] * self.num_recurrent_units

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W event tensor
        :return: N x num_output_channels x H/2 x W/2 output 90, 45, 135, 0, s0, s1, s2, i, a, d
        """
        # split input
        x_90 = x[:, :, 0::2, 0::2]
        x_45 = x[:, :, 0::2, 1::2]
        x_135 = x[:, :, 1::2, 0::2]
        x_0 = x[:, :, 1::2, 1::2]

        # 90 branch
        x_90 = self.head_90(x_90)
        x_90 = self.G1_90(x_90, self._states_90[0])
        self._states_90[0] = x_90
        x_90 = self.R1_90(x_90)
        x_90 = self.G2_90(x_90, self._states_90[1])
        self._states_90[1] = x_90
        x_90 = self.R2_90(x_90)
        i_90 = self.pred_90(x_90)
        # 45 branch
        x_45 = self.head_45(x_45)
        x_45 = self.G1_45(x_45, self._states_45[0])
        self._states_45[0] = x_45
        x_45 = self.R1_45(x_45)
        x_45 = self.G2_45(x_45, self._states_45[1])
        self._states_45[1] = x_45
        x_45 = self.R2_45(x_45)
        i_45 = self.pred_45(x_45)
        # 135 branch
        x_135 = self.head_135(x_135)
        x_135 = self.G1_135(x_135, self._states_135[0])
        self._states_135[0] = x_135
        x_135 = self.R1_135(x_135)
        x_135 = self.G2_135(x_135, self._states_135[1])
        self._states_135[1] = x_135
        x_135 = self.R2_135(x_135)
        i_135 = self.pred_135(x_135)
        # 0 branch
        x_0 = self.head_0(x_0)
        x_0 = self.G1_0(x_0, self._states_0[0])
        self._states_0[0] = x_0
        x_0 = self.R1_0(x_0)
        x_0 = self.G2_0(x_0, self._states_0[1])
        self._states_0[1] = x_0
        x_0 = self.R2_0(x_0)
        i_0 = self.pred_0(x_0)

        # s0 branch
        x_s0 = (x_0 + x_90 + x_45 + x_135) / 2
        x_s0 = self.G3_s0(x_s0, self._states_s0[0])
        self._states_s0[0] = x_s0
        x_s0 = self.R3_s0(x_s0)
        x_s0 = self.G4_s0(x_s0, self._states_s0[1])
        self._states_s0[1] = x_s0
        x_s0 = self.R4_s0(x_s0)
        s0 = self.pred_s0(x_s0)
        # s1 branch
        x_s1 = x_0 - x_90
        x_s1 = self.G3_s1(x_s1, self._states_s1[0])
        self._states_s1[0] = x_s1
        x_s1 = self.R3_s1(x_s1)
        x_s1 = self.G4_s1(x_s1, self._states_s1[1])
        self._states_s1[1] = x_s1
        x_s1 = self.R4_s1(x_s1)
        s1 = self.pred_s1(x_s1)
        # s2 branch
        x_s2 = x_45 - x_135
        x_s2 = self.G3_s2(x_s2, self._states_s2[0])
        self._states_s2[0] = x_s2
        x_s2 = self.R3_s2(x_s2)
        x_s2 = self.G4_s2(x_s2, self._states_s2[1])
        self._states_s2[1] = x_s2
        x_s2 = self.R4_s2(x_s2)
        s2 = self.pred_s2(x_s2)

        # intensity branch
        x_intensity = x_s0 / 2
        x_intensity = self.G5_intensity(x_intensity, self._states_i[0])
        self._states_i[0] = x_intensity
        x_intensity = self.R5_intensity(x_intensity)
        x_intensity = self.G6_intensity(x_intensity, self._states_i[1])
        self._states_i[1] = x_intensity
        x_intensity = self.R6_intensity(x_intensity)
        intensity = self.pred_intensity(x_intensity)
        # aolp branch
        x_aolp = 0.5 * torch.arctan2(x_s2, x_s1)
        x_aolp = self.G5_aolp(x_aolp, self._states_a[0])
        self._states_a[0] = x_aolp
        x_aolp = self.R5_aolp(x_aolp)
        x_aolp = self.G6_aolp(x_aolp, self._states_a[1])
        self._states_a[1] = x_aolp
        x_aolp = self.R6_aolp(x_aolp)
        aolp = self.pred_aolp(x_aolp)
        # dolp branch
        x_dolp = torch.full_like(x_s0, fill_value=0)
        mask = (x_s0 != 0)
        x_dolp[mask] = torch.div(torch.sqrt(torch.square(x_s1[mask]) + torch.square(x_s2[mask])), x_s0[mask])
        x_dolp = self.G5_dolp(x_dolp, self._states_d[0])
        self._states_d[0] = x_dolp
        x_dolp = self.R5_dolp(x_dolp)
        x_dolp = self.G6_dolp(x_dolp, self._states_d[1])
        self._states_d[1] = x_dolp
        x_dolp = self.R6_dolp(x_dolp)
        dolp = self.pred_dolp(x_dolp)

        return {
            'i_90': i_90,
            'i_45': i_45,
            'i_135': i_135,
            'i_0': i_0,
            's0': s0,
            's1': s1,
            's2': s2,
            'i': intensity,
            'a': aolp,
            'd': dolp
        }


class M27(BaseModel):
    """
    Split input and use three separate branches to estimate iad respectively.
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        padding = kernel_size // 2
        # i
        self.head_i = ConvLayer(self.num_bins * 4, base_num_channels, kernel_size, padding=padding)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_i = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_i = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        # a
        self.head_a = ConvLayer(self.num_bins * 4, base_num_channels, kernel_size, padding=padding)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_a = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_a = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        # d
        self.head_d = ConvLayer(self.num_bins * 4, base_num_channels, kernel_size, padding=padding)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_d = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_d = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
        """
        # split input
        # x_90 = x[:, :, 0::2, 0::2]
        # x_45 = x[:, :, 0::2, 1::2]
        # x_135 = x[:, :, 1::2, 0::2]
        # x_0 = x[:, :, 1::2, 1::2]
        # for real data test
        x_135 = x[:, :, 0::2, 0::2]
        x_0 = x[:, :, 0::2, 1::2]
        x_90 = x[:, :, 1::2, 0::2]
        x_45 = x[:, :, 1::2, 1::2]

        x_four = torch.cat([x_90, x_45, x_135, x_0], 1)

        # i branch
        x_i = self.head_i(x_four)
        x_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_i
        x_i = self.R1_i(x_i)
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i
        x_i = self.R2_i(x_i)
        i = self.pred_i(x_i)

        # a branch
        x_a = self.head_a(x_four)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a
        x_a = self.R1_a(x_a)
        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a
        x_a = self.R2_a(x_a)
        a = self.pred_a(x_a)

        # d branch
        x_d = self.head_d(x_four)
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


class M28(BaseModel):
    """
    Using FireNet to predict aolp or dolp.
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        padding = kernel_size // 2
        self.head = ConvLayer(self.num_bins * 4, base_num_channels, kernel_size, padding=padding)
        self.G1 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1 = ResidualBlock(base_num_channels, base_num_channels)
        self.G2 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2 = ResidualBlock(base_num_channels, base_num_channels)
        self.pred = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
        self.reset_states()

    @property
    def states(self):
        return copy_states(self._states)

    @states.setter
    def states(self, states):
        self._states = states

    def reset_states(self):
        self._states = [None] * self.num_recurrent_units

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W event tensor
        :return: N x num_output_channels x H/2 x W/2 aolp
        """
        # split input
        x_90 = x[:, :, 0::2, 0::2]
        x_45 = x[:, :, 0::2, 1::2]
        x_135 = x[:, :, 1::2, 0::2]
        x_0 = x[:, :, 1::2, 1::2]

        x_four = torch.cat([x_90, x_45, x_135, x_0], 1)

        x = self.head(x_four)
        x = self.G1(x, self._states[0])
        self._states[0] = x
        x = self.R1(x)
        x = self.G2(x, self._states[1])
        self._states[1] = x
        x = self.R2(x)
        return {'image': self.pred(x)}


class M30(BaseModel):
    """
    Use three separate branches to estimate iad respectively.
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        padding = kernel_size // 2
        # i
        self.head_i = ConvLayer(self.num_bins, base_num_channels, kernel_size=2, stride=2, padding=0)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_i = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_i = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        # a
        self.head_a = ConvLayer(self.num_bins, base_num_channels, kernel_size=2, stride=2, padding=0)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_a = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_a = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        # d
        self.head_d = ConvLayer(self.num_bins, base_num_channels, kernel_size=2, stride=2, padding=0)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_d = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_d = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
        """
        # i branch
        x_i = self.head_i(x)
        x_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_i
        x_i = self.R1_i(x_i)
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i
        x_i = self.R2_i(x_i)
        i = self.pred_i(x_i)

        # a branch
        x_a = self.head_a(x)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a
        x_a = self.R1_a(x_a)
        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a
        x_a = self.R2_a(x_a)
        a = self.pred_a(x_a)

        # d branch
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


class M31(BaseModel):
    """
    Split input and use three separate encoder-decoder to estimate iad respectively.
    """
    def __init__(self, num_bins=5, base_num_channels=64, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        padding = kernel_size // 2
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.channel_1 = base_num_channels
        self.channel_2 = base_num_channels * 2
        self.channel_4 = base_num_channels * 4
        self.channel_8 = base_num_channels * 8
        # i
        self.head_i = ConvLayer(self.num_bins * 4, self.channel_1, kernel_size, padding=padding, norm='IN')
        self.C1_i = ConvLayer(self.channel_1, self.channel_2, kernel_size=4, stride=2, padding=1, norm='IN')
        self.G1_i = ConvGRU(self.channel_2, self.channel_2, kernel_size)
        self.C2_i = ConvLayer(self.channel_2, self.channel_4, kernel_size=4, stride=2, padding=1, norm='IN')
        self.G2_i = ConvGRU(self.channel_4, self.channel_4, kernel_size)
        self.C3_i = ConvLayer(self.channel_4, self.channel_8, kernel_size=4, stride=2, padding=1, norm='IN')
        self.G3_i = ConvGRU(self.channel_8, self.channel_8, kernel_size)
        self.C4_i = ResidualBlock(self.channel_8, self.channel_8)
        self.G4_i = ConvGRU(self.channel_8, self.channel_8, kernel_size)
        self.C5_i = ConvLayer(self.channel_8, self.channel_4, kernel_size=3, stride=1, padding=1, norm='IN')
        self.G5_i = ConvGRU(self.channel_4, self.channel_4, kernel_size)
        self.C6_i = ConvLayer(self.channel_4, self.channel_2, kernel_size=3, stride=1, padding=1, norm='IN')
        self.G6_i = ConvGRU(self.channel_2, self.channel_2, kernel_size)
        self.C7_i = ConvLayer(self.channel_2, self.channel_1, kernel_size=3, stride=1, padding=1, norm='IN')
        self.pred_i = ConvLayer(self.channel_1, out_channels=1, kernel_size=1, activation=None)
        # a
        self.head_a = ConvLayer(self.num_bins * 4, self.channel_1, kernel_size, padding=padding, norm='IN')
        self.C1_a = ConvLayer(self.channel_1, self.channel_2, kernel_size=4, stride=2, padding=1, norm='IN')
        self.G1_a = ConvGRU(self.channel_2, self.channel_2, kernel_size)
        self.C2_a = ConvLayer(self.channel_2, self.channel_4, kernel_size=4, stride=2, padding=1, norm='IN')
        self.G2_a = ConvGRU(self.channel_4, self.channel_4, kernel_size)
        self.C3_a = ConvLayer(self.channel_4, self.channel_8, kernel_size=4, stride=2, padding=1, norm='IN')
        self.G3_a = ConvGRU(self.channel_8, self.channel_8, kernel_size)
        self.C4_a = ResidualBlock(self.channel_8, self.channel_8)
        self.G4_a = ConvGRU(self.channel_8, self.channel_8, kernel_size)
        self.C5_a = ConvLayer(self.channel_8, self.channel_4, kernel_size=3, stride=1, padding=1, norm='IN')
        self.G5_a = ConvGRU(self.channel_4, self.channel_4, kernel_size)
        self.C6_a = ConvLayer(self.channel_4, self.channel_2, kernel_size=3, stride=1, padding=1, norm='IN')
        self.G6_a = ConvGRU(self.channel_2, self.channel_2, kernel_size)
        self.C7_a = ConvLayer(self.channel_2, self.channel_1, kernel_size=3, stride=1, padding=1, norm='IN')
        self.pred_a = ConvLayer(self.channel_1, out_channels=1, kernel_size=1, activation=None)
        # d
        self.head_d = ConvLayer(self.num_bins * 4, self.channel_1, kernel_size, padding=padding, norm='IN')
        self.C1_d = ConvLayer(self.channel_1, self.channel_2, kernel_size=4, stride=2, padding=1, norm='IN')
        self.G1_d = ConvGRU(self.channel_2, self.channel_2, kernel_size)
        self.C2_d = ConvLayer(self.channel_2, self.channel_4, kernel_size=4, stride=2, padding=1, norm='IN')
        self.G2_d = ConvGRU(self.channel_4, self.channel_4, kernel_size)
        self.C3_d = ConvLayer(self.channel_4, self.channel_8, kernel_size=4, stride=2, padding=1, norm='IN')
        self.G3_d = ConvGRU(self.channel_8, self.channel_8, kernel_size)
        self.C4_d = ResidualBlock(self.channel_8, self.channel_8)
        self.G4_d = ConvGRU(self.channel_8, self.channel_8, kernel_size)
        self.C5_d = ConvLayer(self.channel_8, self.channel_4, kernel_size=3, stride=1, padding=1, norm='IN')
        self.G5_d = ConvGRU(self.channel_4, self.channel_4, kernel_size)
        self.C6_d = ConvLayer(self.channel_4, self.channel_2, kernel_size=3, stride=1, padding=1, norm='IN')
        self.G6_d = ConvGRU(self.channel_2, self.channel_2, kernel_size)
        self.C7_d = ConvLayer(self.channel_2, self.channel_1, kernel_size=3, stride=1, padding=1, norm='IN')
        self.pred_d = ConvLayer(self.channel_1, out_channels=1, kernel_size=1, activation=None)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
        self.reset_states_i2()
        self.reset_states_i4()
        self.reset_states_i8()
        self.reset_states_a2()
        self.reset_states_a4()
        self.reset_states_a8()
        self.reset_states_d2()
        self.reset_states_d4()
        self.reset_states_d8()

    def reset_states_i2(self):
        self._states_i2 = [None] * self.num_recurrent_units

    def reset_states_i4(self):
        self._states_i4 = [None] * self.num_recurrent_units

    def reset_states_i8(self):
        self._states_i8 = [None] * self.num_recurrent_units

    def reset_states_a2(self):
        self._states_a2 = [None] * self.num_recurrent_units

    def reset_states_a4(self):
        self._states_a4 = [None] * self.num_recurrent_units

    def reset_states_a8(self):
        self._states_a8 = [None] * self.num_recurrent_units

    def reset_states_d2(self):
        self._states_d2 = [None] * self.num_recurrent_units

    def reset_states_d4(self):
        self._states_d4 = [None] * self.num_recurrent_units

    def reset_states_d8(self):
        self._states_d8 = [None] * self.num_recurrent_units

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W event tensor
        :return: N x num_output_channels x H/2 x W/2 iad
        """
        # split input
        x_90 = x[:, :, 0::2, 0::2]
        x_45 = x[:, :, 0::2, 1::2]
        x_135 = x[:, :, 1::2, 0::2]
        x_0 = x[:, :, 1::2, 1::2]

        x_four = torch.cat([x_90, x_45, x_135, x_0], 1)
        _, _, h_original, w_original = x_four.size()
        h = (h_original // 8) * 8
        w = (w_original // 8) * 8
        x_four = nn.UpsamplingBilinear2d((h, w))(x_four)

        # i branch
        x_i = self.head_i(x_four)
        skip_i1 = x_i

        x_i = self.C1_i(x_i)
        x_i = self.G1_i(x_i, self._states_i2[0])
        self._states_i2[0] = x_i
        skip_i2 = x_i

        x_i = self.C2_i(x_i)
        x_i = self.G2_i(x_i, self._states_i4[0])
        self._states_i4[0] = x_i
        skip_i3 = x_i

        x_i = self.C3_i(x_i)
        x_i = self.G3_i(x_i, self._states_i8[0])
        self._states_i8[0] = x_i

        x_i = self.C4_i(x_i)
        x_i = self.G4_i(x_i, self._states_i8[1])
        self._states_i8[1] = x_i

        x_i = self.up(self.C5_i(x_i)) + skip_i3
        x_i = self.G5_i(x_i, self._states_i4[1])
        self._states_i4[1] = x_i

        x_i = self.up(self.C6_i(x_i)) + skip_i2
        x_i = self.G6_i(x_i, self._states_i2[1])
        self._states_i2[1] = x_i

        x_i = self.up(self.C7_i(x_i)) + skip_i1
        i = self.pred_i(x_i)

        # a branch
        x_a = self.head_a(x_four)
        skip_a1 = x_a

        x_a = self.C1_a(x_a)
        x_a = self.G1_a(x_a, self._states_a2[0])
        self._states_a2[0] = x_a
        skip_a2 = x_a

        x_a = self.C2_a(x_a)
        x_a = self.G2_a(x_a, self._states_a4[0])
        self._states_a4[0] = x_a
        skip_a3 = x_a

        x_a = self.C3_a(x_a)
        x_a = self.G3_a(x_a, self._states_a8[0])
        self._states_a8[0] = x_a

        x_a = self.C4_a(x_a)
        x_a = self.G4_a(x_a, self._states_a8[1])
        self._states_a8[1] = x_a

        x_a = self.up(self.C5_a(x_a)) + skip_a3
        x_a = self.G5_a(x_a, self._states_a4[1])
        self._states_a4[1] = x_a

        x_a = self.up(self.C6_a(x_a)) + skip_a2
        x_a = self.G6_a(x_a, self._states_a2[1])
        self._states_a2[1] = x_a

        x_a = self.up(self.C7_a(x_a)) + skip_a1
        a = self.pred_a(x_a)

        # d branch
        x_d = self.head_d(x_four)
        skip_d1 = x_d

        x_d = self.C1_d(x_d)
        x_d = self.G1_d(x_d, self._states_d2[0])
        self._states_d2[0] = x_d
        skip_d2 = x_d

        x_d = self.C2_d(x_d)
        x_d = self.G2_d(x_d, self._states_d4[0])
        self._states_d4[0] = x_d
        skip_d3 = x_d

        x_d = self.C3_d(x_d)
        x_d = self.G3_d(x_d, self._states_d8[0])
        self._states_d8[0] = x_d

        x_d = self.C4_d(x_d)
        x_d = self.G4_d(x_d, self._states_d8[1])
        self._states_d8[1] = x_d

        x_d = self.up(self.C5_d(x_d)) + skip_d3
        x_d = self.G5_d(x_d, self._states_d4[1])
        self._states_d4[1] = x_d

        x_d = self.up(self.C6_d(x_d)) + skip_d2
        x_d = self.G6_d(x_d, self._states_d2[1])
        self._states_d2[1] = x_d

        x_d = self.up(self.C7_d(x_d)) + skip_d1
        d = self.pred_d(x_d)

        i = F.interpolate(i, size=(h_original, w_original), mode='bilinear', align_corners=True)
        a = F.interpolate(a, size=(h_original, w_original), mode='bilinear', align_corners=True)
        d = F.interpolate(d, size=(h_original, w_original), mode='bilinear', align_corners=True)

        return {
            'i': i,
            'a': a,
            'd': d
        }


class M32(BaseModel):
    """
    Split input and use three separate encoder-decoder to estimate iad respectively.
    Fusing iad features in the deepest layer.
    """
    def __init__(self, num_bins=5, base_num_channels=64, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        padding = kernel_size // 2
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.channel_1 = base_num_channels
        self.channel_2 = base_num_channels * 2
        self.channel_4 = base_num_channels * 4
        self.channel_8 = base_num_channels * 8
        # i
        self.head_i = ConvLayer(self.num_bins * 4, self.channel_1, kernel_size, padding=padding, norm='IN')
        self.C1_i = ConvLayer(self.channel_1, self.channel_2, kernel_size=4, stride=2, padding=1, norm='IN')
        self.G1_i = ConvGRU(self.channel_2, self.channel_2, kernel_size)
        self.C2_i = ConvLayer(self.channel_2, self.channel_4, kernel_size=4, stride=2, padding=1, norm='IN')
        self.G2_i = ConvGRU(self.channel_4, self.channel_4, kernel_size)
        self.C3_i = ConvLayer(self.channel_4, self.channel_8, kernel_size=4, stride=2, padding=1, norm='IN')
        self.G3_i = ConvGRU(self.channel_8, self.channel_8, kernel_size)
        self.F_i = ConvLayer(self.channel_8 * 3, self.channel_8, kernel_size=1, stride=1, padding=0, norm='IN')
        self.C4_i = ResidualBlock(self.channel_8, self.channel_8)
        self.G4_i = ConvGRU(self.channel_8, self.channel_8, kernel_size)
        self.C5_i = ConvLayer(self.channel_8, self.channel_4, kernel_size=3, stride=1, padding=1, norm='IN')
        self.G5_i = ConvGRU(self.channel_4, self.channel_4, kernel_size)
        self.C6_i = ConvLayer(self.channel_4, self.channel_2, kernel_size=3, stride=1, padding=1, norm='IN')
        self.G6_i = ConvGRU(self.channel_2, self.channel_2, kernel_size)
        self.C7_i = ConvLayer(self.channel_2, self.channel_1, kernel_size=3, stride=1, padding=1, norm='IN')
        self.pred_i = ConvLayer(self.channel_1, out_channels=1, kernel_size=1, activation=None)
        # a
        self.head_a = ConvLayer(self.num_bins * 4, self.channel_1, kernel_size, padding=padding, norm='IN')
        self.C1_a = ConvLayer(self.channel_1, self.channel_2, kernel_size=4, stride=2, padding=1, norm='IN')
        self.G1_a = ConvGRU(self.channel_2, self.channel_2, kernel_size)
        self.C2_a = ConvLayer(self.channel_2, self.channel_4, kernel_size=4, stride=2, padding=1, norm='IN')
        self.G2_a = ConvGRU(self.channel_4, self.channel_4, kernel_size)
        self.C3_a = ConvLayer(self.channel_4, self.channel_8, kernel_size=4, stride=2, padding=1, norm='IN')
        self.G3_a = ConvGRU(self.channel_8, self.channel_8, kernel_size)
        self.F_a = ConvLayer(self.channel_8 * 3, self.channel_8, kernel_size=1, stride=1, padding=0, norm='IN')
        self.C4_a = ResidualBlock(self.channel_8, self.channel_8)
        self.G4_a = ConvGRU(self.channel_8, self.channel_8, kernel_size)
        self.C5_a = ConvLayer(self.channel_8, self.channel_4, kernel_size=3, stride=1, padding=1, norm='IN')
        self.G5_a = ConvGRU(self.channel_4, self.channel_4, kernel_size)
        self.C6_a = ConvLayer(self.channel_4, self.channel_2, kernel_size=3, stride=1, padding=1, norm='IN')
        self.G6_a = ConvGRU(self.channel_2, self.channel_2, kernel_size)
        self.C7_a = ConvLayer(self.channel_2, self.channel_1, kernel_size=3, stride=1, padding=1, norm='IN')
        self.pred_a = ConvLayer(self.channel_1, out_channels=1, kernel_size=1, activation=None)
        # d
        self.head_d = ConvLayer(self.num_bins * 4, self.channel_1, kernel_size, padding=padding, norm='IN')
        self.C1_d = ConvLayer(self.channel_1, self.channel_2, kernel_size=4, stride=2, padding=1, norm='IN')
        self.G1_d = ConvGRU(self.channel_2, self.channel_2, kernel_size)
        self.C2_d = ConvLayer(self.channel_2, self.channel_4, kernel_size=4, stride=2, padding=1, norm='IN')
        self.G2_d = ConvGRU(self.channel_4, self.channel_4, kernel_size)
        self.C3_d = ConvLayer(self.channel_4, self.channel_8, kernel_size=4, stride=2, padding=1, norm='IN')
        self.G3_d = ConvGRU(self.channel_8, self.channel_8, kernel_size)
        self.F_d = ConvLayer(self.channel_8 * 3, self.channel_8, kernel_size=1, stride=1, padding=0, norm='IN')
        self.C4_d = ResidualBlock(self.channel_8, self.channel_8)
        self.G4_d = ConvGRU(self.channel_8, self.channel_8, kernel_size)
        self.C5_d = ConvLayer(self.channel_8, self.channel_4, kernel_size=3, stride=1, padding=1, norm='IN')
        self.G5_d = ConvGRU(self.channel_4, self.channel_4, kernel_size)
        self.C6_d = ConvLayer(self.channel_4, self.channel_2, kernel_size=3, stride=1, padding=1, norm='IN')
        self.G6_d = ConvGRU(self.channel_2, self.channel_2, kernel_size)
        self.C7_d = ConvLayer(self.channel_2, self.channel_1, kernel_size=3, stride=1, padding=1, norm='IN')
        self.pred_d = ConvLayer(self.channel_1, out_channels=1, kernel_size=1, activation=None)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
        self.reset_states_i2()
        self.reset_states_i4()
        self.reset_states_i8()
        self.reset_states_a2()
        self.reset_states_a4()
        self.reset_states_a8()
        self.reset_states_d2()
        self.reset_states_d4()
        self.reset_states_d8()

    def reset_states_i2(self):
        self._states_i2 = [None] * self.num_recurrent_units

    def reset_states_i4(self):
        self._states_i4 = [None] * self.num_recurrent_units

    def reset_states_i8(self):
        self._states_i8 = [None] * self.num_recurrent_units

    def reset_states_a2(self):
        self._states_a2 = [None] * self.num_recurrent_units

    def reset_states_a4(self):
        self._states_a4 = [None] * self.num_recurrent_units

    def reset_states_a8(self):
        self._states_a8 = [None] * self.num_recurrent_units

    def reset_states_d2(self):
        self._states_d2 = [None] * self.num_recurrent_units

    def reset_states_d4(self):
        self._states_d4 = [None] * self.num_recurrent_units

    def reset_states_d8(self):
        self._states_d8 = [None] * self.num_recurrent_units

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W event tensor
        :return: N x num_output_channels x H/2 x W/2 iad
        """
        # split input
        x_90 = x[:, :, 0::2, 0::2]
        x_45 = x[:, :, 0::2, 1::2]
        x_135 = x[:, :, 1::2, 0::2]
        x_0 = x[:, :, 1::2, 1::2]

        x_four = torch.cat([x_90, x_45, x_135, x_0], 1)
        _, _, h_original, w_original = x_four.size()
        h = (h_original // 8) * 8
        w = (w_original // 8) * 8
        x_four = nn.UpsamplingBilinear2d((h, w))(x_four)

        # Encoder
        # i branch
        x_i = self.head_i(x_four)
        skip_i1 = x_i

        x_i = self.C1_i(x_i)
        x_i = self.G1_i(x_i, self._states_i2[0])
        self._states_i2[0] = x_i
        skip_i2 = x_i

        x_i = self.C2_i(x_i)
        x_i = self.G2_i(x_i, self._states_i4[0])
        self._states_i4[0] = x_i
        skip_i3 = x_i

        x_i = self.C3_i(x_i)
        x_i = self.G3_i(x_i, self._states_i8[0])
        self._states_i8[0] = x_i

        # a branch
        x_a = self.head_a(x_four)
        skip_a1 = x_a

        x_a = self.C1_a(x_a)
        x_a = self.G1_a(x_a, self._states_a2[0])
        self._states_a2[0] = x_a
        skip_a2 = x_a

        x_a = self.C2_a(x_a)
        x_a = self.G2_a(x_a, self._states_a4[0])
        self._states_a4[0] = x_a
        skip_a3 = x_a

        x_a = self.C3_a(x_a)
        x_a = self.G3_a(x_a, self._states_a8[0])
        self._states_a8[0] = x_a

        # d branch
        x_d = self.head_d(x_four)
        skip_d1 = x_d

        x_d = self.C1_d(x_d)
        x_d = self.G1_d(x_d, self._states_d2[0])
        self._states_d2[0] = x_d
        skip_d2 = x_d

        x_d = self.C2_d(x_d)
        x_d = self.G2_d(x_d, self._states_d4[0])
        self._states_d4[0] = x_d
        skip_d3 = x_d

        x_d = self.C3_d(x_d)
        x_d = self.G3_d(x_d, self._states_d8[0])
        self._states_d8[0] = x_d

        # Fusion
        concat = torch.cat([x_i, x_a, x_d], 1)

        # Decoder
        # i branch
        x_i = self.F_i(concat)
        x_i = self.C4_i(x_i)
        x_i = self.G4_i(x_i, self._states_i8[1])
        self._states_i8[1] = x_i

        x_i = self.up(self.C5_i(x_i)) + skip_i3
        x_i = self.G5_i(x_i, self._states_i4[1])
        self._states_i4[1] = x_i

        x_i = self.up(self.C6_i(x_i)) + skip_i2
        x_i = self.G6_i(x_i, self._states_i2[1])
        self._states_i2[1] = x_i

        x_i = self.up(self.C7_i(x_i)) + skip_i1
        i = self.pred_i(x_i)

        # a branch
        x_a = self.F_a(concat)
        x_a = self.C4_a(x_a)
        x_a = self.G4_a(x_a, self._states_a8[1])
        self._states_a8[1] = x_a

        x_a = self.up(self.C5_a(x_a)) + skip_a3
        x_a = self.G5_a(x_a, self._states_a4[1])
        self._states_a4[1] = x_a

        x_a = self.up(self.C6_a(x_a)) + skip_a2
        x_a = self.G6_a(x_a, self._states_a2[1])
        self._states_a2[1] = x_a

        x_a = self.up(self.C7_a(x_a)) + skip_a1
        a = self.pred_a(x_a)

        # d branch
        x_d = self.F_d(concat)
        x_d = self.C4_d(x_d)
        x_d = self.G4_d(x_d, self._states_d8[1])
        self._states_d8[1] = x_d

        x_d = self.up(self.C5_d(x_d)) + skip_d3
        x_d = self.G5_d(x_d, self._states_d4[1])
        self._states_d4[1] = x_d

        x_d = self.up(self.C6_d(x_d)) + skip_d2
        x_d = self.G6_d(x_d, self._states_d2[1])
        self._states_d2[1] = x_d

        x_d = self.up(self.C7_d(x_d)) + skip_d1
        d = self.pred_d(x_d)

        # re-scale
        i = F.interpolate(i, size=(h_original, w_original), mode='bilinear', align_corners=True)
        a = F.interpolate(a, size=(h_original, w_original), mode='bilinear', align_corners=True)
        d = F.interpolate(d, size=(h_original, w_original), mode='bilinear', align_corners=True)

        return {
            'i': i,
            'a': a,
            'd': d
        }


class M34(BaseModel):
    """
    Split input, compute difference, and use three separate encoder-decoder to estimate iad respectively.
    Fusing iad features in the deepest layer.
    Remove ConvGRU in the Decoder part.
    """
    def __init__(self, num_bins=5, base_num_channels=64, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        padding = kernel_size // 2
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.channel_1 = base_num_channels
        self.channel_2 = base_num_channels * 2
        self.channel_4 = base_num_channels * 4
        self.channel_8 = base_num_channels * 8
        # i
        self.head_i = ConvLayer(self.num_bins * 1, self.channel_1, kernel_size, padding=padding, norm='IN')
        self.C1_i = ConvLayer(self.channel_1, self.channel_2, kernel_size=4, stride=2, padding=1, norm='IN')
        self.G1_i = ConvGRU(self.channel_2, self.channel_2, kernel_size)
        self.C2_i = ConvLayer(self.channel_2, self.channel_4, kernel_size=4, stride=2, padding=1, norm='IN')
        self.G2_i = ConvGRU(self.channel_4, self.channel_4, kernel_size)
        self.C3_i = ConvLayer(self.channel_4, self.channel_8, kernel_size=4, stride=2, padding=1, norm='IN')
        self.G3_i = ConvGRU(self.channel_8, self.channel_8, kernel_size)
        self.F_i = ConvLayer(self.channel_8 * 3, self.channel_8, kernel_size=1, stride=1, padding=0, norm='IN')
        self.C4_i = ResidualBlock(self.channel_8, self.channel_8)
        self.C5_i = ConvLayer(self.channel_8, self.channel_4, kernel_size=3, stride=1, padding=1, norm='IN')
        self.C6_i = ConvLayer(self.channel_4, self.channel_2, kernel_size=3, stride=1, padding=1, norm='IN')
        self.C7_i = ConvLayer(self.channel_2, self.channel_1, kernel_size=3, stride=1, padding=1, norm='IN')
        self.pred_i = ConvLayer(self.channel_1, out_channels=1, kernel_size=1, activation=None)
        # a
        self.head_a = ConvLayer(self.num_bins * 2, self.channel_1, kernel_size, padding=padding, norm='IN')
        self.C1_a = ConvLayer(self.channel_1, self.channel_2, kernel_size=4, stride=2, padding=1, norm='IN')
        self.G1_a = ConvGRU(self.channel_2, self.channel_2, kernel_size)
        self.C2_a = ConvLayer(self.channel_2, self.channel_4, kernel_size=4, stride=2, padding=1, norm='IN')
        self.G2_a = ConvGRU(self.channel_4, self.channel_4, kernel_size)
        self.C3_a = ConvLayer(self.channel_4, self.channel_8, kernel_size=4, stride=2, padding=1, norm='IN')
        self.G3_a = ConvGRU(self.channel_8, self.channel_8, kernel_size)
        self.F_a = ConvLayer(self.channel_8 * 3, self.channel_8, kernel_size=1, stride=1, padding=0, norm='IN')
        self.C4_a = ResidualBlock(self.channel_8, self.channel_8)
        self.C5_a = ConvLayer(self.channel_8, self.channel_4, kernel_size=3, stride=1, padding=1, norm='IN')
        self.C6_a = ConvLayer(self.channel_4, self.channel_2, kernel_size=3, stride=1, padding=1, norm='IN')
        self.C7_a = ConvLayer(self.channel_2, self.channel_1, kernel_size=3, stride=1, padding=1, norm='IN')
        self.pred_a = ConvLayer(self.channel_1, out_channels=1, kernel_size=1, activation=None)
        # d
        self.head_d = ConvLayer(self.num_bins * 3, self.channel_1, kernel_size, padding=padding, norm='IN')
        self.C1_d = ConvLayer(self.channel_1, self.channel_2, kernel_size=4, stride=2, padding=1, norm='IN')
        self.G1_d = ConvGRU(self.channel_2, self.channel_2, kernel_size)
        self.C2_d = ConvLayer(self.channel_2, self.channel_4, kernel_size=4, stride=2, padding=1, norm='IN')
        self.G2_d = ConvGRU(self.channel_4, self.channel_4, kernel_size)
        self.C3_d = ConvLayer(self.channel_4, self.channel_8, kernel_size=4, stride=2, padding=1, norm='IN')
        self.G3_d = ConvGRU(self.channel_8, self.channel_8, kernel_size)
        self.F_d = ConvLayer(self.channel_8 * 3, self.channel_8, kernel_size=1, stride=1, padding=0, norm='IN')
        self.C4_d = ResidualBlock(self.channel_8, self.channel_8)
        self.C5_d = ConvLayer(self.channel_8, self.channel_4, kernel_size=3, stride=1, padding=1, norm='IN')
        self.C6_d = ConvLayer(self.channel_4, self.channel_2, kernel_size=3, stride=1, padding=1, norm='IN')
        self.C7_d = ConvLayer(self.channel_2, self.channel_1, kernel_size=3, stride=1, padding=1, norm='IN')
        self.pred_d = ConvLayer(self.channel_1, out_channels=1, kernel_size=1, activation=None)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 1
        self.reset_states_i2()
        self.reset_states_i4()
        self.reset_states_i8()
        self.reset_states_a2()
        self.reset_states_a4()
        self.reset_states_a8()
        self.reset_states_d2()
        self.reset_states_d4()
        self.reset_states_d8()

    def reset_states_i2(self):
        self._states_i2 = [None] * self.num_recurrent_units

    def reset_states_i4(self):
        self._states_i4 = [None] * self.num_recurrent_units

    def reset_states_i8(self):
        self._states_i8 = [None] * self.num_recurrent_units

    def reset_states_a2(self):
        self._states_a2 = [None] * self.num_recurrent_units

    def reset_states_a4(self):
        self._states_a4 = [None] * self.num_recurrent_units

    def reset_states_a8(self):
        self._states_a8 = [None] * self.num_recurrent_units

    def reset_states_d2(self):
        self._states_d2 = [None] * self.num_recurrent_units

    def reset_states_d4(self):
        self._states_d4 = [None] * self.num_recurrent_units

    def reset_states_d8(self):
        self._states_d8 = [None] * self.num_recurrent_units

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W event tensor
        :return: N x num_output_channels x H/2 x W/2 iad
        """
        # split input
        x_90 = x[:, :, 0::2, 0::2]
        x_45 = x[:, :, 0::2, 1::2]
        x_135 = x[:, :, 1::2, 0::2]
        x_0 = x[:, :, 1::2, 1::2]

        _, _, h_original, w_original = x_90.size()
        h = (h_original // 8) * 8
        w = (w_original // 8) * 8
        x_90 = nn.UpsamplingBilinear2d((h, w))(x_90)
        x_45 = nn.UpsamplingBilinear2d((h, w))(x_45)
        x_135 = nn.UpsamplingBilinear2d((h, w))(x_135)
        x_0 = nn.UpsamplingBilinear2d((h, w))(x_0)

        x_s0 = (x_90 + x_45 + x_135 + x_0) / 2
        x_s1 = x_0 - x_90
        x_s2 = x_45 - x_135

        x_i = x_s0 / 2
        x_a = torch.cat([x_s1, x_s2], 1)
        x_d = torch.cat([x_s0, x_s1, x_s2], 1)

        # Encoder
        # i branch
        x_i = self.head_i(x_i)
        skip_i1 = x_i

        x_i = self.C1_i(x_i)
        x_i = self.G1_i(x_i, self._states_i2[0])
        self._states_i2[0] = x_i
        skip_i2 = x_i

        x_i = self.C2_i(x_i)
        x_i = self.G2_i(x_i, self._states_i4[0])
        self._states_i4[0] = x_i
        skip_i3 = x_i

        x_i = self.C3_i(x_i)
        x_i = self.G3_i(x_i, self._states_i8[0])
        self._states_i8[0] = x_i

        # a branch
        x_a = self.head_a(x_a)
        skip_a1 = x_a

        x_a = self.C1_a(x_a)
        x_a = self.G1_a(x_a, self._states_a2[0])
        self._states_a2[0] = x_a
        skip_a2 = x_a

        x_a = self.C2_a(x_a)
        x_a = self.G2_a(x_a, self._states_a4[0])
        self._states_a4[0] = x_a
        skip_a3 = x_a

        x_a = self.C3_a(x_a)
        x_a = self.G3_a(x_a, self._states_a8[0])
        self._states_a8[0] = x_a

        # d branch
        x_d = self.head_d(x_d)
        skip_d1 = x_d

        x_d = self.C1_d(x_d)
        x_d = self.G1_d(x_d, self._states_d2[0])
        self._states_d2[0] = x_d
        skip_d2 = x_d

        x_d = self.C2_d(x_d)
        x_d = self.G2_d(x_d, self._states_d4[0])
        self._states_d4[0] = x_d
        skip_d3 = x_d

        x_d = self.C3_d(x_d)
        x_d = self.G3_d(x_d, self._states_d8[0])
        self._states_d8[0] = x_d

        # Fusion
        concat = torch.cat([x_i, x_a, x_d], 1)

        # Decoder
        # i branch
        x_i = self.F_i(concat)
        x_i = self.C4_i(x_i)
        x_i = self.up(self.C5_i(x_i)) + skip_i3
        x_i = self.up(self.C6_i(x_i)) + skip_i2
        x_i = self.up(self.C7_i(x_i)) + skip_i1
        i = self.pred_i(x_i)

        # a branch
        x_a = self.F_a(concat)
        x_a = self.C4_a(x_a)
        x_a = self.up(self.C5_a(x_a)) + skip_a3
        x_a = self.up(self.C6_a(x_a)) + skip_a2
        x_a = self.up(self.C7_a(x_a)) + skip_a1
        a = self.pred_a(x_a)

        # d branch
        x_d = self.F_d(concat)
        x_d = self.C4_d(x_d)
        x_d = self.up(self.C5_d(x_d)) + skip_d3
        x_d = self.up(self.C6_d(x_d)) + skip_d2
        x_d = self.up(self.C7_d(x_d)) + skip_d1
        d = self.pred_d(x_d)

        # re-scale
        i = F.interpolate(i, size=(h_original, w_original), mode='bilinear', align_corners=True)
        a = F.interpolate(a, size=(h_original, w_original), mode='bilinear', align_corners=True)
        d = F.interpolate(d, size=(h_original, w_original), mode='bilinear', align_corners=True)

        return {
            'i': i,
            'a': a,
            'd': d
        }


class M35(BaseModel):
    """
    Split input and use three separate branches to estimate iad respectively.
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        padding = kernel_size // 2
        # i
        self.head_i = ConvLayer(self.num_bins * 1, base_num_channels, kernel_size, padding=padding)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_i = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_i = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        # a
        self.head_a = ConvLayer(self.num_bins * 2, base_num_channels, kernel_size, padding=padding)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_a = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_a = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        # d
        self.head_d = ConvLayer(self.num_bins * 3, base_num_channels, kernel_size, padding=padding)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_d = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_d = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
        """
        # split input
        x_90 = x[:, :, 0::2, 0::2]
        x_45 = x[:, :, 0::2, 1::2]
        x_135 = x[:, :, 1::2, 0::2]
        x_0 = x[:, :, 1::2, 1::2]

        x_s0 = (x_90 + x_45 + x_135 + x_0) / 2
        x_s1 = x_0 - x_90
        x_s2 = x_45 - x_135

        x_i = x_s0 / 2
        x_a = torch.cat([x_s1, x_s2], 1)
        x_d = torch.cat([x_s0, x_s1, x_s2], 1)

        # i branch
        x_i = self.head_i(x_i)
        x_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_i
        x_i = self.R1_i(x_i)
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i
        x_i = self.R2_i(x_i)
        i = self.pred_i(x_i)

        # a branch
        x_a = self.head_a(x_a)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a
        x_a = self.R1_a(x_a)
        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a
        x_a = self.R2_a(x_a)
        a = self.pred_a(x_a)

        # d branch
        x_d = self.head_d(x_d)
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


class M36(BaseModel):
    """
    Split input and use three separate branches to estimate iad respectively.
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        padding = kernel_size // 2
        # i
        self.head_i = ConvLayer(self.num_bins * 4, base_num_channels, kernel_size, padding=padding)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_i = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_i = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        # a
        self.head_a = ConvLayer(self.num_bins * 10, base_num_channels, kernel_size, padding=padding)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_a = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_a = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        # d
        self.head_d = ConvLayer(self.num_bins * 10, base_num_channels, kernel_size, padding=padding)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_d = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_d = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
        """
        # split input
        x_90 = x[:, :, 0::2, 0::2]
        x_45 = x[:, :, 0::2, 1::2]
        x_135 = x[:, :, 1::2, 0::2]
        x_0 = x[:, :, 1::2, 1::2]

        x_four = torch.cat([x_0, x_45, x_90, x_135], 1)
        x_contrast1 = x_0 - x_45
        x_contrast2 = x_0 - x_90
        x_contrast3 = x_0 - x_135
        x_contrast4 = x_45 - x_90
        x_contrast5 = x_45 - x_135
        x_contrast6 = x_90 - x_135

        x_ad = torch.cat([x_contrast1, x_contrast2, x_contrast3, x_contrast4, x_contrast5, x_contrast6, x_four], 1)

        # i branch
        x_i = self.head_i(x_four)
        x_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_i
        x_i = self.R1_i(x_i)
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i
        x_i = self.R2_i(x_i)
        i = self.pred_i(x_i)

        # a branch
        x_a = self.head_a(x_ad)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a
        x_a = self.R1_a(x_a)
        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a
        x_a = self.R2_a(x_a)
        a = self.pred_a(x_a)

        # d branch
        x_d = self.head_d(x_ad)
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


class M37(BaseModel):
    """
    Split input and use three separate branches to estimate iad respectively.
    Use multi-scale perception module to substitute the residual block.
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        padding = kernel_size // 2
        # i
        self.head_i = ConvLayer(self.num_bins * 4, base_num_channels, kernel_size, padding=padding)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_i = MSP(base_num_channels)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_i = MSP(base_num_channels)
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        # a
        self.head_a = ConvLayer(self.num_bins * 10, base_num_channels, kernel_size, padding=padding)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_a = MSP(base_num_channels)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_a = MSP(base_num_channels)
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        # d
        self.head_d = ConvLayer(self.num_bins * 10, base_num_channels, kernel_size, padding=padding)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_d = MSP(base_num_channels)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_d = MSP(base_num_channels)
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
        """
        # split input
        x_90 = x[:, :, 0::2, 0::2]
        x_45 = x[:, :, 0::2, 1::2]
        x_135 = x[:, :, 1::2, 0::2]
        x_0 = x[:, :, 1::2, 1::2]

        x_four = torch.cat([x_0, x_45, x_90, x_135], 1)
        x_contrast1 = x_0 - x_45
        x_contrast2 = x_0 - x_90
        x_contrast3 = x_0 - x_135
        x_contrast4 = x_45 - x_90
        x_contrast5 = x_45 - x_135
        x_contrast6 = x_90 - x_135

        x_ad = torch.cat([x_contrast1, x_contrast2, x_contrast3, x_contrast4, x_contrast5, x_contrast6, x_four], 1)

        # i branch
        x_i = self.head_i(x_four)
        x_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_i
        x_i = self.R1_i(x_i)
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i
        x_i = self.R2_i(x_i)
        i = self.pred_i(x_i)

        # a branch
        x_a = self.head_a(x_ad)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a
        x_a = self.R1_a(x_a)
        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a
        x_a = self.R2_a(x_a)
        a = self.pred_a(x_a)

        # d branch
        x_d = self.head_d(x_ad)
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


class M38(BaseModel):
    """
    Split input and use three separate branches to estimate iad respectively.
    Contrast input.
    Merge five channels.
    Use multi-scale perception module to substitute the residual block.
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        padding = kernel_size // 2
        # merge
        self.merge_0 = ConvLayer(5, 1, kernel_size=1, stride=1, padding=0)
        self.merge_45 = ConvLayer(5, 1, kernel_size=1, stride=1, padding=0)
        self.merge_90 = ConvLayer(5, 1, kernel_size=1, stride=1, padding=0)
        self.merge_135 = ConvLayer(5, 1, kernel_size=1, stride=1, padding=0)
        self.merge_contrast1 = ConvLayer(5, 1, kernel_size=1, stride=1, padding=0)
        self.merge_contrast2 = ConvLayer(5, 1, kernel_size=1, stride=1, padding=0)
        self.merge_contrast3 = ConvLayer(5, 1, kernel_size=1, stride=1, padding=0)
        self.merge_contrast4 = ConvLayer(5, 1, kernel_size=1, stride=1, padding=0)
        self.merge_contrast5 = ConvLayer(5, 1, kernel_size=1, stride=1, padding=0)
        self.merge_contrast6 = ConvLayer(5, 1, kernel_size=1, stride=1, padding=0)
        # i
        self.head_i = ConvLayer(4, base_num_channels, kernel_size, padding=padding)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_i = MSP(base_num_channels)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_i = MSP(base_num_channels)
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        # a
        self.head_a = ConvLayer(10, base_num_channels, kernel_size, padding=padding)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_a = MSP(base_num_channels)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_a = MSP(base_num_channels)
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        # d
        self.head_d = ConvLayer(10, base_num_channels, kernel_size, padding=padding)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_d = MSP(base_num_channels)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_d = MSP(base_num_channels)
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
        """
        # split input
        x_90 = x[:, :, 0::2, 0::2]
        x_45 = x[:, :, 0::2, 1::2]
        x_135 = x[:, :, 1::2, 0::2]
        x_0 = x[:, :, 1::2, 1::2]

        x_contrast1 = x_0 - x_45
        x_contrast2 = x_0 - x_90
        x_contrast3 = x_0 - x_135
        x_contrast4 = x_45 - x_90
        x_contrast5 = x_45 - x_135
        x_contrast6 = x_90 - x_135

        x_merge_0 = self.merge_0(x_0)
        x_merge_45 = self.merge_45(x_45)
        x_merge_90 = self.merge_90(x_90)
        x_merge_135 = self.merge_135(x_135)

        x_merge_contrast1 = self.merge_contrast1(x_contrast1)
        x_merge_contrast2 = self.merge_contrast2(x_contrast2)
        x_merge_contrast3 = self.merge_contrast3(x_contrast3)
        x_merge_contrast4 = self.merge_contrast4(x_contrast4)
        x_merge_contrast5 = self.merge_contrast5(x_contrast5)
        x_merge_contrast6 = self.merge_contrast6(x_contrast6)

        x_four = torch.cat([x_merge_0, x_merge_45, x_merge_90, x_merge_135], 1)
        x_ad = torch.cat([x_merge_contrast1, x_merge_contrast2, x_merge_contrast3, x_merge_contrast4, x_merge_contrast5, x_merge_contrast6, x_four], 1)

        # i branch
        x_i = self.head_i(x_four)
        x_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_i
        x_i = self.R1_i(x_i)
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i
        x_i = self.R2_i(x_i)
        i = self.pred_i(x_i)

        # a branch
        x_a = self.head_a(x_ad)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a
        x_a = self.R1_a(x_a)
        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a
        x_a = self.R2_a(x_a)
        a = self.pred_a(x_a)

        # d branch
        x_d = self.head_d(x_ad)
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


class M39(BaseModel):
    """
    Split input and use three separate branches to estimate iad respectively.
    Contrast input.
    Merge five channels.
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        padding = kernel_size // 2
        # merge
        self.merge_0 = ConvLayer(5, 1, kernel_size=1, stride=1, padding=0)
        self.merge_45 = ConvLayer(5, 1, kernel_size=1, stride=1, padding=0)
        self.merge_90 = ConvLayer(5, 1, kernel_size=1, stride=1, padding=0)
        self.merge_135 = ConvLayer(5, 1, kernel_size=1, stride=1, padding=0)
        self.merge_contrast1 = ConvLayer(5, 1, kernel_size=1, stride=1, padding=0)
        self.merge_contrast2 = ConvLayer(5, 1, kernel_size=1, stride=1, padding=0)
        self.merge_contrast3 = ConvLayer(5, 1, kernel_size=1, stride=1, padding=0)
        self.merge_contrast4 = ConvLayer(5, 1, kernel_size=1, stride=1, padding=0)
        self.merge_contrast5 = ConvLayer(5, 1, kernel_size=1, stride=1, padding=0)
        self.merge_contrast6 = ConvLayer(5, 1, kernel_size=1, stride=1, padding=0)
        # i
        self.head_i = ConvLayer(4, base_num_channels, kernel_size, padding=padding)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_i = MSP(base_num_channels)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_i = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        # a
        self.head_a = ConvLayer(10, base_num_channels, kernel_size, padding=padding)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_a = MSP(base_num_channels)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_a = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        # d
        self.head_d = ConvLayer(10, base_num_channels, kernel_size, padding=padding)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_d = MSP(base_num_channels)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_d = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
        """
        # split input
        x_90 = x[:, :, 0::2, 0::2]
        x_45 = x[:, :, 0::2, 1::2]
        x_135 = x[:, :, 1::2, 0::2]
        x_0 = x[:, :, 1::2, 1::2]

        x_contrast1 = x_0 - x_45
        x_contrast2 = x_0 - x_90
        x_contrast3 = x_0 - x_135
        x_contrast4 = x_45 - x_90
        x_contrast5 = x_45 - x_135
        x_contrast6 = x_90 - x_135

        x_merge_0 = self.merge_0(x_0)
        x_merge_45 = self.merge_45(x_45)
        x_merge_90 = self.merge_90(x_90)
        x_merge_135 = self.merge_135(x_135)

        x_merge_contrast1 = self.merge_contrast1(x_contrast1)
        x_merge_contrast2 = self.merge_contrast2(x_contrast2)
        x_merge_contrast3 = self.merge_contrast3(x_contrast3)
        x_merge_contrast4 = self.merge_contrast4(x_contrast4)
        x_merge_contrast5 = self.merge_contrast5(x_contrast5)
        x_merge_contrast6 = self.merge_contrast6(x_contrast6)

        x_four = torch.cat([x_merge_0, x_merge_45, x_merge_90, x_merge_135], 1)
        x_ad = torch.cat([x_merge_contrast1, x_merge_contrast2, x_merge_contrast3, x_merge_contrast4, x_merge_contrast5, x_merge_contrast6, x_four], 1)

        # i branch
        x_i = self.head_i(x_four)
        x_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_i
        x_i = self.R1_i(x_i)
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i
        x_i = self.R2_i(x_i)
        i = self.pred_i(x_i)

        # a branch
        x_a = self.head_a(x_ad)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a
        x_a = self.R1_a(x_a)
        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a
        x_a = self.R2_a(x_a)
        a = self.pred_a(x_a)

        # d branch
        x_d = self.head_d(x_ad)
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


class M40(BaseModel):
    """
    Split input and use three separate branches to estimate iad respectively.
    Contrast input.
    Merge five channels.
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        padding = kernel_size // 2
        # merge
        self.merge_0 = ConvLayer(5, 1, kernel_size=1, stride=1, padding=0)
        self.merge_45 = ConvLayer(5, 1, kernel_size=1, stride=1, padding=0)
        self.merge_90 = ConvLayer(5, 1, kernel_size=1, stride=1, padding=0)
        self.merge_135 = ConvLayer(5, 1, kernel_size=1, stride=1, padding=0)
        self.merge_contrast1 = ConvLayer(5, 1, kernel_size=1, stride=1, padding=0)
        self.merge_contrast2 = ConvLayer(5, 1, kernel_size=1, stride=1, padding=0)
        self.merge_contrast3 = ConvLayer(5, 1, kernel_size=1, stride=1, padding=0)
        self.merge_contrast4 = ConvLayer(5, 1, kernel_size=1, stride=1, padding=0)
        self.merge_contrast5 = ConvLayer(5, 1, kernel_size=1, stride=1, padding=0)
        self.merge_contrast6 = ConvLayer(5, 1, kernel_size=1, stride=1, padding=0)
        # i
        self.head_i = ConvLayer(4, base_num_channels, kernel_size, padding=padding)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_i = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_i = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        # a
        self.head_a = ConvLayer(10, base_num_channels, kernel_size, padding=padding)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_a = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_a = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        # d
        self.head_d = ConvLayer(10, base_num_channels, kernel_size, padding=padding)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_d = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_d = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
        """
        # split input
        x_90 = x[:, :, 0::2, 0::2]
        x_45 = x[:, :, 0::2, 1::2]
        x_135 = x[:, :, 1::2, 0::2]
        x_0 = x[:, :, 1::2, 1::2]

        x_contrast1 = x_0 - x_45
        x_contrast2 = x_0 - x_90
        x_contrast3 = x_0 - x_135
        x_contrast4 = x_45 - x_90
        x_contrast5 = x_45 - x_135
        x_contrast6 = x_90 - x_135

        x_merge_0 = self.merge_0(x_0)
        x_merge_45 = self.merge_45(x_45)
        x_merge_90 = self.merge_90(x_90)
        x_merge_135 = self.merge_135(x_135)

        x_merge_contrast1 = self.merge_contrast1(x_contrast1)
        x_merge_contrast2 = self.merge_contrast2(x_contrast2)
        x_merge_contrast3 = self.merge_contrast3(x_contrast3)
        x_merge_contrast4 = self.merge_contrast4(x_contrast4)
        x_merge_contrast5 = self.merge_contrast5(x_contrast5)
        x_merge_contrast6 = self.merge_contrast6(x_contrast6)

        x_four = torch.cat([x_merge_0, x_merge_45, x_merge_90, x_merge_135], 1)
        x_ad = torch.cat([x_merge_contrast1, x_merge_contrast2, x_merge_contrast3, x_merge_contrast4, x_merge_contrast5, x_merge_contrast6, x_four], 1)

        # i branch
        x_i = self.head_i(x_four)
        x_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_i
        x_i = self.R1_i(x_i)
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i
        x_i = self.R2_i(x_i)
        i = self.pred_i(x_i)

        # a branch
        x_a = self.head_a(x_ad)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a
        x_a = self.R1_a(x_a)
        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a
        x_a = self.R2_a(x_a)
        a = self.pred_a(x_a)

        # d branch
        x_d = self.head_d(x_ad)
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


class M41(BaseModel):
    """
    i2p
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        padding = kernel_size // 2
        # 90
        self.head_90 = ConvLayer(self.num_bins, base_num_channels, kernel_size, padding=padding)
        self.G1_90 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_90 = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_90 = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        # 45
        self.head_45 = ConvLayer(self.num_bins, base_num_channels, kernel_size, padding=padding)
        self.G1_45 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_45 = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_45 = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        # 135
        self.head_135 = ConvLayer(self.num_bins, base_num_channels, kernel_size, padding=padding)
        self.G1_135 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_135 = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_135 = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        # 0
        self.head_0 = ConvLayer(self.num_bins, base_num_channels, kernel_size, padding=padding)
        self.G1_0 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_0 = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_0 = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)

        # i
        self.head_i = ConvLayer(base_num_channels * 4, base_num_channels, kernel_size, padding=padding)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_i = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        # a
        self.head_a = ConvLayer(base_num_channels * 6, base_num_channels * 2, kernel_size, padding=padding)
        self.G2_a = ConvGRU(base_num_channels * 2, base_num_channels * 2, kernel_size)
        self.R2_a = ResidualBlock(base_num_channels * 2, base_num_channels * 2)
        self.pred_a = ConvLayer(base_num_channels * 2, out_channels=1, kernel_size=1, activation=None)
        # d
        self.head_d = ConvLayer(base_num_channels * 6, base_num_channels * 2, kernel_size, padding=padding)
        self.G2_d = ConvGRU(base_num_channels * 2, base_num_channels * 2, kernel_size)
        self.R2_d = ResidualBlock(base_num_channels * 2, base_num_channels * 2)
        self.pred_d = ConvLayer(base_num_channels * 2, out_channels=1, kernel_size=1, activation=None)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 1
        self.reset_states_90()
        self.reset_states_45()
        self.reset_states_135()
        self.reset_states_0()
        self.reset_states_i()
        self.reset_states_a()
        self.reset_states_d()

    # 90
    def reset_states_90(self):
        self._states_90 = [None] * self.num_recurrent_units

    # 45
    def reset_states_45(self):
        self._states_45 = [None] * self.num_recurrent_units

    # 135
    def reset_states_135(self):
        self._states_135 = [None] * self.num_recurrent_units

    # 0
    def reset_states_0(self):
        self._states_0 = [None] * self.num_recurrent_units

    # i
    def reset_states_i(self):
        self._states_i = [None] * self.num_recurrent_units

    # a
    def reset_states_a(self):
        self._states_a = [None] * self.num_recurrent_units

    # d
    def reset_states_d(self):
        self._states_d = [None] * self.num_recurrent_units

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W event tensor
        :return: N x num_output_channels x H/2 x W/2 four intensity images and iad
        """
        # split input
        x_90 = x[:, :, 0::2, 0::2]
        x_45 = x[:, :, 0::2, 1::2]
        x_135 = x[:, :, 1::2, 0::2]
        x_0 = x[:, :, 1::2, 1::2]

        # 90 branch
        x_90 = self.head_90(x_90)
        x_90 = self.G1_90(x_90, self._states_90[0])
        self._states_90[0] = x_90
        x_90 = self.R1_90(x_90)
        i_90 = self.pred_90(x_90)

        # 45 branch
        x_45 = self.head_45(x_45)
        x_45 = self.G1_45(x_45, self._states_45[0])
        self._states_45[0] = x_45
        x_45 = self.R1_45(x_45)
        i_45 = self.pred_45(x_45)

        # 135 branch
        x_135 = self.head_135(x_135)
        x_135 = self.G1_135(x_135, self._states_135[0])
        self._states_135[0] = x_135
        x_135 = self.R1_135(x_135)
        i_135 = self.pred_135(x_135)

        # 0 branch
        x_0 = self.head_0(x_0)
        x_0 = self.G1_0(x_0, self._states_0[0])
        self._states_0[0] = x_0
        x_0 = self.R1_0(x_0)
        i_0 = self.pred_0(x_0)

        # middle process
        x_four = torch.cat([x_0, x_45, x_90, x_135], 1)

        x_contrast1 = x_0 - x_45
        x_contrast2 = x_0 - x_90
        x_contrast3 = x_0 - x_135
        x_contrast4 = x_45 - x_90
        x_contrast5 = x_45 - x_135
        x_contrast6 = x_90 - x_135

        x_contrast = torch.cat([x_contrast1, x_contrast2, x_contrast3, x_contrast4, x_contrast5, x_contrast6], 1)

        # i branch
        x_i = self.head_i(x_four)
        x_i = self.G2_i(x_i, self._states_i[0])
        self._states_i[0] = x_i
        x_i = self.R2_i(x_i)
        i = self.pred_i(x_i)

        # a branch
        x_a = self.head_a(x_contrast)
        x_a = self.G2_a(x_a, self._states_a[0])
        self._states_a[0] = x_a
        x_a = self.R2_a(x_a)
        a = self.pred_a(x_a)

        # d branch
        x_d = self.head_d(x_contrast)
        x_d = self.G2_d(x_d, self._states_d[0])
        self._states_d[0] = x_d
        x_d = self.R2_d(x_d)
        d = self.pred_d(x_d)

        return {
            'i_90': i_90,
            'i_45': i_45,
            'i_135': i_135,
            'i_0': i_0,
            'i': i,
            'a': a,
            'd': d,
        }


class M42(BaseModel):
    """
    Compatible with E2VID_lightweight
    Recurrent, UNet-like architecture where each encoder is followed by a ConvLSTM or ConvGRU.
    """
    def __init__(self, unet_kwargs):
        super().__init__()
        self.num_bins = unet_kwargs['num_bins']  # legacy
        self.num_encoders = unet_kwargs['num_encoders']  # legacy
        self.unetrecurrent_i = UNetRecurrent(unet_kwargs)
        self.unetrecurrent_a = UNetRecurrent(unet_kwargs)
        self.unetrecurrent_d = UNetRecurrent(unet_kwargs)

    def reset_states_i(self):
        self.unetrecurrent_i.states = [None] * self.unetrecurrent_i.num_encoders

    def reset_states_a(self):
        self.unetrecurrent_a.states = [None] * self.unetrecurrent_a.num_encoders

    def reset_states_d(self):
        self.unetrecurrent_d.states = [None] * self.unetrecurrent_d.num_encoders

    def forward(self, x):
        """
        :param x: N x num_bins x H x W
        :return: iad.
        """
        # split input
        x_90 = x[:, :, 0::2, 0::2]
        x_45 = x[:, :, 0::2, 1::2]
        x_135 = x[:, :, 1::2, 0::2]
        x_0 = x[:, :, 1::2, 1::2]

        x_four = torch.cat([x_90, x_45, x_135, x_0], 1)
        _, _, h_original, w_original = x_four.size()
        h = (h_original // 4) * 4
        w = (w_original // 4) * 4
        x_four = nn.UpsamplingBilinear2d((h, w))(x_four)

        i = self.unetrecurrent_i.forward(x_four)
        a = self.unetrecurrent_a.forward(x_four)
        d = self.unetrecurrent_d.forward(x_four)

        i = F.interpolate(i, size=(h_original, w_original), mode='bilinear', align_corners=True)
        a = F.interpolate(a, size=(h_original, w_original), mode='bilinear', align_corners=True)
        d = F.interpolate(d, size=(h_original, w_original), mode='bilinear', align_corners=True)

        return {
            'i': i,
            'a': a,
            'd': d
        }


class M43(BaseModel):
    """
    i2s
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        padding = kernel_size // 2
        # 90
        self.head_90 = ConvLayer(self.num_bins, base_num_channels, kernel_size, padding=padding)
        self.G1_90 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_90 = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_90 = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        # 45
        self.head_45 = ConvLayer(self.num_bins, base_num_channels, kernel_size, padding=padding)
        self.G1_45 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_45 = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_45 = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        # 135
        self.head_135 = ConvLayer(self.num_bins, base_num_channels, kernel_size, padding=padding)
        self.G1_135 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_135 = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_135 = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        # 0
        self.head_0 = ConvLayer(self.num_bins, base_num_channels, kernel_size, padding=padding)
        self.G1_0 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_0 = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_0 = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)

        # s0
        self.G2_s0 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_s0 = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_s0 = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        # s1
        self.G2_s1 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_s1 = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_s1 = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        # s2
        self.G2_s2 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_s2 = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_s2 = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 1
        self.reset_states_90()
        self.reset_states_45()
        self.reset_states_135()
        self.reset_states_0()
        self.reset_states_s0()
        self.reset_states_s1()
        self.reset_states_s2()

    # 90
    def reset_states_90(self):
        self._states_90 = [None] * self.num_recurrent_units

    # 45
    def reset_states_45(self):
        self._states_45 = [None] * self.num_recurrent_units

    # 135
    def reset_states_135(self):
        self._states_135 = [None] * self.num_recurrent_units

    # 0
    def reset_states_0(self):
        self._states_0 = [None] * self.num_recurrent_units

    # i
    def reset_states_s0(self):
        self._states_s0 = [None] * self.num_recurrent_units

    # a
    def reset_states_s1(self):
        self._states_s1 = [None] * self.num_recurrent_units

    # d
    def reset_states_s2(self):
        self._states_s2 = [None] * self.num_recurrent_units

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W event tensor
        :return: N x num_output_channels x H/2 x W/2 four intensity images and iad
        """
        # split input
        x_90 = x[:, :, 0::2, 0::2]
        x_45 = x[:, :, 0::2, 1::2]
        x_135 = x[:, :, 1::2, 0::2]
        x_0 = x[:, :, 1::2, 1::2]

        # 90 branch
        x_90 = self.head_90(x_90)
        x_90 = self.G1_90(x_90, self._states_90[0])
        self._states_90[0] = x_90
        x_90 = self.R1_90(x_90)
        i_90 = self.pred_90(x_90)

        # 45 branch
        x_45 = self.head_45(x_45)
        x_45 = self.G1_45(x_45, self._states_45[0])
        self._states_45[0] = x_45
        x_45 = self.R1_45(x_45)
        i_45 = self.pred_45(x_45)

        # 135 branch
        x_135 = self.head_135(x_135)
        x_135 = self.G1_135(x_135, self._states_135[0])
        self._states_135[0] = x_135
        x_135 = self.R1_135(x_135)
        i_135 = self.pred_135(x_135)

        # 0 branch
        x_0 = self.head_0(x_0)
        x_0 = self.G1_0(x_0, self._states_0[0])
        self._states_0[0] = x_0
        x_0 = self.R1_0(x_0)
        i_0 = self.pred_0(x_0)

        # middle process
        x_s0 = x_0 + x_90
        x_s1 = x_0 - x_90
        x_s2 = x_45 - x_135

        # s0 branch
        x_s0 = self.G2_s0(x_s0, self._states_s0[0])
        self._states_s0[0] = x_s0
        x_s0 = self.R2_s0(x_s0)
        s0 = self.pred_s0(x_s0)

        # s1 branch
        x_s1 = self.G2_s1(x_s1, self._states_s1[0])
        self._states_s1[0] = x_s1
        x_s1 = self.R2_s1(x_s1)
        s1 = self.pred_s1(x_s1)

        # s2 branch
        x_s2 = self.G2_s2(x_s2, self._states_s2[0])
        self._states_s2[0] = x_s2
        x_s2 = self.R2_s2(x_s2)
        s2 = self.pred_s2(x_s2)

        return {
            'i_90': i_90,
            'i_45': i_45,
            'i_135': i_135,
            'i_0': i_0,
            's0': s0,
            's1': s1,
            's2': s2,
        }


class M44(BaseModel):
    """
    2x2 conv, common and separately
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = ConvLayer(self.num_bins, base_num_channels, kernel_size=2, stride=2, padding=0)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_i = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_i = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # a
        self.head_a = ConvLayer(self.num_bins, base_num_channels, kernel_size=2, stride=2, padding=0)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_a = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_a = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # d
        self.head_d = ConvLayer(self.num_bins, base_num_channels, kernel_size=2, stride=2, padding=0)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_d = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_d = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
        """
        # i branch
        x_i = self.head_i(x)
        x_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_i
        x_i = self.R1_i(x_i)
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i
        x_i = self.R2_i(x_i)
        i = self.pred_i(x_i)

        # a branch
        x_a = self.head_a(x)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a
        x_a = self.R1_a(x_a)
        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a
        x_a = self.R2_a(x_a)
        a = self.pred_a(x_a)

        # d branch
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


class M45(BaseModel):
    """
    2x2 conv, common and separately.
    Large kernel.
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = ConvLayer(self.num_bins, base_num_channels, kernel_size=2, stride=2, padding=0)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_i = Block(base_num_channels)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_i = Block(base_num_channels)
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # a
        self.head_a = ConvLayer(self.num_bins, base_num_channels, kernel_size=2, stride=2, padding=0)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_a = Block(base_num_channels)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_a = Block(base_num_channels)
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # d
        self.head_d = ConvLayer(self.num_bins, base_num_channels, kernel_size=2, stride=2, padding=0)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_d = Block(base_num_channels)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_d = Block(base_num_channels)
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
        """
        # i branch
        x_i = self.head_i(x)
        x_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_i
        x_i = self.R1_i(x_i)
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i
        x_i = self.R2_i(x_i)
        i = self.pred_i(x_i)

        # a branch
        x_a = self.head_a(x)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a
        x_a = self.R1_a(x_a)
        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a
        x_a = self.R2_a(x_a)
        a = self.pred_a(x_a)

        # d branch
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


class M46(BaseModel):
    """
    2x2 conv, common and separately
    My large kernel
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = ConvLayer(self.num_bins, base_num_channels, kernel_size=2, stride=2, padding=0)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_i = ResidualBlock_LK(base_num_channels)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_i = ResidualBlock_LK(base_num_channels)
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # a
        self.head_a = ConvLayer(self.num_bins, base_num_channels, kernel_size=2, stride=2, padding=0)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_a = ResidualBlock_LK(base_num_channels)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_a = ResidualBlock_LK(base_num_channels)
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # d
        self.head_d = ConvLayer(self.num_bins, base_num_channels, kernel_size=2, stride=2, padding=0)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_d = ResidualBlock_LK(base_num_channels)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_d = ResidualBlock_LK(base_num_channels)
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
        """
        # i branch
        x_i = self.head_i(x)
        x_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_i
        x_i = self.R1_i(x_i)
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i
        x_i = self.R2_i(x_i)
        i = self.pred_i(x_i)

        # a branch
        x_a = self.head_a(x)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a
        x_a = self.R1_a(x_a)
        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a
        x_a = self.R2_a(x_a)
        a = self.pred_a(x_a)

        # d branch
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


class M47(BaseModel):
    """
    Rich contrast input
    My large kernel
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        padding = kernel_size // 2
        # i
        self.head_i = ConvLayer(self.num_bins * 4, base_num_channels, kernel_size, padding=padding)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_i = ResidualBlock_LK(base_num_channels)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_i = ResidualBlock_LK(base_num_channels)
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # a
        self.head_a = ConvLayer(self.num_bins * 10, base_num_channels, kernel_size, padding=padding)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_a = ResidualBlock_LK(base_num_channels)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_a = ResidualBlock_LK(base_num_channels)
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # d
        self.head_d = ConvLayer(self.num_bins * 10, base_num_channels, kernel_size, padding=padding)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_d = ResidualBlock_LK(base_num_channels)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_d = ResidualBlock_LK(base_num_channels)
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
        """
        # split input
        x_90 = x[:, :, 0::2, 0::2]
        x_45 = x[:, :, 0::2, 1::2]
        x_135 = x[:, :, 1::2, 0::2]
        x_0 = x[:, :, 1::2, 1::2]

        x_four = torch.cat([x_0, x_45, x_90, x_135], 1)
        x_contrast1 = x_0 - x_45
        x_contrast2 = x_0 - x_90
        x_contrast3 = x_0 - x_135
        x_contrast4 = x_45 - x_90
        x_contrast5 = x_45 - x_135
        x_contrast6 = x_90 - x_135

        x_ad = torch.cat([x_contrast1, x_contrast2, x_contrast3, x_contrast4, x_contrast5, x_contrast6, x_four], 1)

        # i branch
        x_i = self.head_i(x_four)
        x_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_i
        x_i = self.R1_i(x_i)
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i
        x_i = self.R2_i(x_i)
        i = self.pred_i(x_i)

        # a branch
        x_a = self.head_a(x_ad)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a
        x_a = self.R1_a(x_a)
        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a
        x_a = self.R2_a(x_a)
        a = self.pred_a(x_a)

        # d branch
        x_d = self.head_d(x_ad)
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


class M48(BaseModel):
    """
    Rich contrast input
    Extreme large kernel
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        padding = kernel_size // 2
        # i
        self.head_i = ConvLayer(self.num_bins * 4, base_num_channels, kernel_size, padding=padding)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_i = ResidualBlock_ELK(base_num_channels)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_i = ResidualBlock_ELK(base_num_channels)
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # a
        self.head_a = ConvLayer(self.num_bins * 10, base_num_channels, kernel_size, padding=padding)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_a = ResidualBlock_ELK(base_num_channels)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_a = ResidualBlock_ELK(base_num_channels)
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # d
        self.head_d = ConvLayer(self.num_bins * 10, base_num_channels, kernel_size, padding=padding)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_d = ResidualBlock_ELK(base_num_channels)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_d = ResidualBlock_ELK(base_num_channels)
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
        """
        # split input
        x_90 = x[:, :, 0::2, 0::2]
        x_45 = x[:, :, 0::2, 1::2]
        x_135 = x[:, :, 1::2, 0::2]
        x_0 = x[:, :, 1::2, 1::2]

        x_four = torch.cat([x_0, x_45, x_90, x_135], 1)
        x_contrast1 = x_0 - x_45
        x_contrast2 = x_0 - x_90
        x_contrast3 = x_0 - x_135
        x_contrast4 = x_45 - x_90
        x_contrast5 = x_45 - x_135
        x_contrast6 = x_90 - x_135

        x_ad = torch.cat([x_contrast1, x_contrast2, x_contrast3, x_contrast4, x_contrast5, x_contrast6, x_four], 1)

        # i branch
        x_i = self.head_i(x_four)
        x_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_i
        x_i = self.R1_i(x_i)
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i
        x_i = self.R2_i(x_i)
        i = self.pred_i(x_i)

        # a branch
        x_a = self.head_a(x_ad)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a
        x_a = self.R1_a(x_a)
        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a
        x_a = self.R2_a(x_a)
        a = self.pred_a(x_a)

        # d branch
        x_d = self.head_d(x_ad)
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


class M49(BaseModel):
    """
    Rich contrast input
    My large kernel
    Use global branch to estimate angle offset
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        padding = kernel_size // 2
        # i
        self.head_i = ConvLayer(self.num_bins * 4, base_num_channels, kernel_size, padding=padding)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_i = ResidualBlock_LK(base_num_channels)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_i = ResidualBlock_LK(base_num_channels)
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # a
        self.head_a = ConvLayer(self.num_bins * 10, base_num_channels, kernel_size, padding=padding)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_a = ResidualBlock_LK(base_num_channels)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_a = ResidualBlock_LK(base_num_channels)
        self.offset = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Conv2d(base_num_channels, base_num_channels // 4, 1),
                                    nn.ReLU(),
                                    nn.Conv2d(base_num_channels // 4, 1, 1))
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # d
        self.head_d = ConvLayer(self.num_bins * 10, base_num_channels, kernel_size, padding=padding)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_d = ResidualBlock_LK(base_num_channels)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_d = ResidualBlock_LK(base_num_channels)
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
        """
        # split input
        x_90 = x[:, :, 0::2, 0::2]
        x_45 = x[:, :, 0::2, 1::2]
        x_135 = x[:, :, 1::2, 0::2]
        x_0 = x[:, :, 1::2, 1::2]

        x_four = torch.cat([x_0, x_45, x_90, x_135], 1)
        x_contrast1 = x_0 - x_45
        x_contrast2 = x_0 - x_90
        x_contrast3 = x_0 - x_135
        x_contrast4 = x_45 - x_90
        x_contrast5 = x_45 - x_135
        x_contrast6 = x_90 - x_135

        x_ad = torch.cat([x_contrast1, x_contrast2, x_contrast3, x_contrast4, x_contrast5, x_contrast6, x_four], 1)

        # i branch
        x_i = self.head_i(x_four)
        x_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_i
        x_i = self.R1_i(x_i)
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i
        x_i = self.R2_i(x_i)
        i = self.pred_i(x_i)

        # a branch
        x_a = self.head_a(x_ad)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a
        x_a = self.R1_a(x_a)
        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a
        x_a = self.R2_a(x_a)
        offset = self.offset(x_a)
        a = self.pred_a(x_a)
        a = a + offset

        # d branch
        x_d = self.head_d(x_ad)
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


class M50(BaseModel):
    """
    2x2 conv, common and separately
    my large kernel
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = ConvLayer(self.num_bins, base_num_channels, kernel_size=2, stride=2, padding=0)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_i = ResidualBlock_LK(base_num_channels)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_i = ResidualBlock_LK(base_num_channels)
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # a
        self.head_a = ConvLayer(self.num_bins, base_num_channels, kernel_size=2, stride=2, padding=0)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_a = ResidualBlock_LK(base_num_channels)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_a = ResidualBlock_LK(base_num_channels)
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # d
        self.head_d = ConvLayer(self.num_bins, base_num_channels, kernel_size=2, stride=2, padding=0)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_d = ResidualBlock_LK(base_num_channels)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_d = ResidualBlock_LK(base_num_channels)
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
        """
        # i branch
        x_i = self.head_i(x)
        x_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_i
        x_i = self.R1_i(x_i)
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i
        x_i = self.R2_i(x_i)
        i = self.pred_i(x_i)

        # a branch
        x_a = self.head_a(x)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a
        x_a = self.R1_a(x_a)
        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a
        x_a = self.R2_a(x_a)
        a = self.pred_a(x_a)

        # d branch
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


class M51(BaseModel):
    """
    2x2 conv, common and separately
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = ConvLayer(self.num_bins, base_num_channels, kernel_size=2, stride=2, padding=0)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_i = ResidualBlock_CT(base_num_channels)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_i = ResidualBlock_CT(base_num_channels)
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # a
        self.head_a = ConvLayer(self.num_bins, base_num_channels, kernel_size=2, stride=2, padding=0)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_a = ResidualBlock_CT(base_num_channels)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_a = ResidualBlock_CT(base_num_channels)
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # d
        self.head_d = ConvLayer(self.num_bins, base_num_channels, kernel_size=2, stride=2, padding=0)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_d = ResidualBlock_CT(base_num_channels)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_d = ResidualBlock_CT(base_num_channels)
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
        """
        # i branch
        x_i = self.head_i(x)
        x_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_i
        x_i = self.R1_i(x_i)
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i
        x_i = self.R2_i(x_i)
        i = self.pred_i(x_i)

        # a branch
        x_a = self.head_a(x)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a
        x_a = self.R1_a(x_a)
        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a
        x_a = self.R2_a(x_a)
        a = self.pred_a(x_a)

        # d branch
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


class M52(BaseModel):
    """
    2x2 conv, common and separately
    global offset estimation for adolp
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = ConvLayer(self.num_bins, base_num_channels, kernel_size=2, stride=2, padding=0)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_i = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_i = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # a
        self.head_a = ConvLayer(self.num_bins, base_num_channels, kernel_size=2, stride=2, padding=0)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_a = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_a = ResidualBlock(base_num_channels, base_num_channels)
        self.offset_a = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                    nn.Conv2d(base_num_channels, base_num_channels // 4, 1),
                                    nn.ReLU(),
                                    nn.Conv2d(base_num_channels // 4, 1, 1))
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # d
        self.head_d = ConvLayer(self.num_bins, base_num_channels, kernel_size=2, stride=2, padding=0)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_d = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_d = ResidualBlock(base_num_channels, base_num_channels)
        self.offset_d = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                    nn.Conv2d(base_num_channels, base_num_channels // 4, 1),
                                    nn.ReLU(),
                                    nn.Conv2d(base_num_channels // 4, 1, 1))
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
        """
        # i branch
        x_i = self.head_i(x)
        x_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_i
        x_i = self.R1_i(x_i)
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i
        x_i = self.R2_i(x_i)
        i = self.pred_i(x_i)

        # a branch
        x_a = self.head_a(x)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a
        x_a = self.R1_a(x_a)
        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a
        x_a = self.R2_a(x_a)
        offset_a = self.offset_a(x_a)
        a = self.pred_a(x_a)
        a = a - offset_a

        # d branch
        x_d = self.head_d(x)
        x_d = self.G1_d(x_d, self._states_d[0])
        self._states_d[0] = x_d
        x_d = self.R1_d(x_d)
        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d
        x_d = self.R2_d(x_d)
        offset_d = self.offset_d(x_d)
        d = self.pred_d(x_d)
        d = d - offset_d

        return {
            'i': i,
            'a': a,
            'd': d
        }


class M53(BaseModel):
    """
    2x2 conv, common and separately
    element-wise offset estimation for adolp
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = ConvLayer(self.num_bins, base_num_channels, kernel_size=2, stride=2, padding=0)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_i = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_i = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # a
        self.head_a = ConvLayer(self.num_bins, base_num_channels, kernel_size=2, stride=2, padding=0)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_a = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_a = ResidualBlock(base_num_channels, base_num_channels)
        self.offset_a = nn.Sequential(nn.Conv2d(1, 1, kernel_size=(25, 1), padding=(12, 0)),
                                      nn.Conv2d(1, 1, kernel_size=(1, 25), padding=(0, 12)),
                                      nn.ReLU(),
                                      nn.Conv2d(1, 1, 1)
                                      )
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # d
        self.head_d = ConvLayer(self.num_bins, base_num_channels, kernel_size=2, stride=2, padding=0)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_d = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_d = ResidualBlock(base_num_channels, base_num_channels)
        self.offset_d = nn.Sequential(nn.Conv2d(1, 1, kernel_size=(25, 1), padding=(12, 0)),
                                      nn.Conv2d(1, 1, kernel_size=(1, 25), padding=(0, 12)),
                                      nn.ReLU(),
                                      nn.Conv2d(1, 1, 1)
                                      )
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
        """
        # i branch
        x_i = self.head_i(x)
        x_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_i
        x_i = self.R1_i(x_i)
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i
        x_i = self.R2_i(x_i)
        i = self.pred_i(x_i)

        # a branch
        x_a = self.head_a(x)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a
        x_a = self.R1_a(x_a)
        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a
        x_a = self.R2_a(x_a)
        offset_a = self.offset_a(torch.mean(x_a, 1, keepdim=True))
        a = self.pred_a(x_a)
        a = a - offset_a

        # d branch
        x_d = self.head_d(x)
        x_d = self.G1_d(x_d, self._states_d[0])
        self._states_d[0] = x_d
        x_d = self.R1_d(x_d)
        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d
        x_d = self.R2_d(x_d)
        offset_d = self.offset_d(torch.mean(x_d, 1, keepdim=True))
        d = self.pred_d(x_d)
        d = d - offset_d

        return {
            'i': i,
            'a': a,
            'd': d
        }


class M54(BaseModel):
    """
    2x2 conv, common and separately
    iad cross feature enhancement
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = nn.Sequential(nn.Conv2d(self.num_bins, base_num_channels, kernel_size=2, stride=2, padding=0),
                                    nn.BatchNorm2d(base_num_channels), nn.ReLU())
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_i = ResidualBlock(base_num_channels, base_num_channels)
        self.cross_i = nn.Sequential(nn.Conv2d(base_num_channels * 3, base_num_channels, 1),
                                     nn.BatchNorm2d(base_num_channels), nn.ReLU())
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_i = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # a
        self.head_a = nn.Sequential(nn.Conv2d(self.num_bins, base_num_channels, kernel_size=2, stride=2, padding=0),
                                    nn.BatchNorm2d(base_num_channels), nn.ReLU())
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_a = ResidualBlock(base_num_channels, base_num_channels)
        self.cross_a = nn.Sequential(nn.Conv2d(base_num_channels * 3, base_num_channels, 1),
                                     nn.BatchNorm2d(base_num_channels), nn.ReLU())
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_a = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # d
        self.head_d = nn.Sequential(nn.Conv2d(self.num_bins, base_num_channels, kernel_size=2, stride=2, padding=0),
                                    nn.BatchNorm2d(base_num_channels), nn.ReLU())
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_d = ResidualBlock(base_num_channels, base_num_channels)
        self.cross_d = nn.Sequential(nn.Conv2d(base_num_channels * 3, base_num_channels, 1),
                                     nn.BatchNorm2d(base_num_channels), nn.ReLU())
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_d = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
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

        # cross stage
        cross_i = self.cross_i(torch.cat([x_i, x_a, x_d], 1))
        cross_a = self.cross_a(torch.cat([x_a, x_i, x_d], 1))
        cross_d = self.cross_d(torch.cat([x_d, x_i, x_a], 1))

        # second stage
        x_i = self.G2_i(cross_i, self._states_i[1])
        self._states_i[1] = x_i
        x_i = self.R2_i(x_i)
        i = self.pred_i(x_i)

        x_a = self.G2_a(cross_a, self._states_a[1])
        self._states_a[1] = x_a
        x_a = self.R2_a(x_a)
        a = self.pred_a(x_a)

        x_d = self.G2_d(cross_d, self._states_d[1])
        self._states_d[1] = x_d
        x_d = self.R2_d(x_d)
        d = self.pred_d(x_d)

        return {
            'i': i,
            'a': a,
            'd': d
        }


class M55(BaseModel):
    """
    2x2 conv, common and separately
    iad cross feature enhancement
    gemm
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = nn.Sequential(nn.Conv2d(self.num_bins, base_num_channels, kernel_size=2, stride=2, padding=0),
                                    nn.BatchNorm2d(base_num_channels), nn.ReLU())
        self.gemm_i = DepthWiseConv2dImplicitGEMM(base_num_channels, 15, True)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_i = ResidualBlock(base_num_channels, base_num_channels)
        self.cross_i = nn.Sequential(nn.Conv2d(base_num_channels * 3, base_num_channels, 1),
                                     nn.BatchNorm2d(base_num_channels), nn.ReLU())
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_i = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # a
        self.head_a = nn.Sequential(nn.Conv2d(self.num_bins, base_num_channels, kernel_size=2, stride=2, padding=0),
                                    nn.BatchNorm2d(base_num_channels), nn.ReLU())
        self.gemm_a = DepthWiseConv2dImplicitGEMM(base_num_channels, 15, True)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_a = ResidualBlock(base_num_channels, base_num_channels)
        self.cross_a = nn.Sequential(nn.Conv2d(base_num_channels * 3, base_num_channels, 1),
                                     nn.BatchNorm2d(base_num_channels), nn.ReLU())
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_a = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # d
        self.head_d = nn.Sequential(nn.Conv2d(self.num_bins, base_num_channels, kernel_size=2, stride=2, padding=0),
                                    nn.BatchNorm2d(base_num_channels), nn.ReLU())
        self.gemm_d = DepthWiseConv2dImplicitGEMM(base_num_channels, 15, True)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_d = ResidualBlock(base_num_channels, base_num_channels)
        self.cross_d = nn.Sequential(nn.Conv2d(base_num_channels * 3, base_num_channels, 1),
                                     nn.BatchNorm2d(base_num_channels), nn.ReLU())
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_d = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
        """
        # first stage
        x_i = self.head_i(x)
        x_i = self.gemm_i(x_i)
        x_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_i
        x_i = self.R1_i(x_i)

        x_a = self.head_a(x)
        x_a = self.gemm_a(x_a)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a
        x_a = self.R1_a(x_a)

        x_d = self.head_d(x)
        x_d = self.gemm_d(x_d)
        x_d = self.G1_d(x_d, self._states_d[0])
        self._states_d[0] = x_d
        x_d = self.R1_d(x_d)

        # cross stage
        cross_i = self.cross_i(torch.cat([x_i, x_a, x_d], 1))
        cross_a = self.cross_a(torch.cat([x_a, x_i, x_d], 1))
        cross_d = self.cross_d(torch.cat([x_d, x_i, x_a], 1))

        # second stage
        x_i = self.G2_i(cross_i, self._states_i[1])
        self._states_i[1] = x_i
        x_i = self.R2_i(x_i)
        i = self.pred_i(x_i)

        x_a = self.G2_a(cross_a, self._states_a[1])
        self._states_a[1] = x_a
        x_a = self.R2_a(x_a)
        a = self.pred_a(x_a)

        x_d = self.G2_d(cross_d, self._states_d[1])
        self._states_d[1] = x_d
        x_d = self.R2_d(x_d)
        d = self.pred_d(x_d)

        return {
            'i': i,
            'a': a,
            'd': d
        }


class M56(BaseModel):
    """
    2x2 conv, common and separately
    iad cross feature enhancement
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = nn.Sequential(nn.Conv2d(self.num_bins, base_num_channels, kernel_size=2, stride=2, padding=0),
                                    nn.BatchNorm2d(base_num_channels), nn.ReLU())
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_i = ResidualBlock(base_num_channels, base_num_channels)
        self.cross_i = nn.Sequential(nn.Conv2d(base_num_channels * 3, base_num_channels, 1),
                                     nn.BatchNorm2d(base_num_channels), nn.ReLU(), CBAM(base_num_channels))
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_i = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # a
        self.head_a = nn.Sequential(nn.Conv2d(self.num_bins, base_num_channels, kernel_size=2, stride=2, padding=0),
                                    nn.BatchNorm2d(base_num_channels), nn.ReLU())
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_a = ResidualBlock(base_num_channels, base_num_channels)
        self.cross_a = nn.Sequential(nn.Conv2d(base_num_channels * 3, base_num_channels, 1),
                                     nn.BatchNorm2d(base_num_channels), nn.ReLU(), CBAM(base_num_channels))
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_a = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # d
        self.head_d = nn.Sequential(nn.Conv2d(self.num_bins, base_num_channels, kernel_size=2, stride=2, padding=0),
                                    nn.BatchNorm2d(base_num_channels), nn.ReLU())
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_d = ResidualBlock(base_num_channels, base_num_channels)
        self.cross_d = nn.Sequential(nn.Conv2d(base_num_channels * 3, base_num_channels, 1),
                                     nn.BatchNorm2d(base_num_channels), nn.ReLU(), CBAM(base_num_channels))
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_d = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
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

        # cross stage
        cross_i = self.cross_i(torch.cat([x_i, x_a, x_d], 1))
        cross_a = self.cross_a(torch.cat([x_a, x_i, x_d], 1))
        cross_d = self.cross_d(torch.cat([x_d, x_i, x_a], 1))

        # second stage
        x_i = self.G2_i(cross_i, self._states_i[1])
        self._states_i[1] = x_i
        x_i = self.R2_i(x_i)
        i = self.pred_i(x_i)

        x_a = self.G2_a(cross_a, self._states_a[1])
        self._states_a[1] = x_a
        x_a = self.R2_a(x_a)
        a = self.pred_a(x_a)

        x_d = self.G2_d(cross_d, self._states_d[1])
        self._states_d[1] = x_d
        x_d = self.R2_d(x_d)
        d = self.pred_d(x_d)

        return {
            'i': i,
            'a': a,
            'd': d
        }


class M57(BaseModel):
    """
    2x2 conv, common and separately
    early iad cross feature enhancement
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = nn.Sequential(nn.Conv2d(self.num_bins, base_num_channels, kernel_size=2, stride=2, padding=0),
                                    nn.BatchNorm2d(base_num_channels), nn.ReLU())
        self.cross_i = nn.Sequential(nn.Conv2d(base_num_channels * 3, base_num_channels, 1),
                                     nn.BatchNorm2d(base_num_channels), nn.ReLU(), CBAM(base_num_channels))
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_i = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_i = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # a
        self.head_a = nn.Sequential(nn.Conv2d(self.num_bins, base_num_channels, kernel_size=2, stride=2, padding=0),
                                    nn.BatchNorm2d(base_num_channels), nn.ReLU())
        self.cross_a = nn.Sequential(nn.Conv2d(base_num_channels * 3, base_num_channels, 1),
                                     nn.BatchNorm2d(base_num_channels), nn.ReLU(), CBAM(base_num_channels))
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_a = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_a = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # d
        self.head_d = nn.Sequential(nn.Conv2d(self.num_bins, base_num_channels, kernel_size=2, stride=2, padding=0),
                                    nn.BatchNorm2d(base_num_channels), nn.ReLU())
        self.cross_d = nn.Sequential(nn.Conv2d(base_num_channels * 3, base_num_channels, 1),
                                     nn.BatchNorm2d(base_num_channels), nn.ReLU(), CBAM(base_num_channels))
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_d = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_d = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
        """
        # first stage
        x_i = self.head_i(x)
        x_a = self.head_a(x)
        x_d = self.head_d(x)

        # cross stage
        cross_i = self.cross_i(torch.cat([x_i, x_a, x_d], 1))
        cross_a = self.cross_a(torch.cat([x_a, x_i, x_d], 1))
        cross_d = self.cross_d(torch.cat([x_d, x_i, x_a], 1))

        # second stage
        x_i = self.G1_i(cross_i, self._states_i[0])
        self._states_i[0] = x_i
        x_i = self.R1_i(x_i)
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i
        x_i = self.R2_i(x_i)
        i = self.pred_i(x_i)

        x_a = self.G1_a(cross_a, self._states_a[0])
        self._states_a[0] = x_a
        x_a = self.R1_a(x_a)
        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a
        x_a = self.R2_a(x_a)
        a = self.pred_a(x_a)

        x_d = self.G1_d(cross_d, self._states_d[0])
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


class M58(BaseModel):
    """
    2x2 conv + context residual.
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = nn.Sequential(ConvLayer(self.num_bins, base_num_channels, kernel_size=2, stride=2, padding=0),
                                    ResidualBlock(base_num_channels, base_num_channels))
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_i = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_i = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # a
        self.head_a = nn.Sequential(ConvLayer(self.num_bins, base_num_channels, kernel_size=2, stride=2, padding=0),
                                    ResidualBlock(base_num_channels, base_num_channels))
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_a = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_a = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # d
        self.head_d = nn.Sequential(ConvLayer(self.num_bins, base_num_channels, kernel_size=2, stride=2, padding=0),
                                    ResidualBlock(base_num_channels, base_num_channels))
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_d = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_d = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
        """
        # i branch
        x_i = self.head_i(x)
        x_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_i
        x_i = self.R1_i(x_i)
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i
        x_i = self.R2_i(x_i)
        i = self.pred_i(x_i)

        # a branch
        x_a = self.head_a(x)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a
        x_a = self.R1_a(x_a)
        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a
        x_a = self.R2_a(x_a)
        a = self.pred_a(x_a)

        # d branch
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


class M59(BaseModel):
    """
    2x2 conv + context conv
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = nn.Sequential(ConvLayer(self.num_bins, base_num_channels * 2, kernel_size=2, stride=2, padding=0),
                                    nn.Conv2d(base_num_channels * 2, base_num_channels, 3, 1, 1))
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_i = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_i = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # a
        self.head_a = nn.Sequential(ConvLayer(self.num_bins, base_num_channels * 2, kernel_size=2, stride=2, padding=0),
                                    nn.Conv2d(base_num_channels * 2, base_num_channels, 3, 1, 1))
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_a = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_a = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # d
        self.head_d = nn.Sequential(ConvLayer(self.num_bins, base_num_channels * 2, kernel_size=2, stride=2, padding=0),
                                    nn.Conv2d(base_num_channels * 2, base_num_channels, 3, 1, 1))
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_d = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_d = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
        """
        # i branch
        x_i = self.head_i(x)
        x_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_i
        x_i = self.R1_i(x_i)
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i
        x_i = self.R2_i(x_i)
        i = self.pred_i(x_i)

        # a branch
        x_a = self.head_a(x)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a
        x_a = self.R1_a(x_a)
        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a
        x_a = self.R2_a(x_a)
        a = self.pred_a(x_a)

        # d branch
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


class M60(BaseModel):
    """
    2x2 conv, common and separately.
    use external attention to bridge correlations between iad.
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = ConvLayer(self.num_bins, base_num_channels, kernel_size=2, stride=2, padding=0)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_i = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_i = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # a
        self.head_a = ConvLayer(self.num_bins, base_num_channels, kernel_size=2, stride=2, padding=0)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_a = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_a = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # d
        self.head_d = ConvLayer(self.num_bins, base_num_channels, kernel_size=2, stride=2, padding=0)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_d = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_d = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)

        self.ExternalAttention = ExternalAttention(d_model=base_num_channels)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
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

        x_i, x_a, x_d = self.ExternalAttention(x_i, x_a, x_d)

        # second stage
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i
        x_i = self.R2_i(x_i)
        i = self.pred_i(x_i)

        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a
        x_a = self.R2_a(x_a)
        a = self.pred_a(x_a)

        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d
        x_d = self.R2_d(x_d)
        d = self.pred_d(x_d)

        return {
            'i': i,
            'a': a,
            'd': d
        }


class M61(BaseModel):
    """
    2x2 conv.
    separately.
    sk iad fusion.
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = ConvLayer(self.num_bins, base_num_channels, kernel_size=2, stride=2, padding=0)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_i = ResidualBlock(base_num_channels, base_num_channels)
        self.sk_i = SKAttention(channel=base_num_channels, reduction=2)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_i = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # a
        self.head_a = ConvLayer(self.num_bins, base_num_channels, kernel_size=2, stride=2, padding=0)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_a = ResidualBlock(base_num_channels, base_num_channels)
        self.sk_a = SKAttention(channel=base_num_channels, reduction=2)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_a = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # d
        self.head_d = ConvLayer(self.num_bins, base_num_channels, kernel_size=2, stride=2, padding=0)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_d = ResidualBlock(base_num_channels, base_num_channels)
        self.sk_d = SKAttention(channel=base_num_channels, reduction=2)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_d = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
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

        # fusion
        x_i = self.sk_i(x_i, x_a, x_d)
        x_a = self.sk_a(x_i, x_a, x_d)
        x_d = self.sk_d(x_i, x_a, x_d)

        # second stage
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i
        x_i = self.R2_i(x_i)
        i = self.pred_i(x_i)

        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a
        x_a = self.R2_a(x_a)
        a = self.pred_a(x_a)

        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d
        x_d = self.R2_d(x_d)
        d = self.pred_d(x_d)

        return {
            'i': i,
            'a': a,
            'd': d
        }


class M63(BaseModel):
    """
    2x2 conv + psa.
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = ConvLayer(self.num_bins, base_num_channels, kernel_size=2, stride=2, padding=0)
        self.psa_i = SequentialPolarizedSelfAttention(channel=base_num_channels)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_i = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_i = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # a
        self.head_a = ConvLayer(self.num_bins, base_num_channels, kernel_size=2, stride=2, padding=0)
        self.psa_a = SequentialPolarizedSelfAttention(channel=base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_a = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_a = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # d
        self.head_d = ConvLayer(self.num_bins, base_num_channels, kernel_size=2, stride=2, padding=0)
        self.psa_d = SequentialPolarizedSelfAttention(channel=base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_d = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_d = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
        """
        # i branch
        x_i = self.head_i(x)
        x_i = self.psa_i(x_i)
        x_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_i
        x_i = self.R1_i(x_i)
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i
        x_i = self.R2_i(x_i)
        i = self.pred_i(x_i)

        # a branch
        x_a = self.head_a(x)
        x_a = self.psa_a(x_a)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a
        x_a = self.R1_a(x_a)
        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a
        x_a = self.R2_a(x_a)
        a = self.pred_a(x_a)

        # d branch
        x_d = self.head_d(x)
        x_d = self.psa_d(x_d)
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


class M64(BaseModel):
    """
    rppp (4 + sca)
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = RPPP(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_i = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_i = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # a
        self.head_a = RPPP(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_a = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_a = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # d
        self.head_d = RPPP(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_d = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_d = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
        """
        # i branch
        x_i = self.head_i(x)
        x_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_i
        x_i = self.R1_i(x_i)
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i
        x_i = self.R2_i(x_i)
        i = self.pred_i(x_i)

        # a branch
        x_a = self.head_a(x)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a
        x_a = self.R1_a(x_a)
        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a
        x_a = self.R2_a(x_a)
        a = self.pred_a(x_a)

        # d branch
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


class M65(BaseModel):
    """
    rppp (4 + psa)
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = RPPP_PSA(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_i = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_i = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # a
        self.head_a = RPPP_PSA(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_a = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_a = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # d
        self.head_d = RPPP_PSA(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_d = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_d = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
        """
        # i branch
        x_i = self.head_i(x)
        x_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_i
        x_i = self.R1_i(x_i)
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i
        x_i = self.R2_i(x_i)
        i = self.pred_i(x_i)

        # a branch
        x_a = self.head_a(x)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a
        x_a = self.R1_a(x_a)
        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a
        x_a = self.R2_a(x_a)
        a = self.pred_a(x_a)

        # d branch
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


class M66(BaseModel):
    """
    rppp (4 + 1x1 conv)
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = RPPP_Conv(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_i = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_i = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # a
        self.head_a = RPPP_Conv(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_a = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_a = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # d
        self.head_d = RPPP_Conv(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_d = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_d = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
        """
        # i branch
        x_i = self.head_i(x)
        x_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_i
        x_i = self.R1_i(x_i)
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i
        x_i = self.R2_i(x_i)
        i = self.pred_i(x_i)

        # a branch
        x_a = self.head_a(x)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a
        x_a = self.R1_a(x_a)
        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a
        x_a = self.R2_a(x_a)
        a = self.pred_a(x_a)

        # d branch
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


class M67(BaseModel):
    """
    rppp (2+3+4 + 1x1 conv)
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = RPPP_234(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_i = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_i = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # a
        self.head_a = RPPP_234(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_a = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_a = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # d
        self.head_d = RPPP_234(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_d = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_d = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
        """
        # i branch
        x_i = self.head_i(x)
        x_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_i
        x_i = self.R1_i(x_i)
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i
        x_i = self.R2_i(x_i)
        i = self.pred_i(x_i)

        # a branch
        x_a = self.head_a(x)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a
        x_a = self.R1_a(x_a)
        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a
        x_a = self.R2_a(x_a)
        a = self.pred_a(x_a)

        # d branch
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


class M68(BaseModel):
    """
    rppp (2+3 + 1x1 conv)
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = RPPP_23(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_i = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_i = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # a
        self.head_a = RPPP_23(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_a = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_a = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # d
        self.head_d = RPPP_23(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_d = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_d = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
        """
        # i branch
        x_i = self.head_i(x)
        x_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_i
        x_i = self.R1_i(x_i)
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i
        x_i = self.R2_i(x_i)
        i = self.pred_i(x_i)

        # a branch
        x_a = self.head_a(x)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a
        x_a = self.R1_a(x_a)
        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a
        x_a = self.R2_a(x_a)
        a = self.pred_a(x_a)

        # d branch
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


class M69(BaseModel):
    """
    rppp (2+3 + 3x3 conv)
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = RPPP_23_3(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_i = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_i = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # a
        self.head_a = RPPP_23_3(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_a = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_a = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # d
        self.head_d = RPPP_23_3(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_d = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_d = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
        """
        # i branch
        x_i = self.head_i(x)
        x_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_i
        x_i = self.R1_i(x_i)
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i
        x_i = self.R2_i(x_i)
        i = self.pred_i(x_i)

        # a branch
        x_a = self.head_a(x)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a
        x_a = self.R1_a(x_a)
        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a
        x_a = self.R2_a(x_a)
        a = self.pred_a(x_a)

        # d branch
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


class M70(BaseModel):
    """
    rppp (2+3 + 3x3 conv)
    frequency-based residual block
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = RPPP_23_3(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_i = ResidualBlock_Frequency(base_num_channels, base_num_channels)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_i = ResidualBlock_Frequency(base_num_channels, base_num_channels)
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # a
        self.head_a = RPPP_23_3(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_a = ResidualBlock_Frequency(base_num_channels, base_num_channels)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_a = ResidualBlock_Frequency(base_num_channels, base_num_channels)
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # d
        self.head_d = RPPP_23_3(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_d = ResidualBlock_Frequency(base_num_channels, base_num_channels)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_d = ResidualBlock_Frequency(base_num_channels, base_num_channels)
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
        """
        # i branch
        x_i = self.head_i(x)
        x_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_i
        x_i = self.R1_i(x_i)
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i
        x_i = self.R2_i(x_i)
        i = self.pred_i(x_i)

        # a branch
        x_a = self.head_a(x)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a
        x_a = self.R1_a(x_a)
        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a
        x_a = self.R2_a(x_a)
        a = self.pred_a(x_a)

        # d branch
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


class M71(BaseModel):
    """
    rppp (2+3 + triplet attention)
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = RPPP_23_T(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_i = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_i = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # a
        self.head_a = RPPP_23_T(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_a = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_a = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # d
        self.head_d = RPPP_23_T(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_d = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_d = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
        """
        # i branch
        x_i = self.head_i(x)
        x_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_i
        x_i = self.R1_i(x_i)
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i
        x_i = self.R2_i(x_i)
        i = self.pred_i(x_i)

        # a branch
        x_a = self.head_a(x)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a
        x_a = self.R1_a(x_a)
        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a
        x_a = self.R2_a(x_a)
        a = self.pred_a(x_a)

        # d branch
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


class M72(BaseModel):
    """
    rppp (2+3 + triplet attention + dp conv)
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = RPPP_23_T_DS(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_i = nn.Sequential(ResidualBlock(base_num_channels, base_num_channels), TripletAttention())
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_i = nn.Sequential(ResidualBlock(base_num_channels, base_num_channels), TripletAttention())
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, stride=1, padding=0, activation=None)
        # a
        self.head_a = RPPP_23_T_DS(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_a = nn.Sequential(ResidualBlock(base_num_channels, base_num_channels), TripletAttention())
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_a = nn.Sequential(ResidualBlock(base_num_channels, base_num_channels), TripletAttention())
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, stride=1, padding=0, activation=None)
        # d
        self.head_d = RPPP_23_T_DS(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_d = nn.Sequential(ResidualBlock(base_num_channels, base_num_channels), TripletAttention())
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_d = nn.Sequential(ResidualBlock(base_num_channels, base_num_channels), TripletAttention())
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, stride=1, padding=0, activation=None)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
        """
        # i branch
        x_i = self.head_i(x)
        x_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_i
        x_i = self.R1_i(x_i)
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i
        x_i = self.R2_i(x_i)
        i = self.pred_i(x_i)

        # a branch
        x_a = self.head_a(x)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a
        x_a = self.R1_a(x_a)
        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a
        x_a = self.R2_a(x_a)
        a = self.pred_a(x_a)

        # d branch
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


class M73(BaseModel):
    """
    rppp (2+3 + triplet attention)
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = RPPP_23_T(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_i = nn.Sequential(ResidualBlock(base_num_channels, base_num_channels), TripletAttention())
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_i = nn.Sequential(ResidualBlock(base_num_channels, base_num_channels), TripletAttention())
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, stride=1, padding=0, activation=None)
        # a
        self.head_a = RPPP_23_T(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_a = nn.Sequential(ResidualBlock(base_num_channels, base_num_channels), TripletAttention())
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_a = nn.Sequential(ResidualBlock(base_num_channels, base_num_channels), TripletAttention())
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, stride=1, padding=0, activation=None)
        # d
        self.head_d = RPPP_23_T(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_d = nn.Sequential(ResidualBlock(base_num_channels, base_num_channels), TripletAttention())
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_d = nn.Sequential(ResidualBlock(base_num_channels, base_num_channels), TripletAttention())
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, stride=1, padding=0, activation=None)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
        """
        # i branch
        x_i = self.head_i(x)
        x_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_i
        x_i = self.R1_i(x_i)
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i
        x_i = self.R2_i(x_i)
        i = self.pred_i(x_i)

        # a branch
        x_a = self.head_a(x)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a
        x_a = self.R1_a(x_a)
        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a
        x_a = self.R2_a(x_a)
        a = self.pred_a(x_a)

        # d branch
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


class M74(BaseModel):
    """
    rppp (2+3 + add fusion)
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = RPPP_23_plus(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_i = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_i = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # a
        self.head_a = RPPP_23_plus(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_a = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_a = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # d
        self.head_d = RPPP_23_plus(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_d = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_d = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
        """
        # i branch
        x_i = self.head_i(x)
        x_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_i
        x_i = self.R1_i(x_i)
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i
        x_i = self.R2_i(x_i)
        i = self.pred_i(x_i)

        # a branch
        x_a = self.head_a(x)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a
        x_a = self.R1_a(x_a)
        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a
        x_a = self.R2_a(x_a)
        a = self.pred_a(x_a)

        # d branch
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


class M75(BaseModel):
    """
    rppp (2+3 + 1x1 conv)
    hfd
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = RPPP_23(self.num_bins, base_num_channels)
        self.hfd_i = HFD(base_num_channels, 64)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_i = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_i = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # a
        self.head_a = RPPP_23(self.num_bins, base_num_channels)
        self.hfd_a = HFD(base_num_channels, 64)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_a = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_a = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # d
        self.head_d = RPPP_23(self.num_bins, base_num_channels)
        self.hfd_d = HFD(base_num_channels, 64)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_d = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_d = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
        """
        # i branch
        x_i = self.head_i(x)
        hfd_features_i, hfd_i = self.hfd_i(x_i)
        x_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_i
        x_i = self.R1_i(x_i)
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i
        x_i = self.R2_i(x_i)
        i = self.pred_i(x_i)
        final_i = i + hfd_i

        # a branch
        x_a = self.head_a(x)
        hfd_features_a, hfd_a = self.hfd_a(x_a)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a
        x_a = self.R1_a(x_a)
        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a
        x_a = self.R2_a(x_a)
        a = self.pred_a(x_a)
        final_a = a + hfd_a

        # d branch
        x_d = self.head_d(x)
        hfd_features_d, hfd_d = self.hfd_d(x_d)
        x_d = self.G1_d(x_d, self._states_d[0])
        self._states_d[0] = x_d
        x_d = self.R1_d(x_d)
        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d
        x_d = self.R2_d(x_d)
        d = self.pred_d(x_d)
        final_d = d + hfd_d

        return {
            'i': i,
            'a': a,
            'd': d,
            'hfd_features_i': hfd_features_i,
            'hfd_features_a': hfd_features_a,
            'hfd_features_d': hfd_features_d,
            'final_i': final_i,
            'final_a': final_a,
            'final_d': final_d
        }


class M76(BaseModel):
    """
    rppp (2+3 + 1x1 conv)
    auxiliary loss
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = RPPP_23(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_i = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_i = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_i_aux1 = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        self.pred_i_aux2 = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # a
        self.head_a = RPPP_23(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_a = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_a = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_a_aux1 = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        self.pred_a_aux2 = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # d
        self.head_d = RPPP_23(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_d = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_d = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_d_aux1 = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        self.pred_d_aux2 = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
        """
        # i branch
        x_i = self.head_i(x)
        i_aux1 = self.pred_i_aux1(x_i)
        x_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_i
        x_i = self.R1_i(x_i)
        i_aux2 = self.pred_i_aux2(x_i)
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i
        x_i = self.R2_i(x_i)
        i = self.pred_i(x_i)

        # a branch
        x_a = self.head_a(x)
        a_aux1 = self.pred_a_aux1(x_a)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a
        x_a = self.R1_a(x_a)
        a_aux2 = self.pred_a_aux2(x_a)
        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a
        x_a = self.R2_a(x_a)
        a = self.pred_a(x_a)

        # d branch
        x_d = self.head_d(x)
        d_aux1 = self.pred_d_aux1(x_d)
        x_d = self.G1_d(x_d, self._states_d[0])
        self._states_d[0] = x_d
        x_d = self.R1_d(x_d)
        d_aux2 = self.pred_d_aux2(x_d)
        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d
        x_d = self.R2_d(x_d)
        d = self.pred_d(x_d)

        return {
            'i_aux1': i_aux1,
            'i_aux2': i_aux2,
            'i': i,
            'a_aux1': a_aux1,
            'a_aux2': a_aux2,
            'a': a,
            'd_aux1': d_aux1,
            'd_aux2': d_aux2,
            'd': d
        }


class M77(BaseModel):
    """
    rppp (2+3 + 1x1 conv)
    frequency branch
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = RPPP_23(self.num_bins, base_num_channels)
        self.cr1_i = nn.Conv2d(base_num_channels, 1, 1, 1, 0)
        self.f1_i = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64), nn.ReLU(),
                                nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0))
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_i = ResidualBlock(base_num_channels, base_num_channels)
        self.cr2_i = nn.Conv2d(base_num_channels, 1, 1, 1, 0)
        self.f2_i = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64), nn.ReLU(),
                                  nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0))
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_i = ResidualBlock(base_num_channels, base_num_channels)
        self.cr3_i = nn.Conv2d(base_num_channels, 1, 1, 1, 0)
        self.f3_i = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64), nn.ReLU(),
                                  nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0))
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # a
        self.head_a = RPPP_23(self.num_bins, base_num_channels)
        self.cr1_a = nn.Conv2d(base_num_channels, 1, 1, 1, 0)
        self.f1_a = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64), nn.ReLU(),
                                  nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0))
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_a = ResidualBlock(base_num_channels, base_num_channels)
        self.cr2_a = nn.Conv2d(base_num_channels, 1, 1, 1, 0)
        self.f2_a = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64), nn.ReLU(),
                                  nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0))
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_a = ResidualBlock(base_num_channels, base_num_channels)
        self.cr3_a = nn.Conv2d(base_num_channels, 1, 1, 1, 0)
        self.f3_a = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64), nn.ReLU(),
                                  nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0))
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # d
        self.head_d = RPPP_23(self.num_bins, base_num_channels)
        self.cr1_d = nn.Conv2d(base_num_channels, 1, 1, 1, 0)
        self.f1_d = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64), nn.ReLU(),
                                  nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0))
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_d = ResidualBlock(base_num_channels, base_num_channels)
        self.cr2_d = nn.Conv2d(base_num_channels, 1, 1, 1, 0)
        self.f2_d = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64), nn.ReLU(),
                                  nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0))
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_d = ResidualBlock(base_num_channels, base_num_channels)
        self.cr3_d = nn.Conv2d(base_num_channels, 1, 1, 1, 0)
        self.f3_d = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64), nn.ReLU(),
                                  nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0))
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
        """
        b, c, h, w = x.shape
        h = h // 2
        w = w // 2
        h_small = math.ceil(h / 8)
        w_small = math.ceil(w / 8)
        h_pad = h_small * 8 - h
        w_pad = w_small * 8 - w

        x_pad = nn.ZeroPad2d((0, w_pad, 0, h_pad))(x)

        # i branch
        x_i = self.head_i(x_pad)
        cr1_i = self.cr1_i(x_i)
        dct1_i = block_dct(blockify(cr1_i, 8))
        dct1_i = dct1_i.view(b, 1, h_small, w_small, 8, 8).view(b, 1, h_small, w_small, 64).permute(0, 1, 4, 2, 3).squeeze(1)
        f1_i = self.f1_i(dct1_i)
        x_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_i
        x_i = self.R1_i(x_i)
        cr2_i = self.cr2_i(x_i)
        dct2_i = block_dct(blockify(cr2_i, 8))
        dct2_i = dct2_i.view(b, 1, h_small, w_small, 8, 8).view(b, 1, h_small, w_small, 64).permute(0, 1, 4, 2, 3).squeeze(1)
        f2_i = self.f2_i(f1_i + dct2_i)
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i
        x_i = self.R2_i(x_i)
        cr3_i = self.cr3_i(x_i)
        dct3_i = block_dct(blockify(cr3_i, 8))
        dct3_i = dct3_i.view(b, 1, h_small, w_small, 8, 8).view(b, 1, h_small, w_small, 64).permute(0, 1, 4, 2, 3).squeeze(1)
        f3_i = self.f3_i(f2_i + dct3_i)
        i_f = deblockify(block_idct(
            f3_i[:, None, :, :, :].permute(0, 1, 3, 4, 2).view(b, 1, h_small * w_small, 64).
            view(b, 1, h_small * w_small, 8, 8)), (h_small * 8, w_small * 8))
        i_s = self.pred_i(x_i)
        i = i_f + i_s
        if h_pad != 0 and w_pad != 0:
            i = i[:, :, :-h_pad, :-w_pad]

        # a branch
        x_a = self.head_a(x_pad)
        cr1_a = self.cr1_a(x_a)
        dct1_a = block_dct(blockify(cr1_a, 8))
        dct1_a = dct1_a.view(b, 1, h_small, w_small, 8, 8).view(b, 1, h_small, w_small, 64).permute(0, 1, 4, 2, 3).squeeze(1)
        f1_a = self.f1_a(dct1_a)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a
        x_a = self.R1_a(x_a)
        cr2_a = self.cr2_a(x_a)
        dct2_a = block_dct(blockify(cr2_a, 8))
        dct2_a = dct2_a.view(b, 1, h_small, w_small, 8, 8).view(b, 1, h_small, w_small, 64).permute(0, 1, 4, 2, 3).squeeze(1)
        f2_a = self.f2_a(f1_a + dct2_a)
        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a
        x_a = self.R2_a(x_a)
        cr3_a = self.cr3_a(x_a)
        dct3_a = block_dct(blockify(cr3_a, 8))
        dct3_a = dct3_a.view(b, 1, h_small, w_small, 8, 8).view(b, 1, h_small, w_small, 64).permute(0, 1, 4, 2, 3).squeeze(1)
        f3_a = self.f3_a(f2_a + dct3_a)
        a_f = deblockify(block_idct(
            f3_a[:, None, :, :, :].permute(0, 1, 3, 4, 2).view(b, 1, h_small * w_small, 64).
            view(b, 1, h_small * w_small, 8, 8)), (h_small * 8, w_small * 8))
        a_s = self.pred_a(x_a)
        a = a_f + a_s
        if h_pad != 0 and w_pad != 0:
            a = a[:, :, :-h_pad, :-w_pad]

        # d branch
        x_d = self.head_d(x_pad)
        cr1_d = self.cr1_d(x_d)
        dct1_d = block_dct(blockify(cr1_d, 8))
        dct1_d = dct1_d.view(b, 1, h_small, w_small, 8, 8).view(b, 1, h_small, w_small, 64).permute(0, 1, 4, 2, 3).squeeze(1)
        f1_d = self.f1_d(dct1_d)
        x_d = self.G1_d(x_d, self._states_d[0])
        self._states_d[0] = x_d
        x_d = self.R1_d(x_d)
        cr2_d = self.cr2_d(x_d)
        dct2_d = block_dct(blockify(cr2_d, 8))
        dct2_d = dct2_d.view(b, 1, h_small, w_small, 8, 8).view(b, 1, h_small, w_small, 64).permute(0, 1, 4, 2, 3).squeeze(1)
        f2_d = self.f2_d(f1_d + dct2_d)
        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d
        x_d = self.R2_d(x_d)
        cr3_d = self.cr3_d(x_d)
        dct3_d = block_dct(blockify(cr3_d, 8))
        dct3_d = dct3_d.view(b, 1, h_small, w_small, 8, 8).view(b, 1, h_small, w_small, 64).permute(0, 1, 4, 2, 3).squeeze(1)
        f3_d = self.f3_d(f2_d + dct3_d)
        d_f = deblockify(block_idct(
            f3_d[:, None, :, :, :].permute(0, 1, 3, 4, 2).view(b, 1, h_small * w_small, 64).
            view(b, 1, h_small * w_small, 8, 8)), (h_small * 8, w_small * 8))
        d_s = self.pred_d(x_d)
        d = d_f + d_s
        if h_pad != 0 and w_pad != 0:
            d = d[:, :, :-h_pad, :-w_pad]

        return {
            'i': i,
            'a': a,
            'd': d
        }


class M78(BaseModel):
    """
    rppp (2+3 + 1x1 conv)
    frequency branch
    three supervision in total for each domain
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = RPPP_23(self.num_bins, base_num_channels)
        self.cr1_i = nn.Conv2d(base_num_channels, 1, 1, 1, 0)
        self.f1_i = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64), nn.ReLU(),
                                nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0))
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_i = ResidualBlock(base_num_channels, base_num_channels)
        self.cr2_i = nn.Conv2d(base_num_channels, 1, 1, 1, 0)
        self.f2_i = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64), nn.ReLU(),
                                  nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0))
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_i = ResidualBlock(base_num_channels, base_num_channels)
        self.cr3_i = nn.Conv2d(base_num_channels, 1, 1, 1, 0)
        self.f3_i = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64), nn.ReLU(),
                                  nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0))
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        self.fusion_i = nn.Conv2d(2, 1, 1, 1, 0)
        # a
        self.head_a = RPPP_23(self.num_bins, base_num_channels)
        self.cr1_a = nn.Conv2d(base_num_channels, 1, 1, 1, 0)
        self.f1_a = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64), nn.ReLU(),
                                  nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0))
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_a = ResidualBlock(base_num_channels, base_num_channels)
        self.cr2_a = nn.Conv2d(base_num_channels, 1, 1, 1, 0)
        self.f2_a = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64), nn.ReLU(),
                                  nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0))
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_a = ResidualBlock(base_num_channels, base_num_channels)
        self.cr3_a = nn.Conv2d(base_num_channels, 1, 1, 1, 0)
        self.f3_a = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64), nn.ReLU(),
                                  nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0))
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        self.fusion_a = nn.Conv2d(2, 1, 1, 1, 0)
        # d
        self.head_d = RPPP_23(self.num_bins, base_num_channels)
        self.cr1_d = nn.Conv2d(base_num_channels, 1, 1, 1, 0)
        self.f1_d = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64), nn.ReLU(),
                                  nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0))
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_d = ResidualBlock(base_num_channels, base_num_channels)
        self.cr2_d = nn.Conv2d(base_num_channels, 1, 1, 1, 0)
        self.f2_d = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64), nn.ReLU(),
                                  nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0))
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_d = ResidualBlock(base_num_channels, base_num_channels)
        self.cr3_d = nn.Conv2d(base_num_channels, 1, 1, 1, 0)
        self.f3_d = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64), nn.ReLU(),
                                  nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0))
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        self.fusion_d = nn.Conv2d(2, 1, 1, 1, 0)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
        """
        b, c, h, w = x.shape
        h = h // 2
        w = w // 2
        h_small = math.ceil(h / 8)
        w_small = math.ceil(w / 8)
        h_pad = h_small * 8 - h
        w_pad = w_small * 8 - w

        x_pad = nn.ZeroPad2d((0, w_pad, 0, h_pad))(x)

        # i branch
        x_i = self.head_i(x_pad)
        cr1_i = self.cr1_i(x_i)
        dct1_i = block_dct(blockify(cr1_i, 8))
        dct1_i = dct1_i.view(b, 1, h_small, w_small, 8, 8).view(b, 1, h_small, w_small, 64).permute(0, 1, 4, 2, 3).squeeze(1)
        f1_i = self.f1_i(dct1_i)
        x_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_i
        x_i = self.R1_i(x_i)
        cr2_i = self.cr2_i(x_i)
        dct2_i = block_dct(blockify(cr2_i, 8))
        dct2_i = dct2_i.view(b, 1, h_small, w_small, 8, 8).view(b, 1, h_small, w_small, 64).permute(0, 1, 4, 2, 3).squeeze(1)
        f2_i = self.f2_i(f1_i + dct2_i)
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i
        x_i = self.R2_i(x_i)
        cr3_i = self.cr3_i(x_i)
        dct3_i = block_dct(blockify(cr3_i, 8))
        dct3_i = dct3_i.view(b, 1, h_small, w_small, 8, 8).view(b, 1, h_small, w_small, 64).permute(0, 1, 4, 2, 3).squeeze(1)
        f3_i = self.f3_i(f2_i + dct3_i)
        i_f = deblockify(block_idct(
            f3_i[:, None, :, :, :].permute(0, 1, 3, 4, 2).view(b, 1, h_small * w_small, 64).
            view(b, 1, h_small * w_small, 8, 8)), (h_small * 8, w_small * 8))
        i_s = self.pred_i(x_i)
        if h_pad != 0 and w_pad != 0:
            i_f = i_f[:, :, :-h_pad, :-w_pad]
            i_s = i_s[:, :, :-h_pad, :-w_pad]
        i = self.fusion_i(torch.cat((i_f, i_s), 1))

        # a branch
        x_a = self.head_a(x_pad)
        cr1_a = self.cr1_a(x_a)
        dct1_a = block_dct(blockify(cr1_a, 8))
        dct1_a = dct1_a.view(b, 1, h_small, w_small, 8, 8).view(b, 1, h_small, w_small, 64).permute(0, 1, 4, 2, 3).squeeze(1)
        f1_a = self.f1_a(dct1_a)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a
        x_a = self.R1_a(x_a)
        cr2_a = self.cr2_a(x_a)
        dct2_a = block_dct(blockify(cr2_a, 8))
        dct2_a = dct2_a.view(b, 1, h_small, w_small, 8, 8).view(b, 1, h_small, w_small, 64).permute(0, 1, 4, 2, 3).squeeze(1)
        f2_a = self.f2_a(f1_a + dct2_a)
        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a
        x_a = self.R2_a(x_a)
        cr3_a = self.cr3_a(x_a)
        dct3_a = block_dct(blockify(cr3_a, 8))
        dct3_a = dct3_a.view(b, 1, h_small, w_small, 8, 8).view(b, 1, h_small, w_small, 64).permute(0, 1, 4, 2, 3).squeeze(1)
        f3_a = self.f3_a(f2_a + dct3_a)
        a_f = deblockify(block_idct(
            f3_a[:, None, :, :, :].permute(0, 1, 3, 4, 2).view(b, 1, h_small * w_small, 64).
            view(b, 1, h_small * w_small, 8, 8)), (h_small * 8, w_small * 8))
        a_s = self.pred_a(x_a)
        if h_pad != 0 and w_pad != 0:
            a_f = a_f[:, :, :-h_pad, :-w_pad]
            a_s = a_s[:, :, :-h_pad, :-w_pad]
        a = self.fusion_a(torch.cat((a_f, a_s), 1))

        # d branch
        x_d = self.head_d(x_pad)
        cr1_d = self.cr1_d(x_d)
        dct1_d = block_dct(blockify(cr1_d, 8))
        dct1_d = dct1_d.view(b, 1, h_small, w_small, 8, 8).view(b, 1, h_small, w_small, 64).permute(0, 1, 4, 2, 3).squeeze(1)
        f1_d = self.f1_d(dct1_d)
        x_d = self.G1_d(x_d, self._states_d[0])
        self._states_d[0] = x_d
        x_d = self.R1_d(x_d)
        cr2_d = self.cr2_d(x_d)
        dct2_d = block_dct(blockify(cr2_d, 8))
        dct2_d = dct2_d.view(b, 1, h_small, w_small, 8, 8).view(b, 1, h_small, w_small, 64).permute(0, 1, 4, 2, 3).squeeze(1)
        f2_d = self.f2_d(f1_d + dct2_d)
        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d
        x_d = self.R2_d(x_d)
        cr3_d = self.cr3_d(x_d)
        dct3_d = block_dct(blockify(cr3_d, 8))
        dct3_d = dct3_d.view(b, 1, h_small, w_small, 8, 8).view(b, 1, h_small, w_small, 64).permute(0, 1, 4, 2, 3).squeeze(1)
        f3_d = self.f3_d(f2_d + dct3_d)
        d_f = deblockify(block_idct(
            f3_d[:, None, :, :, :].permute(0, 1, 3, 4, 2).view(b, 1, h_small * w_small, 64).
            view(b, 1, h_small * w_small, 8, 8)), (h_small * 8, w_small * 8))
        d_s = self.pred_d(x_d)
        if h_pad != 0 and w_pad != 0:
            d_f = d_f[:, :, :-h_pad, :-w_pad]
            d_s = d_s[:, :, :-h_pad, :-w_pad]
        d = self.fusion_d(torch.cat((d_f, d_s), 1))

        return {
            'i_s': i_s,
            'i_f': i_f,
            'i': i,
            'a_s': a_s,
            'a_f': a_f,
            'a': a,
            'd_s': d_s,
            'd_f': d_f,
            'd': d
        }


class M79(BaseModel):
    """
    rppp (2+3 + 1x1 conv)
    rich contrast module
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = RPPP_23(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_i = ResidualBlock_CT(base_num_channels)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_i = ResidualBlock_CT(base_num_channels)
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # a
        self.head_a = RPPP_23(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_a = ResidualBlock_CT(base_num_channels)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_a = ResidualBlock_CT(base_num_channels)
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # d
        self.head_d = RPPP_23(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_d = ResidualBlock_CT(base_num_channels)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_d = ResidualBlock_CT(base_num_channels)
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
        """
        # i branch
        x_i = self.head_i(x)
        x_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_i
        x_i = self.R1_i(x_i)
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i
        x_i = self.R2_i(x_i)
        i = self.pred_i(x_i)

        # a branch
        x_a = self.head_a(x)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a
        x_a = self.R1_a(x_a)
        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a
        x_a = self.R2_a(x_a)
        a = self.pred_a(x_a)

        # d branch
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


class M80(BaseModel):
    """
    rppp (2+3 + 1x1 conv)
    rich contrast module
    iad mask attention
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = RPPP_23(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_i = ResidualBlock_CT(base_num_channels)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_i = ResidualBlock_CT(base_num_channels)
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # a
        self.head_a = RPPP_23(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_a = ResidualBlock_CT(base_num_channels)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_a = ResidualBlock_CT(base_num_channels)
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # d
        self.head_d = RPPP_23(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_d = ResidualBlock_CT(base_num_channels)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_d = ResidualBlock_CT(base_num_channels)
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)

        self.iad_attention0 = IAD_Attention(base_num_channels)
        self.iad_attention1 = IAD_Attention(base_num_channels)
        self.iad_attention2 = IAD_Attention(base_num_channels)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
        """
        # zero stage
        x_i = self.head_i(x)
        x_a = self.head_a(x)
        x_d = self.head_d(x)

        x_i, x_a, x_d = self.iad_attention0(x_i, x_a, x_d)

        # first stage
        x_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_i
        x_i = self.R1_i(x_i)

        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a
        x_a = self.R1_a(x_a)

        x_d = self.G1_d(x_d, self._states_d[0])
        self._states_d[0] = x_d
        x_d = self.R1_d(x_d)

        x_i, x_a, x_d = self.iad_attention1(x_i, x_a, x_d)

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

        x_i, x_a, x_d = self.iad_attention2(x_i, x_a, x_d)

        # prediction stage
        i = self.pred_i(x_i)
        a = self.pred_a(x_a)
        d = self.pred_d(x_d)

        return {
            'i': i,
            'a': a,
            'd': d
        }


class M81(BaseModel):
    """
    rppp (2+3 + 1x1 conv)
    multiscale perception module
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = RPPP_23(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_i = ResidualBlock_MS(base_num_channels)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_i = ResidualBlock_MS(base_num_channels)
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # a
        self.head_a = RPPP_23(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_a = ResidualBlock_MS(base_num_channels)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_a = ResidualBlock_MS(base_num_channels)
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # d
        self.head_d = RPPP_23(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_d = ResidualBlock_MS(base_num_channels)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_d = ResidualBlock_MS(base_num_channels)
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
        """
        # i branch
        x_i = self.head_i(x)
        x_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_i
        x_i = self.R1_i(x_i)
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i
        x_i = self.R2_i(x_i)
        i = self.pred_i(x_i)

        # a branch
        x_a = self.head_a(x)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a
        x_a = self.R1_a(x_a)
        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a
        x_a = self.R2_a(x_a)
        a = self.pred_a(x_a)

        # d branch
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


class M82(BaseModel):
    """
    rppp (2+3 + 1x1 conv)
    iad feature fusion
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = RPPP_23(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_i = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_i = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # a
        self.head_a = RPPP_23(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_a = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_a = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # d
        self.head_d = RPPP_23(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_d = ResidualBlock(base_num_channels, base_num_channels)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_d = ResidualBlock(base_num_channels, base_num_channels)
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)

        self.iad_attention0 = IAD_Fusion(base_num_channels)
        self.iad_attention1 = IAD_Fusion(base_num_channels)
        self.iad_attention2 = IAD_Fusion(base_num_channels)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
        """
        # zero stage
        x_i = self.head_i(x)
        x_a = self.head_a(x)
        x_d = self.head_d(x)

        x_i, x_a, x_d = self.iad_attention0(x_i, x_a, x_d)

        # first stage
        x_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_i
        x_i = self.R1_i(x_i)

        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a
        x_a = self.R1_a(x_a)

        x_d = self.G1_d(x_d, self._states_d[0])
        self._states_d[0] = x_d
        x_d = self.R1_d(x_d)

        x_i, x_a, x_d = self.iad_attention1(x_i, x_a, x_d)

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

        x_i, x_a, x_d = self.iad_attention2(x_i, x_a, x_d)

        # prediction stage
        i = self.pred_i(x_i)
        a = self.pred_a(x_a)
        d = self.pred_d(x_d)

        return {
            'i': i,
            'a': a,
            'd': d
        }


class M83(BaseModel):
    """
    rppp (2+3 + 1x1 conv)
    rlfb
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = RPPP_23(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_i = RLFB(base_num_channels)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_i = RLFB(base_num_channels)
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # a
        self.head_a = RPPP_23(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_a = RLFB(base_num_channels)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_a = RLFB(base_num_channels)
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # d
        self.head_d = RPPP_23(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_d = RLFB(base_num_channels)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_d = RLFB(base_num_channels)
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
        """
        # i branch
        x_i = self.head_i(x)
        x_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_i
        x_i = self.R1_i(x_i)
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i
        x_i = self.R2_i(x_i)
        i = self.pred_i(x_i)

        # a branch
        x_a = self.head_a(x)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a
        x_a = self.R1_a(x_a)
        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a
        x_a = self.R2_a(x_a)
        a = self.pred_a(x_a)

        # d branch
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


class M84(BaseModel):
    """
    rppp (2+3 + 1x1 conv)
    rlfbf
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = RPPP_23(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_i = RLFBF(base_num_channels)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_i = RLFBF(base_num_channels)
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # a
        self.head_a = RPPP_23(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_a = RLFBF(base_num_channels)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_a = RLFBF(base_num_channels)
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # d
        self.head_d = RPPP_23(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1_d = RLFBF(base_num_channels)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2_d = RLFBF(base_num_channels)
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
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

        # RLFBF 1
        x_i = self.R1_i(self._states_i[0], self._states_a[0], self._states_d[0])
        x_a = self.R1_a(self._states_a[0], self._states_i[0], self._states_d[0])
        x_d = self.R1_d(self._states_d[0], self._states_i[0], self._states_a[0])

        # second stage
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i

        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a

        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d

        # RLFBF 2
        x_i = self.R2_i(self._states_i[1], self._states_a[1], self._states_d[1])
        x_a = self.R2_a(self._states_a[1], self._states_i[1], self._states_d[1])
        x_d = self.R2_d(self._states_d[1], self._states_i[1], self._states_a[1])

        # prediction stage
        i = self.pred_i(x_i)
        a = self.pred_a(x_a)
        d = self.pred_d(x_d)

        return {
            'i': i,
            'a': a,
            'd': d
        }


class M85(BaseModel):
    """
    rppp (2+3 + 1x1 conv)
    scfe
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = RPPP_23(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # a
        self.head_a = RPPP_23(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # d
        self.head_d = RPPP_23(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)

        self.scfe1 = SCFE()
        self.scfe2 = SCFE()

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
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

        # SCFE 1
        x_i, x_a, x_d = self.scfe1(x_i, x_a, x_d)

        # second stage
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i

        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a

        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d

        # RLFBF 2
        x_i, x_a, x_d = self.scfe2(x_i, x_a, x_d)

        # prediction stage
        i = self.pred_i(x_i)
        a = self.pred_a(x_a)
        d = self.pred_d(x_d)

        return {
            'i': i,
            'a': a,
            'd': d
        }


class M86(BaseModel):
    """
    rppp (2+3 + 1x1 conv)
    scfe
    esa
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = RPPP_23(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.ESA1_i = ESA(base_num_channels // 2, base_num_channels, nn.Conv2d)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.ESA2_i = ESA(base_num_channels // 2, base_num_channels, nn.Conv2d)
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # a
        self.head_a = RPPP_23(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.ESA1_a = ESA(base_num_channels // 2, base_num_channels, nn.Conv2d)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.ESA2_a = ESA(base_num_channels // 2, base_num_channels, nn.Conv2d)
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # d
        self.head_d = RPPP_23(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.ESA1_d = ESA(base_num_channels // 2, base_num_channels, nn.Conv2d)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.ESA2_d = ESA(base_num_channels // 2, base_num_channels, nn.Conv2d)
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)

        self.scfe1 = SCFE()
        self.scfe2 = SCFE()

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
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

        # SCFE 1
        x_i, x_a, x_d = self.scfe1(x_i, x_a, x_d)
        # ESA 1
        x_i = self.ESA1_i(x_i)
        x_a = self.ESA1_a(x_a)
        x_d = self.ESA1_d(x_d)

        # second stage
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i

        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a

        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d

        # RLFBF 2
        x_i, x_a, x_d = self.scfe2(x_i, x_a, x_d)
        # ESA 2
        x_i = self.ESA2_i(x_i)
        x_a = self.ESA2_a(x_a)
        x_d = self.ESA2_d(x_d)

        # prediction stage
        i = self.pred_i(x_i)
        a = self.pred_a(x_a)
        d = self.pred_d(x_d)

        return {
            'i': i,
            'a': a,
            'd': d
        }


class M87(BaseModel):
    """
    rppp (2+3 + 1x1 conv)
    scfe
    esaf
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = RPPP_23(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.ESAF1_i = ESAF(base_num_channels // 2, base_num_channels, nn.Conv2d)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.ESAF2_i = ESAF(base_num_channels // 2, base_num_channels, nn.Conv2d)
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # a
        self.head_a = RPPP_23(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.ESAF1_a = ESAF(base_num_channels // 2, base_num_channels, nn.Conv2d)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.ESAF2_a = ESAF(base_num_channels // 2, base_num_channels, nn.Conv2d)
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # d
        self.head_d = RPPP_23(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.ESAF1_d = ESAF(base_num_channels // 2, base_num_channels, nn.Conv2d)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.ESAF2_d = ESAF(base_num_channels // 2, base_num_channels, nn.Conv2d)
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)

        self.scfe1 = SCFE()
        self.scfe2 = SCFE()

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
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

        # SCFE 1
        x_i, x_a, x_d = self.scfe1(x_i, x_a, x_d)
        # ESAF 1
        x1 = x_i
        y1 = x_a
        z1 = x_d
        x_i = self.ESAF1_i(x1, y1, z1)
        x_a = self.ESAF1_a(y1, x1, z1)
        x_d = self.ESAF1_d(z1, x1, y1)

        # second stage
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i

        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a

        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d

        # RLFBF 2
        x_i, x_a, x_d = self.scfe2(x_i, x_a, x_d)
        # ESA 2
        x2 = x_i
        y2 = x_a
        z2 = x_d
        x_i = self.ESAF2_i(x2, y2, z2)
        x_a = self.ESAF2_a(y2, x2, z2)
        x_d = self.ESAF2_d(z2, x2, y2)

        # prediction stage
        i = self.pred_i(x_i)
        a = self.pred_a(x_a)
        d = self.pred_d(x_d)

        return {
            'i': i,
            'a': a,
            'd': d
        }


class M88(BaseModel):
    """
    rppp (2+3 + 1x1 conv)
    jrlfb
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = RPPP_23(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_i = nn.Conv2d(base_num_channels, 1, 1, 1, 0)
        # a
        self.head_a = RPPP_23(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_a = nn.Conv2d(base_num_channels, 1, 1, 1, 0)
        # d
        self.head_d = RPPP_23(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_d = nn.Conv2d(base_num_channels, 1, 1, 1, 0)

        self.jrlfb1 = JRLFB(base_num_channels)
        self.jrlfb2 = JRLFB(base_num_channels)
        # self.jrlfb1 = JRLFB(base_num_channels, n_layer=1)
        # self.jrlfb2 = JRLFB(base_num_channels, n_layer=1)
        # self.jrlfb1 = JRLFB(base_num_channels, n_layer=3)
        # self.jrlfb2 = JRLFB(base_num_channels, n_layer=3)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
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

        # JRLFB 1
        x_i, x_a, x_d = self.jrlfb1(x_i, x_a, x_d)

        # second stage
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i

        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a

        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d

        # JRLFB 2
        x_i, x_a, x_d = self.jrlfb2(x_i, x_a, x_d)

        # prediction stage
        i = self.pred_i(x_i)
        a = self.pred_a(x_a)
        d = self.pred_d(x_d)

        return {
            'i': i,
            'a': a,
            'd': d
        }


class M89(BaseModel):
    """
    rppp (2+3 + 1x1 conv)
    jrlfb
    only one stage
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = RPPP_23(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, stride=1, padding=0, activation=None)
        # a
        self.head_a = RPPP_23(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, stride=1, padding=0, activation=None)
        # d
        self.head_d = RPPP_23(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, stride=1, padding=0, activation=None)

        self.jrlfb1 = JRLFB(base_num_channels)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 1
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
        :return: N x num_output_channels x H/2 x W/2 iad
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

        # JRLFB 1
        x_i, x_a, x_d = self.jrlfb1(x_i, x_a, x_d)

        # prediction stage
        i = self.pred_i(x_i)
        a = self.pred_a(x_a)
        d = self.pred_d(x_d)

        return {
            'i': i,
            'a': a,
            'd': d
        }


class M90(BaseModel):
    """
    rppp (2+3 + 1x1 conv)
    jrlfb
    only one stage
    add 4x4 kernel
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = RPPP_234(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, stride=1, padding=0, activation=None)
        # a
        self.head_a = RPPP_234(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, stride=1, padding=0, activation=None)
        # d
        self.head_d = RPPP_234(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, stride=1, padding=0, activation=None)

        self.jrlfb1 = JRLFB(base_num_channels)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 1
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
        :return: N x num_output_channels x H/2 x W/2 iad
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

        # JRLFB 1
        x_i, x_a, x_d = self.jrlfb1(x_i, x_a, x_d)

        # prediction stage
        i = self.pred_i(x_i)
        a = self.pred_a(x_a)
        d = self.pred_d(x_d)

        return {
            'i': i,
            'a': a,
            'd': d
        }


class M91(BaseModel):
    """
    rppp (2+3 + 1x1 conv)
    jrlfb
    only one stage
    add 4x4 kernel
    more residual feature extraction
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = RPPP_234(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, stride=1, padding=0, activation=None)
        # a
        self.head_a = RPPP_234(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, stride=1, padding=0, activation=None)
        # d
        self.head_d = RPPP_234(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, stride=1, padding=0, activation=None)

        self.jrlfb1 = JRLFB_more(base_num_channels)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 1
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
        :return: N x num_output_channels x H/2 x W/2 iad
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

        # JRLFB 1
        x_i, x_a, x_d = self.jrlfb1(x_i, x_a, x_d)

        # prediction stage
        i = self.pred_i(x_i)
        a = self.pred_a(x_a)
        d = self.pred_d(x_d)

        return {
            'i': i,
            'a': a,
            'd': d
        }


class M92(BaseModel):
    """
    rppp (2+3 + 1x1 conv)
    joint
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = RPPP_23(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # a
        self.head_a = RPPP_23(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # d
        self.head_d = RPPP_23(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)

        self.joint1 = SCFE(base_num_channels // 2, base_num_channels, 1)
        self.joint2 = SCFE(base_num_channels // 4, base_num_channels, 1)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
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

        # Joint 1
        x_i, x_a, x_d = self.joint1(x_i, x_a, x_d)

        # second stage
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i

        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a

        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d

        # Joint 2
        x_i, x_a, x_d = self.joint2(x_i, x_a, x_d)

        # prediction stage
        i = self.pred_i(x_i)
        a = self.pred_a(x_a)
        d = self.pred_d(x_d)

        return {
            'i': i,
            'a': a,
            'd': d
        }


class M93(BaseModel):
    """
    rppp (2+3 + 1x1 conv)
    joint
    detail enhancement
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = RPPP_23(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # a
        self.head_a = RPPP_23(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # d
        self.head_d = RPPP_23(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)

        self.joint1 = SCFE(base_num_channels // 2, base_num_channels, 1)
        self.joint2 = SCFE(base_num_channels // 4, base_num_channels, 1)

        self.de1_i = DE(base_num_channels, base_num_channels // 2)
        self.de2_i = DE(base_num_channels, base_num_channels // 2)

        self.de1_a = DE(base_num_channels, base_num_channels // 2)
        self.de2_a = DE(base_num_channels, base_num_channels // 2)

        self.de1_d = DE(base_num_channels, base_num_channels // 2)
        self.de2_d = DE(base_num_channels, base_num_channels // 2)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
        """
        # first stage
        x_i = self.head_i(x)
        x_features_i = x_i
        x_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_i

        x_a = self.head_a(x)
        x_features_a = x_a
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a

        x_d = self.head_d(x)
        x_features_d = x_d
        x_d = self.G1_d(x_d, self._states_d[0])
        self._states_d[0] = x_d

        # Joint 1
        x_i, x_a, x_d = self.joint1(x_i, x_a, x_d)

        # Detail Enhancement 1
        x_i = self.de1_i(x_i, x_features_i)
        x_a = self.de1_a(x_a, x_features_a)
        x_d = self.de1_d(x_d, x_features_d)

        # second stage
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i

        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a

        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d

        # Joint 2
        x_i, x_a, x_d = self.joint2(x_i, x_a, x_d)

        # Detail Enhancement 2
        x_i = self.de2_i(x_i, x_features_i)
        x_a = self.de2_a(x_a, x_features_a)
        x_d = self.de2_d(x_d, x_features_d)

        # prediction stage
        i = self.pred_i(x_i)
        a = self.pred_a(x_a)
        d = self.pred_d(x_d)

        return {
            'i': i,
            'a': a,
            'd': d
        }


class M94(BaseModel):
    """
    rppp (2+3 + 1x1 conv)
    joint
    detail enhancement
    share the weights of stages
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = RPPP_23(self.num_bins, base_num_channels)
        self.G_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # a
        self.head_a = RPPP_23(self.num_bins, base_num_channels)
        self.G_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # d
        self.head_d = RPPP_23(self.num_bins, base_num_channels)
        self.G_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)

        self.joint = SCFE(base_num_channels // 4, base_num_channels, 1)

        self.de_i = DE(base_num_channels, base_num_channels // 2)
        self.de_a = DE(base_num_channels, base_num_channels // 2)
        self.de_d = DE(base_num_channels, base_num_channels // 2)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 3
        print(self.num_recurrent_units)
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
        :return: N x num_output_channels x H/2 x W/2 iad
        """
        # first stage
        x_i = self.head_i(x)
        x_features_i = x_i

        x_a = self.head_a(x)
        x_features_a = x_a

        x_d = self.head_d(x)
        x_features_d = x_d

        for j in range(self.num_recurrent_units):
            x_i = self.G_i(x_i, self._states_i[j])
            self._states_i[j] = x_i

            x_a = self.G_a(x_a, self._states_a[j])
            self._states_a[j] = x_a

            x_d = self.G_d(x_d, self._states_d[j])
            self._states_d[j] = x_d

            x_i, x_a, x_d = self.joint(x_i, x_a, x_d)

            x_i = self.de_i(x_i, x_features_i)
            x_a = self.de_a(x_a, x_features_a)
            x_d = self.de_d(x_d, x_features_d)

        # prediction stage
        i = self.pred_i(x_i)
        a = self.pred_a(x_a)
        d = self.pred_d(x_d)

        return {
            'i': i,
            'a': a,
            'd': d
        }


class M95(BaseModel):
    """
    rppp_23_rcab
    joint
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = RPPP_23_RCAB(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # a
        self.head_a = RPPP_23_RCAB(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # d
        self.head_d = RPPP_23_RCAB(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)

        self.joint1 = SCFE(base_num_channels // 4, base_num_channels, 1)
        self.joint2 = SCFE(base_num_channels // 4, base_num_channels, 1)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
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

        # Joint 1
        x_i, x_a, x_d = self.joint1(x_i, x_a, x_d)

        # second stage
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i

        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a

        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d

        # Joint 2
        x_i, x_a, x_d = self.joint2(x_i, x_a, x_d)

        # prediction stage
        i = self.pred_i(x_i)
        a = self.pred_a(x_a)
        d = self.pred_d(x_d)

        return {
            'i': i,
            'a': a,
            'd': d
        }

from utils.util import torch2cv2
class M95_test(BaseModel):
    """
    rppp_23_rcab
    joint
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = RPPP_23_RCAB(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # a
        self.head_a = RPPP_23_RCAB(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # d
        self.head_d = RPPP_23_RCAB(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)

        self.joint1 = SCFE(base_num_channels // 4, base_num_channels, 1)
        self.joint2 = SCFE(base_num_channels // 4, base_num_channels, 1)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
        """
        # f0 = torch2cv2(x[0, 0, :, :])
        # f1 = torch2cv2(x[0, 1, :, :])
        # f2 = torch2cv2(x[0, 2, :, :])
        # f3 = torch2cv2(x[0, 3, :, :])
        # f4 = torch2cv2(x[0, 4, :, :])
        #
        # f = np.hconcat([f0, f1, f2, f3, f4])
        # cv2.imshow('f', f)
        # cv2.waitKey(0)

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

        # Joint 1
        x_i, x_a, x_d = self.joint1(x_i, x_a, x_d)

        # second stage
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i

        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a

        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d

        # Joint 2
        x_i, x_a, x_d = self.joint2(x_i, x_a, x_d)

        # prediction stage
        i = self.pred_i(x_i)
        a = self.pred_a(x_a)
        d = self.pred_d(x_d)

        return {
            'i': i,
            'a': a,
            'd': d
        }


class M96(BaseModel):
    """
    rppp + ca + conv 1x1
    joint
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = RPPP_CA_Conv1x1(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # a
        self.head_a = RPPP_CA_Conv1x1(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # d
        self.head_d = RPPP_CA_Conv1x1(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)

        self.joint1 = SCFE(base_num_channels // 4, base_num_channels, 1)
        self.joint2 = SCFE(base_num_channels // 4, base_num_channels, 1)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
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

        # Joint 1
        x_i, x_a, x_d = self.joint1(x_i, x_a, x_d)

        # second stage
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i

        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a

        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d

        # Joint 2
        x_i, x_a, x_d = self.joint2(x_i, x_a, x_d)

        # prediction stage
        i = self.pred_i(x_i)
        a = self.pred_a(x_a)
        d = self.pred_d(x_d)

        return {
            'i': i,
            'a': a,
            'd': d
        }


class M97(BaseModel):
    """
    contrasted rppp + ca
    joint
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = RPPP_Contrasted(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # a
        self.head_a = RPPP_Contrasted(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # d
        self.head_d = RPPP_Contrasted(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)

        self.joint1 = SCFE(base_num_channels // 4, base_num_channels, 1)
        self.joint2 = SCFE(base_num_channels // 4, base_num_channels, 1)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
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

        # Joint 1
        x_i, x_a, x_d = self.joint1(x_i, x_a, x_d)

        # second stage
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i

        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a

        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d

        # Joint 2
        x_i, x_a, x_d = self.joint2(x_i, x_a, x_d)

        # prediction stage
        i = self.pred_i(x_i)
        a = self.pred_a(x_a)
        d = self.pred_d(x_d)

        return {
            'i': i,
            'a': a,
            'd': d
        }


class M98(BaseModel):
    """
    contrasted rppp + ca
    hor
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = RPPP_Contrasted(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # a
        self.head_a = RPPP_Contrasted(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # d
        self.head_d = RPPP_Contrasted(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)

        self.joint1 = gnconv_iad(base_num_channels)
        self.joint2 = gnconv_iad(base_num_channels)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
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

        # Joint 1
        x_i, x_a, x_d = self.joint1(x_i, x_a, x_d)

        # second stage
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i

        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a

        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d

        # Joint 2
        x_i, x_a, x_d = self.joint2(x_i, x_a, x_d)

        # prediction stage
        i = self.pred_i(x_i)
        a = self.pred_a(x_a)
        d = self.pred_d(x_d)

        return {
            'i': i,
            'a': a,
            'd': d
        }


class M99(BaseModel):
    """
    contrasted rppp + ca
    scfe + hor
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = RPPP_Contrasted(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # a
        self.head_a = RPPP_Contrasted(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # d
        self.head_d = RPPP_Contrasted(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)

        self.joint1 = SCFE_HOR(base_num_channels // 2, base_num_channels, 1)
        self.joint2 = SCFE_HOR(base_num_channels // 2, base_num_channels, 1)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
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

        # Joint 1
        x_i, x_a, x_d = self.joint1(x_i, x_a, x_d)

        # second stage
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i

        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a

        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d

        # Joint 2
        x_i, x_a, x_d = self.joint2(x_i, x_a, x_d)

        # prediction stage
        i = self.pred_i(x_i)
        a = self.pred_a(x_a)
        d = self.pred_d(x_d)

        return {
            'i': i,
            'a': a,
            'd': d
        }


class M100(BaseModel):
    """
    contrasted rppp + ca
    scfe + hor
    global filter detail enhancement
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = RPPP_Contrasted(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # a
        self.head_a = RPPP_Contrasted(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # d
        self.head_d = RPPP_Contrasted(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)

        self.joint1 = SCFE_HOR(base_num_channels // 2, base_num_channels, 1)
        self.joint2 = SCFE_HOR(base_num_channels // 2, base_num_channels, 1)

        self.gfde = GFDE(base_num_channels)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
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

        # Joint 1
        x_i, x_a, x_d = self.joint1(x_i, x_a, x_d)

        # second stage
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i

        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a

        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d

        # Joint 2
        x_i, x_a, x_d = self.joint2(x_i, x_a, x_d)

        # detail enhancement
        d_i, d_a, d_d = self.gfde(x_i, x_a, x_d)

        # prediction stage
        i = self.pred_i(x_i)
        a = self.pred_a(x_a)
        d = self.pred_d(x_d)

        return {
            'i': i + d_i,
            'a': a + d_a,
            'd': d + d_d
        }


class M101(BaseModel):
    """
    contrasted rppp + ca
    scfe + hor
    output global filter detail enhancement
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = RPPP_Contrasted(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        # a
        self.head_a = RPPP_Contrasted(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        # d
        self.head_d = RPPP_Contrasted(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)

        self.joint1 = SCFE_HOR(base_num_channels // 2, base_num_channels, 1)
        self.joint2 = SCFE_HOR(base_num_channels // 2, base_num_channels, 1)

        self.gfde = GFDE(base_num_channels)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
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

        # Joint 1
        x_i, x_a, x_d = self.joint1(x_i, x_a, x_d)

        # second stage
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i

        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a

        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d

        # Joint 2
        x_i, x_a, x_d = self.joint2(x_i, x_a, x_d)

        # detail enhancement
        d_i, d_a, d_d = self.gfde(x_i, x_a, x_d)

        return {
            'i': d_i,
            'a': d_a,
            'd': d_d
        }


class M102(BaseModel):
    """
    rppp 23 cr ca
    7711 dw+pw
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = RPPP_23_cr_ca(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU_woinit(base_num_channels, base_num_channels, kernel_size)
        self.G2_i = ConvGRU_woinit(base_num_channels, base_num_channels, kernel_size)
        self.pred_i = nn.Conv2d(base_num_channels, 1, 1, 1, 0)
        # a
        self.head_a = RPPP_23_cr_ca(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU_woinit(base_num_channels, base_num_channels, kernel_size)
        self.G2_a = ConvGRU_woinit(base_num_channels, base_num_channels, kernel_size)
        self.pred_a = nn.Conv2d(base_num_channels, 1, 1, 1, 0)
        # d
        self.head_d = RPPP_23_cr_ca(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU_woinit(base_num_channels, base_num_channels, kernel_size)
        self.G2_d = ConvGRU_woinit(base_num_channels, base_num_channels, kernel_size)
        self.pred_d = nn.Conv2d(base_num_channels, 1, 1, 1, 0)

        self.ce1 = DWPW(base_num_channels)
        self.ce2 = DWPW(base_num_channels)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
        self.reset_states_i()
        self.reset_states_a()
        self.reset_states_d()

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def reset_states_i(self):
        self._states_i = [None] * self.num_recurrent_units

    def reset_states_a(self):
        self._states_a = [None] * self.num_recurrent_units

    def reset_states_d(self):
        self._states_d = [None] * self.num_recurrent_units

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W event tensor
        :return: N x num_output_channels x H/2 x W/2 iad
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

        # Context Exploration 1
        x_i, x_a, x_d = self.ce1(x_i, x_a, x_d)

        # second stage
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i

        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a

        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d

        # Context Exploration 2
        x_i, x_a, x_d = self.ce2(x_i, x_a, x_d)

        # prediction stage
        i = self.pred_i(x_i)
        a = self.pred_a(x_a)
        d = self.pred_d(x_d)

        return {
            'i': i,
            'a': a,
            'd': d
        }


class M103(BaseModel):
    """
    rppp 23 cr ca
    interaction
    7711 dw+pw
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = RPPP_23_cr_ca(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU_woinit(base_num_channels, base_num_channels, kernel_size)
        self.G2_i = ConvGRU_woinit(base_num_channels, base_num_channels, kernel_size)
        self.pred_i = nn.Conv2d(base_num_channels, 1, 1, 1, 0)
        # a
        self.head_a = RPPP_23_cr_ca(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU_woinit(base_num_channels, base_num_channels, kernel_size)
        self.G2_a = ConvGRU_woinit(base_num_channels, base_num_channels, kernel_size)
        self.pred_a = nn.Conv2d(base_num_channels, 1, 1, 1, 0)
        # d
        self.head_d = RPPP_23_cr_ca(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU_woinit(base_num_channels, base_num_channels, kernel_size)
        self.G2_d = ConvGRU_woinit(base_num_channels, base_num_channels, kernel_size)
        self.pred_d = nn.Conv2d(base_num_channels, 1, 1, 1, 0)

        self.ce1 = IAD_DWPW(base_num_channels)
        self.ce2 = IAD_DWPW(base_num_channels)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
        self.reset_states_i()
        self.reset_states_a()
        self.reset_states_d()

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def reset_states_i(self):
        self._states_i = [None] * self.num_recurrent_units

    def reset_states_a(self):
        self._states_a = [None] * self.num_recurrent_units

    def reset_states_d(self):
        self._states_d = [None] * self.num_recurrent_units

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W event tensor
        :return: N x num_output_channels x H/2 x W/2 iad
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

        # Context Exploration 1
        x_i, x_a, x_d = self.ce1(x_i, x_a, x_d)

        # second stage
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i

        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a

        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d

        # Context Exploration 2
        x_i, x_a, x_d = self.ce2(x_i, x_a, x_d)

        # prediction stage
        i = self.pred_i(x_i)
        a = self.pred_a(x_a)
        d = self.pred_d(x_d)

        return {
            'i': i,
            'a': a,
            'd': d
        }


class M104(BaseModel):
    """
    rppp 23 cr ca
    interaction
    7711 dw+pw
    gfde
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = RPPP_23_cr_ca(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU_woinit(base_num_channels, base_num_channels, kernel_size)
        self.G2_i = ConvGRU_woinit(base_num_channels, base_num_channels, kernel_size)
        self.pred_i = nn.Conv2d(base_num_channels, 1, 1, 1, 0)
        # a
        self.head_a = RPPP_23_cr_ca(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU_woinit(base_num_channels, base_num_channels, kernel_size)
        self.G2_a = ConvGRU_woinit(base_num_channels, base_num_channels, kernel_size)
        self.pred_a = nn.Conv2d(base_num_channels, 1, 1, 1, 0)
        # d
        self.head_d = RPPP_23_cr_ca(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU_woinit(base_num_channels, base_num_channels, kernel_size)
        self.G2_d = ConvGRU_woinit(base_num_channels, base_num_channels, kernel_size)
        self.pred_d = nn.Conv2d(base_num_channels, 1, 1, 1, 0)

        self.ce1 = IAD_DWPW(base_num_channels)
        self.ce2 = IAD_DWPW(base_num_channels)

        self.gfde = GFDE(base_num_channels)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
        self.reset_states_i()
        self.reset_states_a()
        self.reset_states_d()

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def reset_states_i(self):
        self._states_i = [None] * self.num_recurrent_units

    def reset_states_a(self):
        self._states_a = [None] * self.num_recurrent_units

    def reset_states_d(self):
        self._states_d = [None] * self.num_recurrent_units

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W event tensor
        :return: N x num_output_channels x H/2 x W/2 iad
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

        # Context Exploration 1
        x_i, x_a, x_d = self.ce1(x_i, x_a, x_d)

        # second stage
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i

        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a

        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d

        # Context Exploration 2
        x_i, x_a, x_d = self.ce2(x_i, x_a, x_d)

        # detail enhancement
        d_i, d_a, d_d = self.gfde(x_i, x_a, x_d)

        # prediction stage
        i = self.pred_i(x_i)
        a = self.pred_a(x_a)
        d = self.pred_d(x_d)

        return {
            'i': i + d_i,
            'a': a + d_a,
            'd': d + d_d
        }


class M105(BaseModel):
    """
    rppp_23_cr_ca
    scfe_ca_sa
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = RPPP_23_cr_ca(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU_woinit(base_num_channels, base_num_channels, kernel_size)
        self.G2_i = ConvGRU_woinit(base_num_channels, base_num_channels, kernel_size)
        self.pred_i = nn.Conv2d(base_num_channels, 1, 1, 1, 0)
        # a
        self.head_a = RPPP_23_cr_ca(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU_woinit(base_num_channels, base_num_channels, kernel_size)
        self.G2_a = ConvGRU_woinit(base_num_channels, base_num_channels, kernel_size)
        self.pred_a = nn.Conv2d(base_num_channels, 1, 1, 1, 0)
        # d
        self.head_d = RPPP_23_cr_ca(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU_woinit(base_num_channels, base_num_channels, kernel_size)
        self.G2_d = ConvGRU_woinit(base_num_channels, base_num_channels, kernel_size)
        self.pred_d = nn.Conv2d(base_num_channels, 1, 1, 1, 0)

        self.ce1 = SCFE_CASA(base_num_channels // 4, base_num_channels, 1)
        self.ce2 = SCFE_CASA(base_num_channels // 4, base_num_channels, 1)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
        self.reset_states_i()
        self.reset_states_a()
        self.reset_states_d()

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def reset_states_i(self):
        self._states_i = [None] * self.num_recurrent_units

    def reset_states_a(self):
        self._states_a = [None] * self.num_recurrent_units

    def reset_states_d(self):
        self._states_d = [None] * self.num_recurrent_units

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W event tensor
        :return: N x num_output_channels x H/2 x W/2 iad
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

        # Context Exploration 1
        x_i, x_a, x_d = self.ce1(x_i, x_a, x_d)

        # second stage
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i

        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a

        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d

        # Context Exploration 2
        x_i, x_a, x_d = self.ce2(x_i, x_a, x_d)

        # prediction stage
        i = self.pred_i(x_i)
        a = self.pred_a(x_a)
        d = self.pred_d(x_d)

        return {
            'i': i,
            'a': a,
            'd': d
        }


class M106(BaseModel):
    """
    rppp_23_cr_ca
    scfe_ca_sa
    dct_prediction
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = RPPP_23_cr_ca(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU_woinit(base_num_channels, base_num_channels, kernel_size)
        self.G2_i = ConvGRU_woinit(base_num_channels, base_num_channels, kernel_size)
        self.pred_i = DCT_Prediction(base_num_channels)
        # a
        self.head_a = RPPP_23_cr_ca(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU_woinit(base_num_channels, base_num_channels, kernel_size)
        self.G2_a = ConvGRU_woinit(base_num_channels, base_num_channels, kernel_size)
        self.pred_a = DCT_Prediction(base_num_channels)
        # d
        self.head_d = RPPP_23_cr_ca(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU_woinit(base_num_channels, base_num_channels, kernel_size)
        self.G2_d = ConvGRU_woinit(base_num_channels, base_num_channels, kernel_size)
        self.pred_d = DCT_Prediction(base_num_channels)

        self.ce1 = SCFE_CASA(base_num_channels // 4, base_num_channels, 1)
        self.ce2 = SCFE_CASA(base_num_channels // 4, base_num_channels, 1)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
        self.reset_states_i()
        self.reset_states_a()
        self.reset_states_d()

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def reset_states_i(self):
        self._states_i = [None] * self.num_recurrent_units

    def reset_states_a(self):
        self._states_a = [None] * self.num_recurrent_units

    def reset_states_d(self):
        self._states_d = [None] * self.num_recurrent_units

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W event tensor
        :return: N x num_output_channels x H/2 x W/2 iad
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

        # Context Exploration 1
        x_i, x_a, x_d = self.ce1(x_i, x_a, x_d)

        # second stage
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i

        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a

        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d

        # Context Exploration 2
        x_i, x_a, x_d = self.ce2(x_i, x_a, x_d)

        # prediction stage
        i = self.pred_i(x_i)
        a = self.pred_a(x_a)
        d = self.pred_d(x_d)

        return {
            'i': i,
            'a': a,
            'd': d
        }


class M107(BaseModel):
    """
    rppp_23_cr_ca
    scfe_gf_ca_sa
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = RPPP_23_cr_ca(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU_woinit(base_num_channels, base_num_channels, kernel_size)
        self.G2_i = ConvGRU_woinit(base_num_channels, base_num_channels, kernel_size)
        self.pred_i = nn.Conv2d(base_num_channels, 1, 1, 1, 0)
        # a
        self.head_a = RPPP_23_cr_ca(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU_woinit(base_num_channels, base_num_channels, kernel_size)
        self.G2_a = ConvGRU_woinit(base_num_channels, base_num_channels, kernel_size)
        self.pred_a = nn.Conv2d(base_num_channels, 1, 1, 1, 0)
        # d
        self.head_d = RPPP_23_cr_ca(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU_woinit(base_num_channels, base_num_channels, kernel_size)
        self.G2_d = ConvGRU_woinit(base_num_channels, base_num_channels, kernel_size)
        self.pred_d = nn.Conv2d(base_num_channels, 1, 1, 1, 0)

        self.ce1 = SCFE_GF_CASA(base_num_channels // 4, base_num_channels, 1)
        self.ce2 = SCFE_GF_CASA(base_num_channels // 4, base_num_channels, 1)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
        self.reset_states_i()
        self.reset_states_a()
        self.reset_states_d()

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def reset_states_i(self):
        self._states_i = [None] * self.num_recurrent_units

    def reset_states_a(self):
        self._states_a = [None] * self.num_recurrent_units

    def reset_states_d(self):
        self._states_d = [None] * self.num_recurrent_units

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W event tensor
        :return: N x num_output_channels x H/2 x W/2 iad
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

        # Context Exploration 1
        x_i, x_a, x_d = self.ce1(x_i, x_a, x_d)

        # second stage
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i

        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a

        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d

        # Context Exploration 2
        x_i, x_a, x_d = self.ce2(x_i, x_a, x_d)

        # prediction stage
        i = self.pred_i(x_i)
        a = self.pred_a(x_a)
        d = self.pred_d(x_d)

        return {
            'i': i,
            'a': a,
            'd': d
        }


class M108(BaseModel):
    """
    rppp_23_cr_ca
    scfe_ca_sa
    contrast_prediction
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = RPPP_23_cr_ca(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU_woinit(base_num_channels, base_num_channels, kernel_size)
        self.G2_i = ConvGRU_woinit(base_num_channels, base_num_channels, kernel_size)
        self.pred_i = Contrast_Prediction(base_num_channels)
        # a
        self.head_a = RPPP_23_cr_ca(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU_woinit(base_num_channels, base_num_channels, kernel_size)
        self.G2_a = ConvGRU_woinit(base_num_channels, base_num_channels, kernel_size)
        self.pred_a = Contrast_Prediction(base_num_channels)
        # d
        self.head_d = RPPP_23_cr_ca(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU_woinit(base_num_channels, base_num_channels, kernel_size)
        self.G2_d = ConvGRU_woinit(base_num_channels, base_num_channels, kernel_size)
        self.pred_d = Contrast_Prediction(base_num_channels)

        self.ce1 = SCFE_CASA(base_num_channels // 2, base_num_channels, 1)
        self.ce2 = SCFE_CASA(base_num_channels // 2, base_num_channels, 1)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
        self.reset_states_i()
        self.reset_states_a()
        self.reset_states_d()

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def reset_states_i(self):
        self._states_i = [None] * self.num_recurrent_units

    def reset_states_a(self):
        self._states_a = [None] * self.num_recurrent_units

    def reset_states_d(self):
        self._states_d = [None] * self.num_recurrent_units

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W event tensor
        :return: N x num_output_channels x H/2 x W/2 iad
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

        # Context Exploration 1
        x_i, x_a, x_d = self.ce1(x_i, x_a, x_d)

        # second stage
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i

        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a

        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d

        # Context Exploration 2
        x_i, x_a, x_d = self.ce2(x_i, x_a, x_d)

        # prediction stage
        i = self.pred_i(x_i)
        a = self.pred_a(x_a)
        d = self.pred_d(x_d)

        return {
            'i': i,
            'a': a,
            'd': d
        }


class M109(BaseModel):
    """
    rppp_23
    scfe
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = RPPP_23(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU_woinit(base_num_channels, base_num_channels, kernel_size)
        self.G2_i = ConvGRU_woinit(base_num_channels, base_num_channels, kernel_size)
        self.pred_i = nn.Conv2d(base_num_channels, 1, 1, 1, 0)
        # a
        self.head_a = RPPP_23(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU_woinit(base_num_channels, base_num_channels, kernel_size)
        self.G2_a = ConvGRU_woinit(base_num_channels, base_num_channels, kernel_size)
        self.pred_a = nn.Conv2d(base_num_channels, 1, 1, 1, 0)
        # d
        self.head_d = RPPP_23(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU_woinit(base_num_channels, base_num_channels, kernel_size)
        self.G2_d = ConvGRU_woinit(base_num_channels, base_num_channels, kernel_size)
        self.pred_d = nn.Conv2d(base_num_channels, 1, 1, 1, 0)

        self.ce1 = SCFE(base_num_channels // 4, base_num_channels, 1)
        self.ce2 = SCFE(base_num_channels // 4, base_num_channels, 1)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
        self.reset_states_i()
        self.reset_states_a()
        self.reset_states_d()

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def reset_states_i(self):
        self._states_i = [None] * self.num_recurrent_units

    def reset_states_a(self):
        self._states_a = [None] * self.num_recurrent_units

    def reset_states_d(self):
        self._states_d = [None] * self.num_recurrent_units

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W event tensor
        :return: N x num_output_channels x H/2 x W/2 iad
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

        # Context Exploration 1
        x_i, x_a, x_d = self.ce1(x_i, x_a, x_d)

        # second stage
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i

        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a

        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d

        # Context Exploration 2
        x_i, x_a, x_d = self.ce2(x_i, x_a, x_d)

        # prediction stage
        i = self.pred_i(x_i)
        a = self.pred_a(x_a)
        d = self.pred_d(x_d)

        return {
            'i': i,
            'a': a,
            'd': d
        }


class M110(BaseModel):
    """
    rppp (2+3 + 1x1 conv)
    fast fourier convolution
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = RPPP_23(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_i = nn.Conv2d(base_num_channels, 1, 1, 1, 0)
        # a
        self.head_a = RPPP_23(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_a = nn.Conv2d(base_num_channels, 1, 1, 1, 0)
        # d
        self.head_d = RPPP_23(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_d = nn.Conv2d(base_num_channels, 1, 1, 1, 0)

        self.ce1 = FFC_IAD(base_num_channels, block_number=2)
        self.ce2 = FFC_IAD(base_num_channels, block_number=2)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
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

        # context exploration 1
        x_i, x_a, x_d = self.ce1(x_i, x_a, x_d)

        # second stage
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i

        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a

        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d

        # context exploration 2
        x_i, x_a, x_d = self.ce2(x_i, x_a, x_d)

        # prediction stage
        i = self.pred_i(x_i)
        a = self.pred_a(x_a)
        d = self.pred_d(x_d)

        return {
            'i': i,
            'a': a,
            'd': d
        }


class M111(BaseModel):
    """
    rppp (2+3 + 1x1 conv) 32
    fast fourier convolution one block
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = RPPP_23_large(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_i = nn.Conv2d(base_num_channels, 1, 1, 1, 0)
        # a
        self.head_a = RPPP_23_large(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_a = nn.Conv2d(base_num_channels, 1, 1, 1, 0)
        # d
        self.head_d = RPPP_23_large(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_d = nn.Conv2d(base_num_channels, 1, 1, 1, 0)

        self.ce1 = FFC_IAD(base_num_channels, block_number=1)
        self.ce2 = FFC_IAD(base_num_channels, block_number=1)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
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

        # context exploration 1
        x_i, x_a, x_d = self.ce1(x_i, x_a, x_d)

        # second stage
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i

        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a

        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d

        # context exploration 2
        x_i, x_a, x_d = self.ce2(x_i, x_a, x_d)

        # prediction stage
        i = self.pred_i(x_i)
        a = self.pred_a(x_a)
        d = self.pred_d(x_d)

        return {
            'i': i,
            'a': a,
            'd': d
        }


class M112(BaseModel):
    """
    rppp (2+3 + 1x1 conv) 32
    iad gated enhancement
    fast fourier convolution one block
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = RPPP_23_large(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_i = nn.Conv2d(base_num_channels, 1, 1, 1, 0)
        # a
        self.head_a = RPPP_23_large(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_a = nn.Conv2d(base_num_channels, 1, 1, 1, 0)
        # d
        self.head_d = RPPP_23_large(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_d = nn.Conv2d(base_num_channels, 1, 1, 1, 0)

        self.ce1 = FFC_IAD_Gated(base_num_channels, block_number=1, gate=True)
        self.ce2 = FFC_IAD_Gated(base_num_channels, block_number=1, gate=False)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
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

        # context exploration 1
        x_i, x_a, x_d = self.ce1(x_i, x_a, x_d)

        # second stage
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i

        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a

        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d

        # context exploration 2
        x_i, x_a, x_d = self.ce2(x_i, x_a, x_d)

        # prediction stage
        i = self.pred_i(x_i)
        a = self.pred_a(x_a)
        d = self.pred_d(x_d)

        return {
            'i': i,
            'a': a,
            'd': d
        }


class M113(BaseModel):
    """
    rppp (2+3 + 1x1 conv) 32
    iad gated enhancement
    fast fourier convolution one block
    5x5 prediction
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = RPPP_23_large(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_i = nn.Conv2d(base_num_channels, 1, 5, 1, 2)
        # a
        self.head_a = RPPP_23_large(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_a = nn.Conv2d(base_num_channels, 1, 5, 1, 2)
        # d
        self.head_d = RPPP_23_large(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_d = nn.Conv2d(base_num_channels, 1, 5, 1, 2)

        self.ce1 = FFC_IAD_Gated(base_num_channels, block_number=1, gate=True)
        self.ce2 = FFC_IAD_Gated(base_num_channels, block_number=1, gate=False)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
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

        # context exploration 1
        x_i, x_a, x_d = self.ce1(x_i, x_a, x_d)

        # second stage
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i

        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a

        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d

        # context exploration 2
        x_i, x_a, x_d = self.ce2(x_i, x_a, x_d)

        # prediction stage
        i = self.pred_i(x_i)
        a = self.pred_a(x_a)
        d = self.pred_d(x_d)

        return {
            'i': i,
            'a': a,
            'd': d
        }


class M114(BaseModel):
    """
    rppp (2+3 + 1x1 conv) 32
    joint
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = RPPP_23_large(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_i = nn.Conv2d(base_num_channels, 1, 1, 1, 0)
        # a
        self.head_a = RPPP_23_large(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_a = nn.Conv2d(base_num_channels, 1, 1, 1, 0)
        # d
        self.head_d = RPPP_23_large(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_d = nn.Conv2d(base_num_channels, 1, 1, 1, 0)

        self.ce1 = JRLFB(base_num_channels)
        self.ce2 = JRLFB(base_num_channels)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
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

        # context exploration 1
        x_i, x_a, x_d = self.ce1(x_i, x_a, x_d)

        # second stage
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i

        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a

        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d

        # context exploration 2
        x_i, x_a, x_d = self.ce2(x_i, x_a, x_d)

        # prediction stage
        i = self.pred_i(x_i)
        a = self.pred_a(x_a)
        d = self.pred_d(x_d)

        return {
            'i': i,
            'a': a,
            'd': d
        }


class M115(BaseModel):
    """
    rppp (2+3 + 1x1 conv) 32
    joint
    5x5 prediction
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = RPPP_23_large(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_i = nn.Conv2d(base_num_channels, 1, 5, 1, 2)
        # a
        self.head_a = RPPP_23_large(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_a = nn.Conv2d(base_num_channels, 1, 5, 1, 2)
        # d
        self.head_d = RPPP_23_large(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_d = nn.Conv2d(base_num_channels, 1, 5, 1, 2)

        self.ce1 = JRLFB(base_num_channels)
        self.ce2 = JRLFB(base_num_channels)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
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

        # context exploration 1
        x_i, x_a, x_d = self.ce1(x_i, x_a, x_d)

        # second stage
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i

        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a

        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d

        # context exploration 2
        x_i, x_a, x_d = self.ce2(x_i, x_a, x_d)

        # prediction stage
        i = self.pred_i(x_i)
        a = self.pred_a(x_a)
        d = self.pred_d(x_d)

        return {
            'i': i,
            'a': a,
            'd': d
        }


class M116(BaseModel):
    """
    rppp (2+3 + 1x1 conv) 32
    joint
    5x5 prediction
    lstm
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = RPPP_23_large(self.num_bins, base_num_channels)
        self.G1_i = RecurrentConvLayer(base_num_channels, base_num_channels, kernel_size)
        self.G2_i = RecurrentConvLayer(base_num_channels, base_num_channels, kernel_size)
        self.pred_i = nn.Conv2d(base_num_channels, 1, 5, 1, 2)
        # a
        self.head_a = RPPP_23_large(self.num_bins, base_num_channels)
        self.G1_a = RecurrentConvLayer(base_num_channels, base_num_channels, kernel_size)
        self.G2_a = RecurrentConvLayer(base_num_channels, base_num_channels, kernel_size)
        self.pred_a = nn.Conv2d(base_num_channels, 1, 5, 1, 2)
        # d
        self.head_d = RPPP_23_large(self.num_bins, base_num_channels)
        self.G1_d = RecurrentConvLayer(base_num_channels, base_num_channels, kernel_size)
        self.G2_d = RecurrentConvLayer(base_num_channels, base_num_channels, kernel_size)
        self.pred_d = nn.Conv2d(base_num_channels, 1, 5, 1, 2)

        self.ce1 = JRLFB(base_num_channels)
        self.ce2 = JRLFB(base_num_channels)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
        """
        # first stage
        x_i = self.head_i(x)
        x_i, x_state_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_state_i

        x_a = self.head_a(x)
        x_a, x_state_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_state_a

        x_d = self.head_d(x)
        x_d, x_state_d = self.G1_d(x_d, self._states_d[0])
        self._states_d[0] = x_state_d

        # context exploration 1
        x_i, x_a, x_d = self.ce1(x_i, x_a, x_d)

        # second stage
        x_i, x_state_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_state_i

        x_a, x_state_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_state_a

        x_d, x_state_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_state_d

        # context exploration 2
        x_i, x_a, x_d = self.ce2(x_i, x_a, x_d)

        # prediction stage
        i = self.pred_i(x_i)
        a = self.pred_a(x_a)
        d = self.pred_d(x_d)

        return {
            'i': i,
            'a': a,
            'd': d
        }


class M117(BaseModel):
    """
    rppp (2+3 + 1x1 conv)
    jrlfb
    5x5 prediction
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = RPPP_23(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_i = nn.Conv2d(base_num_channels, 1, 5, 1, 2)
        # a
        self.head_a = RPPP_23(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_a = nn.Conv2d(base_num_channels, 1, 5, 1, 2)
        # d
        self.head_d = RPPP_23(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_d = nn.Conv2d(base_num_channels, 1, 5, 1, 2)

        self.ce1 = JRLFB(base_num_channels)
        self.ce2 = JRLFB(base_num_channels)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
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

        # context exploration 1
        x_i, x_a, x_d = self.ce1(x_i, x_a, x_d)

        # second stage
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i

        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a

        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d

        # context exploration 2
        x_i, x_a, x_d = self.ce2(x_i, x_a, x_d)

        # prediction stage
        i = self.pred_i(x_i)
        a = self.pred_a(x_a)
        d = self.pred_d(x_d)

        return {
            'i': i,
            'a': a,
            'd': d
        }


class M118(BaseModel):
    """
    rppp (2+3 + 1x1 conv)
    jrlfb
    rich contrast prediction
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = RPPP_23(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_i = Rich_Contrast_Prediction(base_num_channels)
        # a
        self.head_a = RPPP_23(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_a = Rich_Contrast_Prediction(base_num_channels)
        # d
        self.head_d = RPPP_23(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_d = Rich_Contrast_Prediction(base_num_channels)

        self.ce1 = JRLFB(base_num_channels)
        self.ce2 = JRLFB(base_num_channels)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
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

        # context exploration 1
        x_i, x_a, x_d = self.ce1(x_i, x_a, x_d)

        # second stage
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i

        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a

        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d

        # context exploration 2
        x_i, x_a, x_d = self.ce2(x_i, x_a, x_d)

        # prediction stage
        i = self.pred_i(x_i)
        a = self.pred_a(x_a)
        d = self.pred_d(x_d)

        return {
            'i': i,
            'a': a,
            'd': d
        }


class M119(BaseModel):
    """
    rppp (2+3 + 1x1 conv)
    jrlfb
    multiscale prediction
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = RPPP_23(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_i = Multiscale_Prediction(base_num_channels)
        # a
        self.head_a = RPPP_23(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_a = Multiscale_Prediction(base_num_channels)
        # d
        self.head_d = RPPP_23(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_d = Multiscale_Prediction(base_num_channels)

        self.ce1 = JRLFB(base_num_channels)
        self.ce2 = JRLFB(base_num_channels)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
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

        # context exploration 1
        x_i, x_a, x_d = self.ce1(x_i, x_a, x_d)

        # second stage
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i

        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a

        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d

        # context exploration 2
        x_i, x_a, x_d = self.ce2(x_i, x_a, x_d)

        # prediction stage
        i = self.pred_i(x_i)
        a = self.pred_a(x_a)
        d = self.pred_d(x_d)

        return {
            'i': i,
            'a': a,
            'd': d
        }


class M120(BaseModel):
    """
    2x2 avg pooling for i
    rppp (2+3 + 1x1 conv) for ad
    jrlfb
    multiscale prediction
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = RPPP_22avg(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_i = Multiscale_Prediction(base_num_channels)
        # a
        self.head_a = RPPP_23(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_a = Multiscale_Prediction(base_num_channels)
        # d
        self.head_d = RPPP_23(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_d = Multiscale_Prediction(base_num_channels)

        self.ce1 = JRLFB(base_num_channels)
        self.ce2 = JRLFB(base_num_channels)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
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

        # context exploration 1
        x_i, x_a, x_d = self.ce1(x_i, x_a, x_d)

        # second stage
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i

        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a

        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d

        # context exploration 2
        x_i, x_a, x_d = self.ce2(x_i, x_a, x_d)

        # prediction stage
        i = self.pred_i(x_i)
        a = self.pred_a(x_a)
        d = self.pred_d(x_d)

        return {
            'i': i,
            'a': a,
            'd': d
        }


class M121(BaseModel):
    """
    2x2 avg pooling + 3x3 conv, for i
    rppp (2+3 + 3x3 conv), for ad
    two spatial frequency blocks
    multiscale prediction
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = RPPP_22avg(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_i = Multiscale_Prediction(base_num_channels)
        # a
        self.head_a = RPPP_23(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_a = Multiscale_Prediction(base_num_channels)
        # d
        self.head_d = RPPP_23(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_d = Multiscale_Prediction(base_num_channels)

        self.ce1 = SFB_IAD(base_num_channels)
        self.ce2 = SFB_IAD(base_num_channels)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
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

        # context exploration 1
        x_i, x_a, x_d = self.ce1(x_i, x_a, x_d)

        # second stage
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i

        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a

        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d

        # context exploration 2
        x_i, x_a, x_d = self.ce2(x_i, x_a, x_d)

        # prediction stage
        i = self.pred_i(x_i)
        a = self.pred_a(x_a)
        d = self.pred_d(x_d)

        return {
            'i': i,
            'a': a,
            'd': d
        }


class M122(BaseModel):
    """
    2x2 avg pooling + 3x3 conv, for i
    rppp (2+3 + 3x3 conv), for ad
    two spatial frequency blocks
    shared iad kernels in spatial domain
    multiscale prediction
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = RPPP_22avg(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_i = Multiscale_Prediction(base_num_channels)
        # a
        self.head_a = RPPP_23(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_a = Multiscale_Prediction(base_num_channels)
        # d
        self.head_d = RPPP_23(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_d = Multiscale_Prediction(base_num_channels)

        self.ce1 = SFB_IAD_Coupled(base_num_channels)
        self.ce2 = SFB_IAD_Coupled(base_num_channels)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
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

        # context exploration 1
        x_i, x_a, x_d = self.ce1(x_i, x_a, x_d)

        # second stage
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i

        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a

        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d

        # context exploration 2
        x_i, x_a, x_d = self.ce2(x_i, x_a, x_d)

        # prediction stage
        i = self.pred_i(x_i)
        a = self.pred_a(x_a)
        d = self.pred_d(x_d)

        return {
            'i': i,
            'a': a,
            'd': d
        }


class M123(BaseModel):
    """
    2x2 avg pooling + 3x3 conv, for i
    rppp (2+3 + 3x3 conv), for ad
    two spatial frequency blocks
    shared iad kernels in spatial domain
    esa + multiscale prediction
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = RPPP_22avg(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_i = ESA_Multiscale_Prediction(base_num_channels)
        # a
        self.head_a = RPPP_23(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_a = ESA_Multiscale_Prediction(base_num_channels)
        # d
        self.head_d = RPPP_23(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_d = ESA_Multiscale_Prediction(base_num_channels)

        self.ce1 = SFB_IAD_Coupled(base_num_channels)
        self.ce2 = SFB_IAD_Coupled(base_num_channels)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
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

        # context exploration 1
        x_i, x_a, x_d = self.ce1(x_i, x_a, x_d)

        # second stage
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i

        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a

        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d

        # context exploration 2
        x_i, x_a, x_d = self.ce2(x_i, x_a, x_d)

        # prediction stage
        i = self.pred_i(x_i)
        a = self.pred_a(x_a)
        d = self.pred_d(x_d)

        return {
            'i': i,
            'a': a,
            'd': d
        }


class M88_A(BaseModel):
    """
    no rppp, only a 3x3 conv + 2x2 avg pooling
    no jrlfb, only two 3x3 convs
    """
    def __init__(self, num_bins=5, base_num_channels=32, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = nn.Sequential(nn.Conv2d(num_bins, base_num_channels, 3, 1, 1), nn.ReLU(True), nn.AvgPool2d((2, 2), stride=2))
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # a
        self.head_a = nn.Sequential(nn.Conv2d(num_bins, base_num_channels, 3, 1, 1), nn.ReLU(True), nn.AvgPool2d((2, 2), stride=2))
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # d
        self.head_d = nn.Sequential(nn.Conv2d(num_bins, base_num_channels, 3, 1, 1), nn.ReLU(True), nn.AvgPool2d((2, 2), stride=2))
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)

        self.ce1 = ResidualBlock_IAD(base_num_channels)
        self.ce2 = ResidualBlock_IAD(base_num_channels)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
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

        # CE 1
        x_i, x_a, x_d = self.ce1(x_i, x_a, x_d)

        # second stage
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i

        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a

        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d

        # CE 2
        x_i, x_a, x_d = self.ce2(x_i, x_a, x_d)

        # prediction stage
        i = self.pred_i(x_i)
        a = self.pred_a(x_a)
        d = self.pred_d(x_d)

        return {
            'i': i,
            'a': a,
            'd': d
        }


class M88_B(BaseModel):
    """
    no rppp, only a 2x2 with stride of 2
    no jrlfb, only two 3x3 convs
    """
    def __init__(self, num_bins=5, base_num_channels=32, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = nn.Sequential(nn.Conv2d(num_bins, base_num_channels, 2, 2, 0), nn.ReLU(True))
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # a
        self.head_a = nn.Sequential(nn.Conv2d(num_bins, base_num_channels, 2, 2, 0), nn.ReLU(True))
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # d
        self.head_d = nn.Sequential(nn.Conv2d(num_bins, base_num_channels, 2, 2, 0), nn.ReLU(True))
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)

        self.ce1 = ResidualBlock_IAD(base_num_channels)
        self.ce2 = ResidualBlock_IAD(base_num_channels)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
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

        # CE 1
        x_i, x_a, x_d = self.ce1(x_i, x_a, x_d)

        # second stage
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i

        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a

        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d

        # CE 2
        x_i, x_a, x_d = self.ce2(x_i, x_a, x_d)

        # prediction stage
        i = self.pred_i(x_i)
        a = self.pred_a(x_a)
        d = self.pred_d(x_d)

        return {
            'i': i,
            'a': a,
            'd': d
        }


class M88_C(BaseModel):
    """
    no rppp, split and concate + 3x3 conv
    no jrlfb, only two 3x3 convs
    """
    def __init__(self, num_bins=5, base_num_channels=32, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = nn.Sequential(nn.Conv2d(num_bins * 4, base_num_channels, 3, 1, 1), nn.ReLU(True))
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # a
        self.head_a = nn.Sequential(nn.Conv2d(num_bins * 4, base_num_channels, 3, 1, 1), nn.ReLU(True))
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # d
        self.head_d = nn.Sequential(nn.Conv2d(num_bins * 4, base_num_channels, 3, 1, 1), nn.ReLU(True))
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)

        self.ce1 = ResidualBlock_IAD(base_num_channels)
        self.ce2 = ResidualBlock_IAD(base_num_channels)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
        """
        x_90 = x[:, :, 0::2, 0::2]
        x_45 = x[:, :, 0::2, 1::2]
        x_135 = x[:, :, 1::2, 0::2]
        x_0 = x[:, :, 1::2, 1::2]

        x_cat = torch.cat((x_90, x_45, x_135, x_0), 1)

        # first stage
        x_i = self.head_i(x_cat)
        x_i = self.G1_i(x_i, self._states_i[0])
        self._states_i[0] = x_i

        x_a = self.head_a(x_cat)
        x_a = self.G1_a(x_a, self._states_a[0])
        self._states_a[0] = x_a

        x_d = self.head_d(x_cat)
        x_d = self.G1_d(x_d, self._states_d[0])
        self._states_d[0] = x_d

        # CE 1
        x_i, x_a, x_d = self.ce1(x_i, x_a, x_d)

        # second stage
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i

        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a

        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d

        # CE 2
        x_i, x_a, x_d = self.ce2(x_i, x_a, x_d)

        # prediction stage
        i = self.pred_i(x_i)
        a = self.pred_a(x_a)
        d = self.pred_d(x_d)

        return {
            'i': i,
            'a': a,
            'd': d
        }


class M88_D(BaseModel):
    """
    rppp
    no jrlfb, only two 3x3 convs
    """
    def __init__(self, num_bins=5, base_num_channels=32, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        # i
        self.head_i = RPPP_23(self.num_bins, base_num_channels)
        self.G1_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_i = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_i = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # a
        self.head_a = RPPP_23(self.num_bins, base_num_channels)
        self.G1_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_a = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_a = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
        # d
        self.head_d = RPPP_23(self.num_bins, base_num_channels)
        self.G1_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.G2_d = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.pred_d = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)

        self.ce1 = ResidualBlock_IAD(base_num_channels)
        self.ce2 = ResidualBlock_IAD(base_num_channels)

        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
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
        :return: N x num_output_channels x H/2 x W/2 iad
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

        # CE 1
        x_i, x_a, x_d = self.ce1(x_i, x_a, x_d)

        # second stage
        x_i = self.G2_i(x_i, self._states_i[1])
        self._states_i[1] = x_i

        x_a = self.G2_a(x_a, self._states_a[1])
        self._states_a[1] = x_a

        x_d = self.G2_d(x_d, self._states_d[1])
        self._states_d[1] = x_d

        # CE 2
        x_i, x_a, x_d = self.ce2(x_i, x_a, x_d)

        # prediction stage
        i = self.pred_i(x_i)
        a = self.pred_a(x_a)
        d = self.pred_d(x_d)

        return {
            'i': i,
            'a': a,
            'd': d
        }
