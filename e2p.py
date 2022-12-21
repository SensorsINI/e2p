"""
 @Time    : 06.10.22 10:30
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : e2p.py
 @Function:
 
"""

from base_model import BaseModel
from submodules import *


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
            'd': d
        }

