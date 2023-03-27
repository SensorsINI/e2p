import torch
import torch.nn as nn
# import torch.nn.functional as f
import torch.nn.functional as F
from torch.nn import init
import math
import train.dct as dct
from collections import OrderedDict

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation='relu', norm=None,
                 BN_momentum=0.1):
        super(ConvLayer, self).__init__()

        bias = False if norm == 'BN' else True
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        if activation is not None:
            self.activation = getattr(torch, activation)
        else:
            self.activation = None

        self.norm = norm
        if norm == 'BN':
            self.norm_layer = nn.BatchNorm2d(out_channels, momentum=BN_momentum)
        elif norm == 'IN':
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):
        out = self.conv2d(x)

        if self.norm in ['BN', 'IN']:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out


class TransposedConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation='relu', norm=None):
        super(TransposedConvLayer, self).__init__()

        bias = False if norm == 'BN' else True
        self.transposed_conv2d = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=2, padding=padding, output_padding=1, bias=bias)

        if activation is not None:
            self.activation = getattr(torch, activation)
        else:
            self.activation = None

        self.norm = norm
        if norm == 'BN':
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm == 'IN':
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):
        out = self.transposed_conv2d(x)

        if self.norm in ['BN', 'IN']:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out


class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation='relu', norm=None):
        super(UpsampleConvLayer, self).__init__()

        bias = False if norm == 'BN' else True
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

        if activation is not None:
            self.activation = getattr(torch, activation)
        else:
            self.activation = None

        self.norm = norm
        if norm == 'BN':
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm == 'IN':
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):
        x_upsampled = f.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        out = self.conv2d(x_upsampled)

        if self.norm in ['BN', 'IN']:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out


class RecurrentConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 recurrent_block_type='convlstm', activation='relu', norm=None, BN_momentum=0.1):
        super(RecurrentConvLayer, self).__init__()

        assert(recurrent_block_type in ['convlstm', 'convgru'])
        self.recurrent_block_type = recurrent_block_type
        if self.recurrent_block_type == 'convlstm':
            RecurrentBlock = ConvLSTM
        else:
            RecurrentBlock = ConvGRU
        self.conv = ConvLayer(in_channels, out_channels, kernel_size, stride, padding, activation, norm,
                              BN_momentum=BN_momentum)
        self.recurrent_block = RecurrentBlock(input_size=out_channels, hidden_size=out_channels, kernel_size=3)

    def forward(self, x, prev_state):
        x = self.conv(x)
        state = self.recurrent_block(x, prev_state)
        x = state[0] if self.recurrent_block_type == 'convlstm' else state
        return x, state


class RecurrentConvLayer_Split_Input(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 recurrent_block_type='convlstm', activation='relu', norm=None, BN_momentum=0.1):
        super(RecurrentConvLayer_Split_Input, self).__init__()

        assert(recurrent_block_type in ['convlstm', 'convgru'])
        self.recurrent_block_type = recurrent_block_type
        if self.recurrent_block_type == 'convlstm':
            RecurrentBlock = ConvLSTM
        else:
            RecurrentBlock = ConvGRU
        self.conv_90 = ConvLayer(in_channels, out_channels // 4, kernel_size, stride, padding, activation, norm, BN_momentum=BN_momentum)
        self.conv_45 = ConvLayer(in_channels, out_channels // 4, kernel_size, stride, padding, activation, norm, BN_momentum=BN_momentum)
        self.conv_135 = ConvLayer(in_channels, out_channels // 4, kernel_size, stride, padding, activation, norm, BN_momentum=BN_momentum)
        self.conv_0 = ConvLayer(in_channels, out_channels // 4, kernel_size, stride, padding, activation, norm, BN_momentum=BN_momentum)
        self.recurrent_block = RecurrentBlock(input_size=out_channels, hidden_size=out_channels, kernel_size=3)

    def forward(self, x, prev_state):
        x_90 = x[:, :, 0::2, 0::2]
        x_45 = x[:, :, 0::2, 1::2]
        x_135 = x[:, :, 1::2, 0::2]
        x_0 = x[:, :, 1::2, 1::2]

        x_90 = self.conv_90(x_90)
        x_45 = self.conv_45(x_45)
        x_135 = self.conv_135(x_135)
        x_0 = self.conv_0(x_0)

        x = torch.cat([x_90, x_45, x_135, x_0], 1)

        state = self.recurrent_block(x, prev_state)
        x = state[0] if self.recurrent_block_type == 'convlstm' else state
        return x, state


class DownsampleRecurrentConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, recurrent_block_type='convlstm', padding=0, activation='relu'):
        super(DownsampleRecurrentConvLayer, self).__init__()

        self.activation = getattr(torch, activation)

        assert(recurrent_block_type in ['convlstm', 'convgru'])
        self.recurrent_block_type = recurrent_block_type
        if self.recurrent_block_type == 'convlstm':
            RecurrentBlock = ConvLSTM
        else:
            RecurrentBlock = ConvGRU
        self.recurrent_block = RecurrentBlock(input_size=in_channels, hidden_size=out_channels, kernel_size=kernel_size)

    def forward(self, x, prev_state):
        state = self.recurrent_block(x, prev_state)
        x = state[0] if self.recurrent_block_type == 'convlstm' else state
        x = f.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        return self.activation(x), state


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, norm=None,
                 BN_momentum=0.1):
        super(ResidualBlock, self).__init__()
        bias = False if norm == 'BN' else True
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=bias)
        # self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, dilation=2, padding=2, bias=bias)
        self.norm = norm
        if norm == 'BN':
            self.bn1 = nn.BatchNorm2d(out_channels, momentum=BN_momentum)
            self.bn2 = nn.BatchNorm2d(out_channels, momentum=BN_momentum)
        elif norm == 'IN':
            self.bn1 = nn.InstanceNorm2d(out_channels)
            self.bn2 = nn.InstanceNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        # self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, dilation=2, padding=2, bias=bias)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        if self.norm in ['BN', 'IN']:
            out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.norm in ['BN', 'IN']:
            out = self.bn2(out)

        if self.downsample:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ResidualBlock_IAD(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock_IAD, self).__init__()
        self.block_i = ResidualBlock(dim, dim)
        self.block_a = ResidualBlock(dim, dim)
        self.block_d = ResidualBlock(dim, dim)

    def forward(self, x_i, x_a, x_d):
        y_i = self.block_i(x_i)
        y_a = self.block_a(x_a)
        y_d = self.block_d(x_d)

        return y_i, y_a, y_d


class ConvLSTM(nn.Module):
    """Adapted from: https://github.com/Atcold/pytorch-CortexNet/blob/master/model/ConvLSTMCell.py """

    def __init__(self, input_size, hidden_size, kernel_size):
        super(ConvLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        pad = kernel_size // 2

        # cache a tensor filled with zeros to avoid reallocating memory at each inference step if --no-recurrent is enabled
        self.zero_tensors = {}

        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, kernel_size, padding=pad)

    def forward(self, input_, prev_state=None):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:

            # create the zero tensor if it has not been created already
            state_size = tuple([batch_size, self.hidden_size] + list(spatial_size))
            if state_size not in self.zero_tensors:
                # allocate a tensor with size `spatial_size`, filled with zero (if it has not been allocated already)
                self.zero_tensors[state_size] = (
                    torch.zeros(state_size, dtype=input_.dtype).to(input_.device),
                    torch.zeros(state_size, dtype=input_.dtype).to(input_.device)
                )

            prev_state = self.zero_tensors[tuple(state_size)]

        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        return hidden, cell


class ConvGRU(nn.Module):
    """
    Generate a convolutional GRU cell
    Adapted from: https://github.com/jacobkimmel/pytorch_convgru/blob/master/convgru.py
    """

    def __init__(self, input_size, hidden_size, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reset_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.out_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)

        init.orthogonal_(self.reset_gate.weight)
        init.orthogonal_(self.update_gate.weight)
        init.orthogonal_(self.out_gate.weight)
        init.constant_(self.reset_gate.bias, 0.)
        init.constant_(self.update_gate.bias, 0.)
        init.constant_(self.out_gate.bias, 0.)

    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = torch.zeros(state_size, dtype=input_.dtype).to(input_.device)

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_state], dim=1)
        update = torch.sigmoid(self.update_gate(stacked_inputs))
        reset = torch.sigmoid(self.reset_gate(stacked_inputs))
        out_inputs = torch.tanh(self.out_gate(torch.cat([input_, prev_state * reset], dim=1)))
        new_state = prev_state * (1 - update) + out_inputs * update

        return new_state


class ConvGRU_woinit(nn.Module):
    """
    Generate a convolutional GRU cell
    Adapted from: https://github.com/jacobkimmel/pytorch_convgru/blob/master/convgru.py
    """

    def __init__(self, input_size, hidden_size, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reset_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.out_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)

    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = torch.zeros(state_size, dtype=input_.dtype).to(input_.device)

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_state], dim=1)
        update = torch.sigmoid(self.update_gate(stacked_inputs))
        reset = torch.sigmoid(self.reset_gate(stacked_inputs))
        out_inputs = torch.tanh(self.out_gate(torch.cat([input_, prev_state * reset], dim=1)))
        new_state = prev_state * (1 - update) + out_inputs * update

        return new_state


class RecurrentResidualLayer(nn.Module):
    def __init__(self, in_channels, out_channels,
                 recurrent_block_type='convlstm', norm=None, BN_momentum=0.1):
        super(RecurrentResidualLayer, self).__init__()

        assert(recurrent_block_type in ['convlstm', 'convgru'])
        self.recurrent_block_type = recurrent_block_type
        if self.recurrent_block_type == 'convlstm':
            RecurrentBlock = ConvLSTM
        else:
            RecurrentBlock = ConvGRU
        self.conv = ResidualBlock(in_channels=in_channels,
                                  out_channels=out_channels,
                                  norm=norm,
                                  BN_momentum=BN_momentum)
        self.recurrent_block = RecurrentBlock(input_size=out_channels,
                                              hidden_size=out_channels,
                                              kernel_size=3)

    def forward(self, x, prev_state):
        x = self.conv(x)
        state = self.recurrent_block(x, prev_state)
        x = state[0] if self.recurrent_block_type == 'convlstm' else state
        return x, state


########################### Added by Haiyang Mei #######################################
from torch.nn import Softmax


def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)


class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""

    def __init__(self, in_dim):
        super(CrissCrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2,
                                                                                                                 1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2,
                                                                                                                 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width)).view(m_batchsize, width, height, height).permute(0, 2, 1, 3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        # print(concate)
        # print(att_H)
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)
        # print(out_H.size(),out_W.size())
        return self.gamma * (out_H + out_W) + x


# Spatially Separable Convolution Based Residual Block
class ResidualBlock_SSC(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock_SSC, self).__init__()
        self.channels = in_channels
        self.conv1_vh = nn.Sequential(nn.Conv2d(self.channels, self.channels, (9, 1), 1, padding=(4, 0)),
                                      nn.Conv2d(self.channels, self.channels, (1, 9), 1, padding=(0, 4)),
                                      nn.BatchNorm2d(self.channels))
        self.conv1_hv = nn.Sequential(nn.Conv2d(self.channels, self.channels, (1, 9), 1, padding=(0, 4)),
                                      nn.Conv2d(self.channels, self.channels, (9, 1), 1, padding=(4, 0)),
                                      nn.BatchNorm2d(self.channels))
        self.relu = nn.ReLU(inplace=True)
        self.conv2_vh = nn.Sequential(nn.Conv2d(self.channels, self.channels, (9, 1), 1, padding=(4, 0)),
                                      nn.Conv2d(self.channels, self.channels, (1, 9), 1, padding=(0, 4)),
                                      nn.BatchNorm2d(self.channels))
        self.conv2_hv = nn.Sequential(nn.Conv2d(self.channels, self.channels, (1, 9), 1, padding=(0, 4)),
                                      nn.Conv2d(self.channels, self.channels, (9, 1), 1, padding=(4, 0)),
                                      nn.BatchNorm2d(self.channels))

    def forward(self, x):
        conv1 = self.conv1_vh(x) + self.conv1_hv(x)
        conv1 = self.relu(conv1)

        conv2 = self.conv2_vh(conv1) + self.conv2_hv(conv1)
        out = conv2 + x
        out = self.relu(out)

        return out


# SE Residual Block
# def conv3x3(in_planes, out_planes, stride=1):
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualBlock_SE(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(ResidualBlock_SE, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


########################### Added by Haiyang Mei #######################################
########################### Multi-Scale Perception #####################################
BN_MOMENTUM = 0.1

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out

class MSP(nn.Module):
    def __init__(self, in_planes):
        super(MSP, self).__init__()
        self.down1 = nn.Sequential(nn.Conv2d(in_planes, in_planes, kernel_size=1, stride=1, padding=0),
                                   nn.BatchNorm2d(in_planes, momentum=BN_MOMENTUM), nn.ReLU(inplace=True))
        self.down2 = nn.Sequential(nn.Conv2d(in_planes, in_planes, kernel_size=2, stride=2, padding=0),
                                   nn.BatchNorm2d(in_planes, momentum=BN_MOMENTUM), nn.ReLU(inplace=True))
        self.down4 = nn.Sequential(nn.Conv2d(in_planes, in_planes, kernel_size=4, stride=4, padding=0),
                                   nn.BatchNorm2d(in_planes, momentum=BN_MOMENTUM), nn.ReLU(inplace=True))
        self.down8 = nn.Sequential(nn.Conv2d(in_planes, in_planes, kernel_size=8, stride=8, padding=0),
                                   nn.BatchNorm2d(in_planes, momentum=BN_MOMENTUM), nn.ReLU(inplace=True))

        self.mapping1 = BasicBlock(in_planes, in_planes)
        self.mapping2 = BasicBlock(in_planes, in_planes)
        self.mapping4 = BasicBlock(in_planes, in_planes)
        self.mapping8 = BasicBlock(in_planes, in_planes)

        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.up8 = nn.UpsamplingBilinear2d(scale_factor=8)

        self.fusion = nn.Sequential(nn.Conv2d(in_planes * 4, in_planes, kernel_size=1, stride=1, padding=0),
                                    nn.BatchNorm2d(in_planes, momentum=BN_MOMENTUM), nn.ReLU(inplace=True))

    def forward(self, x):
        down1 = self.down1(x)
        down2 = self.down2(x)
        down4 = self.down4(x)
        down8 = self.down8(x)

        mapping1 = self.mapping1(down1)
        mapping2 = self.mapping2(down2)
        mapping4 = self.mapping4(down4)
        mapping8 = self.mapping8(down8)

        mapping2 = self.up2(mapping2)
        mapping4 = self.up4(mapping4)
        mapping8 = self.up8(mapping8)

        mapping = torch.cat([mapping1, mapping2, mapping4, mapping8], 1)

        fusion = self.fusion(mapping)

        return fusion


########################### My Large Kernel Residual Block #######################################
class ResidualBlock_LK(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock_LK, self).__init__()
        self.conv1_v = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=(9, 3), stride=1, padding=(4, 1)),
                                     nn.BatchNorm2d(dim))
        self.conv1_h = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=(3, 9), stride=1, padding=(1, 4)),
                                     nn.BatchNorm2d(dim))
        self.conv1_l = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=(3, 3), stride=1, padding=(1, 1)),
                                     nn.BatchNorm2d(dim))
        self.relu = nn.ReLU(inplace=True)

        self.conv2_v = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=(9, 3), stride=1, padding=(4, 1)),
                                     nn.BatchNorm2d(dim))
        self.conv2_h = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=(3, 9), stride=1, padding=(1, 4)),
                                     nn.BatchNorm2d(dim))
        self.conv2_l = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=(3, 3), stride=1, padding=(1, 1)),
                                     nn.BatchNorm2d(dim))

    def forward(self, x):
        residual = x

        conv1 = self.conv1_v(x) + self.conv1_h(x) + self.conv1_l(x)
        conv1 = self.relu(conv1)

        conv2 = self.conv2_v(conv1) + self.conv2_h(conv1) + self.conv2_l(conv1)
        conv2 = self.relu(conv2 + residual)

        return conv2


########################### My Contrast Residual Block #######################################
class ResidualBlock_CT(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock_CT, self).__init__()
        self.conv1_local = nn.Conv2d(dim, dim, kernel_size=3, stride=1, dilation=1, padding=1)
        self.conv1_context = nn.Conv2d(dim, dim, kernel_size=3, stride=1, dilation=2, padding=2)

        self.conv2_local = nn.Conv2d(dim, dim, kernel_size=3, stride=1, dilation=1, padding=1)
        self.conv2_context = nn.Conv2d(dim, dim, kernel_size=3, stride=1, dilation=2, padding=2)

        self.bn1 = nn.BatchNorm2d(dim)
        self.bn2 = nn.BatchNorm2d(dim)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        conv1 = self.conv1_local(x) - self.conv1_context(x)
        conv1 = self.relu(self.bn1(conv1))

        conv2 = self.conv2_local(conv1) - self.conv2_context(conv1)
        conv2 = self.relu(self.bn2(conv2) + residual)

        return conv2


########################### My Multiscale Perception Residual Block #######################################
class ResidualBlock_MS(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock_MS, self).__init__()
        self.conv1_1 = nn.Conv2d(dim, dim, kernel_size=1, stride=1, dilation=1, padding=0)
        self.conv1_3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, dilation=1, padding=1)
        self.conv1_5 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, dilation=2, padding=2)
        self.conv1_7 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, dilation=3, padding=3)

        self.conv2_1 = nn.Conv2d(dim, dim, kernel_size=1, stride=1, dilation=1, padding=0)
        self.conv2_3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, dilation=1, padding=1)
        self.conv2_5 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, dilation=2, padding=2)
        self.conv2_7 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, dilation=3, padding=3)

        self.bn1 = nn.BatchNorm2d(dim)
        self.bn2 = nn.BatchNorm2d(dim)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        conv1_1 = self.conv1_1(x)
        conv1_3 = self.conv1_3(x)
        conv1_5 = self.conv1_5(x)
        conv1_7 = self.conv1_7(x)

        conv1 = self.relu(self.bn1(conv1_1 + conv1_3 + conv1_5 + conv1_7))

        conv2_1 = self.conv2_1(conv1)
        conv2_3 = self.conv2_3(conv1)
        conv2_5 = self.conv2_5(conv1)
        conv2_7 = self.conv2_7(conv1)

        conv2 = self.relu(self.bn2(conv2_1 + conv2_3 + conv2_5 + conv2_7) + residual)

        return conv2


########################### My IAD Attention Block #######################################
class IAD_Attention(nn.Module):
    def __init__(self, dim):
        super(IAD_Attention, self).__init__()
        self.fusion = nn.Sequential(nn.Conv2d(dim * 3, dim * 3, 3, 1, 1, groups=dim * 3), nn.ReLU(),
                                    nn.Conv2d(dim * 3, dim, 1, 1, 0), nn.ReLU())
        self.mask_i = nn.Sequential(nn.Conv2d(dim, 1, 3, 1, 1), nn.Sigmoid())
        self.mask_a = nn.Sequential(nn.Conv2d(dim, 1, 3, 1, 1), nn.Sigmoid())
        self.mask_d = nn.Sequential(nn.Conv2d(dim, 1, 3, 1, 1), nn.Sigmoid())

    def forward(self, x_i, x_a, x_d):
        fusion = self.fusion(torch.cat((x_i, x_a, x_d), 1))
        mask_i = self.mask_i(fusion)
        mask_a = self.mask_a(fusion)
        mask_d = self.mask_d(fusion)

        output_i = x_i * mask_i
        output_a = x_a * mask_a
        output_d = x_d * mask_d

        return output_i, output_a, output_d


########################### My IAD Attention Block #######################################
class IAD_Fusion(nn.Module):
    def __init__(self, dim):
        super(IAD_Fusion, self).__init__()
        self.fusion_i = nn.Sequential(nn.Conv2d(dim * 3, dim * 3, 3, 1, 1, groups=dim * 3), nn.ReLU(),
                                    nn.Conv2d(dim * 3, dim, 1, 1, 0), nn.ReLU())
        self.fusion_a = nn.Sequential(nn.Conv2d(dim * 3, dim * 3, 3, 1, 1, groups=dim * 3), nn.ReLU(),
                                      nn.Conv2d(dim * 3, dim, 1, 1, 0), nn.ReLU())
        self.fusion_d = nn.Sequential(nn.Conv2d(dim * 3, dim * 3, 3, 1, 1, groups=dim * 3), nn.ReLU(),
                                      nn.Conv2d(dim * 3, dim, 1, 1, 0), nn.ReLU())

    def forward(self, x_i, x_a, x_d):
        fusion_i = self.fusion_i(torch.cat((x_i, x_a, x_d), 1))
        fusion_a = self.fusion_a(torch.cat((x_i, x_a, x_d), 1))
        fusion_d = self.fusion_d(torch.cat((x_i, x_a, x_d), 1))

        return fusion_i, fusion_a, fusion_d


########################### Large Kernel Convolution #######################################
# import torch.nn.functional as F
# from timm.models.layers import trunc_normal_, DropPath
# from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM
#
# use_sync_bn = False
#
# def get_conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
#     return DepthWiseConv2dImplicitGEMM(in_channels, kernel_size, bias=bias)
#
#
# def get_bn(channels):
#     if use_sync_bn:
#         return nn.SyncBatchNorm(channels)
#     else:
#         return nn.BatchNorm2d(channels)
#
#
# def conv_bn_relu(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1):
#     if padding is None:
#         padding = kernel_size // 2
#     result = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
#                      stride=stride, padding=padding, groups=groups, dilation=dilation)
#     result.add_module('nonlinear', nn.ReLU())
#     return result
#
#
# def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1):
#     if padding is None:
#         padding = kernel_size // 2
#     result = nn.Sequential()
#     result.add_module('conv', get_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
#                                          stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False))
#     result.add_module('bn', get_bn(out_channels))
#     return result
#
#
# def fuse_bn(conv, bn):
#     kernel = conv.weight
#     running_mean = bn.running_mean
#     running_var = bn.running_var
#     gamma = bn.weight
#     beta = bn.bias
#     eps = bn.eps
#     std = (running_var + eps).sqrt()
#     t = (gamma / std).reshape(-1, 1, 1, 1)
#     return kernel * t, beta - running_mean * gamma / std
#
#
# class LayerNorm(nn.Module):
#     r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
#     The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
#     shape (batch_size, height, width, channels) while channels_first corresponds to inputs
#     with shape (batch_size, channels, height, width).
#     """
#
#     def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(normalized_shape))
#         self.bias = nn.Parameter(torch.zeros(normalized_shape))
#         self.eps = eps
#         self.data_format = data_format
#         if self.data_format not in ["channels_last", "channels_first"]:
#             raise NotImplementedError
#         self.normalized_shape = (normalized_shape,)
#
#     def forward(self, x):
#         if self.data_format == "channels_last":
#             return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
#         elif self.data_format == "channels_first":
#             u = x.mean(1, keepdim=True)
#             s = (x - u).pow(2).mean(1, keepdim=True)
#             x = (x - u) / torch.sqrt(s + self.eps)
#             x = self.weight[:, None, None] * x + self.bias[:, None, None]
#             return x
#
#
# class ReparamLargeKernelConv(nn.Module):
#
#     def __init__(self, in_channels, out_channels, kernel_size,
#                  stride, groups,
#                  small_kernel,
#                  small_kernel_merged=False, Decom=False):
#         super(ReparamLargeKernelConv, self).__init__()
#         self.kernel_size = kernel_size
#         self.small_kernel = small_kernel
#         self.Decom = Decom
#         # We assume the conv does not change the feature map size, so padding = k//2. Otherwise, you may configure padding as you wish, and change the padding of small_conv accordingly.
#         padding = kernel_size // 2
#         if small_kernel_merged:
#             self.lkb_reparam = get_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
#                                           stride=stride, padding=padding, dilation=1, groups=groups, bias=True)
#         else:
#             if self.Decom:
#                 self.LoRA1 = conv_bn(in_channels=in_channels, out_channels=out_channels,
#                                      kernel_size=(kernel_size, small_kernel),
#                                      stride=stride, padding=padding, dilation=1, groups=groups)
#                 self.LoRA2 = conv_bn(in_channels=in_channels, out_channels=out_channels,
#                                      kernel_size=(small_kernel, kernel_size),
#                                      stride=stride, padding=padding, dilation=1, groups=groups)
#             else:
#                 self.lkb_origin = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
#                                           stride=stride, padding=padding, dilation=1, groups=groups)
#
#             if (small_kernel is not None) and small_kernel < kernel_size:
#                 self.small_conv = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=small_kernel,
#                                           stride=stride, padding=small_kernel // 2, groups=groups, dilation=1)
#
#     def forward(self, inputs):
#         if hasattr(self, 'lkb_reparam'):
#             out = self.lkb_reparam(inputs)
#         elif self.Decom:
#             out = self.LoRA1(inputs) + self.LoRA2(inputs)
#             if hasattr(self, 'small_conv'):
#                 out += self.small_conv(inputs)
#         else:
#             out = self.lkb_origin(inputs)
#             if hasattr(self, 'small_conv'):
#                 out += self.small_conv(inputs)
#         return out
#
#     def get_equivalent_kernel_bias(self):
#         eq_k, eq_b = fuse_bn(self.lkb_origin.conv, self.lkb_origin.bn)
#         if hasattr(self, 'small_conv'):
#             small_k, small_b = fuse_bn(self.small_conv.conv, self.small_conv.bn)
#             eq_b += small_b
#             #   add to the central part
#             eq_k += nn.functional.pad(small_k, [(self.kernel_size - self.small_kernel) // 2] * 4)
#         return eq_k, eq_b
#
#     def merge_kernel(self):
#         eq_k, eq_b = self.get_equivalent_kernel_bias()
#         self.lkb_reparam = get_conv2d(in_channels=self.lkb_origin.conv.in_channels,
#                                       out_channels=self.lkb_origin.conv.out_channels,
#                                       kernel_size=self.lkb_origin.conv.kernel_size, stride=self.lkb_origin.conv.stride,
#                                       padding=self.lkb_origin.conv.padding, dilation=self.lkb_origin.conv.dilation,
#                                       groups=self.lkb_origin.conv.groups, bias=True)
#         self.lkb_reparam.weight.data = eq_k
#         self.lkb_reparam.bias.data = eq_b
#         self.__delattr__('lkb_origin')
#         if hasattr(self, 'small_conv'):
#             self.__delattr__('small_conv')
#
#
# class ResidualBlock_ELK(nn.Module):
#     r""" SLaK Block. There are two equivalent implementations:
#     (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
#     (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
#     We use (2) as we find it slightly faster in PyTorch
#
#     Args:
#         dim (int): Number of input channels.
#         drop_path (float): Stochastic depth rate. Default: 0.0
#         layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
#     """
#
#     def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, kernel_size=(51, 5), Decom=None):
#         super().__init__()
#
#         self.large_kernel = ReparamLargeKernelConv(in_channels=dim, out_channels=dim,
#                                                    kernel_size=kernel_size[0],
#                                                    stride=1, groups=dim, small_kernel=kernel_size[1],
#                                                    small_kernel_merged=False, Decom=Decom)
#
#         self.norm = LayerNorm(dim, eps=1e-6)
#         self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
#         self.act = nn.GELU()
#         self.pwconv2 = nn.Linear(4 * dim, dim)
#         self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
#                                   requires_grad=True) if layer_scale_init_value > 0 else None
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#
#     def forward(self, x):
#         input = x
#         x = self.large_kernel(x)
#         x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
#         x = self.norm(x)
#         x = self.pwconv1(x)
#         x = self.act(x)
#         x = self.pwconv2(x)
#         if self.gamma is not None:
#             x = self.gamma * x
#         x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
#
#         x = input + self.drop_path(x)
#         return x


###################################################################
# ########################## CBAM #################################
###################################################################
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    def forward(self, x):
        # original
        # return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        # max
        # torch.max(x, 1)[0].unsqueeze(1)
        # avg
        return torch.mean(x, 1).unsqueeze(1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(1, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)  # broadcasting
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels=128, reduction_ratio=16, pool_types=['avg'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


########################### external attention #######################################
class ExternalAttention(nn.Module):

    def __init__(self, d_model, S=32):
        super().__init__()
        self.mk=nn.Linear(d_model,S,bias=False)
        self.mv=nn.Linear(S,d_model,bias=False)
        self.softmax=nn.Softmax(dim=1)
        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, i, a, d):
        batchsize, C, height, width = i.size()
        i = i.view(batchsize, C, -1).permute(0, 2, 1)
        a = a.view(batchsize, C, -1).permute(0, 2, 1)
        d = d.view(batchsize, C, -1).permute(0, 2, 1)

        attn_i = self.mk(i)
        attn_i = self.softmax(attn_i)
        attn_i = attn_i / torch.sum(attn_i, dim=2, keepdim=True)
        i = self.mv(attn_i)
        i = i.permute(0, 2, 1).view(batchsize, C, height, width)

        attn_a = self.mk(a)
        attn_a = self.softmax(attn_a)
        attn_a = attn_a / torch.sum(attn_a, dim=2, keepdim=True)
        a = self.mv(attn_a)
        a = a.permute(0, 2, 1).view(batchsize, C, height, width)

        attn_d = self.mk(d)
        attn_d = self.softmax(attn_d)
        attn_d = attn_d / torch.sum(attn_d, dim=2, keepdim=True)
        d = self.mv(attn_d)
        d = d.permute(0, 2, 1).view(batchsize, C, height, width)

        return i, a, d


class SKAttention(nn.Module):

    def __init__(self, channel=32, reduction=2):
        super().__init__()
        self.d = channel // reduction
        self.fc0 = nn.Linear(channel, self.d)
        self.fc1 = nn.Linear(self.d, channel)
        self.fc2 = nn.Linear(self.d, channel)
        self.fc3 = nn.Linear(self.d, channel)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, i, a, d):
        iad = i + a + d
        iad = iad.mean(-1).mean(-1)
        fc0 = self.fc0(iad)
        fc1 = self.fc1(fc0)
        fc2 = self.fc2(fc0)
        fc3 = self.fc3(fc0)

        fc123 = torch.cat([fc1[:, :, None], fc2[:, :, None], fc3[:, :, None]], 2)
        weight = self.softmax(fc123)
        output = i * weight[:, :, 0][:, :, None, None] + a * weight[:, :, 1][:, :, None, None] + d * weight[:, :, 2][:, :, None, None]

        return output


class SequentialPolarizedSelfAttention(nn.Module):

    def __init__(self, channel=32):
        super().__init__()
        self.ch_wv=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.ch_wq=nn.Conv2d(channel,1,kernel_size=(1,1))
        self.softmax_channel=nn.Softmax(1)
        self.softmax_spatial=nn.Softmax(-1)
        self.ch_wz=nn.Conv2d(channel//2,channel,kernel_size=(1,1))
        self.ln=nn.LayerNorm(channel)
        self.sigmoid=nn.Sigmoid()
        self.sp_wv=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.sp_wq=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.agp=nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        b, c, h, w = x.size()

        #Channel-only Self-Attention
        channel_wv=self.ch_wv(x) #bs,c//2,h,w
        channel_wq=self.ch_wq(x) #bs,1,h,w
        channel_wv=channel_wv.reshape(b,c//2,-1) #bs,c//2,h*w
        channel_wq=channel_wq.reshape(b,-1,1) #bs,h*w,1
        channel_wq=self.softmax_channel(channel_wq)
        channel_wz=torch.matmul(channel_wv,channel_wq).unsqueeze(-1) #bs,c//2,1,1
        channel_weight=self.sigmoid(self.ln(self.ch_wz(channel_wz).reshape(b,c,1).permute(0,2,1))).permute(0,2,1).reshape(b,c,1,1) #bs,c,1,1
        channel_out=channel_weight*x

        #Spatial-only Self-Attention
        spatial_wv=self.sp_wv(channel_out) #bs,c//2,h,w
        spatial_wq=self.sp_wq(channel_out) #bs,c//2,h,w
        spatial_wq=self.agp(spatial_wq) #bs,c//2,1,1
        spatial_wv=spatial_wv.reshape(b,c//2,-1) #bs,c//2,h*w
        spatial_wq=spatial_wq.permute(0,2,3,1).reshape(b,1,c//2) #bs,1,c//2
        spatial_wq=self.softmax_spatial(spatial_wq)
        spatial_wz=torch.matmul(spatial_wq,spatial_wv) #bs,1,h*w
        spatial_weight=self.sigmoid(spatial_wz.reshape(b,1,h,w)) #bs,1,h,w
        spatial_out=spatial_weight*channel_out
        return spatial_out


class RPPP(nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.pad1 = nn.ZeroPad2d((1, 0, 1, 0))
        self.pad2 = nn.ZeroPad2d((0, 1, 1, 0))
        self.pad3 = nn.ZeroPad2d((1, 0, 0, 1))
        self.pad4 = nn.ZeroPad2d((0, 1, 0, 1))

        dim_mid = dim_out // 4
        self.pattern1 = nn.Sequential(nn.Conv2d(dim_in, dim_mid, 3, stride=2, padding=0), nn.ReLU())
        self.pattern2 = nn.Sequential(nn.Conv2d(dim_in, dim_mid, 3, stride=2, padding=0), nn.ReLU())
        self.pattern3 = nn.Sequential(nn.Conv2d(dim_in, dim_mid, 3, stride=2, padding=0), nn.ReLU())
        self.pattern4 = nn.Sequential(nn.Conv2d(dim_in, dim_mid, 3, stride=2, padding=0), nn.ReLU())

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=5, padding=2)
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x_1 = self.pattern1(self.pad1(x))
        x_2 = self.pattern2(self.pad2(x))
        x_3 = self.pattern3(self.pad3(x))
        x_4 = self.pattern4(self.pad4(x))

        x_c = torch.cat([x_1, x_2, x_3, x_4], 1)

        y=self.gap(x_c) #bs,c,1,1
        y=y.squeeze(-1).permute(0,2,1) #bs,1,c
        y=self.conv(y) #bs,1,c
        y=self.sigmoid(y) #bs,1,c
        y=y.permute(0,2,1).unsqueeze(-1) #bs,c,1,1
        return x_c*y.expand_as(x_c)


class RPPP_PSA(nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.pad1 = nn.ZeroPad2d((1, 0, 1, 0))
        self.pad2 = nn.ZeroPad2d((0, 1, 1, 0))
        self.pad3 = nn.ZeroPad2d((1, 0, 0, 1))
        self.pad4 = nn.ZeroPad2d((0, 1, 0, 1))

        dim_mid = dim_out // 4
        self.pattern1 = nn.Sequential(nn.Conv2d(dim_in, dim_mid, 3, stride=2, padding=0), nn.ReLU())
        self.pattern2 = nn.Sequential(nn.Conv2d(dim_in, dim_mid, 3, stride=2, padding=0), nn.ReLU())
        self.pattern3 = nn.Sequential(nn.Conv2d(dim_in, dim_mid, 3, stride=2, padding=0), nn.ReLU())
        self.pattern4 = nn.Sequential(nn.Conv2d(dim_in, dim_mid, 3, stride=2, padding=0), nn.ReLU())

        self.psa = SequentialPolarizedSelfAttention(dim_out)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x_1 = self.pattern1(self.pad1(x))
        x_2 = self.pattern2(self.pad2(x))
        x_3 = self.pattern3(self.pad3(x))
        x_4 = self.pattern4(self.pad4(x))

        x_c = torch.cat([x_1, x_2, x_3, x_4], 1)

        x_psa = self.psa(x_c)

        return x_psa


class RPPP_Conv(nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.pad1 = nn.ZeroPad2d((1, 0, 1, 0))
        self.pad2 = nn.ZeroPad2d((0, 1, 1, 0))
        self.pad3 = nn.ZeroPad2d((1, 0, 0, 1))
        self.pad4 = nn.ZeroPad2d((0, 1, 0, 1))

        self.pattern1 = nn.Sequential(nn.Conv2d(dim_in, dim_out, 3, stride=2, padding=0), nn.ReLU())
        self.pattern2 = nn.Sequential(nn.Conv2d(dim_in, dim_out, 3, stride=2, padding=0), nn.ReLU())
        self.pattern3 = nn.Sequential(nn.Conv2d(dim_in, dim_out, 3, stride=2, padding=0), nn.ReLU())
        self.pattern4 = nn.Sequential(nn.Conv2d(dim_in, dim_out, 3, stride=2, padding=0), nn.ReLU())

        self.conv = nn.Sequential(nn.Conv2d(dim_out * 4, dim_out, 1, 1, 0), nn.ReLU())

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x_1 = self.pattern1(self.pad1(x))
        x_2 = self.pattern2(self.pad2(x))
        x_3 = self.pattern3(self.pad3(x))
        x_4 = self.pattern4(self.pad4(x))

        x_c = torch.cat([x_1, x_2, x_3, x_4], 1)

        x_conv = self.conv(x_c)

        return x_conv


class RPPP_234(nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.pad1 = nn.ZeroPad2d((1, 0, 1, 0))
        self.pad2 = nn.ZeroPad2d((0, 1, 1, 0))
        self.pad3 = nn.ZeroPad2d((1, 0, 0, 1))
        self.pad4 = nn.ZeroPad2d((0, 1, 0, 1))

        dim_mid = dim_out // 4
        self.pattern1 = nn.Sequential(nn.Conv2d(dim_in, dim_mid, 3, stride=2, padding=0), nn.ReLU())
        self.pattern2 = nn.Sequential(nn.Conv2d(dim_in, dim_mid, 3, stride=2, padding=0), nn.ReLU())
        self.pattern3 = nn.Sequential(nn.Conv2d(dim_in, dim_mid, 3, stride=2, padding=0), nn.ReLU())
        self.pattern4 = nn.Sequential(nn.Conv2d(dim_in, dim_mid, 3, stride=2, padding=0), nn.ReLU())

        self.pattern0 = nn.Sequential(nn.Conv2d(dim_in, dim_mid, 2, stride=2, padding=0), nn.ReLU())
        self.pattern5 = nn.Sequential(nn.Conv2d(dim_in, dim_mid, 4, stride=2, padding=1), nn.ReLU())

        self.conv = nn.Sequential(nn.Conv2d(dim_mid * 6, dim_out, 1, 1, 0), nn.ReLU())

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x_0 = self.pattern0(x)

        x_1 = self.pattern1(self.pad1(x))
        x_2 = self.pattern2(self.pad2(x))
        x_3 = self.pattern3(self.pad3(x))
        x_4 = self.pattern4(self.pad4(x))

        x_5 = self.pattern5(x)

        x_c = torch.cat([x_0, x_1, x_2, x_3, x_4, x_5], 1)

        x_conv = self.conv(x_c)

        return x_conv


class RPPP_23(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.pad1 = nn.ZeroPad2d((1, 0, 1, 0))
        self.pad2 = nn.ZeroPad2d((0, 1, 1, 0))
        self.pad3 = nn.ZeroPad2d((1, 0, 0, 1))
        self.pad4 = nn.ZeroPad2d((0, 1, 0, 1))

        dim_mid = dim_out // 4
        self.pattern1 = nn.Sequential(nn.Conv2d(dim_in, dim_mid, 3, stride=2, padding=0), nn.ReLU(True))
        self.pattern2 = nn.Sequential(nn.Conv2d(dim_in, dim_mid, 3, stride=2, padding=0), nn.ReLU(True))
        self.pattern3 = nn.Sequential(nn.Conv2d(dim_in, dim_mid, 3, stride=2, padding=0), nn.ReLU(True))
        self.pattern4 = nn.Sequential(nn.Conv2d(dim_in, dim_mid, 3, stride=2, padding=0), nn.ReLU(True))

        self.pattern0 = nn.Sequential(nn.Conv2d(dim_in, dim_mid, 2, stride=2, padding=0), nn.ReLU(True))

        self.conv = nn.Sequential(nn.Conv2d(dim_mid * 5, dim_out, 1, 1, 0), nn.ReLU(True))

    def forward(self, x):
        x_0 = self.pattern0(x)

        x_1 = self.pattern1(self.pad1(x))
        x_2 = self.pattern2(self.pad2(x))
        x_3 = self.pattern3(self.pad3(x))
        x_4 = self.pattern4(self.pad4(x))

        x_c = torch.cat([x_0, x_1, x_2, x_3, x_4], 1)

        x_conv = self.conv(x_c)

        return x_conv


class RPPP_22avg(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.avg = nn.AvgPool2d(2, stride=2)

        self.conv = nn.Sequential(nn.Conv2d(dim_in, dim_out, 3, 1, 1), nn.ReLU())

    def forward(self, x):
        avg = self.avg(x)
        conv = self.conv(avg)

        return conv


class RPPP_23_large(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.pad1 = nn.ZeroPad2d((1, 0, 1, 0))
        self.pad2 = nn.ZeroPad2d((0, 1, 1, 0))
        self.pad3 = nn.ZeroPad2d((1, 0, 0, 1))
        self.pad4 = nn.ZeroPad2d((0, 1, 0, 1))

        dim_mid = dim_out
        self.pattern1 = nn.Sequential(nn.Conv2d(dim_in, dim_mid, 3, stride=2, padding=0), nn.ReLU(True))
        self.pattern2 = nn.Sequential(nn.Conv2d(dim_in, dim_mid, 3, stride=2, padding=0), nn.ReLU(True))
        self.pattern3 = nn.Sequential(nn.Conv2d(dim_in, dim_mid, 3, stride=2, padding=0), nn.ReLU(True))
        self.pattern4 = nn.Sequential(nn.Conv2d(dim_in, dim_mid, 3, stride=2, padding=0), nn.ReLU(True))

        self.pattern0 = nn.Sequential(nn.Conv2d(dim_in, dim_mid, 2, stride=2, padding=0), nn.ReLU(True))

        self.conv = nn.Sequential(nn.Conv2d(dim_mid * 5, dim_out, 1, 1, 0), nn.ReLU(True))

    def forward(self, x):
        x_0 = self.pattern0(x)

        x_1 = self.pattern1(self.pad1(x))
        x_2 = self.pattern2(self.pad2(x))
        x_3 = self.pattern3(self.pad3(x))
        x_4 = self.pattern4(self.pad4(x))

        x_c = torch.cat([x_0, x_1, x_2, x_3, x_4], 1)

        x_conv = self.conv(x_c)

        return x_conv


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)

        return x * y


## Spatial Attention (SA) Layer
class SALayer(nn.Module):
    def __init__(self):
        super(SALayer, self).__init__()
        self.conv = nn.Conv2d(1, 1, 7, 1, 3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = torch.mean(x, 1, keepdim=True)
        y = self.conv(y)
        y = self.sigmoid(y)

        return x * y


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res


class RPPP_23_RCAB(nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.pad1 = nn.ZeroPad2d((1, 0, 1, 0))
        self.pad2 = nn.ZeroPad2d((0, 1, 1, 0))
        self.pad3 = nn.ZeroPad2d((1, 0, 0, 1))
        self.pad4 = nn.ZeroPad2d((0, 1, 0, 1))

        dim_mid = dim_out // 4
        self.pattern1 = nn.Sequential(nn.Conv2d(dim_in, dim_mid, 3, stride=2, padding=0), nn.ReLU())
        self.pattern2 = nn.Sequential(nn.Conv2d(dim_in, dim_mid, 3, stride=2, padding=0), nn.ReLU())
        self.pattern3 = nn.Sequential(nn.Conv2d(dim_in, dim_mid, 3, stride=2, padding=0), nn.ReLU())
        self.pattern4 = nn.Sequential(nn.Conv2d(dim_in, dim_mid, 3, stride=2, padding=0), nn.ReLU())

        self.pattern0 = nn.Sequential(nn.Conv2d(dim_in, dim_mid, 2, stride=2, padding=0), nn.ReLU())

        self.conv = nn.Sequential(nn.Conv2d(dim_mid * 5, dim_out, 1, 1, 0), nn.ReLU(), RCAB(nn.Conv2d, 32, 1, 4))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x_0 = self.pattern0(x)

        x_1 = self.pattern1(self.pad1(x))
        x_2 = self.pattern2(self.pad2(x))
        x_3 = self.pattern3(self.pad3(x))
        x_4 = self.pattern4(self.pad4(x))

        x_c = torch.cat([x_0, x_1, x_2, x_3, x_4], 1)

        x_conv = self.conv(x_c)

        return x_conv


class RPPP_CA_Conv1x1(nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.pad1 = nn.ZeroPad2d((1, 0, 1, 0))
        self.pad2 = nn.ZeroPad2d((0, 1, 1, 0))
        self.pad3 = nn.ZeroPad2d((1, 0, 0, 1))
        self.pad4 = nn.ZeroPad2d((0, 1, 0, 1))

        dim_mid = dim_out // 4
        self.pattern1 = nn.Sequential(nn.Conv2d(dim_in, dim_mid, 3, stride=2, padding=0), nn.ReLU())
        self.pattern2 = nn.Sequential(nn.Conv2d(dim_in, dim_mid, 3, stride=2, padding=0), nn.ReLU())
        self.pattern3 = nn.Sequential(nn.Conv2d(dim_in, dim_mid, 3, stride=2, padding=0), nn.ReLU())
        self.pattern4 = nn.Sequential(nn.Conv2d(dim_in, dim_mid, 3, stride=2, padding=0), nn.ReLU())

        self.pattern0 = nn.Sequential(nn.Conv2d(dim_in, dim_mid, 2, stride=2, padding=0), nn.ReLU())

        self.conv = nn.Sequential(CALayer(dim_mid * 5, reduction=5), nn.Conv2d(dim_mid * 5, dim_out, 1, 1, 0), nn.ReLU())

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x_0 = self.pattern0(x)

        x_1 = self.pattern1(self.pad1(x))
        x_2 = self.pattern2(self.pad2(x))
        x_3 = self.pattern3(self.pad3(x))
        x_4 = self.pattern4(self.pad4(x))

        x_c = torch.cat([x_0, x_1, x_2, x_3, x_4], 1)

        x_conv = self.conv(x_c)

        return x_conv


class RPPP_23_cr_ca(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.pad1 = nn.ZeroPad2d((1, 0, 1, 0))
        self.pad2 = nn.ZeroPad2d((0, 1, 1, 0))
        self.pad3 = nn.ZeroPad2d((1, 0, 0, 1))
        self.pad4 = nn.ZeroPad2d((0, 1, 0, 1))

        dim_mid = dim_out // 4
        self.pattern1 = nn.Sequential(nn.Conv2d(dim_in, dim_mid, 3, stride=2, padding=0), nn.ReLU(True))
        self.pattern2 = nn.Sequential(nn.Conv2d(dim_in, dim_mid, 3, stride=2, padding=0), nn.ReLU(True))
        self.pattern3 = nn.Sequential(nn.Conv2d(dim_in, dim_mid, 3, stride=2, padding=0), nn.ReLU(True))
        self.pattern4 = nn.Sequential(nn.Conv2d(dim_in, dim_mid, 3, stride=2, padding=0), nn.ReLU(True))

        self.pattern0 = nn.Sequential(nn.Conv2d(dim_in, dim_mid, 2, stride=2, padding=0), nn.ReLU(True))

        self.conv = nn.Sequential(nn.Conv2d(dim_mid * 5, dim_out, 1, 1, 0), nn.ReLU(True), CALayer(dim_out, reduction=4))

    def forward(self, x):
        x_0 = self.pattern0(x)

        x_1 = self.pattern1(self.pad1(x))
        x_2 = self.pattern2(self.pad2(x))
        x_3 = self.pattern3(self.pad3(x))
        x_4 = self.pattern4(self.pad4(x))

        x_c = torch.cat([x_0, x_1, x_2, x_3, x_4], 1)

        x_conv = self.conv(x_c)

        return x_conv


class RPPP_Contrasted(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.pad1 = nn.ZeroPad2d((1, 0, 1, 0))
        self.pad2 = nn.ZeroPad2d((0, 1, 1, 0))
        self.pad3 = nn.ZeroPad2d((1, 0, 0, 1))
        self.pad4 = nn.ZeroPad2d((0, 1, 0, 1))

        dim_mid = dim_out // 4
        self.pattern1 = nn.Sequential(nn.Conv2d(dim_in, dim_mid, 3, stride=2, padding=0))
        self.pattern2 = nn.Sequential(nn.Conv2d(dim_in, dim_mid, 3, stride=2, padding=0))
        self.pattern3 = nn.Sequential(nn.Conv2d(dim_in, dim_mid, 3, stride=2, padding=0))
        self.pattern4 = nn.Sequential(nn.Conv2d(dim_in, dim_mid, 3, stride=2, padding=0))

        self.center1 = nn.Sequential(nn.Conv2d(dim_in, dim_mid, 1, stride=2, padding=0))
        self.center2 = nn.Sequential(nn.Conv2d(dim_in, dim_mid, 1, stride=2, padding=0))
        self.center3 = nn.Sequential(nn.Conv2d(dim_in, dim_mid, 1, stride=2, padding=0))
        self.center4 = nn.Sequential(nn.Conv2d(dim_in, dim_mid, 1, stride=2, padding=0))

        self.conv = CALayer(dim_mid * 4, reduction=4)

        self.relu = nn.ReLU(inplace=True)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        pad1 = self.pad1(x)
        pad2 = self.pad2(x)
        pad3 = self.pad3(x)
        pad4 = self.pad4(x)

        pattern1 = self.pattern1(pad1)
        pattern2 = self.pattern2(pad2)
        pattern3 = self.pattern3(pad3)
        pattern4 = self.pattern4(pad4)

        center1 = self.center1(pad1[:, :, 1:, 1:])
        center2 = self.center2(pad2[:, :, 1:, 1:])
        center3 = self.center3(pad3[:, :, 1:, 1:])
        center4 = self.center4(pad4[:, :, 1:, 1:])

        contrast1 = self.relu(center1 - pattern1)
        contrast2 = self.relu(center2 - pattern2)
        contrast3 = self.relu(center3 - pattern3)
        contrast4 = self.relu(center4 - pattern4)

        x_c = torch.cat([contrast1, contrast2, contrast3, contrast4], 1)

        x_conv = self.conv(x_c)

        return x_conv


class RPPP_23_3(nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.pad1 = nn.ZeroPad2d((1, 0, 1, 0))
        self.pad2 = nn.ZeroPad2d((0, 1, 1, 0))
        self.pad3 = nn.ZeroPad2d((1, 0, 0, 1))
        self.pad4 = nn.ZeroPad2d((0, 1, 0, 1))

        dim_mid = dim_out // 4
        self.pattern1 = nn.Sequential(nn.Conv2d(dim_in, dim_mid, 3, stride=2, padding=0), nn.ReLU(True))
        self.pattern2 = nn.Sequential(nn.Conv2d(dim_in, dim_mid, 3, stride=2, padding=0), nn.ReLU(True))
        self.pattern3 = nn.Sequential(nn.Conv2d(dim_in, dim_mid, 3, stride=2, padding=0), nn.ReLU(True))
        self.pattern4 = nn.Sequential(nn.Conv2d(dim_in, dim_mid, 3, stride=2, padding=0), nn.ReLU(True))

        self.pattern0 = nn.Sequential(nn.Conv2d(dim_in, dim_mid, 2, stride=2, padding=0), nn.ReLU(True))

        self.conv = nn.Sequential(nn.Conv2d(dim_mid * 5, dim_out, 3, 1, 1), nn.ReLU(True))
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x_0 = self.pattern0(x)

        x_1 = self.pattern1(self.pad1(x))
        x_2 = self.pattern2(self.pad2(x))
        x_3 = self.pattern3(self.pad3(x))
        x_4 = self.pattern4(self.pad4(x))

        x_c = torch.cat([x_0, x_1, x_2, x_3, x_4], 1)

        x_conv = self.conv(x_c)

        return x_conv


class RPPP_23_plus(nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.pad1 = nn.ZeroPad2d((1, 0, 1, 0))
        self.pad2 = nn.ZeroPad2d((0, 1, 1, 0))
        self.pad3 = nn.ZeroPad2d((1, 0, 0, 1))
        self.pad4 = nn.ZeroPad2d((0, 1, 0, 1))

        self.pattern1 = nn.Sequential(nn.Conv2d(dim_in, dim_out, 3, stride=2, padding=0), nn.ReLU())
        self.pattern2 = nn.Sequential(nn.Conv2d(dim_in, dim_out, 3, stride=2, padding=0), nn.ReLU())
        self.pattern3 = nn.Sequential(nn.Conv2d(dim_in, dim_out, 3, stride=2, padding=0), nn.ReLU())
        self.pattern4 = nn.Sequential(nn.Conv2d(dim_in, dim_out, 3, stride=2, padding=0), nn.ReLU())

        self.pattern0 = nn.Sequential(nn.Conv2d(dim_in, dim_out, 2, stride=2, padding=0), nn.ReLU())

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x_0 = self.pattern0(x)

        x_1 = self.pattern1(self.pad1(x))
        x_2 = self.pattern2(self.pad2(x))
        x_3 = self.pattern3(self.pad3(x))
        x_4 = self.pattern4(self.pad4(x))

        x_out = x_0 + x_1 + x_2 + x_3 + x_4

        return x_out


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)

class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale

class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial=no_spatial
        if not no_spatial:
            self.hw = AttentionGate()
    def forward(self, x):
        x_perm1 = x.permute(0,2,1,3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0,2,1,3).contiguous()
        x_perm2 = x.permute(0,3,2,1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0,3,2,1).contiguous()
        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = 1/3 * (x_out + x_out11 + x_out21)
        else:
            x_out = 1/2 * (x_out11 + x_out21)
        return x_out


class RPPP_23_T(nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.pad1 = nn.ZeroPad2d((1, 0, 1, 0))
        self.pad2 = nn.ZeroPad2d((0, 1, 1, 0))
        self.pad3 = nn.ZeroPad2d((1, 0, 0, 1))
        self.pad4 = nn.ZeroPad2d((0, 1, 0, 1))

        dim_mid = dim_out // 4
        self.pattern1 = nn.Sequential(nn.Conv2d(dim_in, dim_mid, 3, stride=2, padding=0), nn.ReLU())
        self.pattern2 = nn.Sequential(nn.Conv2d(dim_in, dim_mid, 3, stride=2, padding=0), nn.ReLU())
        self.pattern3 = nn.Sequential(nn.Conv2d(dim_in, dim_mid, 3, stride=2, padding=0), nn.ReLU())
        self.pattern4 = nn.Sequential(nn.Conv2d(dim_in, dim_mid, 3, stride=2, padding=0), nn.ReLU())

        self.pattern0 = nn.Sequential(nn.Conv2d(dim_in, dim_mid, 2, stride=2, padding=0), nn.ReLU())

        self.conv = nn.Sequential(nn.Conv2d(dim_mid * 5, dim_out, 1, 1, 0), nn.ReLU(), TripletAttention())

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x_0 = self.pattern0(x)

        x_1 = self.pattern1(self.pad1(x))
        x_2 = self.pattern2(self.pad2(x))
        x_3 = self.pattern3(self.pad3(x))
        x_4 = self.pattern4(self.pad4(x))

        x_c = torch.cat([x_0, x_1, x_2, x_3, x_4], 1)

        x_conv = self.conv(x_c)

        return x_conv


# for m72
class DepthwiseSeparableConvolution(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_ch
        )
        self.pointwise_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

    def forward(self, x):
        out = self.depthwise_conv(x)
        out = self.pointwise_conv(out)
        return out


class RPPP_23_T_DS(nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.pad1 = nn.ZeroPad2d((1, 0, 1, 0))
        self.pad2 = nn.ZeroPad2d((0, 1, 1, 0))
        self.pad3 = nn.ZeroPad2d((1, 0, 0, 1))
        self.pad4 = nn.ZeroPad2d((0, 1, 0, 1))

        dim_mid = dim_out // 4
        self.pattern1 = nn.Sequential(DepthwiseSeparableConvolution(dim_in, dim_mid, 3, stride=2, padding=0), nn.ReLU())
        self.pattern2 = nn.Sequential(DepthwiseSeparableConvolution(dim_in, dim_mid, 3, stride=2, padding=0), nn.ReLU())
        self.pattern3 = nn.Sequential(DepthwiseSeparableConvolution(dim_in, dim_mid, 3, stride=2, padding=0), nn.ReLU())
        self.pattern4 = nn.Sequential(DepthwiseSeparableConvolution(dim_in, dim_mid, 3, stride=2, padding=0), nn.ReLU())

        self.pattern0 = nn.Sequential(DepthwiseSeparableConvolution(dim_in, dim_mid, 2, stride=2, padding=0), nn.ReLU())

        self.conv = self.conv = nn.Sequential(nn.Conv2d(dim_mid * 5, dim_out, 1, 1, 0), nn.ReLU(), TripletAttention())

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x_0 = self.pattern0(x)

        x_1 = self.pattern1(self.pad1(x))
        x_2 = self.pattern2(self.pad2(x))
        x_3 = self.pattern3(self.pad3(x))
        x_4 = self.pattern4(self.pad4(x))

        x_c = torch.cat([x_0, x_1, x_2, x_3, x_4], 1)

        x_conv = self.conv(x_c)

        return x_conv



# Frequency Residual Block
def apply_complex(fr, fi, input, dtype = torch.complex64):
    return (fr(input.real)-fi(input.imag)).type(dtype) \
            + 1j*(fr(input.imag)+fi(input.real)).type(dtype)


class ComplexConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 dilation=1, groups=1, bias=True):
        super(ComplexConv2d, self).__init__()
        self.conv_r = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_i = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input):
        return apply_complex(self.conv_r, self.conv_i, input)

def complex_relu(input):
    return nn.ReLU()(input.real).type(torch.complex64)+1j*nn.ReLU()(input.imag).type(torch.complex64)

class ResidualBlock_Frequency(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock_Frequency, self).__init__()
        self.conv1 = ComplexConv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = ComplexConv2d(in_channels, out_channels, 3, 1, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        _, _, a, b = x.shape
        residual = x

        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')

        out = self.conv1(x)
        out = complex_relu(out)
        out = self.conv2(out)

        out = torch.fft.irfft2(out, s=(a, b), dim=(2, 3), norm='ortho')

        out += residual
        out = self.relu(out)
        return out


class HFD(nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()

        dim_oneeighth = dim_out // 8
        dim_quarter = dim_out // 4
        dim_half = dim_out // 2

        self.s2f = nn.Sequential(nn.Conv2d(dim_in, dim_oneeighth, 1, 1, 0), nn.ReLU())

        self.down1 = nn.Sequential(nn.Conv2d(dim_oneeighth, dim_quarter, 2, 2, 0), nn.ReLU())
        self.down2 = nn.Sequential(nn.Conv2d(dim_quarter, dim_half, 2, 2, 0), nn.ReLU())
        self.down3 = nn.Sequential(nn.Conv2d(dim_half, dim_out, 2, 2, 0), nn.ReLU())

        self.predict = nn.Sequential(nn.Conv2d(dim_out, dim_out, 3, 1, 1), nn.ReLU(),
                                     nn.Conv2d(dim_out, dim_out, 1, 1, 0))

    def forward(self, x):
        b, c, h, w = x.shape
        h_small = math.ceil(h / 8)
        w_small = math.ceil(w / 8)
        h_pad = h_small * 8 - h
        w_pad = w_small * 8 - w

        x_pad = nn.ZeroPad2d((0, w_pad, 0, h_pad))(x)

        s2f = self.s2f(x_pad)
        down1 = self.down1(s2f)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        predict = self.predict(down3)

        z = predict.view(b, 64, h_small*w_small).transpose(1, 2).view(b, 1, h_small*w_small, 8, 8)
        # print(z.shape)
        z = dct.block_idct(z)
        z = dct.deblockify(z, (h_small * 8, w_small * 8))
        # print(z.shape)
        if h_pad != 0 and w_pad != 0:
            z = z[:, :, :-h_pad, :-w_pad]
        # print(z.shape)

        return predict, z


########################### Residual Local Feature Block #######################################


def _make_pair(value):
    if isinstance(value, int):
        value = (value,) * 2
    return value


def conv_layer(in_channels,
               out_channels,
               kernel_size,
               bias=True):
    """
    Re-write convolution layer for adaptive `padding`.
    """
    kernel_size = _make_pair(kernel_size)
    padding = (int((kernel_size[0] - 1) / 2),
               int((kernel_size[1] - 1) / 2))
    return nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size,
                     padding=padding,
                     bias=bias)


class ESA(nn.Module):
    """
    Modification of Enhanced Spatial Attention (ESA), which is proposed by
    `Residual Feature Aggregation Network for Image Super-Resolution`
    Note: `conv_max` and `conv3_` are NOT used here, so the corresponding codes
    are deleted.
    """

    def __init__(self, esa_channels, n_feats, conv):
        super(ESA, self).__init__()
        f = esa_channels
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = torch.nn.functional.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = self.conv3(v_max)
        c3 = torch.nn.functional.interpolate(c3, (x.size(2), x.size(3)),
                           mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return x * m


class ESAF(nn.Module):
    """
    Modification of Enhanced Spatial Attention (ESA), which is proposed by
    `Residual Feature Aggregation Network for Image Super-Resolution`
    Note: `conv_max` and `conv3_` are NOT used here, so the corresponding codes
    are deleted.
    """

    def __init__(self, esa_channels, n_feats, conv):
        super(ESAF, self).__init__()
        f = esa_channels
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv1_y = conv(n_feats, f, kernel_size=1)
        self.conv1_z = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv2 = conv(f * 3, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y, z):
        c1_ = (self.conv1(x))
        c1_y = (self.conv1_y(y))
        c1_z = (self.conv1_z(z))
        c1 = self.conv2(torch.cat((c1_, c1_y, c1_z), 1))
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = self.conv3(v_max)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)),
                           mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return x * m


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    """
    Activation functions for ['relu', 'lrelu', 'prelu'].
    Parameters
    ----------
    act_type: str
        one of ['relu', 'lrelu', 'prelu'].
    inplace: bool
        whether to use inplace operator.
    neg_slope: float
        slope of negative region for `lrelu` or `prelu`.
    n_prelu: int
        `num_parameters` for `prelu`.
    ----------
    """
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError(
            'activation layer [{:s}] is not found'.format(act_type))
    return layer


class RLFB(nn.Module):
    """
    Residual Local Feature Block (RLFB).
    """

    def __init__(self,
                 in_channels,
                 mid_channels=None,
                 out_channels=None,
                 esa_channels=16):
        super(RLFB, self).__init__()

        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.c1_r = conv_layer(in_channels, mid_channels, 3)
        self.c2_r = conv_layer(mid_channels, mid_channels, 3)
        self.c3_r = conv_layer(mid_channels, in_channels, 3)

        self.c5 = conv_layer(in_channels, out_channels, 1)
        self.esa = ESA(esa_channels, out_channels, nn.Conv2d)

        self.act = activation('lrelu', neg_slope=0.05)

    def forward(self, x):
        out = (self.c1_r(x))
        out = self.act(out)

        out = (self.c2_r(out))
        out = self.act(out)

        out = (self.c3_r(out))
        out = self.act(out)

        out = out + x
        out = self.esa(self.c5(out))

        return out


class RLFBF(nn.Module):
    """
    Residual Local Feature Block Fusion (RLFBF).
    """

    def __init__(self,
                 in_channels,
                 mid_channels=None,
                 out_channels=None,
                 esa_channels=16):
        super(RLFBF, self).__init__()

        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.c1_r = conv_layer(in_channels, mid_channels, 3)
        self.c2_r = conv_layer(mid_channels, mid_channels, 3)
        self.c3_r = conv_layer(mid_channels, in_channels, 3)

        self.c5 = conv_layer(in_channels, out_channels, 1)
        self.esaf = ESAF(esa_channels, out_channels, nn.Conv2d)

        self.act = activation('lrelu', neg_slope=0.05)

    def forward(self, x, y, z):
        out = (self.c1_r(x))
        out = self.act(out)

        out = (self.c2_r(out))
        out = self.act(out)

        out = (self.c3_r(out))
        out = self.act(out)

        out = out + x
        out = self.esaf(self.c5(out), y, z)

        return out

########################### Semi-Coupled Feature Extraction #######################################
class Coupled_Layer(nn.Module):
    def __init__(self,
                 coupled_number=16,
                 n_feats=32,
                 kernel_size=3):
        super(Coupled_Layer, self).__init__()
        self.n_feats = n_feats
        self.coupled_number = coupled_number
        self.kernel_size = kernel_size

        self.kernel_shared_1 = nn.Parameter(nn.init.kaiming_uniform_(
            torch.zeros(size=[self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
        self.kernel_i_1 = nn.Parameter(nn.init.kaiming_uniform_(
            torch.randn(size=[self.n_feats - self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
        self.kernel_a_1 = nn.Parameter(nn.init.kaiming_uniform_(
            torch.randn(size=[self.n_feats - self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
        self.kernel_d_1 = nn.Parameter(nn.init.kaiming_uniform_(
            torch.randn(size=[self.n_feats - self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))

        self.kernel_shared_2 = nn.Parameter(nn.init.kaiming_uniform_(
            torch.randn(size=[self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
        self.kernel_i_2 = nn.Parameter(nn.init.kaiming_uniform_(
            torch.randn(size=[self.n_feats - self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
        self.kernel_a_2 = nn.Parameter(nn.init.kaiming_uniform_(
            torch.randn(size=[self.n_feats - self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
        self.kernel_d_2 = nn.Parameter(nn.init.kaiming_uniform_(
            torch.randn(size=[self.n_feats - self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))

        self.bias_shared_1 = nn.Parameter((torch.zeros(size=[self.coupled_number])))
        self.bias_i_1 = nn.Parameter((torch.zeros(size=[self.n_feats - self.coupled_number])))
        self.bias_a_1 = nn.Parameter((torch.zeros(size=[self.n_feats - self.coupled_number])))
        self.bias_d_1 = nn.Parameter((torch.zeros(size=[self.n_feats - self.coupled_number])))

        self.bias_shared_2 = nn.Parameter((torch.zeros(size=[self.coupled_number])))
        self.bias_i_2 = nn.Parameter((torch.zeros(size=[self.n_feats - self.coupled_number])))
        self.bias_a_2 = nn.Parameter((torch.zeros(size=[self.n_feats - self.coupled_number])))
        self.bias_d_2 = nn.Parameter((torch.zeros(size=[self.n_feats - self.coupled_number])))

    def forward(self, feat_i, feat_a, feat_d):
        # i
        shortCut_i = feat_i
        feat_i = F.conv2d(feat_i,
                            torch.cat([self.kernel_shared_1, self.kernel_i_1], dim=0),
                            torch.cat([self.bias_shared_1, self.bias_i_1], dim=0), padding=1)
        feat_i = F.relu(feat_i, inplace=True)
        feat_i = F.conv2d(feat_i,
                            torch.cat([self.kernel_shared_2, self.kernel_i_2], dim=0),
                            torch.cat([self.bias_shared_2, self.bias_i_2], dim=0), padding=1)
        # for m88
        # feat_i = F.relu(feat_i + shortCut_i, inplace=True)
        # for sfb in m123
        feat_i = feat_i + shortCut_i

        # a
        shortCut_a = feat_a
        feat_a = F.conv2d(feat_a,
                            torch.cat([self.kernel_shared_1, self.kernel_a_1], dim=0),
                            torch.cat([self.bias_shared_1, self.bias_a_1], dim=0), padding=1)
        feat_a = F.relu(feat_a, inplace=True)
        feat_a = F.conv2d(feat_a,
                            torch.cat([self.kernel_shared_2, self.kernel_a_2], dim=0),
                            torch.cat([self.bias_shared_2, self.bias_a_2], dim=0), padding=1)
        # feat_a = F.relu(feat_a + shortCut_a, inplace=True)
        feat_a = feat_a + shortCut_a

        # d
        shortCut_d = feat_d
        feat_d = F.conv2d(feat_d,
                          torch.cat([self.kernel_shared_1, self.kernel_d_1], dim=0),
                          torch.cat([self.bias_shared_1, self.bias_d_1], dim=0), padding=1)
        feat_d = F.relu(feat_d, inplace=True)
        feat_d = F.conv2d(feat_d,
                          torch.cat([self.kernel_shared_2, self.kernel_d_2], dim=0),
                          torch.cat([self.bias_shared_2, self.bias_d_2], dim=0), padding=1)
        # feat_d = F.relu(feat_d + shortCut_d, inplace=True)
        feat_d = feat_d + shortCut_d

        return feat_i, feat_a, feat_d


class SCFE(nn.Module):
    def __init__(self, coupled_number, n_feats, n_layer=1):
        super(SCFE, self).__init__()
        self.n_layer = n_layer
        self.coupled_feat_extractor = nn.ModuleList([Coupled_Layer(coupled_number, n_feats) for _ in range(self.n_layer)])

    def forward(self, feat_i, feat_a, feat_d):
        for layer in self.coupled_feat_extractor:
            feat_i, feat_a, feat_d = layer(feat_i, feat_a, feat_d)
        return feat_i, feat_a, feat_d


class SCFE_CASA(nn.Module):
    def __init__(self, coupled_number, n_feats, n_layer=1):
        super(SCFE_CASA, self).__init__()
        self.n_layer = n_layer
        self.coupled_feat_extractor = nn.ModuleList([Coupled_Layer(coupled_number, n_feats) for _ in range(self.n_layer)])
        self.casa_i = nn.Sequential(CALayer(n_feats, reduction=4), SALayer())
        self.casa_a = nn.Sequential(CALayer(n_feats, reduction=4), SALayer())
        self.casa_d = nn.Sequential(CALayer(n_feats, reduction=4), SALayer())

    def forward(self, feat_i, feat_a, feat_d):
        for layer in self.coupled_feat_extractor:
            feat_i, feat_a, feat_d = layer(feat_i, feat_a, feat_d)
        feat_i = self.casa_i(feat_i)
        feat_a = self.casa_a(feat_a)
        feat_d = self.casa_d(feat_d)

        return feat_i, feat_a, feat_d


class SCFE_GF_CASA(nn.Module):
    def __init__(self, coupled_number, n_feats, n_layer=1):
        super(SCFE_GF_CASA, self).__init__()
        self.n_layer = n_layer
        self.coupled_feat_extractor = nn.ModuleList([Coupled_Layer(coupled_number, n_feats) for _ in range(self.n_layer)])

        self.gf_i = nn.Sequential(GlobalLocalFilter(n_feats), nn.ReLU(True))
        self.gf_a = nn.Sequential(GlobalLocalFilter(n_feats), nn.ReLU(True))
        self.gf_d = nn.Sequential(GlobalLocalFilter(n_feats), nn.ReLU(True))

        self.casa_i = nn.Sequential(CALayer(n_feats, reduction=4), SALayer())
        self.casa_a = nn.Sequential(CALayer(n_feats, reduction=4), SALayer())
        self.casa_d = nn.Sequential(CALayer(n_feats, reduction=4), SALayer())

    def forward(self, feat_i, feat_a, feat_d):
        for layer in self.coupled_feat_extractor:
            feat_i, feat_a, feat_d = layer(feat_i, feat_a, feat_d)

        feat_i = self.gf_i(feat_i)
        feat_a = self.gf_a(feat_a)
        feat_d = self.gf_d(feat_d)

        feat_i = self.casa_i(feat_i)
        feat_a = self.casa_a(feat_a)
        feat_d = self.casa_d(feat_d)

        return feat_i, feat_a, feat_d


class DE(nn.Module):
    """
    Detail Enhancement
    """

    def __init__(self, dim, dim_mid, conv=nn.Conv2d):
        super(DE, self).__init__()
        f = dim_mid
        self.conv1 = conv(dim, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, dim, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, x_features):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = self.conv3(v_max)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)),
                           mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        w = self.sigmoid(c4)

        out = x + w * x_features

        return out


class JRLFB(nn.Module):
    """
    Joint Residual Local Feature Block (JRLFB).
    """
    def __init__(self,
                 in_channels,
                 out_channels=None,
                 n_layer=2,
                 esa_channels=16):
        super(JRLFB, self).__init__()

        if out_channels is None:
            out_channels = in_channels

        self.n_layer = n_layer
        self.coupled_feat_extractor = nn.ModuleList([Coupled_Layer() for i in range(self.n_layer)])

        self.cc_i = nn.Conv2d(in_channels, out_channels, 1)
        self.cc_a = nn.Conv2d(in_channels, out_channels, 1)
        self.cc_d = nn.Conv2d(in_channels, out_channels, 1)

        self.esa_i = ESA(esa_channels, out_channels, nn.Conv2d)
        self.esa_a = ESA(esa_channels, out_channels, nn.Conv2d)
        self.esa_d = ESA(esa_channels, out_channels, nn.Conv2d)

    def forward(self, feat_i, feat_a, feat_d):
        for layer in self.coupled_feat_extractor:
            feat_i, feat_a, feat_d = layer(feat_i, feat_a, feat_d)

        out_i = self.esa_i(self.cc_i(feat_i))
        out_a = self.esa_a(self.cc_a(feat_a))
        out_d = self.esa_d(self.cc_d(feat_d))

        return out_i, out_a, out_d


class JRLFB_more(nn.Module):
    """
    Joint Residual Local Feature Block (JRLFB).
    """
    def __init__(self,
                 in_channels,
                 out_channels=None,
                 n_layer=4,
                 esa_channels=16):
        super(JRLFB_more, self).__init__()

        if out_channels is None:
            out_channels = in_channels

        self.n_layer = n_layer
        self.coupled_feat_extractor = nn.ModuleList([Coupled_Layer() for i in range(self.n_layer)])

        self.cc_i = nn.Conv2d(in_channels, out_channels, 1)
        self.cc_a = nn.Conv2d(in_channels, out_channels, 1)
        self.cc_d = nn.Conv2d(in_channels, out_channels, 1)

        self.esa_i = ESA(esa_channels, out_channels, nn.Conv2d)
        self.esa_a = ESA(esa_channels, out_channels, nn.Conv2d)
        self.esa_d = ESA(esa_channels, out_channels, nn.Conv2d)

    def forward(self, feat_i, feat_a, feat_d):
        for layer in self.coupled_feat_extractor:
            feat_i, feat_a, feat_d = layer(feat_i, feat_a, feat_d)

        out_i = self.esa_i(self.cc_i(feat_i))
        out_a = self.esa_a(self.cc_a(feat_a))
        out_d = self.esa_d(self.cc_d(feat_d))

        return out_i, out_a, out_d


########################### gnconv #######################################
def get_dwconv(dim, kernel, bias):
    return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel-1)//2 ,bias=bias, groups=dim)


class gnconv(nn.Module):
    def __init__(self, dim, order=5, gflayer=None, h=14, w=8, s=1.0):
        super().__init__()
        self.order = order
        self.dims = [dim // 2 ** i for i in range(order)]
        self.dims.reverse()
        self.proj_in = nn.Conv2d(dim, 2 * dim, 1)

        if gflayer is None:
            self.dwconv = get_dwconv(sum(self.dims), 7, True)
        # else:
        #     self.dwconv = gflayer(sum(self.dims), h=h, w=w)

        self.proj_out = nn.Conv2d(dim, dim, 1)

        self.pws = nn.ModuleList(
            [nn.Conv2d(self.dims[i], self.dims[i + 1], 1) for i in range(order - 1)]
        )

        self.scale = s
        print('[gnconv]', order, 'order with dims=', self.dims, 'scale=%.4f' % self.scale)

    def forward(self, x, mask=None, dummy=False):
        B, C, H, W = x.shape

        fused_x = self.proj_in(x)
        pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)

        dw_abc = self.dwconv(abc) * self.scale

        dw_list = torch.split(dw_abc, self.dims, dim=1)
        x = pwa * dw_list[0]

        for i in range(self.order - 1):
            x = self.pws[i](x) * dw_list[i + 1]

        x = self.proj_out(x)

        return x

class gnconv_iad(nn.Module):
    def __init__(self, dim):
        super(gnconv_iad, self).__init__()
        self.gnconv1_i = gnconv(dim)
        self.gnconv1_a = gnconv(dim)
        self.gnconv1_d = gnconv(dim)
        self.gnconv2_i = gnconv(dim)
        self.gnconv2_a = gnconv(dim)
        self.gnconv2_d = gnconv(dim)
        self.relu = nn.ReLU(True)

    def forward(self, feat_i, feat_a, feat_d):
        gnconv1_i = self.gnconv1_i(feat_i)
        feat_i = self.relu(feat_i + gnconv1_i)
        gnconv2_i = self.gnconv2_i(feat_i)
        feat_i = self.relu(feat_i + gnconv2_i)

        gnconv1_a = self.gnconv1_a(feat_a)
        feat_a = self.relu(feat_a + gnconv1_a)
        gnconv2_a = self.gnconv2_a(feat_a)
        feat_a = self.relu(feat_a + gnconv2_a)

        gnconv1_d = self.gnconv1_d(feat_d)
        feat_d = self.relu(feat_d + gnconv1_d)
        gnconv2_d = self.gnconv2_d(feat_d)
        feat_d = self.relu(feat_d + gnconv2_d)

        return feat_i, feat_a, feat_d


class SCFE_HOR(nn.Module):
    def __init__(self, coupled_number, n_feats, n_layer=1):
        super(SCFE_HOR, self).__init__()
        self.n_layer = n_layer
        self.coupled_feat_extractor = nn.ModuleList([Coupled_Layer(coupled_number, n_feats) for _ in range(self.n_layer)])
        self.gnconv1_i = gnconv(n_feats)
        self.gnconv1_a = gnconv(n_feats)
        self.gnconv1_d = gnconv(n_feats)
        self.relu = nn.ReLU(True)

    def forward(self, feat_i, feat_a, feat_d):
        for layer in self.coupled_feat_extractor:
            feat_i, feat_a, feat_d = layer(feat_i, feat_a, feat_d)
        gnconv1_i = self.gnconv1_i(feat_i)
        feat_i = self.relu(feat_i + gnconv1_i)
        gnconv1_a = self.gnconv1_a(feat_a)
        feat_a = self.relu(feat_a + gnconv1_a)
        gnconv1_d = self.gnconv1_d(feat_d)
        feat_d = self.relu(feat_d + gnconv1_d)

        return feat_i, feat_a, feat_d


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

# from timm.models.layers import trunc_normal_
# class GlobalLocalFilter(nn.Module):
#     def __init__(self, dim, h=14, w=8):
#         super().__init__()
#         self.dw = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, bias=False, groups=dim // 2)
#         self.complex_weight = nn.Parameter(torch.randn(dim // 2, h, w, 2, dtype=torch.float32) * 0.02)
#         trunc_normal_(self.complex_weight, std=.02)
#         self.pre_norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')
#         self.post_norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')
#
#     def forward(self, x):
#         x = self.pre_norm(x)
#         x1, x2 = torch.chunk(x, 2, dim=1)
#         x1 = self.dw(x1)
#
#         x2 = x2.to(torch.float32)
#         B, C, a, b = x2.shape
#         x2 = torch.fft.rfft2(x2, dim=(2, 3), norm='ortho')
#
#         weight = self.complex_weight
#         if not weight.shape[1:3] == x2.shape[2:4]:
#             weight = F.interpolate(weight.permute(3,0,1,2), size=x2.shape[2:4], mode='bilinear', align_corners=True).permute(1,2,3,0)
#
#         weight = torch.view_as_complex(weight.contiguous())
#
#         x2 = x2 * weight
#         x2 = torch.fft.irfft2(x2, s=(a, b), dim=(2, 3), norm='ortho')
#
#         x = torch.cat([x1.unsqueeze(2), x2.unsqueeze(2)], dim=2).reshape(B, 2 * C, a, b)
#         x = self.post_norm(x)
#         return x


class GFDE(nn.Module):
    def __init__(self, dim):
        super(GFDE, self).__init__()
        self.relu = nn.ReLU(True)
        self.filter_i = GlobalLocalFilter(dim)
        self.detail_i = nn.Conv2d(dim, 1, 1, 1, 0)
        self.filter_a = GlobalLocalFilter(dim)
        self.detail_a = nn.Conv2d(dim, 1, 1, 1, 0)
        self.filter_d = GlobalLocalFilter(dim)
        self.detail_d = nn.Conv2d(dim, 1, 1, 1, 0)

    def forward(self, feat_i, feat_a, feat_d):
        detail_i = self.detail_i(self.relu(self.filter_i(feat_i)))
        detail_a = self.detail_a(self.relu(self.filter_a(feat_a)))
        detail_d = self.detail_d(self.relu(self.filter_d(feat_d)))

        return detail_i, detail_a, detail_d


# Context Exploration block
class DWPW(nn.Module):
    def __init__(self, dim):
        super(DWPW, self).__init__()
        self.dw1_i = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim)
        self.dw1_a = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim)
        self.dw1_d = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim)

        self.pw1_i = nn.Conv2d(dim, dim, 1, 1, 0)
        self.pw1_a = nn.Conv2d(dim, dim, 1, 1, 0)
        self.pw1_d = nn.Conv2d(dim, dim, 1, 1, 0)

        self.dw2_i = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim)
        self.dw2_a = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim)
        self.dw2_d = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim)

        self.pw2_i = nn.Conv2d(dim, dim, 1, 1, 0)
        self.pw2_a = nn.Conv2d(dim, dim, 1, 1, 0)
        self.pw2_d = nn.Conv2d(dim, dim, 1, 1, 0)

        self.relu = nn.ReLU()

    def forward(self, feat_i, feat_a, feat_d):
        residual_i = feat_i
        feat_i = self.relu(self.pw1_i(self.dw1_i(feat_i)))
        feat_i = self.relu(residual_i + self.pw2_i(self.dw2_i(feat_i)))

        residual_a = feat_a
        feat_a = self.relu(self.pw1_a(self.dw1_a(feat_a)))
        feat_a = self.relu(residual_a + self.pw2_a(self.dw2_a(feat_a)))

        residual_d = feat_d
        feat_d = self.relu(self.pw1_d(self.dw1_d(feat_d)))
        feat_d = self.relu(residual_d + self.pw2_d(self.dw2_d(feat_d)))

        return feat_i, feat_a, feat_d


# IAD Interaction block
class IAD_Interaction(nn.Module):
    def __init__(self, dim):
        super(IAD_Interaction, self).__init__()
        dim_mid = dim // 4
        dim_cc = dim_mid * 7

        self.conv1_i = nn.Sequential(nn.Conv2d(dim, dim, 7, 1, 3, groups=dim),
                                     nn.Conv2d(dim, dim_mid, 1, 1, 0), nn.ReLU(True))
        self.conv1_a = nn.Sequential(nn.Conv2d(dim, dim, 7, 1, 3, groups=dim),
                                     nn.Conv2d(dim, dim_mid, 1, 1, 0), nn.ReLU(True))
        self.conv1_d = nn.Sequential(nn.Conv2d(dim, dim, 7, 1, 3, groups=dim),
                                     nn.Conv2d(dim, dim_mid, 1, 1, 0), nn.ReLU(True))

        self.conv2_i = nn.Sequential(nn.Conv2d(dim_cc, dim_cc, 7, 1, 3, groups=dim_cc),
                                     nn.Conv2d(dim_cc, dim, 1, 1, 0), CALayer(dim, reduction=4), SALayer())
        self.conv2_a = nn.Sequential(nn.Conv2d(dim_cc, dim_cc, 7, 1, 3, groups=dim_cc),
                                     nn.Conv2d(dim_cc, dim, 1, 1, 0), CALayer(dim, reduction=4), SALayer())
        self.conv2_d = nn.Sequential(nn.Conv2d(dim_cc, dim_cc, 7, 1, 3, groups=dim_cc),
                                     nn.Conv2d(dim_cc, dim, 1, 1, 0), CALayer(dim, reduction=4), SALayer())

        self.relu = nn.ReLU(True)

    def forward(self, in_i, in_a, in_d):
        feat_i = self.conv1_i(in_i)
        feat_a = self.conv1_a(in_a)
        feat_d = self.conv1_d(in_d)

        feat_ia = feat_i * feat_a
        feat_id = feat_i * feat_d
        feat_ad = feat_a * feat_d
        feat_iad = feat_i * feat_a * feat_d

        feat_cc = torch.cat([feat_i, feat_a, feat_d, feat_ia, feat_id, feat_ad, feat_iad], 1)

        out_i = self.relu(in_i + self.conv2_i(feat_cc))
        out_a = self.relu(in_a + self.conv2_a(feat_cc))
        out_d = self.relu(in_d + self.conv2_d(feat_cc))

        return out_i, out_a, out_d


# IAD Interaction Context Exploration block
class IAD_DWPW(nn.Module):
    def __init__(self, dim):
        super(IAD_DWPW, self).__init__()
        self.iad = IAD_Interaction(dim)

        self.dw1_i = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim)
        self.dw1_a = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim)
        self.dw1_d = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim)

        self.pw1_i = nn.Conv2d(dim, dim, 1, 1, 0)
        self.pw1_a = nn.Conv2d(dim, dim, 1, 1, 0)
        self.pw1_d = nn.Conv2d(dim, dim, 1, 1, 0)

        self.dw2_i = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim)
        self.dw2_a = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim)
        self.dw2_d = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim)

        self.pw2_i = nn.Conv2d(dim, dim, 1, 1, 0)
        self.pw2_a = nn.Conv2d(dim, dim, 1, 1, 0)
        self.pw2_d = nn.Conv2d(dim, dim, 1, 1, 0)

        self.relu = nn.ReLU(True)

    def forward(self, feat_i, feat_a, feat_d):
        feat_i, feat_a, feat_d = self.iad(feat_i, feat_a, feat_d)

        residual_i = feat_i
        feat_i = self.relu(self.pw1_i(self.dw1_i(feat_i)))
        feat_i = self.relu(residual_i + self.pw2_i(self.dw2_i(feat_i)))

        residual_a = feat_a
        feat_a = self.relu(self.pw1_a(self.dw1_a(feat_a)))
        feat_a = self.relu(residual_a + self.pw2_a(self.dw2_a(feat_a)))

        residual_d = feat_d
        feat_d = self.relu(self.pw1_d(self.dw1_d(feat_d)))
        feat_d = self.relu(residual_d + self.pw2_d(self.dw2_d(feat_d)))

        return feat_i, feat_a, feat_d


class DCT_Prediction(nn.Module):
    def __init__(self, dim):
        super(DCT_Prediction, self).__init__()
        self.filter = GlobalLocalFilter(dim)
        self.relu = nn.ReLU(True)
        self.prediction = nn.Conv2d(dim, 1, 1, 1, 0)

    def forward(self, x):
        filter = self.relu(self.filter(x))
        prediction = self.prediction(filter)

        return prediction


class Contrast_Prediction(nn.Module):
    def __init__(self, dim):
        super(Contrast_Prediction, self).__init__()
        self.local = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.context = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim)

        self.relu = nn.ReLU(True)

        self.prediction = nn.Conv2d(dim, 1, 1, 1, 0)

    def forward(self, x):
        local = self.local(x)
        context = self.context(x)
        contrast = self.relu(local - context)

        prediction = self.prediction(contrast)

        return prediction


class Rich_Contrast_Prediction(nn.Module):
    def __init__(self, dim):
        super(Rich_Contrast_Prediction, self).__init__()
        self.f1 = nn.Conv2d(dim, dim, 3, 1, 1, dilation=1, groups=dim)
        self.f2 = nn.Conv2d(dim, dim, 3, 1, 2, dilation=2, groups=dim)
        self.f3 = nn.Conv2d(dim, dim, 3, 1, 3, dilation=3, groups=dim)
        self.f4 = nn.Conv2d(dim, dim, 3, 1, 4, dilation=4, groups=dim)

        self.relu = nn.ReLU(True)

        self.prediction = nn.Conv2d(dim * 6, 1, 1, 1, 0)

    def forward(self, x):
        f1 = self.f1(x)
        f2 = self.f2(x)
        f3 = self.f3(x)
        f4 = self.f4(x)

        c1 = self.relu(f1 - f2)
        c2 = self.relu(f1 - f3)
        c3 = self.relu(f1 - f4)
        c4 = self.relu(f2 - f3)
        c5 = self.relu(f2 - f4)
        c6 = self.relu(f3 - f4)

        c = torch.cat((c1, c2, c3, c4, c5, c6), 1)

        prediction = self.prediction(c)

        return prediction


class Multiscale_Prediction(nn.Module):
    def __init__(self, dim):
        super(Multiscale_Prediction, self).__init__()
        self.conv1 = nn.Conv2d(dim, 1, 1, 1, 0)
        self.conv3 = nn.Conv2d(dim, 1, 3, 1, 1)
        self.conv5 = nn.Conv2d(dim, 1, 5, 1, 2)
        self.conv7 = nn.Conv2d(dim, 1, 7, 1, 3)

        self.fusion = nn.Conv2d(4, 1, 1, 1, 0)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv3 = self.conv3(x)
        conv5 = self.conv5(x)
        conv7 = self.conv7(x)

        conv1357 = torch.cat((conv1, conv3, conv5, conv7), 1)

        fusion = self.fusion(conv1357)

        return fusion


class ESA_Multiscale_Prediction(nn.Module):
    def __init__(self, dim):
        super(ESA_Multiscale_Prediction, self).__init__()
        self.esa = ESA(dim // 2, dim, nn.Conv2d)
        self.conv1 = nn.Conv2d(dim, 1, 1, 1, 0)
        self.conv3 = nn.Conv2d(dim, 1, 3, 1, 1)
        self.conv5 = nn.Conv2d(dim, 1, 5, 1, 2)
        self.conv7 = nn.Conv2d(dim, 1, 7, 1, 3)

        self.fusion = nn.Conv2d(4, 1, 1, 1, 0)

    def forward(self, x):
        x = self.esa(x)
        conv1 = self.conv1(x)
        conv3 = self.conv3(x)
        conv5 = self.conv5(x)
        conv7 = self.conv7(x)

        conv1357 = torch.cat((conv1, conv3, conv5, conv7), 1)

        fusion = self.fusion(conv1357)

        return fusion


from train.model.ffc import FFCResnetBlock
class FFC_IAD(nn.Module):
    def __init__(self, dim, block_number=2):
        super(FFC_IAD, self).__init__()
        self.block_number = block_number
        module_i = []
        module_a = []
        module_d = []
        for _ in range(self.block_number):
            module_i += [FFCResnetBlock(dim, padding_type='reflect', activation_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, inline=True)]
            module_a += [FFCResnetBlock(dim, padding_type='reflect', activation_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, inline=True)]
            module_d += [FFCResnetBlock(dim, padding_type='reflect', activation_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, inline=True)]
        self.module_i = nn.Sequential(*module_i)
        self.module_a = nn.Sequential(*module_a)
        self.module_d = nn.Sequential(*module_d)

    def forward(self, x_i, x_a, x_d):
        y_i = self.module_i(x_i)
        y_a = self.module_a(x_a)
        y_d = self.module_d(x_d)

        return y_i, y_a, y_d


class FFC_IAD_Gated(nn.Module):
    def __init__(self, dim, block_number=2, gate=False):
        super(FFC_IAD_Gated, self).__init__()
        self.gate = gate
        self.block_number = block_number
        module_i = []
        module_a = []
        module_d = []
        for _ in range(self.block_number):
            module_i += [FFCResnetBlock(dim, padding_type='reflect', activation_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, inline=True)]
            module_a += [FFCResnetBlock(dim, padding_type='reflect', activation_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, inline=True)]
            module_d += [FFCResnetBlock(dim, padding_type='reflect', activation_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, inline=True)]
        self.module_i = nn.Sequential(*module_i)
        self.module_a = nn.Sequential(*module_a)
        self.module_d = nn.Sequential(*module_d)

        if self.gate:
            self.gate_i = nn.Conv2d(dim * 3, 3, 3, 1, 1)
            self.gate_a = nn.Conv2d(dim * 3, 3, 3, 1, 1)
            self.gate_d = nn.Conv2d(dim * 3, 3, 3, 1, 1)

            self.sigmoid = nn.Sigmoid()

    def forward(self, x_i, x_a, x_d):
        y_i = self.module_i(x_i)
        y_a = self.module_a(x_a)
        y_d = self.module_d(x_d)

        if self.gate:
            concat = torch.cat((y_i, y_a, y_d), 1)
            gate_i = self.gate_i(concat)
            gate_i_i, gate_i_a, gate_i_d = torch.split(gate_i, 1, dim=1)
            z_i = y_i + self.sigmoid(gate_i_i) * y_i + self.sigmoid(gate_i_a) * y_a + self.sigmoid(gate_i_d) * y_d

            gate_a = self.gate_a(concat)
            gate_a_i, gate_a_a, gate_a_d = torch.split(gate_a, 1, dim=1)
            z_a = y_a + self.sigmoid(gate_a_i) * y_i + self.sigmoid(gate_a_a) * y_a + self.sigmoid(gate_a_d) * y_d

            gate_d = self.gate_d(concat)
            gate_d_i, gate_d_a, gate_d_d = torch.split(gate_d, 1, dim=1)
            z_d = y_d + self.sigmoid(gate_d_i) * y_i + self.sigmoid(gate_d_a) * y_a + self.sigmoid(gate_d_d) * y_d

            return z_i, z_a, z_d

        return y_i, y_a, y_d


# Modified version of FFC
class SFB(nn.Module):
    def __init__(self, dim):
        super(SFB, self).__init__()
        self.s_clc = nn.Sequential(nn.Conv2d(dim, dim, 3, 1, 1), nn.LeakyReLU(True), nn.Conv2d(dim, dim, 3, 1, 1))

        self.f_cl0 = nn.Sequential(nn.Conv2d(dim, dim, 1, 1, 0), nn.LeakyReLU(True))
        self.f_cl1 = nn.Sequential(nn.Conv2d(dim * 2, dim * 2, 1, 1, 0), nn.LeakyReLU(True))
        self.f_c = nn.Conv2d(dim, dim, 1, 1, 0)

        self.fusion = nn.Sequential(nn.Conv2d(dim * 2, dim, 1, 1, 0), nn.LeakyReLU(True))

    def forward(self, x):
        s = x + self.s_clc(x)

        f_cl0 = self.f_cl0(x)
        # real fft2d
        batch = f_cl0.shape[0]
        fft_dim = (-2, -1)
        ffted = torch.fft.rfftn(f_cl0, dim=fft_dim, norm='ortho')
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        # f_cl1
        ffted = self.f_cl1(ffted)  # (batch, c*2, h, w/2+1)
        # inverse real fft2d
        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])
        ifft_shape_slice = f_cl0.shape[-2:]
        iffted = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm='ortho')
        # f_c
        f = self.f_c(f_cl0 + iffted)

        fusion = self.fusion(torch.cat((s, f), 1))

        return fusion


class SFB_IAD(nn.Module):
    def __init__(self, dim):
        super(SFB_IAD, self).__init__()
        self.sfb1_i = SFB(dim)
        self.sfb2_i = SFB(dim)

        self.sfb1_a = SFB(dim)
        self.sfb2_a = SFB(dim)

        self.sfb1_d = SFB(dim)
        self.sfb2_d = SFB(dim)

    def forward(self, x_i, x_a, x_d):
        sfb1_i = self.sfb1_i(x_i)
        sfb2_i = self.sfb2_i(sfb1_i)

        sfb1_a = self.sfb1_a(x_a)
        sfb2_a = self.sfb2_a(sfb1_a)

        sfb1_d = self.sfb1_d(x_d)
        sfb2_d = self.sfb2_d(sfb1_d)

        return sfb2_i, sfb2_a, sfb2_d


class F_Branch(nn.Module):
    def __init__(self, dim):
        super(F_Branch, self).__init__()
        self.f_cl0 = nn.Sequential(nn.Conv2d(dim, dim, 1, 1, 0), nn.LeakyReLU(True))
        self.f_cl1 = nn.Sequential(nn.Conv2d(dim * 2, dim * 2, 1, 1, 0), nn.LeakyReLU(True))
        self.f_c = nn.Conv2d(dim, dim, 1, 1, 0)

    def forward(self, x):
        f_cl0 = self.f_cl0(x)
        # real fft2d
        batch = f_cl0.shape[0]
        fft_dim = (-2, -1)
        ffted = torch.fft.rfftn(f_cl0, dim=fft_dim, norm='ortho')
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        # f_cl1
        ffted = self.f_cl1(ffted)  # (batch, c*2, h, w/2+1)
        # inverse real fft2d
        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])
        ifft_shape_slice = f_cl0.shape[-2:]
        iffted = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm='ortho')
        # f_c
        f = self.f_c(f_cl0 + iffted)

        return f


class SFB_IAD_Coupled(nn.Module):
    def __init__(self, dim):
        super(SFB_IAD_Coupled, self).__init__()
        self.s1_couple = Coupled_Layer()
        self.f1_i = F_Branch(dim)
        self.f1_a = F_Branch(dim)
        self.f1_d = F_Branch(dim)
        self.sf1_i = nn.Sequential(nn.Conv2d(dim * 2, dim, 1, 1, 0), nn.LeakyReLU(True))
        self.sf1_a = nn.Sequential(nn.Conv2d(dim * 2, dim, 1, 1, 0), nn.LeakyReLU(True))
        self.sf1_d = nn.Sequential(nn.Conv2d(dim * 2, dim, 1, 1, 0), nn.LeakyReLU(True))

        self.s2_couple = Coupled_Layer()
        self.f2_i = F_Branch(dim)
        self.f2_a = F_Branch(dim)
        self.f2_d = F_Branch(dim)
        self.sf2_i = nn.Sequential(nn.Conv2d(dim * 2, dim, 1, 1, 0), nn.LeakyReLU(True))
        self.sf2_a = nn.Sequential(nn.Conv2d(dim * 2, dim, 1, 1, 0), nn.LeakyReLU(True))
        self.sf2_d = nn.Sequential(nn.Conv2d(dim * 2, dim, 1, 1, 0), nn.LeakyReLU(True))

    def forward(self, x_i, x_a, x_d):
        s1_i, s1_a, s1_d = self.s1_couple(x_i, x_a, x_d)
        f1_i = self.f1_i(x_i)
        f1_a = self.f1_a(x_a)
        f1_d = self.f1_d(x_d)
        sf1_i = self.sf1_i(torch.cat((s1_i, f1_i), 1))
        sf1_a = self.sf1_a(torch.cat((s1_a, f1_a), 1))
        sf1_d = self.sf1_d(torch.cat((s1_d, f1_d), 1))

        s2_i, s2_a, s2_d = self.s2_couple(sf1_i, sf1_a, sf1_d)
        f2_i = self.f2_i(sf1_i)
        f2_a = self.f2_a(sf1_a)
        f2_d = self.f2_d(sf1_d)
        sf2_i = self.sf2_i(torch.cat((s2_i, f2_i), 1))
        sf2_a = self.sf2_a(torch.cat((s2_a, f2_a), 1))
        sf2_d = self.sf2_d(torch.cat((s2_d, f2_d), 1))

        return sf2_i, sf2_a, sf2_d


###################################################################################################
############################## Embedding Consistency Loss #########################################
###################################################################################################
class Up(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch, norm_layer, leaky=True, bilinear=True):
        super(Up, self).__init__()

        if leaky is True:
            relu = nn.LeakyReLU
            param = (0.2, True)
        else:
            relu = nn.ReLU
            param = [True]

        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
                norm_layer(out_ch),
                relu(*param)
            )
        else:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
                norm_layer(out_ch),
                relu(*param)
            )

    def forward(self, x):
        x = self.up(x)
        return x


class RB(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch, norm_layer, leaky=True, down_ch=False):
        super(RB, self).__init__()

        if leaky is True:
            relu = nn.LeakyReLU
            param = (0.2, True)
        else:
            relu = nn.ReLU
            param = [True]

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            norm_layer(out_ch),
            relu(*param),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            norm_layer(out_ch))

        if down_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=True),
                norm_layer(out_ch)
            )
        else:
            self.shortcut = nn.Sequential()

        self.relu_out = relu(*param)

    def forward(self, x):

        identity = x

        x = self.conv(x)
        x = x + self.shortcut(identity)
        x = self.relu_out(x)

        return x


class Autoencoder(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, pool_layer=nn.MaxPool2d, leaky=True,
                 upsampling='bilinear', return_att=False, n1=16, iters=3, return_embedding=True):
        super(Autoencoder, self).__init__()

        filters = [n1, n1 * 2, n1 * 4, n1 * 8]
        self.filters = filters

        self.iters = iters

        self.return_att = return_att
        self.return_embedding = return_embedding

        if pool_layer != 'conv':
            self.Maxpool1 = pool_layer(kernel_size=2, stride=2)
            self.Maxpool2 = pool_layer(kernel_size=2, stride=2)
            self.Maxpool3 = pool_layer(kernel_size=2, stride=2)
        else:
            self.Maxpool1 = nn.Conv2d(self.filters[0], self.filters[0], kernel_size=2, stride=2, padding=0)
            self.Maxpool2 = nn.Conv2d(self.filters[1], self.filters[1], kernel_size=2, stride=2, padding=0)
            self.Maxpool3 = nn.Conv2d(self.filters[2], self.filters[2], kernel_size=2, stride=2, padding=0)

        if leaky is True:
            relu = nn.LeakyReLU
            param = (0.2, True)
        else:
            relu = nn.ReLU
            param = [True]

        self.Conv_input = nn.Sequential(nn.Conv2d(1, self.filters[0], kernel_size=3, stride=1, padding=1),
                                        nn.InstanceNorm2d(self.filters[0]),
                                        relu(*param))

        self.Conv1 = RB(self.filters[0], self.filters[0], norm_layer, leaky=leaky, down_ch=False)
        self.Conv2 = RB(self.filters[0], self.filters[1], norm_layer, leaky=leaky, down_ch=True)
        self.Conv3 = RB(self.filters[1], self.filters[2], norm_layer, leaky=leaky, down_ch=True)
        self.Conv4 = RB(self.filters[2], self.filters[3], norm_layer, leaky=leaky, down_ch=True)

        self.Up4 = Up(self.filters[3], self.filters[2], norm_layer, leaky=leaky, bilinear=upsampling == 'bilinear')
        self.Up_conv4 = RB(self.filters[2], self.filters[2], norm_layer, leaky=leaky, down_ch=True)

        self.Up3 = Up(self.filters[2], self.filters[1], norm_layer, leaky=leaky, bilinear=upsampling == 'bilinear')
        self.Up_conv3 = RB(self.filters[1], self.filters[1], norm_layer, leaky=leaky, down_ch=True)

        self.Up2 = Up(self.filters[1], self.filters[0], norm_layer, leaky=leaky, bilinear=upsampling == 'bilinear')
        self.Up_conv2 = RB(self.filters[0], self.filters[0], norm_layer, leaky=leaky, down_ch=True)

        self.Conv = nn.Conv2d(self.filters[0], 1, kernel_size=1, stride=1, padding=0)

    def forward(self, input):

        x = input

        x_in = x

        x_in = self.Conv_input(x_in)

        e1 = self.Conv1(x_in)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)


        d4 = self.Up4(e4)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        x = out

        if self.return_embedding:
            return x, e4
        else:
            return x
